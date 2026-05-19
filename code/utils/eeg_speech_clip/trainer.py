from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import safe_name
from .dataset import (
    SpeechEEGWindowDataset,
    UniquePairKeyBatchSampler,
    WindowRecord,
    collate_torch,
)
from .evaluation import (
    aggregate_by_key,
    material_shift_metrics,
    retrieval_metrics,
    shifted_speech_metrics,
)
from .losses import ClipStyleInfoNCELoss
from .models import SpeechEEGClipModel

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return resolved


def make_loader(
    records: Sequence[WindowRecord],
    dataset: SpeechEEGWindowDataset,
    batch_size: int,
    seed: int,
    num_workers: int,
    shuffle_unique_pairs: bool,
) -> DataLoader:
    if shuffle_unique_pairs:
        sampler = UniquePairKeyBatchSampler(
            records=records,
            batch_size=batch_size,
            seed=seed,
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_torch,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_torch,
    )


@torch.no_grad()
def collect_embeddings(
    model: SpeechEEGClipModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    eeg_all = []
    speech_all = []
    keys: list[str] = []
    material_keys: list[str] = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        speech = batch["speech"].to(device)
        eeg_emb, speech_emb = model(eeg, speech)
        eeg_all.append(eeg_emb.detach().cpu().numpy())
        speech_all.append(speech_emb.detach().cpu().numpy())
        keys.extend(batch["pair_key"])
        material_keys.extend(batch["material_key"])
    if not eeg_all:
        return np.empty((0, 0)), np.empty((0, 0)), [], []
    return np.vstack(eeg_all), np.vstack(speech_all), keys, material_keys


@torch.no_grad()
def eval_loss(
    model: SpeechEEGClipModel,
    loss_fn: ClipStyleInfoNCELoss,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        speech = batch["speech"].to(device)
        eeg_emb, speech_emb = model(eeg, speech)
        loss = loss_fn(eeg_emb, speech_emb)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_one_fold(
    fold_name: str,
    train_records: Sequence[WindowRecord],
    val_records: Sequence[WindowRecord],
    provider,
    eeg_root: Path,
    target_fs: float,
    window_sec: float,
    speech_dim: int,
    out_dir: Path,
    config: Dict[str, object],
) -> dict[str, object]:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 0))

    train_dataset = SpeechEEGWindowDataset(
        train_records,
        provider=provider,
        eeg_root=eeg_root,
        target_fs=target_fs,
        window_sec=window_sec,
    )
    val_dataset = SpeechEEGWindowDataset(
        val_records,
        provider=provider,
        eeg_root=eeg_root,
        target_fs=target_fs,
        window_sec=window_sec,
    )
    train_loader = make_loader(
        records=train_records,
        dataset=train_dataset,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        shuffle_unique_pairs=True,
    )
    val_loader = make_loader(
        records=val_records,
        dataset=val_dataset,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        shuffle_unique_pairs=True,
    )

    model = SpeechEEGClipModel(
        speech_dim=speech_dim,
        n_eeg_channels=31,
        d_proj=int(config.get("d_proj", 128)),
    ).to(device)
    loss_fn = ClipStyleInfoNCELoss(
        init_temperature=float(config.get("init_temperature", 0.07))
    ).to(device)
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    epochs = int(config.get("epochs", 30))
    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 1e-5))
    save_checkpoints = bool(config.get("save_checkpoints", False))

    best_val = float("inf")
    best_state = None
    stale = 0
    history: list[dict[str, object]] = []

    LOGGER.info(
        "Fold %s: train=%d val=%d device=%s", fold_name, len(train_records), len(val_records), device
    )
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            speech = batch["speech"].to(device)
            eeg_emb, speech_emb = model(eeg, speech)
            loss = loss_fn(eeg_emb, speech_emb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = eval_loss(model, loss_fn, val_loader, device)
        row = {
            "fold": fold_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "temperature": loss_fn.temperature(),
        }
        history.append(row)
        LOGGER.info(
            "Fold %s epoch %d/%d train_loss=%.5f val_loss=%.5f temp=%.5f",
            fold_name,
            epoch,
            epochs,
            train_loss,
            val_loss,
            loss_fn.temperature(),
        )

        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "loss": loss_fn.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        loss_fn.load_state_dict(best_state["loss"])

    fold_dir = out_dir / "folds" / safe_name(fold_name)
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_history(fold_dir / "history.csv", history)

    if save_checkpoints and best_state is not None:
        torch.save(best_state, fold_dir / "best.pt")

    eeg_emb, speech_emb, pair_keys, material_keys_raw = collect_embeddings(model, val_loader, device)
    eeg_eval, speech_eval, unique_pair_keys = aggregate_by_key(eeg_emb, speech_emb, pair_keys)
    material_by_pair = {}
    for pair_key, material_key in zip(pair_keys, material_keys_raw):
        material_by_pair.setdefault(pair_key, material_key)
    material_keys_eval = [material_by_pair[key] for key in unique_pair_keys]
    metrics = retrieval_metrics(eeg_eval, speech_eval)
    metrics.update(shifted_speech_metrics(eeg_eval, speech_eval, shift=1, prefix="time_shift"))
    metrics.update(material_shift_metrics(eeg_eval, speech_eval, material_keys_eval, prefix="material_shift"))
    metrics.update(
        {
            "fold": fold_name,
            "n_train_windows": len(train_records),
            "n_val_windows": len(val_records),
            "n_val_pair_keys": len(unique_pair_keys),
            "best_val_loss": float(best_val),
            "best_epoch": int(best_state["epoch"]) if best_state else 0,
            "final_temperature": loss_fn.temperature(),
        }
    )

    np.savez_compressed(
        fold_dir / "val_embeddings.npz",
        eeg=eeg_emb.astype(np.float32),
        speech=speech_emb.astype(np.float32),
        pair_key=np.asarray(pair_keys),
        eeg_pair_mean=eeg_eval.astype(np.float32),
        speech_pair_mean=speech_eval.astype(np.float32),
        unique_pair_key=np.asarray(unique_pair_keys),
    )
    (fold_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics
