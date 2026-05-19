"""
Stage 3: CLIP/CLAP-style speech-EEG contrastive baseline.

This stage is intentionally independent from the TRF/encoding-model analyses.
It keeps EEG preprocessing minimal, uses speaker+emotion as the unique
stimulus material key, and trains a frozen-speech / EEGNet dual encoder with
classic symmetric InfoNCE.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from utils.eeg_speech_clip.config import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SPEECH_MODELS,
    SpeechConfig,
    TrainConfig,
    WindowConfig,
    parse_int_list,
    resolve_speech_model,
    resolve_speech_source,
    resolve_subjects,
    safe_name,
    speech_cache_dir,
)
from utils.eeg_speech_clip.dataset import (
    CachedSpeechProvider,
    ExistingWhisperProvider,
    build_window_records,
    make_folds,
    records_to_frame,
    summarize_records,
)
from utils.eeg_speech_clip.speech_features import (
    build_material_speech_cache,
    materials_for_video_ids,
)
from utils.eeg_whisper_modeling.data_loader import (
    DEFAULT_EEG_ROOT,
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_EMBEDDING_ROOT_V2,
)


LOGGER = logging.getLogger("stage3_speech_eeg_clip")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 speech-EEG CLIP baseline.")
    parser.add_argument("--subjects", default=None, help="Comma-separated EEG listener subject IDs.")
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--exclude-subjects", default="")
    parser.add_argument("--video-ids", default=None, help="Comma-separated video IDs. Default: 1..28.")
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--embedding-root-v2", type=Path, default=DEFAULT_EMBEDDING_ROOT_V2)
    parser.add_argument("--embedding-tag", default="whisper_base_last")

    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument(
        "--eeg-target-fs",
        type=float,
        default=250.0,
        help="Target EEG fs after minimal window preprocessing. 250 keeps original sampling.",
    )
    parser.add_argument(
        "--max-windows-per-trial",
        type=int,
        default=0,
        help="Debug limiter; 0 means use all windows with available speech features.",
    )

    parser.add_argument(
        "--speech-backend",
        default="whisper",
        choices=["whisper", "wav2vec2", "hubert"],
    )
    parser.add_argument(
        "--speech-source",
        default="auto",
        choices=["auto", "existing-whisper", "cache"],
        help="existing-whisper uses current .npy caches; cache uses Stage-3 speech .npz caches.",
    )
    parser.add_argument("--speech-model-id", default=None)
    parser.add_argument("--speech-model-path", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--prepare-speech-cache",
        action="store_true",
        help="Build material-level speech cache before windowing/training.",
    )
    parser.add_argument("--speech-cache-batch-size", type=int, default=64)

    parser.add_argument("--split-unit", default="material", choices=["material", "video"])
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument(
        "--fold-start",
        type=int,
        default=0,
        help="Start index (inclusive) into the sorted fold list for fold-level parallelism.",
    )
    parser.add_argument(
        "--fold-end",
        type=int,
        default=0,
        help="End index (exclusive) into the sorted fold list. 0 means use all remaining folds.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--d-proj", type=int, default=128)
    parser.add_argument("--init-temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is not available instead of falling back to CPU.",
    )
    parser.add_argument("--save-checkpoints", action="store_true")

    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def make_provider(args: argparse.Namespace, speech_source: str, cache_dir: Path):
    if speech_source == "existing-whisper":
        if args.speech_backend != "whisper":
            raise ValueError("--speech-source existing-whisper is only valid with --speech-backend whisper")
        return ExistingWhisperProvider(
            embedding_root=args.embedding_root,
            embedding_root_v2=args.embedding_root_v2,
            embedding_tag=args.embedding_tag,
            min_window_coverage=0.95,
        )
    if speech_source == "cache":
        return CachedSpeechProvider(cache_dir=cache_dir)
    raise ValueError(f"Unsupported speech source: {speech_source}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(args: argparse.Namespace, run_name: str) -> Path:
    out_dir = args.output_root / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir exists and is not empty: {out_dir}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def require_cuda_if_requested(require_cuda: bool) -> None:
    if not require_cuda:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "--require-cuda was set, but torch.cuda.is_available() is False. "
            "Check NVIDIA driver visibility and PyTorch CUDA compatibility."
        )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    video_ids = parse_int_list(args.video_ids, range(1, 29))
    subjects = resolve_subjects(
        eeg_root=args.eeg_root,
        subjects_arg=args.subjects,
        max_subjects=args.max_subjects,
        exclude_subjects=args.exclude_subjects,
    )
    speech_model_id = resolve_speech_model(args.speech_backend, args.speech_model_id)
    speech_source = resolve_speech_source(args.speech_backend, args.speech_source)
    window_cfg = WindowConfig(
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        eeg_target_fs=args.eeg_target_fs,
        max_windows_per_trial=args.max_windows_per_trial,
    )
    speech_cfg = SpeechConfig(
        backend=args.speech_backend,
        source=speech_source,
        model_id=speech_model_id,
        model_path=args.speech_model_path,
        embedding_tag=args.embedding_tag,
        embedding_root=args.embedding_root,
        embedding_root_v2=args.embedding_root_v2,
        local_files_only=args.local_files_only,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        seed=args.seed,
        d_proj=args.d_proj,
        init_temperature=args.init_temperature,
        device=args.device,
        save_checkpoints=args.save_checkpoints,
    )

    model_key = safe_name(speech_model_id)
    run_name = args.run_name or (
        f"clip_{args.speech_backend}_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = prepare_output_dir(args, run_name)
    cache_dir = speech_cache_dir(args.cache_root, args.speech_backend, speech_model_id, window_cfg)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subjects": subjects,
        "video_ids": video_ids,
        "eeg_root": str(args.eeg_root),
        "window": window_cfg.__dict__,
        "speech": {
            **speech_cfg.__dict__,
            "model_path": str(speech_cfg.model_path) if speech_cfg.model_path else None,
            "embedding_root": str(speech_cfg.embedding_root),
            "embedding_root_v2": str(speech_cfg.embedding_root_v2),
        },
        "train": train_cfg.__dict__,
        "require_cuda": bool(args.require_cuda),
        "split_unit": args.split_unit,
        "fold_start": int(args.fold_start),
        "fold_end": int(args.fold_end),
        "cache_dir": str(cache_dir),
        "output_dir": str(out_dir),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.prepare_speech_cache:
        require_cuda_if_requested(bool(args.require_cuda))
        materials = materials_for_video_ids(video_ids)
        build_material_speech_cache(
            materials=materials,
            cache_dir=cache_dir,
            backend=args.speech_backend,
            model_id=speech_model_id,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            batch_size=args.speech_cache_batch_size,
            device=args.device,
            model_path=args.speech_model_path,
            local_files_only=args.local_files_only,
            overwrite=args.overwrite,
        )

    provider = make_provider(args, speech_source=speech_source, cache_dir=cache_dir)
    records = build_window_records(
        subjects=subjects,
        video_ids=video_ids,
        provider=provider,
        eeg_root=args.eeg_root,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        max_windows_per_trial=args.max_windows_per_trial,
    )
    if not records:
        raise RuntimeError(
            "No usable speech-EEG windows were built. Check speech coverage/cache and EEG paths."
        )

    manifest = records_to_frame(records)
    manifest_path = out_dir / "window_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = summarize_records(records)
    (out_dir / "window_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Window summary: %s", summary)

    folds = make_folds(records, split_unit=args.split_unit)
    if args.max_folds and args.max_folds > 0:
        folds = folds[: args.max_folds]
    fold_start = int(args.fold_start) if args.fold_start else 0
    fold_end = int(args.fold_end) if args.fold_end else len(folds)
    if fold_start > 0 or fold_end < len(folds):
        LOGGER.info(
            "Fold slice [%d:%d] of %d total folds (fold-level parallelism).",
            fold_start, fold_end, len(folds),
        )
        folds = folds[fold_start:fold_end]
    fold_rows = [
        {"fold": key, "n_train_windows": len(train_idx), "n_val_windows": len(val_idx)}
        for key, train_idx, val_idx in folds
    ]
    write_csv(out_dir / "fold_plan.csv", fold_rows)
    LOGGER.info("Prepared %d folds by %s", len(folds), args.split_unit)

    if args.dry_run:
        print(json.dumps({"window_summary": summary, "n_folds": len(folds)}, ensure_ascii=False, indent=2))
        return 0

    import torch  # noqa: F401 - fail early with a clear dependency error in train mode
    require_cuda_if_requested(bool(args.require_cuda))
    from utils.eeg_speech_clip.trainer import train_one_fold

    train_options = {
        "epochs": train_cfg.epochs,
        "batch_size": train_cfg.batch_size,
        "lr": train_cfg.lr,
        "weight_decay": train_cfg.weight_decay,
        "patience": train_cfg.patience,
        "min_delta": train_cfg.min_delta,
        "num_workers": train_cfg.num_workers,
        "seed": train_cfg.seed,
        "d_proj": train_cfg.d_proj,
        "init_temperature": train_cfg.init_temperature,
        "device": train_cfg.device,
        "save_checkpoints": train_cfg.save_checkpoints,
    }
    metrics_rows: list[dict[str, object]] = []
    for fold_name, train_idx, val_idx in folds:
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        metrics = train_one_fold(
            fold_name=fold_name,
            train_records=train_records,
            val_records=val_records,
            provider=provider,
            eeg_root=args.eeg_root,
            target_fs=args.eeg_target_fs,
            window_sec=args.window_sec,
            speech_dim=int(summary["speech_dim"]),
            out_dir=out_dir,
            config=train_options,
        )
        metrics_rows.append(metrics)
        write_csv(out_dir / "fold_metrics.csv", metrics_rows)

    numeric_summary = {}
    for key in metrics_rows[0].keys():
        values = []
        for row in metrics_rows:
            value = row.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))
        if values:
            numeric_summary[f"mean_{key}"] = float(np.nanmean(values))
            numeric_summary[f"median_{key}"] = float(np.nanmedian(values))
    (out_dir / "summary.json").write_text(
        json.dumps(numeric_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Done. Results: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
