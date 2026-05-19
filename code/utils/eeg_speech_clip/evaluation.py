from __future__ import annotations

from typing import Dict

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norm, eps, None)


def retrieval_metrics(
    eeg_emb: np.ndarray,
    speech_emb: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    eeg_z = l2_normalize(eeg_emb)
    speech_z = l2_normalize(speech_emb)
    sim = eeg_z @ speech_z.T
    n = int(sim.shape[0])
    if n == 0:
        return {}

    labels = np.arange(n)
    e2s_order = np.argsort(-sim, axis=1)
    s2e_order = np.argsort(-sim.T, axis=1)
    e2s_rank = np.array([int(np.where(e2s_order[i] == i)[0][0]) + 1 for i in labels])
    s2e_rank = np.array([int(np.where(s2e_order[i] == i)[0][0]) + 1 for i in labels])
    pos = np.diag(sim)
    neg_mask = ~np.eye(n, dtype=bool)
    neg = sim[neg_mask]

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}n": float(n),
        f"{p}eeg_to_speech_top1": float(np.mean(e2s_rank == 1)),
        f"{p}speech_to_eeg_top1": float(np.mean(s2e_rank == 1)),
        f"{p}eeg_to_speech_top5": float(np.mean(e2s_rank <= min(5, n))),
        f"{p}speech_to_eeg_top5": float(np.mean(s2e_rank <= min(5, n))),
        f"{p}eeg_to_speech_mean_rank": float(np.mean(e2s_rank)),
        f"{p}speech_to_eeg_mean_rank": float(np.mean(s2e_rank)),
        f"{p}positive_cosine_mean": float(np.mean(pos)),
        f"{p}negative_cosine_mean": float(np.mean(neg)) if neg.size else float("nan"),
        f"{p}pos_minus_neg": float(np.mean(pos) - np.mean(neg)) if neg.size else float("nan"),
    }


def aggregate_by_key(
    eeg_emb: np.ndarray,
    speech_emb: np.ndarray,
    keys: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average duplicate pair-key embeddings before one-positive retrieval."""
    if len(keys) != int(eeg_emb.shape[0]) or len(keys) != int(speech_emb.shape[0]):
        raise ValueError("keys length must match embedding rows")
    order = sorted(set(keys))
    eeg_rows = []
    speech_rows = []
    for key in order:
        idx = [i for i, item in enumerate(keys) if item == key]
        eeg_rows.append(np.mean(eeg_emb[idx], axis=0))
        speech_rows.append(np.mean(speech_emb[idx], axis=0))
    if not eeg_rows:
        return np.empty((0, 0)), np.empty((0, 0)), []
    return np.vstack(eeg_rows), np.vstack(speech_rows), order


def shifted_speech_metrics(
    eeg_emb: np.ndarray,
    speech_emb: np.ndarray,
    shift: int = 1,
    prefix: str = "shift",
) -> Dict[str, float]:
    if len(speech_emb) == 0:
        return {}
    shifted = np.roll(np.asarray(speech_emb), int(shift), axis=0)
    return retrieval_metrics(eeg_emb, shifted, prefix=prefix)


def material_shift_metrics(
    eeg_emb: np.ndarray,
    speech_emb: np.ndarray,
    material_keys: list[str],
    prefix: str = "material_shift",
) -> Dict[str, float]:
    if len(material_keys) != int(speech_emb.shape[0]):
        raise ValueError("material_keys length must match embedding rows")
    if len(speech_emb) == 0:
        return {}
    order = np.arange(len(speech_emb))
    unique = sorted(set(material_keys))
    if len(unique) < 2:
        return retrieval_metrics(eeg_emb, np.roll(speech_emb, 1, axis=0), prefix=prefix)
    shifted = order.copy()
    for i, key in enumerate(material_keys):
        candidates = np.where(np.asarray(material_keys) != key)[0]
        shifted[i] = candidates[i % len(candidates)]
    return retrieval_metrics(eeg_emb, speech_emb[shifted], prefix=prefix)
