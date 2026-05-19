from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy import signal

from utils.eeg_whisper_modeling.data_loader import (
    DEFAULT_EEG_ROOT,
    STIMULUS_ROOT_V1,
    STIMULUS_ROOT_V2,
    STIMULUS_VERSION_V1,
    STIMULUS_VERSION_V2,
    load_eeg_trial_data,
    load_embedding_by_video_id,
    load_narrative_identity,
    normalize_stimulus_version,
)
from utils.eeg_whisper_modeling.feature_processor import flatten_embedding

LOGGER = logging.getLogger(__name__)


def make_material_key(speaker_id: object, emotion: object) -> str:
    speaker = str(speaker_id).strip()
    emo = str(emotion).strip().lower()
    return f"speaker-{speaker}_emotion-{emo}"


def make_pair_key(material_key: str, window_idx: int) -> str:
    return f"{material_key}_win-{int(window_idx):05d}"


def find_audio_path(stimulus_version: str, video_id: int) -> Path:
    root = STIMULUS_ROOT_V1 if normalize_stimulus_version(stimulus_version) == STIMULUS_VERSION_V1 else STIMULUS_ROOT_V2
    for suffix in (".wav", ".mp3", ".flac", ".m4a"):
        path = root / f"{int(video_id)}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No audio file for {stimulus_version} video {video_id} in {root}")


@dataclass(frozen=True)
class MaterialInfo:
    stimulus_version: str
    video_id: int
    speaker_id: str
    emotion: str
    material_key: str
    audio_path: str


@dataclass(frozen=True)
class WindowRecord:
    subject: str
    stimulus_version: str
    video_id: int
    trial_idx: int
    speaker_id: str
    emotion: str
    material_key: str
    window_idx: int
    pair_key: str
    t_start: float
    t_end: float
    eeg_start_sample: int
    eeg_n_samples: int
    eeg_fs: float
    speech_dim: int
    speech_source: str
    source_key: str


class SpeechWindowProvider(Protocol):
    source_name: str

    def get_window_feature(
        self,
        stimulus_version: str,
        video_id: int,
        material_key: str,
        t_start: float,
        t_end: float,
        window_idx: int,
    ) -> Optional[np.ndarray]:
        ...


class ExistingWhisperProvider:
    """Read existing frame-wise Whisper .npy files and mean-pool per window."""

    source_name = "existing-whisper"

    def __init__(
        self,
        embedding_root: Path,
        embedding_root_v2: Path,
        embedding_tag: str,
        min_window_coverage: float = 0.95,
    ) -> None:
        self.embedding_root = Path(embedding_root)
        self.embedding_root_v2 = Path(embedding_root_v2)
        self.embedding_tag = embedding_tag
        self.min_window_coverage = float(min_window_coverage)
        self._material_cache: Dict[str, tuple[np.ndarray, np.ndarray, str]] = {}

    def _load_material(
        self,
        stimulus_version: str,
        video_id: int,
        material_key: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if material_key in self._material_cache:
            return self._material_cache[material_key]

        record = load_embedding_by_video_id(
            video_id=int(video_id),
            embedding_type="speech",
            timestamp=self.embedding_tag,
            embedding_root=self.embedding_root,
            embedding_root_v2=self.embedding_root_v2,
            stimulus_version=stimulus_version,
        )
        features, centers, _ = flatten_embedding(record)
        features = np.asarray(features, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        finite = np.isfinite(centers) & np.isfinite(features).all(axis=1)
        if not np.any(finite):
            raise ValueError(f"No finite Whisper frames in {record.path}")
        payload = (features[finite], centers[finite], str(record.path))
        self._material_cache[material_key] = payload
        return payload

    def get_window_feature(
        self,
        stimulus_version: str,
        video_id: int,
        material_key: str,
        t_start: float,
        t_end: float,
        window_idx: int,
    ) -> Optional[np.ndarray]:
        del window_idx
        features, centers, _ = self._load_material(stimulus_version, video_id, material_key)
        mask = (centers >= float(t_start)) & (centers < float(t_end))
        if not np.any(mask):
            return None
        selected = centers[mask]
        if selected.size > 1:
            step = float(np.nanmedian(np.diff(selected)))
        else:
            step = 0.0
        covered = float(np.nanmax(selected) - np.nanmin(selected) + max(step, 0.0))
        required = float(t_end - t_start) * self.min_window_coverage
        if covered + 1e-6 < required:
            return None
        pooled = np.nanmean(features[mask], axis=0).astype(np.float32)
        if not np.isfinite(pooled).all():
            return None
        return pooled


class CachedSpeechProvider:
    """Read material-level speech window features generated by speech_features.py."""

    source_name = "cache"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._cache: Dict[str, dict[int, np.ndarray]] = {}

    def _load_material(self, material_key: str) -> dict[int, np.ndarray]:
        if material_key in self._cache:
            return self._cache[material_key]
        path = self.cache_dir / f"{material_key}.npz"
        if not path.exists():
            self._cache[material_key] = {}
            return self._cache[material_key]
        data = np.load(path, allow_pickle=False)
        window_idx = data["window_idx"].astype(int)
        features = data["features"].astype(np.float32)
        out = {int(idx): features[i] for i, idx in enumerate(window_idx)}
        self._cache[material_key] = out
        return out

    def get_window_feature(
        self,
        stimulus_version: str,
        video_id: int,
        material_key: str,
        t_start: float,
        t_end: float,
        window_idx: int,
    ) -> Optional[np.ndarray]:
        del stimulus_version, video_id, t_start, t_end
        feature = self._load_material(material_key).get(int(window_idx))
        if feature is None or not np.isfinite(feature).all():
            return None
        return np.asarray(feature, dtype=np.float32)


def load_materials(video_ids: Sequence[int]) -> dict[tuple[str, int], MaterialInfo]:
    materials: dict[tuple[str, int], MaterialInfo] = {}
    for version in (STIMULUS_VERSION_V1, STIMULUS_VERSION_V2):
        identity = load_narrative_identity(version)
        for _, row in identity.iterrows():
            video_id = int(row["video_id"])
            if video_id not in set(video_ids):
                continue
            speaker_id = str(row["subject_id"])
            emotion = str(row["target_emotion"]).lower()
            material_key = make_material_key(speaker_id, emotion)
            try:
                audio_path = str(find_audio_path(version, video_id))
            except FileNotFoundError:
                audio_path = ""
            materials[(version, video_id)] = MaterialInfo(
                stimulus_version=version,
                video_id=video_id,
                speaker_id=speaker_id,
                emotion=emotion,
                material_key=material_key,
                audio_path=audio_path,
            )
    return materials


def build_window_records(
    subjects: Sequence[str],
    video_ids: Sequence[int],
    provider: SpeechWindowProvider,
    eeg_root: Path = DEFAULT_EEG_ROOT,
    window_sec: float = 4.0,
    stride_sec: float = 2.0,
    max_windows_per_trial: int = 0,
) -> list[WindowRecord]:
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window_sec and stride_sec must be positive")

    materials = load_materials(video_ids)
    records: list[WindowRecord] = []
    speech_dim: Optional[int] = None

    for subject in subjects:
        eeg_data = load_eeg_trial_data(subject, "1", data_root=Path(eeg_root))
        stimulus_version = normalize_stimulus_version(eeg_data.trialinfo["stimulus_version"].iloc[0])
        video_index = np.asarray(eeg_data.video_index, dtype=int)

        for video_id in video_ids:
            matches = np.where(video_index == int(video_id))[0]
            if matches.size == 0:
                continue
            trial_idx = int(matches[0])
            material = materials.get((stimulus_version, int(video_id)))
            if material is None:
                continue

            trial = np.asarray(eeg_data.trials[trial_idx])
            eeg_fs = float(eeg_data.fs)
            eeg_n = int(trial.shape[1])
            eeg_duration = eeg_n / eeg_fs
            if eeg_duration < window_sec:
                continue

            n_added = 0
            t_start = 0.0
            while t_start + window_sec <= eeg_duration + 1e-9:
                window_idx = int(round(t_start / stride_sec))
                t_end = t_start + window_sec
                feature = provider.get_window_feature(
                    stimulus_version=stimulus_version,
                    video_id=int(video_id),
                    material_key=material.material_key,
                    t_start=t_start,
                    t_end=t_end,
                    window_idx=window_idx,
                )
                if feature is not None:
                    feature = np.asarray(feature)
                    if speech_dim is None:
                        speech_dim = int(feature.shape[-1])
                    if int(feature.shape[-1]) == speech_dim:
                        eeg_start = int(round(t_start * eeg_fs))
                        eeg_len = int(round(window_sec * eeg_fs))
                        if eeg_start + eeg_len <= eeg_n:
                            pair_key = make_pair_key(material.material_key, window_idx)
                            records.append(
                                WindowRecord(
                                    subject=str(subject).zfill(3),
                                    stimulus_version=stimulus_version,
                                    video_id=int(video_id),
                                    trial_idx=trial_idx,
                                    speaker_id=material.speaker_id,
                                    emotion=material.emotion,
                                    material_key=material.material_key,
                                    window_idx=window_idx,
                                    pair_key=pair_key,
                                    t_start=float(t_start),
                                    t_end=float(t_end),
                                    eeg_start_sample=eeg_start,
                                    eeg_n_samples=eeg_len,
                                    eeg_fs=eeg_fs,
                                    speech_dim=speech_dim,
                                    speech_source=provider.source_name,
                                    source_key=f"{stimulus_version}:video-{int(video_id):02d}",
                                )
                            )
                            n_added += 1
                if max_windows_per_trial and n_added >= max_windows_per_trial:
                    break
                t_start += stride_sec

    return records


def records_to_frame(records: Sequence[WindowRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.__dict__ for record in records])


def summarize_records(records: Sequence[WindowRecord]) -> dict[str, object]:
    if not records:
        return {
            "n_windows": 0,
            "n_subjects": 0,
            "n_materials": 0,
            "n_pair_keys": 0,
            "speech_dim": None,
        }
    return {
        "n_windows": len(records),
        "n_subjects": len({r.subject for r in records}),
        "n_materials": len({r.material_key for r in records}),
        "n_pair_keys": len({r.pair_key for r in records}),
        "n_source_keys": len({r.source_key for r in records}),
        "speech_dim": int(records[0].speech_dim),
        "min_t_start": float(min(r.t_start for r in records)),
        "max_t_end": float(max(r.t_end for r in records)),
    }


def make_folds(
    records: Sequence[WindowRecord],
    split_unit: str = "material",
) -> list[tuple[str, list[int], list[int]]]:
    if split_unit not in {"material", "video"}:
        raise ValueError("split_unit must be 'material' or 'video'")
    if split_unit == "material":
        keys = [r.material_key for r in records]
    else:
        keys = [r.source_key for r in records]
    unique_keys = sorted(set(keys))
    folds: list[tuple[str, list[int], list[int]]] = []
    for key in unique_keys:
        val_idx = [i for i, item in enumerate(keys) if item == key]
        train_idx = [i for i, item in enumerate(keys) if item != key]
        if train_idx and val_idx:
            folds.append((key, train_idx, val_idx))
    return folds


def preprocess_eeg_window(
    eeg: np.ndarray,
    original_fs: float,
    target_fs: float,
    window_sec: float,
) -> np.ndarray:
    x = np.asarray(eeg, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"EEG window must be 2D (channels,time), got {x.shape}")
    if target_fs > 0 and abs(float(target_fs) - float(original_fs)) > 1e-6:
        target_n = int(round(float(window_sec) * float(target_fs)))
        x = signal.resample(x, target_n, axis=1).astype(np.float32)
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x


class SpeechEEGWindowDataset:
    def __init__(
        self,
        records: Sequence[WindowRecord],
        provider: SpeechWindowProvider,
        eeg_root: Path,
        target_fs: float,
        window_sec: float,
    ) -> None:
        self.records = list(records)
        self.provider = provider
        self.eeg_root = Path(eeg_root)
        self.target_fs = float(target_fs)
        self.window_sec = float(window_sec)
        self._subject_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_subject(self, subject: str):
        if subject not in self._subject_cache:
            self._subject_cache[subject] = load_eeg_trial_data(subject, "1", data_root=self.eeg_root)
        return self._subject_cache[subject]

    def __getitem__(self, idx: int) -> dict[str, object]:
        record = self.records[idx]
        eeg_data = self._load_subject(record.subject)
        trial = np.asarray(eeg_data.trials[record.trial_idx], dtype=np.float32)
        start = int(record.eeg_start_sample)
        stop = start + int(record.eeg_n_samples)
        eeg = trial[:, start:stop]
        if eeg.shape[1] != record.eeg_n_samples:
            raise IndexError(f"Short EEG window for {record}")
        eeg = preprocess_eeg_window(
            eeg,
            original_fs=float(record.eeg_fs),
            target_fs=self.target_fs,
            window_sec=self.window_sec,
        )
        speech = self.provider.get_window_feature(
            stimulus_version=record.stimulus_version,
            video_id=record.video_id,
            material_key=record.material_key,
            t_start=record.t_start,
            t_end=record.t_end,
            window_idx=record.window_idx,
        )
        if speech is None:
            raise IndexError(f"Missing speech feature for {record.pair_key}")
        return {
            "eeg": eeg.astype(np.float32),
            "speech": np.asarray(speech, dtype=np.float32),
            "pair_key": record.pair_key,
            "material_key": record.material_key,
            "subject": record.subject,
            "video_id": record.video_id,
            "window_idx": record.window_idx,
        }


def collate_torch(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    import torch

    eeg = torch.from_numpy(np.stack([b["eeg"] for b in batch], axis=0))
    speech = torch.from_numpy(np.stack([b["speech"] for b in batch], axis=0))
    return {
        "eeg": eeg,
        "speech": speech,
        "pair_key": [str(b["pair_key"]) for b in batch],
        "material_key": [str(b["material_key"]) for b in batch],
        "subject": [str(b["subject"]) for b in batch],
        "video_id": [int(b["video_id"]) for b in batch],
        "window_idx": [int(b["window_idx"]) for b in batch],
    }


class UniquePairKeyBatchSampler:
    """Batch sampler that prevents duplicate pair_key values within a batch."""

    def __init__(
        self,
        records: Sequence[WindowRecord],
        batch_size: int,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        groups: dict[str, list[int]] = {}
        for idx, record in enumerate(self.records):
            groups.setdefault(record.pair_key, []).append(idx)
        for values in groups.values():
            rng.shuffle(values)
        active = [key for key, values in groups.items() if values]
        while active:
            rng.shuffle(active)
            selected = active[: self.batch_size]
            batch = [groups[key].pop() for key in selected]
            active = [key for key in active if groups[key]]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.records) // self.batch_size
        return int(math.ceil(len(self.records) / float(self.batch_size)))
