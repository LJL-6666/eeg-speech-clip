from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from utils.eeg_whisper_modeling.data_loader import (
    DEFAULT_EEG_ROOT,
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_EMBEDDING_ROOT_V2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "stage3_speech_eeg_clip"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cache" / "stage3_speech_eeg_clip"
DEFAULT_LOCAL_WHISPER_BASE = PROJECT_ROOT / "models" / "openai_whisper-base"

DEFAULT_SPEECH_MODELS = {
    "whisper": "openai/whisper-base",
    "wav2vec2": "facebook/wav2vec2-xls-r-300m",
    "hubert": "utter-project/mHuBERT-147-base-3rd-iter",
}


@dataclass(frozen=True)
class WindowConfig:
    window_sec: float = 4.0
    stride_sec: float = 2.0
    eeg_target_fs: float = 250.0
    max_windows_per_trial: int = 0


@dataclass(frozen=True)
class SpeechConfig:
    backend: str = "whisper"
    source: str = "auto"
    model_id: Optional[str] = None
    model_path: Optional[Path] = None
    embedding_tag: str = "whisper_base_last"
    embedding_root: Path = DEFAULT_EMBEDDING_ROOT
    embedding_root_v2: Path = DEFAULT_EMBEDDING_ROOT_V2
    local_files_only: bool = False


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 5
    min_delta: float = 1e-5
    num_workers: int = 0
    seed: int = 42
    d_proj: int = 128
    init_temperature: float = 0.07
    device: str = "auto"
    save_checkpoints: bool = False


def resolve_speech_source(backend: str, source: str) -> str:
    if source != "auto":
        return source
    return "existing-whisper" if backend == "whisper" else "cache"


def resolve_speech_model(backend: str, model_id: Optional[str]) -> str:
    if model_id:
        return model_id
    if backend not in DEFAULT_SPEECH_MODELS:
        raise ValueError(f"Unsupported speech backend: {backend}")
    return DEFAULT_SPEECH_MODELS[backend]


def safe_name(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "model"


def parse_int_list(value: Optional[str], default: Iterable[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_subject_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip().zfill(3) for item in value.split(",") if item.strip()]


def resolve_subjects(
    eeg_root: Path,
    subjects_arg: Optional[str],
    max_subjects: int = 0,
    exclude_subjects: Optional[str] = None,
) -> list[str]:
    if subjects_arg:
        subjects = parse_subject_list(subjects_arg)
    else:
        subjects = sorted(path.stem.zfill(3) for path in eeg_root.glob("*.pkl"))
    excluded = set(parse_subject_list(exclude_subjects))
    subjects = [subject for subject in subjects if subject not in excluded]
    if max_subjects and max_subjects > 0:
        subjects = subjects[:max_subjects]
    if not subjects:
        raise FileNotFoundError(f"No EEG subject pkl files found in {eeg_root}")
    return subjects


def speech_cache_dir(cache_root: Path, backend: str, model_id: str, window: WindowConfig) -> Path:
    model_key = safe_name(model_id)
    return (
        cache_root
        / "speech_features"
        / f"{backend}_{model_key}"
        / f"win{window.window_sec:.3f}_stride{window.stride_sec:.3f}"
    )

