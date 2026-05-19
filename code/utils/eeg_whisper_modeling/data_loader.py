"""
Data loading helpers for eeg_whisper_modeling workflow.

Responsibilities:
- Discover and load per-word Whisper embedding files from EmotionExpression project
- Load EEG trial-level signals (channels x time) together with metadata
- Provide utility helpers to match embedding segments with EEG trials

Required Environment: ty_eeg_whisper_modeling
Environment Config: environments/ty_eeg_whisper_modeling.yml
Utility Modules: code/utils/eeg_whisper_modeling/

Last Updated: 2025-01-XX
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = PROJECT_ROOT.parent / "\u53c2\u8003\u4ee3\u7801"
STIMULUS_VERSION_V1 = "v1"
STIMULUS_VERSION_V2 = "v2"
# Embedding data from EmotionExpression project
# Prefer newly regenerated stimulus-versioned caches in this EEG migration
# package, then fall back to the earlier local/original v1 paths.
_stage1_embedding_root = PROJECT_ROOT / "data" / "stage1_audio_whisper_embeddings_v2"
_stage1_embedding_root_v1 = _stage1_embedding_root / "stimulus_v1"
_stage1_embedding_root_v2 = _stage1_embedding_root / "stimulus_v2"
_local_embedding_root = PROJECT_ROOT / "data" / "audio_whisper_embeddings_v2"
_local_embedding_root_v2 = PROJECT_ROOT / "data" / "audio_whisper_embeddings_v2_stimulus_v2"
_linux_embedding_root = Path("/data/liming/学术/项目/Emotion_Expression/Project/data/audio_whisper_embeddings_v2")
_windows_embedding_root = Path(r"D:\Seafile\学术\项目\Emotion_Expression\Project\data\audio_whisper_embeddings_v2")


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except PermissionError:
        LOGGER.warning("No permission to access path: %s", path)
        return False


_embedding_root_candidates = (
    _stage1_embedding_root_v1,
    _local_embedding_root,
    _linux_embedding_root,
    _windows_embedding_root,
)
DEFAULT_EMBEDDING_ROOT = next(
    (path for path in _embedding_root_candidates if _path_exists(path)),
    _stage1_embedding_root_v1,
)
_embedding_root_v2_candidates = (
    _stage1_embedding_root_v2,
    _local_embedding_root_v2,
)
DEFAULT_EMBEDDING_ROOT_V2 = next(
    (path for path in _embedding_root_v2_candidates if _path_exists(path)),
    _stage1_embedding_root_v2,
)
# 默认用原版 EEG；修正版请传 --eeg-root data/eeg_corrected
_DEFAULT_EEG_ORIGINAL = PROJECT_ROOT / "data" / "eeg_original"
_DEFAULT_EEG_CORRECTED = PROJECT_ROOT / "data" / "eeg_corrected"
DEFAULT_EEG_ROOT = (
    _DEFAULT_EEG_ORIGINAL
    if _DEFAULT_EEG_ORIGINAL.exists()
    else Path("/data/liujialing/TY/建模/EEG与语音联动/Preprocessing/output/communication")
)
DEFAULT_EEG_ROOT_CORRECTED = (
    _DEFAULT_EEG_CORRECTED
    if _DEFAULT_EEG_CORRECTED.exists()
    else Path(
        "/data/liujialing/TY/建模/EEG与语音联动/Preprocessing/output/communication_corrected_20260519"
    )
)
# Narrative identity mapping: video_id -> (target_emotion, narrator_subject_id)
NARRATIVE_IDENTITY_CSV_V1 = PROJECT_ROOT / "experiments" / "stimulus_narrative_v1" / "narrative_identity.csv"
REFERENCE_NARRATIVE_IDENTITY_CSV_V1 = REFERENCE_ROOT / "experiments" / "stimulus_narrative_v1" / "narrative_identity.csv"
NARRATIVE_IDENTITY_CSV_V2 = PROJECT_ROOT / "experiments" / "stimulus_narrative_v2" / "narrative_identity.csv"
REFERENCE_NARRATIVE_IDENTITY_CSV_V2 = REFERENCE_ROOT / "experiments" / "stimulus_narrative_v2" / "narrative_identity.csv"
NARRATIVE_IDENTITY_CSV = NARRATIVE_IDENTITY_CSV_V1
STIMULUS_ROOT_V1 = REFERENCE_ROOT / "experiments" / "stimulus_narrative_v1"
STIMULUS_ROOT_V2 = REFERENCE_ROOT / "experiments" / "stimulus_narrative_v2"

EMBEDDING_TYPES = ("acoustic", "speech", "language")
# This project's exp1 corresponds to EmotionExpression's exp2 (narrative videos)
EXPERIMENT_TASK_MAP = {
    "1": ["narrative"],  # This project's exp1
}
# EmotionExpression project's exp2 corresponds to this project's exp1
EMOTION_EXPRESSION_EXP_MAP = {
    "1": "2",  # This project exp1 -> EmotionExpression exp2
}
EEG_FS = 250.0
EEG_N_CHANNELS = 31
EEG_N_TIMEPOINTS = 7500  # legacy 30-second export at 250 Hz
EEG_N_VIDEOS = 28
# Preprocessed Tongyong EEG pkl channel order. These labels replace generic
# numbered placeholders and follow the numeric TY_CHANNEL_REORDER used by
# Preprocessing_tongyong.py when exporting output/communication/*.pkl.
EEG_CHANNEL_LABELS = [
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "P4",
    "FT7",
    "FCz",
    "FC3",
    "FT8",
    "Cz",
    "C3",
    "C4",
    "T3",
    "T4",
    "A1",
    "A2",
    "CP3",
    "CP4",
    "TP7",
    "TP8",
    "T6",
    "Oz",
    "Pz",
    "P3",
    "T5",
    "O1",
    "O2",
    "FC4",
]

@dataclass
class EmbeddingRecord:
    """Container for a single embedding .npy file."""

    subject_id: str  # Narrator subject ID from EmotionExpression project
    experiment: str  # Experiment ID from EmotionExpression project ("2")
    timestamp: str
    emotion: str
    embedding_type: str
    path: Path
    words: List[str]
    embeddings: List[np.ndarray]
    timestamps: List[np.ndarray]
    metadata: Dict[str, object]
    video_id: Optional[int] = None  # Video ID in this project (1-28)
    stimulus_version: str = STIMULUS_VERSION_V1


@dataclass
class EegExperimentData:
    """
    EEG trial-level signals for a subject & experiment.

    Attributes:
        subject_id: Subject identifier as string (listener in this project)
        experiment: Experiment identifier ("1" for this project)
        labels: EEG channel labels
        trials: List of arrays (n_channels, n_timepoints)
        times: List of time vectors (n_timepoints,)
        fs: Sampling rate (Hz)
        trialinfo: Pandas DataFrame with at least {'task','target_emotion','video_index'}
        video_index: np.ndarray with video IDs (1-28)
    """

    subject_id: str
    experiment: str
    labels: List[str]
    trials: List[np.ndarray]
    times: List[np.ndarray]
    fs: float
    trialinfo: pd.DataFrame
    video_index: Optional[np.ndarray]

    def iter_trials(self) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield (trial_idx, data, time_vec) tuples."""
        for idx, (trial, time_vec) in enumerate(zip(self.trials, self.times)):
            yield idx, trial, time_vec


# Backward-compatible alias: the migrated modeling script still expects the
# original class name in a few type hints.
FnirsExperimentData = EegExperimentData


def normalize_stimulus_version(stimulus_version: str) -> str:
    value = str(stimulus_version).strip().lower()
    aliases = {
        "1": STIMULUS_VERSION_V1,
        "v1": STIMULUS_VERSION_V1,
        "cohort1": STIMULUS_VERSION_V1,
        "stimulus_narrative_v1": STIMULUS_VERSION_V1,
        "2": STIMULUS_VERSION_V2,
        "v2": STIMULUS_VERSION_V2,
        "cohort2": STIMULUS_VERSION_V2,
        "stimulus_narrative_v2": STIMULUS_VERSION_V2,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported stimulus version '{stimulus_version}'. Expected v1 or v2.")
    return aliases[value]


def stimulus_version_for_subject(subject_id: str) -> str:
    """Use v1 stimuli for EEG subject IDs <69 and v2 for IDs >=69."""

    s = str(subject_id).strip()
    if not s.isdigit():
        raise ValueError(f"Invalid numeric EEG subject id: {subject_id}")
    return STIMULUS_VERSION_V1 if int(s) < 69 else STIMULUS_VERSION_V2


def cohort_for_subject(subject_id: str) -> str:
    return "cohort1" if stimulus_version_for_subject(subject_id) == STIMULUS_VERSION_V1 else "cohort2"


def identity_csv_for_stimulus_version(stimulus_version: str) -> Path:
    version = normalize_stimulus_version(stimulus_version)
    if version == STIMULUS_VERSION_V1:
        return NARRATIVE_IDENTITY_CSV_V1 if NARRATIVE_IDENTITY_CSV_V1.exists() else REFERENCE_NARRATIVE_IDENTITY_CSV_V1
    return NARRATIVE_IDENTITY_CSV_V2 if NARRATIVE_IDENTITY_CSV_V2.exists() else REFERENCE_NARRATIVE_IDENTITY_CSV_V2


def identity_csv_for_subject(subject_id: str) -> Path:
    return identity_csv_for_stimulus_version(stimulus_version_for_subject(subject_id))


def audio_root_for_stimulus_version(stimulus_version: str) -> Path:
    version = normalize_stimulus_version(stimulus_version)
    return STIMULUS_ROOT_V1 if version == STIMULUS_VERSION_V1 else STIMULUS_ROOT_V2


def audio_root_for_subject(subject_id: str) -> Path:
    return audio_root_for_stimulus_version(stimulus_version_for_subject(subject_id))


def embedding_root_for_stimulus_version(
    stimulus_version: str,
    embedding_root: Optional[Path] = None,
    embedding_root_v2: Optional[Path] = None,
) -> Path:
    version = normalize_stimulus_version(stimulus_version)
    if version == STIMULUS_VERSION_V1:
        return Path(embedding_root) if embedding_root is not None else DEFAULT_EMBEDDING_ROOT
    return Path(embedding_root_v2) if embedding_root_v2 is not None else DEFAULT_EMBEDDING_ROOT_V2


def embedding_root_for_subject(
    subject_id: str,
    embedding_root: Optional[Path] = None,
    embedding_root_v2: Optional[Path] = None,
) -> Path:
    return embedding_root_for_stimulus_version(
        stimulus_version_for_subject(subject_id),
        embedding_root=embedding_root,
        embedding_root_v2=embedding_root_v2,
    )


def load_narrative_identity(stimulus_version: str = STIMULUS_VERSION_V1) -> pd.DataFrame:
    """
    Load narrative identity mapping: video_id -> (target_emotion, narrator_subject_id).

    Returns:
        DataFrame with columns: video_id, target_emotion, subject_id (narrator)
    """
    identity_csv = identity_csv_for_stimulus_version(stimulus_version)
    if not identity_csv.exists():
        raise FileNotFoundError(f"Narrative identity CSV not found: {identity_csv}")
    df = pd.read_csv(identity_csv)
    return df


def load_embedding_by_video_id(
    video_id: int,
    embedding_type: str,
    timestamp: Optional[str] = None,
    embedding_root: Path = DEFAULT_EMBEDDING_ROOT,
    listener_subject_id: Optional[str] = None,
    stimulus_version: Optional[str] = None,
    embedding_root_v2: Optional[Path] = None,
) -> EmbeddingRecord:
    """
    Load embedding for a specific video_id from EmotionExpression project.

    Args:
        video_id: Video ID (1-28) in this project
        embedding_type: One of EMBEDDING_TYPES
        timestamp: Optional timestamp. If omitted, auto-detects when only one file matches.
        embedding_root: Root folder storing video whisper embeddings in EmotionExpression project

    Returns:
        EmbeddingRecord with loaded tensors and metadata

    Raises:
        FileNotFoundError: if no .npy matches
        ValueError: if multiple matches found without explicit timestamp
    """
    resolved_version = (
        normalize_stimulus_version(stimulus_version)
        if stimulus_version is not None
        else (
            stimulus_version_for_subject(listener_subject_id)
            if listener_subject_id is not None
            else STIMULUS_VERSION_V1
        )
    )
    resolved_embedding_root = embedding_root_for_stimulus_version(
        resolved_version,
        embedding_root=embedding_root,
        embedding_root_v2=embedding_root_v2,
    )

    narrative_df = load_narrative_identity(resolved_version)
    video_row = narrative_df[narrative_df["video_id"] == video_id]
    if video_row.empty:
        raise ValueError(f"Video ID {video_id} not found in narrative_identity.csv")
    if len(video_row) > 1:
        raise ValueError(f"Multiple rows found for video_id {video_id}")

    narrator_subject_id = str(video_row.iloc[0]["subject_id"])
    target_emotion = video_row.iloc[0]["target_emotion"].lower()
    emotion_exp_exp = EMOTION_EXPRESSION_EXP_MAP.get("1", "2")  # This project exp1 -> EmotionExpression exp2

    return load_embedding_record(
        subject_id=narrator_subject_id,
        experiment=emotion_exp_exp,
        emotion=target_emotion,
        embedding_type=embedding_type,
        timestamp=timestamp,
        embedding_root=resolved_embedding_root,
        stimulus_version=resolved_version,
    )


def load_embedding_record(
    subject_id: str,
    experiment: str,
    emotion: str,
    embedding_type: str,
    timestamp: Optional[str] = None,
    embedding_root: Path = DEFAULT_EMBEDDING_ROOT,
    stimulus_version: str = STIMULUS_VERSION_V1,
) -> EmbeddingRecord:
    """
    Load a specific embedding .npy file from EmotionExpression project.

    Args:
        subject_id: Narrator subject folder (string or int) from EmotionExpression project
        experiment: Experiment ID from EmotionExpression project ("2")
        emotion: Emotion label (lowercase)
        embedding_type: One of EMBEDDING_TYPES
        timestamp: Optional timestamp. If omitted, auto-detects when only one file matches.
        embedding_root: Root folder storing video whisper embeddings in EmotionExpression project

    Returns:
        EmbeddingRecord with loaded tensors and metadata

    Raises:
        FileNotFoundError: if no .npy matches
        ValueError: if multiple matches found without explicit timestamp
    """

    subject_dir = embedding_root / str(subject_id)
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    embedding_type = embedding_type.lower()
    if embedding_type not in EMBEDDING_TYPES:
        raise ValueError(f"Unsupported embedding_type '{embedding_type}'. Expected {EMBEDDING_TYPES}")

    pattern = f"{embedding_type}_exp{experiment}_*_{emotion.lower()}.npy"
    candidates = sorted(subject_dir.glob(pattern))

    if timestamp:
        file_path = subject_dir / f"{embedding_type}_exp{experiment}_{timestamp}_{emotion.lower()}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
    else:
        if not candidates:
            raise FileNotFoundError(f"No embedding files match pattern {pattern} in {subject_dir}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple files match pattern {pattern}. "
                "Specify timestamp explicitly to disambiguate."
            )
        file_path = candidates[0]
        timestamp = _parse_embedding_timestamp_from_name(
            file_path.name,
            embedding_type=embedding_type,
            experiment=experiment,
            emotion=emotion,
        )

    npy = np.load(file_path, allow_pickle=True).item()

    # Find video_id from narrative_identity.csv
    resolved_version = normalize_stimulus_version(stimulus_version)
    narrative_df = load_narrative_identity(resolved_version)
    video_row = narrative_df[
        (narrative_df["subject_id"] == int(subject_id)) &
        (narrative_df["target_emotion"].str.lower() == emotion.lower())
    ]
    video_id = int(video_row.iloc[0]["video_id"]) if not video_row.empty else None

    return EmbeddingRecord(
        subject_id=str(subject_id),
        experiment=str(experiment),
        emotion=emotion.lower(),
        timestamp=timestamp or npy.get("metadata", {}).get("timestamp", ""),
        embedding_type=embedding_type,
        path=file_path,
        words=list(npy.get("words", [])),
        embeddings=list(npy.get("embeddings", [])),
        timestamps=list(npy.get("timestamps", [])),
        metadata=dict(npy.get("metadata", {})),
        video_id=video_id,
        stimulus_version=resolved_version,
    )


def load_eeg_trial_data(
    subject_id: str,
    experiment: str,
    data_root: Path = DEFAULT_EEG_ROOT,
) -> EegExperimentData:
    """
    Load preprocessed EEG trials for a given subject & experiment.

    Args:
        subject_id: Subject identifier (listener in this project)
        experiment: Experiment number ("1" for this project)
        data_root: Directory containing per-subject pkl files. Supported
            formats are the legacy array export shaped (28, 31, 7500) and
            the full-trial dict export with data/lengths/video_ids fields.

    Returns:
        EegExperimentData with signals, metadata and derived trialinfo DataFrame.
    """

    del experiment  # only exp1 communication is currently represented in this EEG export
    subject_norm = _normalize_subject_id(subject_id)
    pkl_path = data_root / f"{subject_norm}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"EEG pkl file missing: {pkl_path}")

    with pkl_path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        missing = {"data", "lengths", "video_ids"} - set(payload)
        if missing:
            raise ValueError(f"Unsupported EEG dict format for {pkl_path}: missing {sorted(missing)}")
        arr = np.asarray(payload["data"], dtype=float)
        lengths = np.asarray(payload["lengths"], dtype=int)
        video_ids = np.asarray(payload["video_ids"], dtype=int)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported EEG data ndim for {pkl_path}: {arr.ndim}; expected 3")
        if arr.shape[1] != EEG_N_CHANNELS:
            raise ValueError(
                f"Unsupported EEG channel count for {pkl_path}: {arr.shape[1]}; "
                f"expected {EEG_N_CHANNELS}"
            )
        if arr.shape[0] != lengths.size or arr.shape[0] != video_ids.size:
            raise ValueError(
                f"EEG metadata length mismatch for {pkl_path}: data has {arr.shape[0]} trials, "
                f"lengths has {lengths.size}, video_ids has {video_ids.size}"
            )
        if np.any(lengths <= 0) or np.any(lengths > arr.shape[2]):
            raise ValueError(
                f"Invalid EEG trial lengths for {pkl_path}: min={lengths.min()}, "
                f"max={lengths.max()}, padded length={arr.shape[2]}"
            )
    else:
        arr = np.asarray(payload, dtype=float)
        if arr.shape != (EEG_N_VIDEOS, EEG_N_CHANNELS, EEG_N_TIMEPOINTS):
            raise ValueError(
                f"Unsupported EEG data shape for {pkl_path}: {arr.shape}; "
                f"expected {(EEG_N_VIDEOS, EEG_N_CHANNELS, EEG_N_TIMEPOINTS)} "
                "or dict format with data/lengths/video_ids"
            )
        lengths = np.full(arr.shape[0], arr.shape[2], dtype=int)
        video_ids = np.arange(1, arr.shape[0] + 1, dtype=int)

    stimulus_version = stimulus_version_for_subject(subject_norm)
    narrative_df = load_narrative_identity(stimulus_version)
    trialinfo_df = pd.DataFrame(
        {
            "task": ["narrative"] * int(video_ids.size),
            "stimulus_version": [stimulus_version] * int(video_ids.size),
            "cohort": [cohort_for_subject(subject_norm)] * int(video_ids.size),
            "video_index": video_ids,
            "target_emotion": [
                _target_emotion_for_video(narrative_df, int(video_id))
                for video_id in video_ids
            ],
        }
    )

    trials = [arr[i, :, : int(lengths[i])].copy() for i in range(video_ids.size)]
    times = [
        np.arange(int(length), dtype=float) / EEG_FS
        for length in lengths
    ]

    return EegExperimentData(
        subject_id=subject_norm,
        experiment="1",
        labels=list(EEG_CHANNEL_LABELS),
        trials=trials,
        times=list(times),
        fs=EEG_FS,
        trialinfo=trialinfo_df,
        video_index=video_ids,
    )


# Backward-compatible alias for the migrated main script.
load_fnirs_trial_data = load_eeg_trial_data


def match_embedding_to_trial(
    embedding: EmbeddingRecord,
    fnirs_data: FnirsExperimentData,
    preferred_tasks: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """
    Match an embedding file to a trial index within fNIRS data.

    Strategy:
        1. Use video_id if available (preferred method for this project)
        2. Otherwise, use (task, target_emotion) as a unique key to locate the matching trial row.
        3. Explicitly warn whenever no trial (or multiple trials) satisfy the key to avoid silent mismatches.

    Args:
        embedding: EmbeddingRecord to match.
        fnirs_data: Loaded FnirsExperimentData.
        preferred_tasks: Optional manual override for tasks to consider.

    Returns:
        Trial index (0-based) if found, otherwise None.
    """

    df = fnirs_data.trialinfo
    if df is None or df.empty:
        LOGGER.warning(
            "Trialinfo empty for subject=%s exp=%s; cannot match embeddings.",
            fnirs_data.subject_id,
            fnirs_data.experiment,
        )
        return None

    # First try to match by video_id if available
    if embedding.video_id is not None and "video_index" in df.columns:
        video_matches = df[df["video_index"] == embedding.video_id]
        if len(video_matches) == 1:
            return int(video_matches.index[0])
        elif len(video_matches) > 1:
            LOGGER.warning(
                "Multiple trials found for video_id=%d (subject=%s exp=%s). Using emotion matching.",
                embedding.video_id,
                fnirs_data.subject_id,
                fnirs_data.experiment,
            )

    # Fallback to emotion matching
    emotion = embedding.emotion.lower()
    task_key = _infer_embedding_task(embedding, fnirs_data, preferred_tasks)

    if task_key:
        task_mask = df["task"].str.lower() == task_key
        matching = df[task_mask & (df["target_emotion"].str.lower() == emotion)]
    else:
        matching = df[df["target_emotion"].str.lower() == emotion]

    if matching.empty:
        LOGGER.warning(
            "No trial matches subject=%s exp=%s task=%s emotion=%s video_id=%s",
            fnirs_data.subject_id,
            fnirs_data.experiment,
            task_key or "<any>",
            embedding.emotion,
            embedding.video_id or "<unknown>",
        )
        return None

    if len(matching) > 1:
        LOGGER.error(
            "Found %d trials for subject=%s exp=%s task=%s emotion=%s. "
            "Check data consistency and disambiguate manually.",
            len(matching),
            fnirs_data.subject_id,
            fnirs_data.experiment,
            task_key or "<any>",
            embedding.emotion,
        )
        return None

    return int(matching.index[0])


def _infer_embedding_task(
    embedding: EmbeddingRecord,
    fnirs_data: FnirsExperimentData,
    preferred_tasks: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Determine which behavioral task the embedding belongs to.

    Resolution order:
        1. preferred_tasks override (must resolve to a single unique task)
        2. metadata['task'] inside embedding file (if provided)
        3. experiment-level defaults (EXPERIMENT_TASK_MAP)
        4. fallback to single unique task present in trialinfo
    """

    if preferred_tasks:
        unique = {task.strip().lower() for task in preferred_tasks if task}
        if len(unique) == 1:
            return unique.pop()
        if len(unique) > 1:
            LOGGER.error("preferred_tasks must resolve to a single task, got %s", unique)
            return None

    meta_task = str(embedding.metadata.get("task", "")).strip().lower()
    if meta_task:
        return meta_task

    defaults = EXPERIMENT_TASK_MAP.get(fnirs_data.experiment)
    if defaults:
        unique_defaults = {task.strip().lower() for task in defaults if task}
        if len(unique_defaults) == 1:
            return next(iter(unique_defaults))
        LOGGER.warning(
            "Experiment %s has multiple default tasks %s; unable to disambiguate automatically.",
            fnirs_data.experiment,
            defaults,
        )
        return None

    unique_trial_tasks = {task.strip().lower() for task in fnirs_data.trialinfo["task"].unique() if task}
    if len(unique_trial_tasks) == 1:
        return next(iter(unique_trial_tasks))

    LOGGER.warning(
        "Cannot infer task for subject=%s exp=%s; trialinfo tasks=%s",
        fnirs_data.subject_id,
        fnirs_data.experiment,
        sorted(unique_trial_tasks),
    )
    return None


def list_available_embeddings(
    video_ids: Optional[Sequence[int]] = None,
    embedding_type: Optional[str] = None,
    embedding_root: Path = DEFAULT_EMBEDDING_ROOT,
    listener_subject_id: Optional[str] = None,
    stimulus_version: Optional[str] = None,
    embedding_root_v2: Optional[Path] = None,
) -> List[Tuple[int, str, str, str]]:
    """
    List available embeddings for specified video_ids.

    Args:
        video_ids: Optional list of video IDs (1-28). If None, returns all available.
        embedding_type: Optional embedding type filter. If None, returns all types.
        embedding_root: Root folder storing video whisper embeddings in EmotionExpression project

    Returns:
        List of tuples (video_id, embedding_type, timestamp, emotion) for which embedding exists.
    """
    resolved_version = (
        normalize_stimulus_version(stimulus_version)
        if stimulus_version is not None
        else (
            stimulus_version_for_subject(listener_subject_id)
            if listener_subject_id is not None
            else STIMULUS_VERSION_V1
        )
    )
    resolved_embedding_root = embedding_root_for_stimulus_version(
        resolved_version,
        embedding_root=embedding_root,
        embedding_root_v2=embedding_root_v2,
    )
    narrative_df = load_narrative_identity(resolved_version)

    if video_ids is None:
        video_ids = narrative_df["video_id"].tolist()

    records = []
    for video_id in video_ids:
        video_row = narrative_df[narrative_df["video_id"] == video_id]
        if video_row.empty:
            continue
        narrator_subject_id = str(video_row.iloc[0]["subject_id"])
        target_emotion = video_row.iloc[0]["target_emotion"].lower()
        emotion_exp_exp = EMOTION_EXPRESSION_EXP_MAP.get("1", "2")

        subject_dir = resolved_embedding_root / narrator_subject_id
        if not subject_dir.exists():
            continue

        for emb_type in EMBEDDING_TYPES:
            if embedding_type and emb_type != embedding_type:
                continue
            pattern = f"{emb_type}_exp{emotion_exp_exp}_*_{target_emotion}.npy"
            candidates = sorted(subject_dir.glob(pattern))
            for npy_path in candidates:
                timestamp = _parse_embedding_timestamp_from_name(
                    npy_path.name,
                    embedding_type=emb_type,
                    experiment=emotion_exp_exp,
                    emotion=target_emotion,
                )
                records.append((video_id, emb_type, timestamp, target_emotion))

    return sorted(records)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_trialinfo_array(mat_path: Path) -> np.ndarray:
    """Load the `trialinfo` cell array from MAT file."""
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trialinfo = data.get("trialinfo")
    if trialinfo is None:
        raise ValueError(f"'trialinfo' variable missing in {mat_path}")
    return np.atleast_2d(trialinfo)


def _parse_embedding_timestamp_from_name(
    file_name: str,
    *,
    embedding_type: str,
    experiment: str,
    emotion: str,
) -> str:
    prefix = f"{embedding_type}_exp{experiment}_"
    suffix = f"_{emotion.lower()}.npy"
    if file_name.startswith(prefix) and file_name.endswith(suffix):
        return file_name[len(prefix) : -len(suffix)]
    parts = file_name.split("_")
    return parts[2] if len(parts) > 2 else ""


def _trialinfo_to_dataframe(trialinfo: np.ndarray) -> pd.DataFrame:
    """
    Convert FieldTrip `trialinfo` (cell array) to a DataFrame with standard columns.

    Column inference rules:
        - Column 0 always stores task labels (string)
        - Column 1:
            - numeric  => treated as videoIndex
            - string   => treated as targetEmotion
        - Column 2 (if column 1 was numeric) => targetEmotion
        - Remaining columns are stored generically (c0, c1, ...) for downstream use.
    """

    df_raw = pd.DataFrame(trialinfo)
    n_cols = df_raw.shape[1]
    if n_cols == 0:
        raise ValueError("trialinfo array has zero columns")

    result = pd.DataFrame(index=df_raw.index)
    result["task"] = df_raw.iloc[:, 0].astype(str)
    if n_cols == 1:
        result["video_index"] = np.nan
        result["target_emotion"] = ""
        return result

    # Column 1 inference
    first_col1 = df_raw.iloc[:, 1]
    if _is_numeric_series(first_col1):
        result["video_index"] = pd.to_numeric(first_col1, errors="coerce")
        if n_cols > 2:
            result["target_emotion"] = df_raw.iloc[:, 2].astype(str)
        else:
            result["target_emotion"] = ""
        meta_start = 3
    else:
        result["video_index"] = (
            pd.to_numeric(df_raw.iloc[:, 2], errors="coerce") if n_cols > 2 else np.nan
        )
        result["target_emotion"] = df_raw.iloc[:, 1].astype(str)
        meta_start = 2

    # Attach remaining unnamed columns for completeness
    for col_idx in range(meta_start, n_cols):
        result[f"col_{col_idx}"] = df_raw.iloc[:, col_idx]

    return result


def _is_numeric_series(series: pd.Series) -> bool:
    """Heuristic check whether a Series contains numeric values."""
    sample = series.dropna()
    if sample.empty:
        return False
    return all(isinstance(val, (int, float, np.integer, np.floating)) for val in sample)


def _load_trials_from_mat(mat_path: Path):
    """Load FieldTrip trials directly from MAT file without external deps."""

    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials_dc = data.get("trialsDC")
    if trials_dc is None:
        raise ValueError(f"trialsDC struct missing in {mat_path}")

    labels = [str(lbl) for lbl in np.atleast_1d(trials_dc.label)]
    times_list = [np.asarray(t, dtype=float) for t in np.atleast_1d(trials_dc.time)]
    trials_list = [np.asarray(tr, dtype=float) for tr in np.atleast_1d(trials_dc.trial)]

    fs = None
    if hasattr(trials_dc, "hdr") and getattr(trials_dc.hdr, "Fs", None):
        fs = float(trials_dc.hdr.Fs)
    elif hasattr(trials_dc, "fsample"):
        fs = float(trials_dc.fsample)

    video_index = _extract_video_index(data, trials_dc)
    return labels, times_list, trials_list, fs, video_index


def _extract_video_index(data, trials_dc) -> Optional[np.ndarray]:
    """Best-effort extraction of videoIndex column."""

    if hasattr(trials_dc, "trialinfo") and hasattr(trials_dc.trialinfo, "videoIndex"):
        try:
            return np.atleast_1d(trials_dc.trialinfo.videoIndex).astype(int)
        except Exception:
            pass

    candidate = data.get("trialinfo")
    if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
        col1 = candidate[:, 1]
        if _is_numeric_series(pd.Series(col1)):
            return pd.to_numeric(pd.Series(col1), errors="coerce").to_numpy(dtype=int, na_value=-1)
        if candidate.shape[1] > 2:
            col2 = candidate[:, 2]
            if _is_numeric_series(pd.Series(col2)):
                return pd.to_numeric(pd.Series(col2), errors="coerce").to_numpy(dtype=int, na_value=-1)

    return None


def _normalize_subject_id(subject_id: str) -> str:
    """Normalize subject IDs to the zero-padded names used by EEG pkl files."""

    s = str(subject_id).strip()
    if s.isdigit():
        return f"{int(s):03d}"
    return s


def _target_emotion_for_video(narrative_df: pd.DataFrame, video_id: int) -> str:
    row = narrative_df[narrative_df["video_id"] == video_id]
    if row.empty:
        raise ValueError(f"Video ID {video_id} not found in narrative identity table")
    return str(row.iloc[0]["target_emotion"]).lower()
