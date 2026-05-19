"""
Feature processing utilities for fnirs_whisper_modeling workflow.

- Flatten per-word Whisper embeddings into continuous frame sequences
- Standardize and optionally reduce feature dimensionality
- Align embedding sequences to fNIRS sampling grid (11 Hz)

Required Environment: ty_fnirs_whisper_modeling
Environment Config: environments/ty_fnirs_whisper_modeling.yml
Utility Modules: code/utils/fnirs_whisper_modeling/

Last Updated: 2025-01-XX
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import logging

import numpy as np

from .data_loader import EmbeddingRecord

LOGGER = logging.getLogger(__name__)

MIN_EPS = 1e-9


@dataclass
class PCAParams:
    """Parameters required to apply a fitted PCA transform."""

    mean: np.ndarray
    components: np.ndarray
    var_ratio: np.ndarray


@dataclass
class FeatureTransformMeta:
    """Metadata describing preprocessing applied to embeddings."""

    standardize_method: str
    standardize_params: Dict[str, np.ndarray]
    dim_reduction_method: str
    dim_reduction_components: Optional[int]
    pca_params: Optional[PCAParams]


def flatten_embedding(record: EmbeddingRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate per-word embeddings into a frame-wise matrix.

    Returns:
        features: (n_frames, feature_dim)
        centers: (n_frames,) timestamp centers in seconds
        spans: (n_frames, 2) start/end bounds per frame
    """

    frame_list = []
    time_centers = []
    time_spans = []

    for emb, ts in zip(record.embeddings, record.timestamps):
        if emb is None or ts is None:
            continue

        emb_arr = np.asarray(emb)
        ts_arr = np.asarray(ts, dtype=float)

        if emb_arr.ndim == 1:
            # Language embedding: treat vector as constant across its duration
            start = float(np.nanmin(ts_arr)) if ts_arr.size else np.nan
            end = float(np.nanmax(ts_arr)) if ts_arr.size else np.nan
            if not np.isfinite(start) or not np.isfinite(end):
                center = _collapse_timestamp(ts_arr)
                start = end = center
            else:
                if end < start:
                    start, end = end, start
                center = 0.5 * (start + end)

            frame_list.append(emb_arr[None, :])
            time_centers.append(np.array([center]))
            time_spans.append(np.array([[start, end]]))
        else:
            frame_list.append(emb_arr)
            if ts_arr.ndim == 1:
                time_centers.append(ts_arr)
                span = np.column_stack((ts_arr, ts_arr))
                time_spans.append(span)
            elif ts_arr.ndim == 2:
                # Some extractors return (1, n_frames)
                flattened = ts_arr.flatten()
                time_centers.append(flattened)
                span = np.column_stack((flattened, flattened))
                time_spans.append(span)
            else:
                raise ValueError(f"Unsupported timestamp shape {ts_arr.shape} for embedding type {record.embedding_type}")

    if not frame_list:
        raise ValueError(f"No valid embeddings found for {record.path}")

    features = np.vstack(frame_list)
    centers = np.concatenate(time_centers)
    spans = np.vstack(time_spans)

    sort_idx = np.argsort(centers)
    return features[sort_idx], centers[sort_idx], spans[sort_idx]


def standardize_features(
    features: np.ndarray,
    method: str = "zscore",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Standardize feature matrix along feature dimension.

    Args:
        features: (n_samples, n_features)
        method: 'zscore', 'robust', or 'none'

    Returns:
        standardized features, parameter dict (mean/std or median/mad)
    """

    method = method.lower()
    params: Dict[str, np.ndarray] = {}
    feats = np.asarray(features, dtype=float)

    if method == "none":
        return feats, params

    if method == "zscore":
        mean = np.nanmean(feats, axis=0, keepdims=True)
        std = np.nanstd(feats, axis=0, keepdims=True)
        std = np.clip(std, MIN_EPS, None)
        params = {"mean": mean, "std": std}
        return (feats - mean) / std, params

    if method == "robust":
        median = np.nanmedian(feats, axis=0, keepdims=True)
        mad = np.nanmedian(np.abs(feats - median), axis=0, keepdims=True)
        scale = np.clip(mad * 1.4826, MIN_EPS, None)
        params = {"median": median, "mad": mad}
        return (feats - median) / scale, params

    raise ValueError(f"Unsupported standardize method '{method}'")


def reduce_dimension(
    features: np.ndarray,
    method: str = "pca",
    n_components: int = 50,
    random_state: int = 0,
) -> Tuple[np.ndarray, Optional[PCAParams]]:
    """
    Dimensionality reduction wrapper.

    Args:
        features: (n_samples, n_features)
        method: 'pca' or 'none'
        n_components: desired components (ignored when >= original dims)

    Returns:
        (reduced_features, fitted_pca_or_None)
    """

    method = method.lower()
    n_samples, n_features = features.shape

    if method == "none" or n_components is None or n_components >= n_features:
        return features, None

    if method != "pca":
        raise ValueError(f"Unsupported dim reduction method '{method}'")

    effective_components = min(n_components, n_features, n_samples)
    if effective_components < 1:
        raise ValueError("n_components must be >= 1")

    reduced, pca_params = _pca_fit_transform(features, effective_components)
    return reduced, pca_params


def align_features_to_fnirs(
    feature_times: np.ndarray,
    feature_values: np.ndarray,
    fnirs_time_vec: np.ndarray,
    fnirs_fs: float = 11.0,
    window_sec: Optional[float] = None,
    fill_strategy: str = "zero",
    feature_spans: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Align frame-wise embeddings to fNIRS sampling grid.

    Args:
        feature_times: (n_frames,) center timestamps (seconds)
        feature_spans: optional (n_frames, 2) span [start, end) per frame
        feature_values: (n_frames, n_features)
        fnirs_time_vec: (n_timepoints,)
        fnirs_fs: sampling rate (Hz) used to set default window size
        window_sec: aggregation half-window. Default 0.5 / fnirs_fs
        fill_strategy: 'nearest', 'zero', or 'nan'

    Returns:
        aligned feature matrix (n_timepoints, n_features)
    """

    if feature_times.size == 0:
        raise ValueError("feature_times is empty")

    if window_sec is None:
        window_sec = 0.5 / max(fnirs_fs, MIN_EPS)

    aligned = np.empty((fnirs_time_vec.size, feature_values.shape[1]), dtype=float)
    aligned[:] = np.nan

    spans = feature_spans
    if spans is not None and spans.shape != (feature_times.size, 2):
        raise ValueError("feature_spans must have shape (n_frames, 2)")

    for idx, center in enumerate(fnirs_time_vec):
        lower = center - window_sec
        upper = center + window_sec
        if spans is not None:
            mask = (spans[:, 0] < upper) & (spans[:, 1] >= lower)
        else:
            mask = (feature_times >= lower) & (feature_times < upper)
        if np.any(mask):
            aligned[idx] = np.nanmean(feature_values[mask], axis=0)
            continue

        if fill_strategy == "nearest":
            nearest = np.argmin(np.abs(feature_times - center))
            aligned[idx] = feature_values[nearest]
        elif fill_strategy == "zero":
            aligned[idx] = 0.0
        elif fill_strategy == "nan":
            aligned[idx] = np.nan
        else:
            raise ValueError(f"Unsupported fill strategy '{fill_strategy}'")

    return aligned


def prepare_embedding_timeseries(
    record: EmbeddingRecord,
    fnirs_time_vec: np.ndarray,
    fnirs_fs: float = 11.0,
    standardize_method: str = "zscore",
    dim_reduction_method: str = "pca",
    n_components: int = 50,
    window_sec: Optional[float] = None,
) -> Tuple[np.ndarray, FeatureTransformMeta]:
    """
    Full preprocessing: flatten -> standardize -> reduce -> align.

    Returns:
        aligned_features, FeatureTransformMeta
    """

    features, centers, spans = flatten_embedding(record)
    standardized, stats = standardize_features(features, method=standardize_method)
    reduced, pca_params = reduce_dimension(
        standardized,
        method=dim_reduction_method,
        n_components=n_components,
    )
    aligned = align_features_to_fnirs(
        feature_times=centers,
        feature_values=reduced,
        fnirs_time_vec=fnirs_time_vec,
        fnirs_fs=fnirs_fs,
        window_sec=window_sec,
        feature_spans=spans,
    )

    meta = FeatureTransformMeta(
        standardize_method=standardize_method,
        standardize_params=stats,
        dim_reduction_method=dim_reduction_method,
        dim_reduction_components=reduced.shape[1],
        pca_params=pca_params,
    )
    return aligned, meta


def _collapse_timestamp(ts: np.ndarray) -> float:
    """Convert arbitrary timestamp array to a scalar (prefer last element)."""

    if ts.size == 0:
        return np.nan
    if ts.size == 1:
        return float(ts.item())
    # Use last timestamp (word-end) for language embeddings
    return float(ts.reshape(-1)[-1])


def _pca_fit_transform(features: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCAParams]:
    """Lightweight PCA using SVD."""

    feats = np.nan_to_num(features, nan=0.0)
    mean = np.mean(feats, axis=0, keepdims=True)
    centered = feats - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    transformed = centered @ components.T

    total_var = (S ** 2).sum()
    if total_var <= 0:
        var_ratio = np.zeros(n_components)
    else:
        var_ratio = (S[:n_components] ** 2) / total_var

    params = PCAParams(mean=mean, components=components, var_ratio=var_ratio)
    return transformed, params


def fit_feature_transform_meta(
    feature_list: Sequence[np.ndarray],
    standardize_method: str = "zscore",
    dim_reduction_method: str = "pca",
    n_components: int = 50,
) -> FeatureTransformMeta:
    """
    Fit normalization + dimensionality reduction on concatenated features.

    Args:
        feature_list: sequence of (n_samples_i, n_features) arrays
    """

    if not feature_list:
        raise ValueError("feature_list must contain at least one array")

    stacked = np.vstack([np.asarray(feat, dtype=float) for feat in feature_list])
    standardized, stats = standardize_features(stacked, method=standardize_method)
    reduced, pca_params = reduce_dimension(
        standardized,
        method=dim_reduction_method,
        n_components=n_components,
    )

    return FeatureTransformMeta(
        standardize_method=standardize_method,
        standardize_params=stats,
        dim_reduction_method=dim_reduction_method,
        dim_reduction_components=reduced.shape[1],
        pca_params=pca_params,
    )


def apply_feature_transform(features: np.ndarray, meta: FeatureTransformMeta) -> np.ndarray:
    """Apply previously fitted FeatureTransformMeta to new data."""

    transformed = np.asarray(features, dtype=float)

    method = meta.standardize_method.lower()
    if method == "zscore":
        mean = meta.standardize_params.get("mean")
        std = meta.standardize_params.get("std")
        if mean is None or std is None:
            raise ValueError("FeatureTransformMeta missing mean/std for zscore scaling")
        std = np.clip(std, MIN_EPS, None)
        transformed = (transformed - mean) / std
    elif method == "robust":
        median = meta.standardize_params.get("median")
        mad = meta.standardize_params.get("mad")
        if median is None or mad is None:
            raise ValueError("FeatureTransformMeta missing median/mad for robust scaling")
        scale = np.clip(mad * 1.4826, MIN_EPS, None)
        transformed = (transformed - median) / scale
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unsupported standardize method '{meta.standardize_method}'")

    if meta.dim_reduction_method.lower() == "pca" and meta.pca_params:
        transformed = _apply_pca_with_params(transformed, meta.pca_params)

    return transformed


def _apply_pca_with_params(features: np.ndarray, params: PCAParams) -> np.ndarray:
    feats = np.nan_to_num(features, nan=0.0)
    centered = feats - params.mean
    return centered @ params.components.T

