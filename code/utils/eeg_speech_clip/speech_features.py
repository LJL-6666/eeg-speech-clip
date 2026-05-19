from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import DEFAULT_LOCAL_WHISPER_BASE
from .dataset import MaterialInfo, find_audio_path, load_materials

LOGGER = logging.getLogger(__name__)


SAMPLE_RATE = 16000


def import_audio_dependencies():
    try:
        import librosa
        import torch
        from transformers import (
            HubertModel,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
            WhisperFeatureExtractor,
            WhisperModel,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Speech feature extraction requires torch, transformers, and librosa. "
            "Use environments/ty_eeg_speech_stage1.yml."
        ) from exc
    return {
        "librosa": librosa,
        "torch": torch,
        "HubertModel": HubertModel,
        "Wav2Vec2FeatureExtractor": Wav2Vec2FeatureExtractor,
        "Wav2Vec2Model": Wav2Vec2Model,
        "WhisperFeatureExtractor": WhisperFeatureExtractor,
        "WhisperModel": WhisperModel,
    }


def load_audio(librosa_module, audio_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = librosa_module.load(str(audio_path), sr=sample_rate, mono=True)
    if sr != sample_rate:
        raise RuntimeError(f"Unexpected sample rate after load: {sr} != {sample_rate}")
    return np.asarray(audio, dtype=np.float32)


def window_starts(duration_sec: float, window_sec: float, stride_sec: float) -> list[float]:
    starts: list[float] = []
    t = 0.0
    while t + window_sec <= duration_sec + 1e-9:
        starts.append(float(t))
        t += stride_sec
    return starts


def resolve_model_load_path(backend: str, model_id: str, model_path: Optional[Path]):
    if model_path is not None:
        return model_path
    if backend == "whisper" and model_id == "openai/whisper-base" and DEFAULT_LOCAL_WHISPER_BASE.exists():
        return DEFAULT_LOCAL_WHISPER_BASE
    return model_id


class FrozenSpeechWindowEncoder:
    def __init__(
        self,
        backend: str,
        model_id: str,
        device: str,
        model_path: Optional[Path] = None,
        local_files_only: bool = False,
    ) -> None:
        deps = import_audio_dependencies()
        self.torch = deps["torch"]
        self.backend = str(backend)
        self.model_id = str(model_id)
        self.device = device if device != "auto" else ("cuda" if self.torch.cuda.is_available() else "cpu")
        load_path = resolve_model_load_path(self.backend, self.model_id, model_path)
        local_only = bool(local_files_only or model_path is not None or isinstance(load_path, Path))

        if self.backend == "whisper":
            feature_cls = deps["WhisperFeatureExtractor"]
            model_cls = deps["WhisperModel"]
            self.processor = feature_cls.from_pretrained(load_path, local_files_only=local_only)
            full_model = model_cls.from_pretrained(load_path, local_files_only=local_only)
            self.model = full_model.encoder.to(self.device)
            self.hidden_dim = int(full_model.config.d_model)
        elif self.backend == "wav2vec2":
            feature_cls = deps["Wav2Vec2FeatureExtractor"]
            model_cls = deps["Wav2Vec2Model"]
            self.processor = feature_cls.from_pretrained(load_path, local_files_only=local_only)
            self.model = model_cls.from_pretrained(load_path, local_files_only=local_only).to(self.device)
            self.hidden_dim = int(self.model.config.hidden_size)
        elif self.backend == "hubert":
            feature_cls = deps["Wav2Vec2FeatureExtractor"]
            model_cls = deps["HubertModel"]
            self.processor = feature_cls.from_pretrained(load_path, local_files_only=local_only)
            self.model = model_cls.from_pretrained(load_path, local_files_only=local_only).to(self.device)
            self.hidden_dim = int(self.model.config.hidden_size)
        else:
            raise ValueError(f"Unsupported speech backend: {self.backend}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_batch(self, windows: np.ndarray) -> np.ndarray:
        if windows.ndim != 2:
            raise ValueError(f"Expected audio windows (B,N), got {windows.shape}")
        torch = self.torch
        if self.backend == "whisper":
            inputs = self.processor(
                [x for x in windows],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self.device)
            with torch.no_grad():
                out = self.model(input_features, return_dict=True)
            features = out.last_hidden_state.mean(dim=1)
        else:
            inputs = self.processor(
                [x for x in windows],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                out = self.model(input_values, attention_mask=attention_mask, return_dict=True)
            hidden = out.last_hidden_state
            if attention_mask is not None:
                # Approximate feature mask after convolutional downsampling.
                valid = torch.ones(hidden.shape[:2], dtype=hidden.dtype, device=hidden.device)
                lengths = attention_mask.sum(dim=1).float()
                feature_lengths = (lengths / input_values.shape[1] * hidden.shape[1]).ceil().long()
                for i, length in enumerate(feature_lengths):
                    valid[i, int(length):] = 0
                features = (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            else:
                features = hidden.mean(dim=1)
        return features.detach().cpu().numpy().astype(np.float32)


def build_material_speech_cache(
    materials: Sequence[MaterialInfo],
    cache_dir: Path,
    backend: str,
    model_id: str,
    window_sec: float,
    stride_sec: float,
    batch_size: int,
    device: str,
    model_path: Optional[Path] = None,
    local_files_only: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    deps = import_audio_dependencies()
    librosa = deps["librosa"]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder = FrozenSpeechWindowEncoder(
        backend=backend,
        model_id=model_id,
        device=device,
        model_path=model_path,
        local_files_only=local_files_only,
    )
    written: list[Path] = []

    seen: dict[str, MaterialInfo] = {}
    for material in materials:
        seen.setdefault(material.material_key, material)

    for material in seen.values():
        out_path = cache_dir / f"{material.material_key}.npz"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue
        audio_path = Path(material.audio_path) if material.audio_path else find_audio_path(
            material.stimulus_version, material.video_id
        )
        audio = load_audio(librosa, audio_path)
        duration = len(audio) / float(SAMPLE_RATE)
        starts = window_starts(duration, window_sec, stride_sec)
        if not starts:
            LOGGER.warning("No speech windows for %s", material.material_key)
            continue
        win_samples = int(round(window_sec * SAMPLE_RATE))
        windows = []
        for start in starts:
            s = int(round(start * SAMPLE_RATE))
            seg = audio[s : s + win_samples]
            if len(seg) < win_samples:
                seg = np.pad(seg, (0, win_samples - len(seg)), mode="constant")
            windows.append(seg.astype(np.float32))
        features = []
        for i in range(0, len(windows), int(batch_size)):
            features.append(encoder.encode_batch(np.stack(windows[i : i + int(batch_size)], axis=0)))
        feat = np.vstack(features).astype(np.float32)
        window_idx = np.asarray([int(round(st / stride_sec)) for st in starts], dtype=np.int32)
        meta = {
            "backend": backend,
            "model_id": model_id,
            "material_key": material.material_key,
            "stimulus_version": material.stimulus_version,
            "video_id": int(material.video_id),
            "speaker_id": material.speaker_id,
            "emotion": material.emotion,
            "audio_path": str(audio_path),
            "window_sec": float(window_sec),
            "stride_sec": float(stride_sec),
            "sample_rate": SAMPLE_RATE,
            "feature_dim": int(feat.shape[1]),
        }
        tmp_path = out_path.with_name(out_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            window_idx=window_idx,
            starts=np.asarray(starts, dtype=np.float32),
            features=feat,
            meta=json.dumps(meta, ensure_ascii=False),
        )
        os.replace(tmp_path, out_path)
        written.append(out_path)
        LOGGER.info("Saved speech cache: %s shape=%s", out_path, feat.shape)

    return written


def materials_for_video_ids(video_ids: Sequence[int]) -> list[MaterialInfo]:
    return list(load_materials(video_ids).values())

