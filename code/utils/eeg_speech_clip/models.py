from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetEncoder(nn.Module):
    """
    Compact EEGNet-style encoder.

    Input shape is (batch, channels, time). The module applies temporal
    filtering, depthwise spatial filtering, separable temporal filtering, and
    adaptive pooling. It intentionally leaves frequency/noise discovery to the
    learned filters instead of applying an explicit low-frequency bandpass.
    """

    def __init__(
        self,
        n_channels: int = 31,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        temporal_kernel: int = 125,
        separable_kernel: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1
        if separable_kernel % 2 == 0:
            separable_kernel += 1
        self.n_channels = int(n_channels)
        self.f2 = int(f2)

        self.temporal = nn.Conv2d(
            1,
            f1,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(
            f1,
            f1 * depth_multiplier,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f1 * depth_multiplier)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depth = nn.Conv2d(
            f1 * depth_multiplier,
            f1 * depth_multiplier,
            kernel_size=(1, separable_kernel),
            padding=(0, separable_kernel // 2),
            groups=f1 * depth_multiplier,
            bias=False,
        )
        self.sep_point = nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.out_pool = nn.AdaptiveAvgPool2d((1, 1))

    @property
    def output_dim(self) -> int:
        return self.f2

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shape (B,C,T), got {tuple(eeg.shape)}")
        if eeg.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} EEG channels, got {int(eeg.shape[1])}"
            )
        x = eeg.unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = F.elu(self.bn2(self.spatial(x)))
        x = self.drop1(self.pool1(x))
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = F.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))
        x = self.out_pool(x)
        return x.flatten(start_dim=1)


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_proj: int = 128,
        hidden: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        h = int(hidden or max(d_proj, min(512, d_in)))
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, d_proj),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpeechEEGClipModel(nn.Module):
    def __init__(
        self,
        speech_dim: int,
        n_eeg_channels: int = 31,
        d_proj: int = 128,
        eeg_dropout: float = 0.25,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.eeg_encoder = EEGNetEncoder(n_channels=n_eeg_channels, dropout=eeg_dropout)
        self.eeg_proj = ProjectionMLP(
            d_in=self.eeg_encoder.output_dim,
            d_proj=d_proj,
            hidden=128,
            dropout=proj_dropout,
        )
        self.speech_proj = ProjectionMLP(
            d_in=int(speech_dim),
            d_proj=d_proj,
            hidden=max(256, d_proj),
            dropout=proj_dropout,
        )

    def forward(self, eeg: torch.Tensor, speech: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eeg_feat = self.eeg_encoder(eeg)
        eeg_emb = self.eeg_proj(eeg_feat)
        speech_emb = self.speech_proj(speech)
        return eeg_emb, speech_emb

