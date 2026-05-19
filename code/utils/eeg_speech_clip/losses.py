from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipStyleInfoNCELoss(nn.Module):
    """Symmetric CLIP/CLAP-style InfoNCE with normalized embeddings."""

    def __init__(
        self,
        init_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.max_logit_scale = float(max_logit_scale)

    def logits(self, eeg_emb: torch.Tensor, speech_emb: torch.Tensor) -> torch.Tensor:
        eeg_z = F.normalize(eeg_emb, p=2, dim=-1)
        speech_z = F.normalize(speech_emb, p=2, dim=-1)
        scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
        return scale * (eeg_z @ speech_z.t())

    def forward(self, eeg_emb: torch.Tensor, speech_emb: torch.Tensor) -> torch.Tensor:
        if eeg_emb.shape != speech_emb.shape:
            raise ValueError(f"Embedding shape mismatch: {eeg_emb.shape} vs {speech_emb.shape}")
        logits = self.logits(eeg_emb, speech_emb)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_e2s = F.cross_entropy(logits, labels)
        loss_s2e = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_e2s + loss_s2e)

    def temperature(self) -> float:
        return float(1.0 / self.logit_scale.exp().detach().cpu().item())

