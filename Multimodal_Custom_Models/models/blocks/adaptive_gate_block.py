from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConfidenceGate(nn.Module):
    """
    Uncertainty-Aware Adaptive Gated Fusion (SOTA IEEE Aquaculture Paper):
    Dynamically generates per-channel confidence scores ga, gv in [0, 1] for Audio and Video modalities.
    If Video is degraded (water glare/turbidity), g_v drops and model prioritizes Audio.
    If Audio is degraded (motor noise), g_a drops and model prioritizes Video.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
            nn.Sigmoid(),
        )


    def forward(self, f_audio: torch.Tensor, f_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint = torch.cat([f_audio, f_video], dim=-1)
        gates = self.mlp(joint)
        g_audio, g_video = torch.chunk(gates, chunks=2, dim=-1)
        f_fused = g_audio * f_audio + g_video * f_video
        return f_fused, g_audio, g_video
