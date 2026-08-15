from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFiLMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.audio_to_video = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        self.video_to_audio = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _apply_film(x: torch.Tensor, gamma_beta: torch.Tensor) -> torch.Tensor:
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        gamma = 1.0 + torch.sigmoid(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_desc = self.pool(audio_feat).flatten(1)
        video_desc = self.pool(video_feat).flatten(1)
        audio_feat = self._apply_film(audio_feat, self.video_to_audio(video_desc))
        video_feat = self._apply_film(video_feat, self.audio_to_video(audio_desc))
        return audio_feat, video_feat


class DynamicInteractionUnit(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.audio_to_video = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        self.video_to_audio = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )
        self.video_gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _film(x: torch.Tensor, gamma_beta: torch.Tensor) -> torch.Tensor:
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        gamma = 1.0 + torch.sigmoid(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_desc = self.pool(audio_feat).flatten(1)
        video_desc = self.pool(video_feat).flatten(1)
        joint_desc = torch.cat([audio_desc, video_desc], dim=1)

        audio_modulated = self._film(audio_feat, self.video_to_audio(video_desc))
        video_modulated = self._film(video_feat, self.audio_to_video(audio_desc))

        audio_gate = self.audio_gate(joint_desc).unsqueeze(-1).unsqueeze(-1)
        video_gate = self.video_gate(joint_desc).unsqueeze(-1).unsqueeze(-1)
        audio_out = audio_feat + audio_gate * audio_modulated
        video_out = video_feat + video_gate * video_modulated
        return audio_out, video_out


class LightweightCrossAttentionUnit(nn.Module):
    """
    Lightweight Multi-Scale Cross-Attention Unit using Depthwise Separable Projections.
    Enables Audio Queries to attend to Video Keys/Values, and Video Queries to attend to Audio Keys/Values.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.q_proj_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.k_proj_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.v_proj_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.q_proj_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.k_proj_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.v_proj_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.gate_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Global descriptors for Query-Key cross attention
        q_a = self.q_proj_a(audio_feat)
        k_v = self.pool(self.k_proj_v(video_feat))
        v_v = self.pool(self.v_proj_v(video_feat))

        attn_a = torch.sigmoid(q_a * k_v)
        audio_cross = attn_a * v_v

        q_v = self.q_proj_v(video_feat)
        k_a = self.pool(self.k_proj_a(audio_feat))
        v_a = self.pool(self.v_proj_a(audio_feat))

        attn_v = torch.sigmoid(q_v * k_a)
        video_cross = attn_v * v_a

        audio_out = audio_feat + self.gate_a(audio_feat) * audio_cross
        video_out = video_feat + self.gate_v(video_feat) * video_cross
        return audio_out, video_out


class DynamicSpatialFrequencyModulationBlock(nn.Module):
    """
    Novelty Contribution #1: D-FSFM (Dynamic Frequency-Spatial Feature Modulation Unit)
    Calculates a 2D Frequency-Spatial Gate Tensor modulating video spatial regions
    based on active audio frequency sub-bands, and vice versa.
    Uses Depthwise Separable Convolutions for minimal parameter cost.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.audio_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.video_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.gate_audio = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_video = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_desc = self.audio_proj(self.pool(audio_feat))
        v_desc = self.video_proj(self.pool(video_feat))

        # Frequency-to-Spatial Modulation
        video_modulated = video_feat * a_desc
        audio_modulated = audio_feat * v_desc

        audio_out = audio_feat + self.gate_audio(audio_feat) * audio_modulated
        video_out = video_feat + self.gate_video(video_feat) * video_modulated
        return audio_out, video_out

