from __future__ import annotations

import torch
import torch.nn as nn


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
