from __future__ import annotations

import torch
import torch.nn as nn

from config import FusionConfig


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _classifier_block(input_dim: int, hidden_dim: int, num_classes: int, cfg: FusionConfig) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
    if cfg.use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.extend([_activation(cfg.activation), nn.Dropout(cfg.dropout), nn.Linear(hidden_dim, num_classes)])
    return nn.Sequential(*layers)


def _direct_classifier(input_dim: int, num_classes: int, cfg: FusionConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    if cfg.use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    layers.extend([_activation(cfg.activation), nn.Dropout(cfg.dropout), nn.Linear(input_dim, num_classes)])
    return nn.Sequential(*layers)


class RawConcatFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.classifier = _classifier_block(dim_audio + dim_video, cfg.hidden_dim, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([audio_feat, video_feat], dim=-1))


class LinearConcatFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.proj_audio = nn.Linear(dim_audio, cfg.proj_dim)
        self.proj_video = nn.Linear(dim_video, cfg.proj_dim)
        self.classifier = _direct_classifier(cfg.proj_dim * 2, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_audio(audio_feat)
        z_v = self.proj_video(video_feat)
        return self.classifier(torch.cat([z_a, z_v], dim=-1))


class LinearMeanFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.proj_audio = nn.Linear(dim_audio, cfg.proj_dim)
        self.proj_video = nn.Linear(dim_video, cfg.proj_dim)
        self.classifier = _direct_classifier(cfg.proj_dim, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_audio(audio_feat)
        z_v = self.proj_video(video_feat)
        return self.classifier((z_a + z_v) / 2.0)


class GatedFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.proj_audio = nn.Linear(dim_audio, cfg.proj_dim)
        self.proj_video = nn.Linear(dim_video, cfg.proj_dim)
        self.gate_layer = nn.Sequential(nn.Linear(cfg.proj_dim * 2, cfg.proj_dim), nn.Sigmoid())
        self.classifier = _direct_classifier(cfg.proj_dim, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_audio(audio_feat)
        z_v = self.proj_video(video_feat)
        gate = self.gate_layer(torch.cat([z_a, z_v], dim=-1))
        return self.classifier(gate * z_a + (1.0 - gate) * z_v)


class SelfAttentionFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        if cfg.proj_dim % cfg.num_heads != 0:
            raise ValueError(f"fusion.proj_dim={cfg.proj_dim} must be divisible by num_heads={cfg.num_heads}.")
        self.proj_audio = nn.Linear(dim_audio, cfg.proj_dim)
        self.proj_video = nn.Linear(dim_video, cfg.proj_dim)
        self.attention = nn.MultiheadAttention(embed_dim=cfg.proj_dim, num_heads=cfg.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.proj_dim)
        self.classifier = _classifier_block(cfg.proj_dim * 2, cfg.hidden_dim, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_audio(audio_feat)
        z_v = self.proj_video(video_feat)
        tokens = torch.stack([z_a, z_v], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens)
        fused_tokens = self.norm(tokens + attended)
        return self.classifier(fused_tokens.flatten(start_dim=1))


FUSION_HEADS = {
    "raw_concat": RawConcatFusion,
    "linear_concat": LinearConcatFusion,
    "linear_mean": LinearMeanFusion,
    "gated_fusion": GatedFusion,
    "self_attention": SelfAttentionFusion,
}


def build_fusion_head(dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> nn.Module:
    if cfg.type not in FUSION_HEADS:
        raise ValueError(f"Unsupported fusion type '{cfg.type}'. Expected one of {sorted(FUSION_HEADS)}.")
    return FUSION_HEADS[cfg.type](dim_audio=dim_audio, dim_video=dim_video, num_classes=num_classes, cfg=cfg)
