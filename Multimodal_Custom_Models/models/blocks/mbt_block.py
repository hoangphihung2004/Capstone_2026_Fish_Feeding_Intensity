from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MBTCrossAttentionBlock(nn.Module):
    """
    SOTA Multimodal Bottleneck Transformer (MBT) Cross-Attention Block (NeurIPS Paper):
    - Uses K=4 learnable Bottleneck Tokens (B) in latent space.
    - Audio tokens update Bottleneck tokens via Cross-Attention.
    - Bottleneck tokens then update Video tokens via Cross-Attention.
    Reduces quadratic cross-attention complexity O(Na * Nv) to linear O((Na + Nv) * K).
    """

    def __init__(self, channels: int, num_bottlenecks: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_bottlenecks = num_bottlenecks
        self.num_heads = num_heads
        self.head_dim = max(1, channels // num_heads)
        self.scale = self.head_dim ** -0.5

        # K=4 learnable Bottleneck Tokens
        self.bottleneck_tokens = nn.Parameter(torch.randn(1, num_bottlenecks, channels) * 0.02)

        # Cross-Attention: Audio -> Bottleneck
        self.norm_b1 = nn.LayerNorm(channels)
        self.norm_a = nn.LayerNorm(channels)
        self.q_b1 = nn.Linear(channels, channels, bias=False)
        self.k_a = nn.Linear(channels, channels, bias=False)
        self.v_a = nn.Linear(channels, channels, bias=False)

        # Cross-Attention: Bottleneck -> Video
        self.norm_v = nn.LayerNorm(channels)
        self.norm_b2 = nn.LayerNorm(channels)
        self.q_v = nn.Linear(channels, channels, bias=False)
        self.k_b2 = nn.Linear(channels, channels, bias=False)
        self.v_b2 = nn.Linear(channels, channels, bias=False)

        self.gate_v = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, ha, wa = audio_feat.shape
        _, _, hv, wv = video_feat.shape

        # Flatten spatial dimensions into tokens
        tokens_a = audio_feat.flatten(2).transpose(1, 2)  # [B, Ha*Wa, C]
        tokens_v = video_feat.flatten(2).transpose(1, 2)  # [B, Hv*Wv, C]

        # Expand Bottleneck Tokens for current batch
        b_tokens = self.bottleneck_tokens.expand(batch_size, -1, -1)  # [B, K, C]

        # 1. Audio -> Bottleneck Tokens
        q_b = self.q_b1(self.norm_b1(b_tokens)).reshape(batch_size, self.num_bottlenecks, self.num_heads, self.head_dim).transpose(1, 2)
        k_a = self.k_a(self.norm_a(tokens_a)).reshape(batch_size, ha * wa, self.num_heads, self.head_dim).transpose(1, 2)
        v_a = self.v_a(self.norm_a(tokens_a)).reshape(batch_size, ha * wa, self.num_heads, self.head_dim).transpose(1, 2)

        attn_ba = torch.softmax((q_b @ k_a.transpose(-2, -1)) * self.scale, dim=-1)
        b_updated = b_tokens + (attn_ba @ v_a).transpose(1, 2).reshape(batch_size, self.num_bottlenecks, channels)

        # 2. Bottleneck Tokens -> Video
        q_v = self.q_v(self.norm_v(tokens_v)).reshape(batch_size, hv * wv, self.num_heads, self.head_dim).transpose(1, 2)
        k_b = self.k_b2(self.norm_b2(b_updated)).reshape(batch_size, self.num_bottlenecks, self.num_heads, self.head_dim).transpose(1, 2)
        v_b = self.v_b2(self.norm_b2(b_updated)).reshape(batch_size, self.num_bottlenecks, self.num_heads, self.head_dim).transpose(1, 2)

        attn_vb = torch.softmax((q_v @ k_b.transpose(-2, -1)) * self.scale, dim=-1)
        v_cross = (attn_vb @ v_b).transpose(1, 2).reshape(batch_size, hv * wv, channels)
        v_updated = tokens_v + self.gate_v(tokens_v) * v_cross

        # Reshape tokens back to spatial feature maps
        audio_out = audio_feat
        video_out = v_updated.transpose(1, 2).reshape(batch_size, channels, hv, wv)

        return audio_out, video_out

