from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMutualCrossAttentionBlock(nn.Module):
    """
    SOTA Mechanism C: Mutual Information Gated Cross-Attention (V2 Architecture - IEEE T-MM 2025):
    - Bi-directional mutual cross-attention between Audio and Video tokens.
    - Mutual Information Gate (M_av, M_va in [0, 1]) dynamically estimates cross-modal dependency.
    - Suppresses modality-specific background noise (motor sound / sun glare) while preserving correlated feeding cues.
    """

    def __init__(self, channels: int, num_bottlenecks: int = 4, num_heads: int = 4, **kwargs) -> None:
        super().__init__()
        self.channels = channels
        self.num_bottlenecks = num_bottlenecks
        self.num_heads = num_heads
        self.head_dim = max(1, channels // num_heads)
        self.scale = self.head_dim ** -0.5


        # Shared Q/K/V Projections for parameter efficiency and shared latent mapping
        self.norm_a = nn.LayerNorm(channels)
        self.norm_v = nn.LayerNorm(channels)

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)

        # Mutual Information Estimator Gates
        hidden_dim = max(channels // 4, 32)
        self.mi_gate_a = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid(),
        )
        self.mi_gate_v = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid(),
        )

        self.out_proj_a = nn.Linear(channels, channels)
        self.out_proj_v = nn.Linear(channels, channels)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, ha, wa = audio_feat.shape
        _, _, hv, wv = video_feat.shape

        tokens_a = audio_feat.flatten(2).transpose(1, 2)  # [B, Na, C]
        tokens_v = video_feat.flatten(2).transpose(1, 2)  # [B, Nv, C]

        norm_a = self.norm_a(tokens_a)
        norm_v = self.norm_v(tokens_v)

        # 1. Compute Q/K/V for both modalities
        q_a = self.q_proj(norm_a).reshape(batch_size, ha * wa, self.num_heads, self.head_dim).transpose(1, 2)
        k_a = self.k_proj(norm_a).reshape(batch_size, ha * wa, self.num_heads, self.head_dim).transpose(1, 2)
        v_a = self.v_proj(norm_a).reshape(batch_size, ha * wa, self.num_heads, self.head_dim).transpose(1, 2)

        q_v = self.q_proj(norm_v).reshape(batch_size, hv * wv, self.num_heads, self.head_dim).transpose(1, 2)
        k_v = self.k_proj(norm_v).reshape(batch_size, hv * wv, self.num_heads, self.head_dim).transpose(1, 2)
        v_v = self.v_proj(norm_v).reshape(batch_size, hv * wv, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Bi-Directional Cross-Attention
        # Audio attends to Video
        attn_av = torch.softmax((q_a @ k_v.transpose(-2, -1)) * self.scale, dim=-1)
        cross_a = (attn_av @ v_v).transpose(1, 2).reshape(batch_size, ha * wa, channels)

        # Video attends to Audio
        attn_va = torch.softmax((q_v @ k_a.transpose(-2, -1)) * self.scale, dim=-1)
        cross_v = (attn_va @ v_a).transpose(1, 2).reshape(batch_size, hv * wv, channels)

        # 3. Mutual Information Gating
        a_pool = tokens_a.mean(dim=1)
        v_pool = tokens_v.mean(dim=1)

        cross_a_pool = cross_a.mean(dim=1)
        cross_v_pool = cross_v.mean(dim=1)

        m_av = self.mi_gate_a(torch.cat([a_pool, cross_a_pool], dim=-1)).unsqueeze(1)  # [B, 1, C]
        m_va = self.mi_gate_v(torch.cat([v_pool, cross_v_pool], dim=-1)).unsqueeze(1)  # [B, 1, C]

        # 4. Gated Residual Feature Update
        a_updated = tokens_a + self.out_proj_a(m_av * cross_a)
        v_updated = tokens_v + self.out_proj_v(m_va * cross_v)

        audio_out = a_updated.transpose(1, 2).reshape(batch_size, channels, ha, wa)
        video_out = v_updated.transpose(1, 2).reshape(batch_size, channels, hv, wv)

        return audio_out, video_out


# Alias for backward compatibility
MBTCrossAttentionBlock = MIMutualCrossAttentionBlock
