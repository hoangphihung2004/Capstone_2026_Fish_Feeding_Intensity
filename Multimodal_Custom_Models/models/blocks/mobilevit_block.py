from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSelfAttention(nn.Module):
    """
    MobileViTv2 Linear Self-Attention module with O(N) complexity.
    Calculates global context via element-wise expectation weights.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.qkv_proj = nn.Conv2d(embed_dim, 1 + 2 * embed_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        qkv = self.qkv_proj(x)
        q_weight, k, v = torch.split(qkv, [1, self.embed_dim, self.embed_dim], dim=1)

        B, _, H, W = x.shape
        attn_weights = F.softmax(q_weight.view(B, 1, -1), dim=-1).view(B, 1, H, W)
        context = torch.sum(attn_weights * k, dim=(-2, -1), keepdim=True)

        out = F.relu(v * context)
        out = self.out_proj(out)
        return out


class MobileViTv2Block(nn.Module):
    """
    Lightweight MobileViTv2 Block with Depthwise Convolutions and Linear Self-Attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attn_dim: int = 64,
        conv_ksize: int = 3,
    ) -> None:
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=conv_ksize, padding=conv_ksize // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_dim),
            nn.SiLU(inplace=True),
        )

        self.global_attn = LinearSelfAttention(embed_dim=attn_dim)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(attn_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        feat = self.local_conv(x)
        feat = self.global_attn(feat)
        out = self.fusion_conv(feat)
        return F.silu(out + res)
