from __future__ import annotations

import torch
import torch.nn as nn


class TokenSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1, ffn_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens))
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class ECABlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(1, 2)
        weights = self.conv(weights).transpose(1, 2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_weight = self.channel_mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_weight = self.channel_mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        x = x * self.sigmoid(avg_weight + max_weight)
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.amax(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.spatial(torch.cat([spatial_avg, spatial_max], dim=1)))


class BottleneckTokenAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, attn_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, attn_dim * 3, bias=False)
        self.proj = nn.Linear(attn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized)
        qkv = qkv.reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = self.dropout(torch.softmax(attn, dim=-1))
        attended = attn @ value
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, self.num_heads * self.head_dim)
        tokens = tokens + self.dropout(self.proj(attended))
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens
