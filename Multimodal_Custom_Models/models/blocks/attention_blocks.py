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
