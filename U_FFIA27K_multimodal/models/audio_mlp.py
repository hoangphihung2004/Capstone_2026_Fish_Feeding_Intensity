"""MLP audio backbone (ported from the `final` branch): Linear + LayerNorm + GELU + Dropout blocks, then Linear + LayerNorm."""

from typing import Tuple

import torch
from torch import nn


class AudioMLPBackbone(nn.Module):
    """[B, in_features] -> [B, embed_dim]. Defaults (2049 -> 768 -> 512 -> 384) are the `final` branch's ~2.17M-param MLP."""

    def __init__(self, in_features: int = 2049, hidden_dims: Tuple[int, ...] = (768, 512), embed_dim: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_features = in_features
        layers = []
        width = in_features
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        layers += [nn.Linear(width, embed_dim), nn.LayerNorm(embed_dim)]
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"Expected input of shape [B, {self.in_features}], got {tuple(x.shape)}.")
        return self.net(x)
