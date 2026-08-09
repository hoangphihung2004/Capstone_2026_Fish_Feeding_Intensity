from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MobileV2Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 1, pool_size: tuple[int, int] = (2, 2)):
        super().__init__()
        hidden = out_channels * expansion
        self.pool_size = pool_size
        self.use_shortcut = in_channels == out_channels
        self.shortcut = None if self.use_shortcut else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels, hidden, kernel_size=1),
            ConvBNReLU(hidden, hidden, kernel_size=3, groups=hidden),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.block2 = nn.Sequential(
            ConvBNReLU(out_channels, hidden, kernel_size=1),
            ConvBNReLU(hidden, hidden, kernel_size=3, groups=hidden),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_shortcut else self.shortcut(x)
        x = self.relu(residual + self.block1(x))
        x = self.relu(x + self.block2(x))
        return F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)


class DepthwiseSeparableProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, groups=in_channels),
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
