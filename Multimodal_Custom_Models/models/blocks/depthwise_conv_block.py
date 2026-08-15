from __future__ import annotations

import torch
import torch.nn as nn
from models.blocks.attention_blocks import ECABlock


class DepthwiseSeparableConv2d(nn.Module):
    """
    SOTA Depthwise Separable Conv2d block:
    - 3x3 Depthwise Conv (groups=in_channels) for spatial feature extraction
    - 1x1 Pointwise Conv for channel projection
    Reduces parameter count by ~80-85% compared to standard Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class DepthwiseAudioStage(nn.Module):
    """
    Modular Audio Stage Block utilizing Depthwise Separable Convolutions and ECA Attention.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, out_channels, stride=stride),
            ECABlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
