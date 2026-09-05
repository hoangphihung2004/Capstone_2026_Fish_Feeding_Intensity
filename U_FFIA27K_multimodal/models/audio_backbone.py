"""Lightweight multi-axis residual backbone for log-mel audio features."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_norm(in_channels, out_channels, kernel, groups=1, dilation=(1, 1), activation=True):
    padding = ((kernel[0] - 1) * dilation[0] // 2, (kernel[1] - 1) * dilation[1] // 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.01, inplace=True) if activation else nn.Identity(),
    )


class MultiAxisResidualUnit(nn.Module):
    """Mix temporal, frequency and local contexts with sample-adaptive weights."""

    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.expand = conv_norm(in_channels, hidden_channels, (1, 1))
        self.temporal = nn.Sequential(
            conv_norm(hidden_channels, hidden_channels, (5, 1), groups=hidden_channels),
            conv_norm(hidden_channels, hidden_channels, (3, 1), groups=hidden_channels, dilation=(2, 1)),
        )
        self.frequency = nn.Sequential(
            conv_norm(hidden_channels, hidden_channels, (1, 5), groups=hidden_channels),
            conv_norm(hidden_channels, hidden_channels, (1, 3), groups=hidden_channels, dilation=(1, 2)),
        )
        self.local = conv_norm(hidden_channels, hidden_channels, (3, 3), groups=hidden_channels)
        self.branch_selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, 3, kernel_size=1))
        self.project = conv_norm(hidden_channels, out_channels, (1, 1), activation=False)
        self.shortcut = nn.Identity() if in_channels == out_channels else conv_norm(in_channels, out_channels, (1, 1), activation=False)

    def forward(self, x):
        residual = self.shortcut(x)
        base = self.expand(x)
        branches = torch.stack([self.temporal(base), self.frequency(base), self.local(base)], dim=1)
        weights = F.softmax(self.branch_selector(base).flatten(1), dim=1)
        mixed = (branches * weights[:, :, None, None, None]).sum(dim=1)
        return F.leaky_relu(self.project(mixed) + residual, negative_slope=0.01)


class AudioBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__(
            MultiAxisResidualUnit(in_channels, out_channels, hidden_channels),
            MultiAxisResidualUnit(out_channels, out_channels, hidden_channels),
            nn.AvgPool2d(2),
        )


class LightweightAudioEncoder(nn.Module):
    channels = (16, 24, 40, 112, 320)
    hidden_channels = (16, 24, 40, 64, 128)

    def __init__(self):
        super().__init__()
        inputs = (1,) + self.channels[:-1]
        self.blocks = nn.ModuleList(AudioBlock(ci, co, hidden) for ci, co, hidden in zip(inputs, self.channels, self.hidden_channels))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @staticmethod
    def pool(x):
        x = x.mean(dim=3)
        return x.amax(dim=2) + x.mean(dim=2)
