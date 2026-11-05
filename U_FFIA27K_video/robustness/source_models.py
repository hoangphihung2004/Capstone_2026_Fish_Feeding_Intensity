"""Source (surrogate) video models for the black-box transfer attack.

Byte-identical on the video and multimodal black-box branches, so both build the same source model
and generate the same perturbations. The module layout matches `models.VideoModel` of the video
branch (`backbone.model.<torchvision layers>`), so the trained video checkpoints load strictly.
"""

from typing import Callable, Dict

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, resnet18, swin_t

NUM_CLASSES = 4
INPUT_CHANNELS = 6  # first and last frame, RGB each


def _resnet18() -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def _convnext_tiny() -> nn.Module:
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    return model


def _swin_tiny() -> nn.Module:
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    return model


SOURCE_BUILDERS: Dict[str, Callable[[], nn.Module]] = {
    "ConvNeXtTiny": _convnext_tiny,
    "SwinTiny": _swin_tiny,
    "ResNet18": _resnet18,
}


def _widen_first_conv(model: nn.Module, channels: int) -> None:
    """Same replacement as models/surgery.py: the first 3-channel Conv2d takes `channels` inputs."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            wide = nn.Conv2d(
                channels, module.out_channels, module.kernel_size, module.stride, module.padding,
                module.dilation, module.groups, module.bias is not None, module.padding_mode,
            )
            parent = model
            *path, last = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, last, wide)
            return
    raise ValueError("No 3-channel Conv2d found to widen.")


class _Backbone(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.model(video)


class SourceVideoModel(nn.Module):
    """Returns logits for normalized [B, 6, H, W] video."""

    def __init__(self, name: str) -> None:
        super().__init__()
        if name not in SOURCE_BUILDERS:
            raise ValueError(f"Unknown source model '{name}'. Choose one of {sorted(SOURCE_BUILDERS)}.")
        model = SOURCE_BUILDERS[name]()
        _widen_first_conv(model, INPUT_CHANNELS)
        self.backbone = _Backbone(model)
        self.name = name

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.backbone(video)
