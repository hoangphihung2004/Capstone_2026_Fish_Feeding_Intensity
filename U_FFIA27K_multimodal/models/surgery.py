import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def adapt_first_conv_to_multi_channels(model: nn.Module, target_channels: int = 6) -> None:
    """
    Adapt the first 3-channel Conv2d to first-last 6-channel video input.

    The pretrained RGB weights are duplicated for each frame and divided by the
    number of frames, matching the video baseline's activation-scale behavior.
    """
    if target_channels == 3:
        return
    if target_channels % 3 != 0:
        raise ValueError(f"target_channels must be divisible by 3, got {target_channels}")

    num_frames = target_channels // 3
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d) or module.in_channels != 3:
            continue

        logger.info(f"Adapting first conv layer '{name}' to {target_channels} channels.")
        old_conv = module
        new_conv = nn.Conv2d(
            in_channels=target_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )
        with torch.no_grad():
            new_conv.weight.data = torch.cat([old_conv.weight.data] * num_frames, dim=1) / float(num_frames)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data

        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)
        return

    raise ValueError("Could not find a 3-channel Conv2d to adapt.")
