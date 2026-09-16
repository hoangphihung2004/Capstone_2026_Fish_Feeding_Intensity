import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def adapt_first_conv_to_multi_channels(model: nn.Module, target_channels: int) -> None:
    """Adapt the first RGB convolution exactly as U_FFIA27K_video_Multi_channels."""
    if target_channels == 3:
        return
    if target_channels % 3 != 0:
        raise ValueError(f"target_channels must be divisible by 3, got {target_channels}.")

    num_frames = target_channels // 3
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            logger.info("Adapting first conv layer '%s' to %d channels for early fusion.", name, target_channels)
            new_conv = nn.Conv2d(
                in_channels=target_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
            )
            with torch.no_grad():
                new_conv.weight.copy_(torch.cat([module.weight.data] * num_frames, dim=1) / float(num_frames))
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias.data)

            parent = model
            for part in name.split(".")[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split(".")[-1], new_conv)
            return
    logger.warning("Could not find a 3-channel Conv2d to adapt.")
