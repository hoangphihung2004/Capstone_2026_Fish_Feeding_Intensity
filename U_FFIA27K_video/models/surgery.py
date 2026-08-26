import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def adapt_first_conv_to_6ch(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.in_channels == 3:
                logger.info(f"Adapting first conv layer '{name}' to 6 channels for Early Fusion.")
                old_conv = module
                
                new_conv = nn.Conv2d(
                    in_channels=6,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=old_conv.groups,
                    bias=(old_conv.bias is not None),
                    padding_mode=old_conv.padding_mode
                )
                
                with torch.no_grad():
                    w = old_conv.weight.data
                    new_w = torch.cat([w, w], dim=1)
                    new_conv.weight.data = new_w / 2.0
                    
                    if old_conv.bias is not None:
                        new_conv.bias.data = old_conv.bias.data
                
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_conv)
                
                return
    logger.warning("Could not find a 3-channel Conv2d to adapt!")
