import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def adapt_first_conv_to_multi_channels(model, target_channels):
    """
    Find the very first nn.Conv2d layer (which should be 3-channel), 
    duplicate its weights to match target_channels, divide by the duplication factor, and replace it.
    """
    if target_channels == 3:
        return
        
    num_frames = target_channels // 3

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.in_channels == 3:
                logger.info(f"Adapting first conv layer '{name}' to {target_channels} channels for Early Fusion.")
                old_conv = module
                
                new_conv = nn.Conv2d(
                    in_channels=target_channels,
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
                    # Concatenate along channel dim (dim=1)
                    new_w = torch.cat([w] * num_frames, dim=1)
                    # Divide by num_frames to keep activation magnitude consistent
                    new_conv.weight.data = new_w / float(num_frames)
                    
                    if old_conv.bias is not None:
                        new_conv.bias.data = old_conv.bias.data
                
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_conv)
                
                return
    logger.warning("Could not find a 3-channel Conv2d to adapt!")
