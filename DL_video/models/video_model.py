import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.base_backbone import BaseBackbone

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoModel(nn.Module):
    """
    Unified VideoModel Wrapper class (matching AudioModel architecture).
    Wraps a 3D Video Backbone model complying with the BaseBackbone contract.
    Accepts 5D video frame tensor [Batch, Channels (3), Frames (T), Height, Width]
    and returns a dictionary containing classification logits 'clipwise_output' [Batch, Num_Classes].
    """
    def __init__(self, backbone: nn.Module) -> None:
        """
        Initialize VideoModel wrapper.

        Args:
            backbone (nn.Module): Video backbone model (e.g. S3D, ViT3D, ViViT, ResNet3D, MobileVit).
        """
        super(VideoModel, self).__init__()

        if isinstance(backbone, BaseBackbone):
            logger.info(f"Verified backbone model '{backbone.get_name()}' inherits from BaseBackbone.")

        self.backbone = backbone

        logger.info("==================================================")
        logger.info("Initialized unified VideoModel wrapper:")
        logger.info(f"  - Backbone: {self.backbone.__class__.__name__}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward Pass of the unified VideoModel.

        Args:
            input_tensor (torch.Tensor): Video frames tensor [Batch, Channels (3), Frames (T), Height, Width].

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing classification logits 'clipwise_output' [Batch, Num_Classes].
        """
        output = self.backbone(input_tensor)

        # Handle models that return a dict or tuple
        if isinstance(output, dict):
            logits = output.get('clipwise_output', output)
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        # Squeeze temporal dimension if shape is [B, Num_Classes, 1]
        if logits.dim() == 3 and logits.size(2) == 1:
            logits = logits.squeeze(2)

        return {
            "clipwise_output": logits
        }
