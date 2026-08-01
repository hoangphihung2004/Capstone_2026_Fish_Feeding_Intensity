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

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoModel(nn.Module):
    """
    Thin wrapper around image and clip classification backbones.
    Accepts image tensors [B, C, H, W] or video tensors [B, C, T, H, W]
    and returns classification logits under the existing 'clipwise_output' key.
    """
    def __init__(self, backbone: nn.Module) -> None:
        """
        Initialize VideoModel wrapper.

        Args:
            backbone (nn.Module): Torch image classification model.
        """
        super(VideoModel, self).__init__()

        self.backbone = backbone
        self.model_name = getattr(backbone, "model_name", backbone.__class__.__name__.lower())
        self.input_type = getattr(backbone, "input_type", "image")

        logger.info("==================================================")
        logger.info("Initialized unified VideoModel wrapper:")
        logger.info(f"  - Backbone: {self.model_name}")
        logger.info(f"  - Input Type: {self.input_type}")
        logger.info("==================================================")

    def get_name(self) -> str:
        return self.model_name

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward Pass of the unified VideoModel.

        Args:
            input_tensor (torch.Tensor): Image [B, C, H, W] or clip [B, C, T, H, W].

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing classification logits 'clipwise_output' [Batch, Num_Classes].
        """
        expected_dims = 5 if self.input_type == "clip" else 4
        if input_tensor.dim() != expected_dims:
            raise ValueError(
                f"Backbone '{self.model_name}' expects a {expected_dims}D {self.input_type} tensor, "
                f"but received shape {tuple(input_tensor.shape)}."
            )

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
