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
    First+last frame fusion classifier.

    Both frames go through the same (weight-shared) image backbone, their pooled
    embeddings are summed, and a single fully connected layer produces the logits:

        [B, T, 3, H, W] -> backbone -> [B, T, D] -> sum over T -> [B, D] -> FC -> [B, Num_Classes]
    """
    def __init__(self, backbone: nn.Module, classes_num: int = 4) -> None:
        """
        Initialize VideoModel.

        Args:
            backbone (nn.Module): Torch image backbone returning pooled embeddings [N, D]
                and exposing their size as `embed_dim`.
            classes_num (int): Number of output classes. Default: 4.
        """
        super(VideoModel, self).__init__()

        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.fc = nn.Linear(self.embed_dim, classes_num)
        self.model_name = getattr(backbone, "model_name", backbone.__class__.__name__.lower())

        logger.info("==================================================")
        logger.info("Initialized first+last frame fusion VideoModel:")
        logger.info(f"  - Backbone:      {self.model_name} (shared across frames)")
        logger.info(f"  - Embedding Dim: {self.embed_dim}")
        logger.info("  - Fusion:        sum of frame embeddings")
        logger.info(f"  - Classifier:    Linear({self.embed_dim} -> {classes_num})")
        logger.info("==================================================")

    def get_name(self) -> str:
        return self.model_name

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward Pass of the first+last frame fusion model.

        Args:
            input_tensor (torch.Tensor): Frame tensor [Batch, Frames (2), Channels (3), Height, Width].

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing classification logits 'clipwise_output' [Batch, Num_Classes].
        """
        batch_size, num_frames = input_tensor.shape[:2]

        # Run all frames through the backbone in one pass: [B*T, 3, H, W] -> [B*T, D]
        frames = input_tensor.flatten(0, 1)
        embeddings = self.backbone(frames)
        embeddings = embeddings.view(batch_size, num_frames, -1)

        fused = embeddings.sum(dim=1)  # [B, D]
        logits = self.fc(fused)

        return {
            "clipwise_output": logits
        }
