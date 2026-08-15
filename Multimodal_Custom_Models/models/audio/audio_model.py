from __future__ import annotations

import logging
import torch
import torch.nn as nn
from .base_backbone import BaseBackbone

logger = logging.getLogger(__name__)


class AudioModel(nn.Module):
    """
    Unified AudioModel Wrapper class.
    Connects AudioFrontend (GPU spectrogram extractor) with a CNN Backbone model.
    Returns dictionary with classification logits 'clipwise_output' and feature representations 'feature'.
    """
    def __init__(self, frontend: nn.Module, backbone: BaseBackbone) -> None:
        super(AudioModel, self).__init__()
        assert isinstance(backbone, BaseBackbone), "Error: Provided backbone model must inherit from BaseBackbone!"
        self.frontend = frontend
        self.backbone = backbone
        self.model_name = getattr(backbone, "model_name", backbone.__class__.__name__.lower())

    def get_name(self) -> str:
        return self.model_name

    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.frontend(input_tensor)
        
        # Check if backbone supports separate feature extraction
        if hasattr(self.backbone, "forward_features"):
            feature_vec, logits = self.backbone.forward_features(features)
        else:
            logits = self.backbone(features)
            feature_vec = logits

        if isinstance(logits, dict):
            logits = logits.get("clipwise_output", logits)
        elif isinstance(logits, (tuple, list)):
            logits = logits[0]

        return {
            "clipwise_output": logits,
            "logits": logits,
            "feature": feature_vec,
        }
