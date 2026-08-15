from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VideoModel(nn.Module):
    """
    Unified VideoModel Wrapper class.
    Accepts image tensors and returns dictionary containing 'clipwise_output' (logits) and 'feature' embeddings.
    """
    def __init__(self, backbone: nn.Module) -> None:
        super(VideoModel, self).__init__()
        self.backbone = backbone
        self.model_name = getattr(backbone, "model_name", backbone.__class__.__name__.lower())

    def get_name(self) -> str:
        return self.model_name

    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "features") and hasattr(self.backbone.model, "classifier"):
            feat = self.backbone.model.features(input_tensor)
            feat = self.backbone.model.avgpool(feat)
            feature_vec = torch.flatten(feat, 1)
            logits = self.backbone.model.classifier(feature_vec)
        elif hasattr(self.backbone, "forward_features"):
            feature_vec, logits = self.backbone.forward_features(input_tensor)
        else:
            logits = self.backbone(input_tensor)
            feature_vec = logits

        if isinstance(logits, dict):
            logits = logits.get("clipwise_output", logits)
        elif isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.dim() == 3 and logits.size(2) == 1:
            logits = logits.squeeze(2)

        return {
            "clipwise_output": logits,
            "logits": logits,
            "feature": feature_vec,
        }
