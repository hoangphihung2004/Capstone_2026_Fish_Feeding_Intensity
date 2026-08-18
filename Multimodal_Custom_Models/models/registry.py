from __future__ import annotations

import logging
from pathlib import Path

import torch

from config import TrainConfig
from models.multimodal_student import MultimodalStudentModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "MultimodalStudentModel": MultimodalStudentModel,
    "multimodal_student": MultimodalStudentModel,
}


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.model.name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{cfg.model.name}'. Available models: {sorted(MODEL_REGISTRY)}")
    model_cls = MODEL_REGISTRY[cfg.model.name]
    model = model_cls(
        num_classes=cfg.num_classes,
        audio_features_config=cfg.audio_features,
        video_features_config=cfg.video_features,
        pretrained=cfg.model.pretrained,
    )
    if cfg.model.checkpoint_path:
        checkpoint_path = Path(cfg.model.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=True)
        logger.info("==================================================")
        logger.info("WARM-RESTART FINE-TUNING MODE ACTIVATED!")
        logger.info("  - Loaded model weights from checkpoint: %s", checkpoint_path)
        logger.info("==================================================")
    return model

