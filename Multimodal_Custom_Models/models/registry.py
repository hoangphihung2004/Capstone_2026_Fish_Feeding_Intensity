from __future__ import annotations

from pathlib import Path

import torch

from config import TrainConfig
from models.av_mobile_attn_film_net import AVMobileAttnFiLMNet
from models.av_mobile_difm_efficientnet import AVMobileDIFMEfficientNet


MODEL_REGISTRY = {
    "AVMobileDIFMEfficientNet": AVMobileDIFMEfficientNet,
    "AVMobileAttnFiLMNet": AVMobileAttnFiLMNet,
}


def build_model(cfg: TrainConfig):
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
    return model
