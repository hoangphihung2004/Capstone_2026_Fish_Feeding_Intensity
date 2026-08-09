from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from config import TrainConfig
from features.audio_frontend import AudioFrontend
from models.audio import build_audio_backbone
from models.audio.audio_model import AudioModel
from models.video import build_video_backbone
from models.video.video_model import VideoModel

logger = logging.getLogger(__name__)


def _load_checkpoint_strict(model: nn.Module, checkpoint_path: str, teacher_name: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"{teacher_name} checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    logger.info("%s checkpoint loaded with strict state-dict matching: %s", teacher_name, path)


def _set_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad = trainable
    model.train(trainable)


def build_audio_teacher(cfg: TrainConfig) -> Optional[nn.Module]:
    if not cfg.distillation.enabled:
        return None
    teacher_cfg = cfg.distillation.audio_teacher
    frontend = AudioFrontend(cfg.audio_features)
    backbone = build_audio_backbone(teacher_cfg.name, classes_num=cfg.num_classes, pretrained=False)
    teacher = AudioModel(frontend=frontend, backbone=backbone)
    if cfg.distillation.mode == "offline":
        if not teacher_cfg.checkpoint_path:
            raise ValueError("Offline audio distillation requires distillation.audio_teacher.checkpoint_path.")
        _load_checkpoint_strict(teacher, teacher_cfg.checkpoint_path, "Audio teacher")
        _set_trainable(teacher, False)
    else:
        _set_trainable(teacher, True)
    return teacher


def build_video_teacher(cfg: TrainConfig) -> Optional[nn.Module]:
    if not cfg.distillation.enabled:
        return None
    teacher_cfg = cfg.distillation.video_teacher
    backbone = build_video_backbone(teacher_cfg.name, classes_num=cfg.num_classes, pretrained=True)
    teacher = VideoModel(backbone=backbone)
    if cfg.distillation.mode == "offline":
        if not teacher_cfg.checkpoint_path:
            raise ValueError("Offline video distillation requires distillation.video_teacher.checkpoint_path.")
        _load_checkpoint_strict(teacher, teacher_cfg.checkpoint_path, "Video teacher")
        _set_trainable(teacher, False)
    else:
        _set_trainable(teacher, True)
    return teacher
