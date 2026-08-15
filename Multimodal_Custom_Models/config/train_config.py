from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


DEFAULT_IMAGE_CACHE_ROOT = "video_image_cache"
VALID_CACHE_MODES = {"none", "ram", "disk"}


class AudioFeaturesConfig(BaseModel):
    sample_rate: int = 64000
    window_size: int = 2048
    hop_size: int = 1024
    mel_bins: int = 128
    fmin: int = 1
    fmax: int = 32000
    time_drop_width: int = 64
    time_stripes_num: int = 2
    freq_drop_width: int = 8
    freq_stripes_num: int = 2
    freeze_parameters: bool = True


class VideoFeaturesConfig(BaseModel):
    image_size: int = 224


class SplitterConfig(BaseModel):
    dataset_path: str = "/marimo/Fish_Feeding_Intensity_Dataset"
    seed: int = 42
    test_sample_per_class: int = 700
    save_results: bool = False
    include_video: bool = True
    split_strategy: Literal["random_sample", "time_series", "group_random"] = "random_sample"
    evaluation_mode: Literal["holdout", "cross_validation"] = "holdout"
    num_folds: int = 5
    fold_index: Optional[int] = None
    cv_val_ratio: float = 0.2
    output_dir: str = "outputs"


class ModelConfig(BaseModel):
    name: str = "MultimodalStudentModel"
    pretrained: bool = True
    checkpoint_path: str = ""


class TeacherConfig(BaseModel):
    name: str
    checkpoint_path: str = ""
    temperature: float = 4.0
    loss_weight: float = 0.3


class DistillationConfig(BaseModel):
    enabled: bool = False
    mode: Literal["offline", "online"] = "offline"
    hard_label_weight: float = 1.0
    alpha_logit: float = 1.0
    beta_feature: float = 2.0
    lambda_aux: float = 0.3
    audio_teacher: TeacherConfig = Field(default_factory=lambda: TeacherConfig(name="Cnn14MobileV2"))
    video_teacher: TeacherConfig = Field(default_factory=lambda: TeacherConfig(name="EfficientNetB0"))


class DatasetConfig(BaseModel):
    dataset_path: str = "/marimo/Fish_Feeding_Intensity_Dataset"
    cache_audio: bool = True
    cache_video: bool = True
    video_cache_mode: Literal["none", "ram", "disk"] = "ram"
    num_workers: int = -1
    prefetch_factor: Optional[int] = None
    seed: int = 42
    split_strategy: Literal["random_sample", "time_series", "group_random"] = "random_sample"
    test_sample_per_class: int = 700
    num_folds: int = 5
    cv_val_ratio: float = 0.2


class TrainConfig(BaseModel):
    seed: int = 42
    device: str = "cuda"
    num_classes: int = 4
    evaluation_mode: Literal["holdout", "cross_validation"] = "holdout"
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    monitor: Literal["accuracy", "f1_macro", "loss"] = "accuracy"
    early_stopping: bool = True
    patience: int = 70
    delta: float = 0.0
    loss_type: Literal["cross_entropy", "weighted_cross_entropy", "focal_loss"] = "weighted_cross_entropy"
    focal_gamma: float = 2.0
    output_dir: str = "outputs"
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig)
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


def load_train_config(path: str | Path = "config/train_config.json") -> TrainConfig:
    return TrainConfig.from_json(path)
