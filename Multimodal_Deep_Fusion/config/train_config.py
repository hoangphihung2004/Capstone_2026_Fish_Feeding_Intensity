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


class ModalityConfig(BaseModel):
    backbone: str
    pretrained: bool = False
    freeze: bool = False
    checkpoint_path: str = ""


class FusionConfig(BaseModel):
    type: Literal["raw_concat", "linear_concat", "linear_mean", "gated_fusion", "self_attention"] = "raw_concat"
    proj_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.3
    num_heads: int = 4
    activation: Literal["relu", "gelu"] = "relu"
    use_batchnorm: bool = True


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
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    monitor: Literal["accuracy", "f1_macro", "loss"] = "accuracy"
    early_stopping: bool = True
    patience: int = 30
    delta: float = 0.0
    output_dir: str = "outputs"
    audio: ModalityConfig = Field(default_factory=lambda: ModalityConfig(backbone="PANNS_Cnn6"))
    video: ModalityConfig = Field(default_factory=lambda: ModalityConfig(backbone="EfficientNetB0", pretrained=True))
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig)
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


def load_train_config(path: str | Path = "config/train_config.json") -> TrainConfig:
    return TrainConfig.from_json(path)
