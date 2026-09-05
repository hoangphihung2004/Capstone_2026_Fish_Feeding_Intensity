import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_CACHE_ROOT = "/marimo/video_cache"
VALID_CACHE_MODES = ("disk", "ram", "none")


class AudioFeaturesConfig(BaseModel):
    sample_rate: int = Field(default=64000, description="Audio target sample rate.")
    window_size: int = Field(default=2048, description="FFT window size.")
    hop_size: int = Field(default=1024, description="FFT hop size.")
    mel_bins: int = Field(default=128, description="Number of mel bins.")
    fmin: int = Field(default=1, description="Minimum mel frequency.")
    fmax: int = Field(default=32000, description="Maximum mel frequency.")
    time_drop_width: int = Field(default=64, description="SpecAugment time mask width.")
    time_stripes_num: int = Field(default=2, description="SpecAugment time masks.")
    freq_drop_width: int = Field(default=8, description="SpecAugment frequency mask width.")
    freq_stripes_num: int = Field(default=2, description="SpecAugment frequency masks.")


class VideoFeaturesConfig(BaseModel):
    image_size: int = Field(default=224, description="Decoded video frame size.")


class ModelConfig(BaseModel):
    audio_backbone: Literal["LightweightAudioEncoder"] = "LightweightAudioEncoder"
    video_backbone: Literal["EfficientNetB0_S7"] = "EfficientNetB0_S7"
    pretrained_video: bool = Field(default=True, description="Use ImageNet weights for video encoder.")
    pretrained_audio: Literal[False] = False


class SplitterConfig(BaseModel):
    dataset_path: str = Field(default="/marimo/Fish_Feeding_Intensity_Dataset")
    seed: int = Field(default=42, ge=0)
    test_sample_per_class: int = Field(default=700, gt=0)
    save_results: bool = Field(default=False)
    # Multimodal training always requires both paired modalities.
    include_video: Literal[True] = True
    split_strategy: str = Field(default="random_sample")
    evaluation_mode: str = Field(default="holdout")
    num_folds: int = Field(default=5, gt=1)
    fold_index: Optional[int] = Field(default=None)
    cv_val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)


class MultimodalTrainConfig(BaseModel):
    epochs: int = Field(default=250, gt=0)
    batch_size: int = Field(default=128, gt=0)
    dataloader_workers: int = Field(default=-1, ge=-1)
    prefetch_factor: Optional[int] = Field(default=None)
    learning_rate: float = Field(default=1e-4)
    ckpt_dir: str = Field(default="checkpoint/")
    monitor: Literal["loss", "accuracy"] = "loss"
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=40)
    delta: float = Field(default=0.0)
    audio_loss_weight: float = Field(default=1.0, ge=0.0)
    video_loss_weight: float = Field(default=1.0, ge=0.0)
    multimodal_loss_weight: float = Field(default=1.0, ge=0.0)
    cache_audio: bool = Field(default=True)
    cache_video_mode: Literal["disk", "ram", "none"] = Field(default="ram")

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig)
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig)

    @classmethod
    def from_json(cls, path: str = "config/train_config.json") -> "MultimodalTrainConfig":
        logger.info(f"Loading multimodal training configuration from JSON: '{path}'")
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(**json.load(f))


TrainConfig = MultimodalTrainConfig
