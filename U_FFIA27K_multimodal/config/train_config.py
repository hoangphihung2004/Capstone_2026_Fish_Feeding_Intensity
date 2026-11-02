import json
import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

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


class StftFeaturesConfig(BaseModel):
    """High-resolution TKEO-STFT frontend (256 kHz, 2049 linear bins) used by the stft_tkeo_mlp student."""

    sample_rate: int = Field(default=256000, description="Sample rate the student's STFT audio is loaded at.")
    window_size: int = Field(default=4096, description="STFT window size in samples.")
    hop_size: int = Field(default=2048, description="STFT hop size in samples.")
    use_tkeo: bool = Field(default=True, description="Teager-Kaiser Energy Operator adaptive pre-emphasis.")
    alpha_max: float = Field(default=0.99, ge=0.0, le=1.0, description="Max pre-emphasis coefficient.")
    beta: float = Field(default=0.8, ge=0.0, le=1.0, description="Temporal smoothing factor of the pre-emphasis coefficient.")
    use_spectral_aug: bool = Field(default=True, description="1D spectral augmentation (frequency cutout + Gaussian jitter), train only.")
    cutout_width: int = Field(default=24, ge=0, description="Width of the contiguous frequency cutout in bins.")
    cutout_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability of applying the cutout per sample.")
    noise_std: float = Field(default=0.02, ge=0.0, description="Std of the Gaussian spectral jitter.")

    @property
    def stft_bins(self) -> int:
        return self.window_size // 2 + 1


class StudentMLPConfig(BaseModel):
    hidden_dims: Tuple[int, ...] = Field(default=(768, 512), description="Hidden widths; each is Linear + LayerNorm + GELU + Dropout.")
    embed_dim: int = Field(default=384, gt=0, description="Width of the last Linear + LayerNorm before the classifier.")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class StudentConfig(BaseModel):
    type: Literal["conv", "stft_tkeo_mlp"] = Field(
        default="conv",
        description=(
            "conv: LightweightAudioEncoder on log-mel (supports feature KD). "
            "stft_tkeo_mlp: MLP on the 2049-D TKEO-STFT vector of 256 kHz audio (distilled from logits only)."
        ),
    )
    mlp: StudentMLPConfig = Field(default_factory=StudentMLPConfig)


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


class ReduceLROnPlateauConfig(BaseModel):
    enabled: bool = Field(default=True)
    factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    patience: int = Field(default=10, ge=0)
    threshold: float = Field(default=0.001, ge=0.0)
    cooldown: int = Field(default=2, ge=0)
    min_lr: float = Field(default=1e-6, gt=0.0)


class KDConfig(BaseModel):
    """Offline Hinton knowledge distillation from the frozen multimodal teacher."""

    teacher_checkpoint: str = Field(default="", description="Path to the teacher multimodal_best.pt of the selected fold.")
    temperature: float = Field(default=4.0, gt=0.0, description="Softmax temperature applied to teacher and student logits.")
    ce_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the hard-label CE term. 0 turns CE off so only the KD terms train the student.",
    )
    alpha: float = Field(default=0.5, ge=0.0, description="Weight of the logit KD term: loss = ce_weight * CE + alpha * T^2 * KL + beta * L_feat.")
    beta: float = Field(default=0.1, ge=0.0, description="Weight of the feature KD term: loss = ce_weight * CE + alpha * T^2 * KL + beta * L_feat.")
    feature_target: Literal["none", "audio", "video", "both"] = Field(
        default="audio",
        description="Teacher branch(es) whose pooled stage features the student's pooled stage features are regressed onto with MSE.",
    )
    feature_source: Literal["fused", "delta", "update"] = Field(
        default="fused",
        description=(
            "What the student's pooled stage features are regressed onto. fused: the teacher's pooled stage feature after fusion "
            "(stages 1-5). delta: tanh(scale) * update(context), the vector fusion adds to the modality (stages 2-5). "
            "update: update(context) without the scale gate (stages 2-5)."
        ),
    )
    feature_stages: List[int] = Field(
        default_factory=lambda: [5],
        description="Stages (1-5, channels 16/24/40/112/320) used in L_feat. [5] is the final pooled 320-D feature.",
    )
    stage_weights: Optional[List[float]] = Field(
        default=None,
        description="Weight per entry of feature_stages (same order); null means equal weights. L_feat is the weighted mean of the stage MSEs.",
    )
    verify_teacher_split: bool = Field(
        default=True,
        description="Require the teacher's saved splitter_config.json/splits to match the current split when present.",
    )

    @model_validator(mode="after")
    def _check_feature_stages(self) -> "KDConfig":
        if self.ce_weight == 0 and self.alpha == 0:
            raise ValueError("kd.ce_weight and kd.alpha cannot both be 0: nothing would train the classifier head.")
        if self.feature_target != "none" and not self.feature_stages:
            raise ValueError("kd.feature_stages must not be empty unless kd.feature_target is 'none'.")
        if any(stage not in (1, 2, 3, 4, 5) for stage in self.feature_stages) or len(set(self.feature_stages)) != len(self.feature_stages):
            raise ValueError(f"kd.feature_stages must be unique values in 1..5, got {self.feature_stages}.")
        if self.feature_source != "fused" and 1 in self.feature_stages:
            raise ValueError(f"kd.feature_source='{self.feature_source}' exists only for stages 2-5 (stage 1 has no fusion).")
        if self.stage_weights is not None:
            if len(self.stage_weights) != len(self.feature_stages):
                raise ValueError("kd.stage_weights must have the same length as kd.feature_stages.")
            if any(weight < 0 for weight in self.stage_weights) or (self.feature_stages and sum(self.stage_weights) <= 0):
                raise ValueError("kd.stage_weights must be non-negative with a positive sum.")
        return self


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
    cache_audio: bool = Field(default=True)
    cache_video_mode: Literal["disk", "ram", "none"] = Field(default="ram")

    lr_scheduler: ReduceLROnPlateauConfig = Field(default_factory=ReduceLROnPlateauConfig)
    kd: KDConfig = Field(default_factory=KDConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig)
    stft_features: StftFeaturesConfig = Field(default_factory=StftFeaturesConfig)
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig)

    @model_validator(mode="after")
    def _check_student(self) -> "MultimodalTrainConfig":
        if self.student.type != "conv" and self.kd.feature_target != "none":
            raise ValueError(
                f"student.type='{self.student.type}' has no stage feature maps: set kd.feature_target to 'none' "
                "(MLP students are distilled from logits only)."
            )
        return self

    @classmethod
    def from_json(cls, path: str = "config/train_config.json") -> "MultimodalTrainConfig":
        logger.info(f"Loading multimodal training configuration from JSON: '{path}'")
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(**json.load(f))


TrainConfig = MultimodalTrainConfig
