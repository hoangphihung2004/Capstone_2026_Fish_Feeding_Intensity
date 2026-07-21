import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommonVideoFeaturesConfig(BaseModel):
    """
    Common video frame extraction parameters shared across all models.
    """
    image_size: int = Field(default=224, description="Target width and height to resize video frames.")
    frames: int = Field(default=4, description="Number of frames to sample per video clip using Segment-based Sampling.")


class VideoAugmentationConfig(BaseModel):
    """
    Configuration parameters for Video Data Augmentations.
    """
    enable_flip: bool = Field(default=False, description="Enable random horizontal flip augmentation.")
    flip_probability: float = Field(default=0.5, description="Probability of horizontal flip.")
    enable_crop: bool = Field(default=False, description="Enable center crop augmentation.")
    crop_size: List[int] = Field(default_factory=lambda: [196, 196], description="Height and width for center crop.")


class SplitterConfig(BaseModel):
    """
    Configuration parameters for dataset splitting.
    """
    dataset_path: str = Field(
        default='/marimo/Fish_Feeding_Intensity_Dataset',
        description="Absolute path to the raw dataset directory."
    )
    seed: int = Field(
        default=25,
        ge=0,
        description="Random seed for split reproducibility."
    )
    test_sample_per_class: int = Field(
        default=700,
        gt=0,
        description="Number of samples per class designated for test and validation subsets."
    )
    save_results: bool = Field(
        default=True,
        description="Whether to save the splits output results to CSV/JSON files."
    )
    include_video: bool = Field(
        default=True,
        description="Whether to return video paths in RAM alongside audio paths."
    )


class VideoTrainConfig(BaseModel):
    """
    Unified Video training configuration loaded from 3 separate modular JSON files:
    1. video_common.json (General training & dataset parameters)
    2. video_transform.json (Video Data Augmentation parameters)
    3. models/<model_name>.json (Model architecture parameters)
    """
    epochs: int = Field(default=500, description="Maximum training epochs.")
    batch_size: int = Field(default=50, description="Mini-batch size.")
    learning_rate: float = Field(default=1e-3, description="Optimizer learning rate.")
    ckpt_dir: str = Field(default='video_checkpoint/', description="Directory to save checkpoints and CSV logs.")
    monitor: str = Field(default='accuracy', description="Metric to monitor for best model saving ('accuracy' or 'loss').")
    early_stopping: bool = Field(default=True, description="Enable/disable early stopping mechanism.")
    patience: int = Field(default=30, description="Early stopping patience epochs.")
    delta: float = Field(default=0.0, description="Minimum change in monitored metric to qualify as improvement.")
    cache_video: bool = Field(default=True, description="Preload entire raw video dataset directly into RAM at startup.")
    
    # Nested configurations
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="Dataset splitter configurations.")
    video_features: CommonVideoFeaturesConfig = Field(default_factory=CommonVideoFeaturesConfig, description="Common video feature extraction configuration.")
    augmentation: VideoAugmentationConfig = Field(default_factory=VideoAugmentationConfig, description="Video augmentation configuration.")
    model_config: Dict[str, Any] = Field(default_factory=dict, description="Model-specific hyperparameter dictionary.")

    @classmethod
    def load_modular(
        cls,
        common_path: str = 'config/video_common.json',
        aug_path: str = 'config/video_transform.json',
        model_path: str = 'config/models/s3d.json'
    ) -> 'VideoTrainConfig':
        """
        Load video configuration from 3 separate modular JSON files and construct unified VideoTrainConfig.
        """
        logger.info(f"Loading modular Video configs: Common='{common_path}', Transform='{aug_path}', Model='{model_path}'")
        
        with open(common_path, 'r', encoding='utf-8') as f:
            common_data = json.load(f)
            
        with open(aug_path, 'r', encoding='utf-8') as f:
            aug_data = json.load(f)
            
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        common_data['augmentation'] = aug_data
        common_data['model_config'] = model_data

        return cls(**common_data)

    @classmethod
    def from_json(cls, path: str = 'config/video_common.json') -> 'VideoTrainConfig':
        """
        Fallback loader supporting both modular loading and single file loading.
        """
        if os.path.exists('config/video_common.json') and os.path.exists('config/video_transform.json'):
            return cls.load_modular()
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def save_merged_checkpoint_config(self, output_dir: str) -> str:
        """
        Merge all 3 config components (common, transform, model) into EXACTLY 1 unified
        train_config.json file inside the checkpoint output directory after training.
        """
        os.makedirs(output_dir, exist_ok=True)
        merged_file_path = os.path.join(output_dir, 'train_config.json')

        merged_data = {
            "common": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "ckpt_dir": self.ckpt_dir,
                "monitor": self.monitor,
                "early_stopping": self.early_stopping,
                "patience": self.patience,
                "delta": self.delta,
                "cache_video": self.cache_video,
                "dataset_splitter": self.dataset_splitter.model_dump(),
                "video_features": self.video_features.model_dump()
            },
            "transform": self.augmentation.model_dump(),
            "model": self.model_config
        }

        with open(merged_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved merged unified configuration to checkpoint: '{merged_file_path}'")
        return merged_file_path
