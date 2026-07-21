import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
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


class VideoFeaturesConfig(BaseModel):
    """
    Video frame extraction parameters.
    """
    image_size: int = Field(default=224, description="Target width and height to resize video frames.")
    frames: int = Field(default=4, description="Number of frames to sample per video clip using Segment-based Sampling.")


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
    Single unified Video training configuration class matching the audio side structure.
    Complies with OOP design via Pydantic.
    """
    epochs: int = Field(default=500, description="Maximum training epochs.")
    batch_size: int = Field(default=50, description="Mini-batch size.")
    learning_rate: float = Field(default=1e-3, description="Optimizer learning rate.")
    ckpt_dir: str = Field(default='checkpoint/', description="Directory to save checkpoints and CSV logs.")
    monitor: str = Field(default='accuracy', description="Metric to monitor for best model saving ('accuracy' or 'loss').")
    early_stopping: bool = Field(default=True, description="Enable/disable early stopping mechanism.")
    patience: int = Field(default=30, description="Early stopping patience epochs.")
    delta: float = Field(default=0.0, description="Minimum change in monitored metric to qualify as improvement.")
    cache_video: bool = Field(default=True, description="Preload entire raw video dataset directly into RAM at startup.")
    
    # Nested configurations
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="Dataset splitter configurations.")
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig, description="Video feature extraction configuration.")

    @classmethod
    def from_json(cls, path: str = 'config/video_config.json') -> 'VideoTrainConfig':
        """
        Load single unified video training configuration from a JSON file.
        """
        logger.info(f"Loading single unified video configuration from JSON: '{path}'")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
