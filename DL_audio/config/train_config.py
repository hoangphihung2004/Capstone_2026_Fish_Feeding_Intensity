import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
from pydantic import BaseModel, Field

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioFeaturesConfig(BaseModel):
    """
    Detailed configuration parameters for GPU Mel-spectrogram feature extraction.
    """
    sample_rate: int = Field(default=64000, description="Sample rate of raw audio waveforms.")
    window_size: int = Field(default=2048, description="FFT window size (n_fft).")
    hop_size: int = Field(default=1024, description="Hop length between FFT windows.")
    mel_bins: int = Field(default=128, description="Number of Mel frequency bins (n_mels).")
    fmin: int = Field(default=1, description="Minimum filter frequency.")
    fmax: int = Field(default=128000, description="Maximum filter frequency.")
    time_drop_width: int = Field(default=64, description="Maximum time masking width.")
    time_stripes_num: int = Field(default=2, description="Number of masked time stripes.")
    freq_drop_width: int = Field(default=8, description="Maximum frequency masking width.")
    freq_stripes_num: int = Field(default=2, description="Number of masked frequency stripes.")


class SplitterConfig(BaseModel):
    """
    Configuration parameters for dataset splitting.
    """
    dataset_path: str = Field(
        default='C:/Users/manhm/Desktop/Capstone_2026_Fish_Feeding_Intensity_DL/raw_dataset/U_FFIA',
        description="Absolute path to the raw U_FFIA dataset directory."
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


class TrainConfig(BaseModel):
    """
    Unified configuration class for training hyperparameters and model structure.
    Complies with OOP design via Pydantic.
    """
    epochs: int = Field(default=100, description="Maximum training epochs.")
    batch_size: int = Field(default=40, description="Mini-batch size.")
    learning_rate: float = Field(default=1e-3, description="Optimizer learning rate.")
    ckpt_dir: str = Field(default='checkpoint/', description="Directory to save checkpoints and CSV logs.")
    monitor: str = Field(default='accuracy', description="Metric to monitor for best model saving ('accuracy' or 'loss').")
    early_stopping: bool = Field(default=True, description="Enable/disable early stopping mechanism.")
    patience: int = Field(default=30, description="Early stopping patience epochs.")
    delta: float = Field(default=0.0, description="Minimum change in monitored metric to qualify as improvement.")
    cache_audio: bool = Field(default=True, description="Preload entire audio dataset to RAM cache.")
    
    # Nested configurations
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="Dataset splitter configurations.")
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig, description="Mel-spectrogram extractor configuration.")

    @classmethod
    def from_json(cls, path: str = 'config/train_config.json') -> 'TrainConfig':
        """
        Load unified training configuration from a JSON file.
        """
        logger.info(f"Loading unified training configuration from JSON: '{path}'")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
