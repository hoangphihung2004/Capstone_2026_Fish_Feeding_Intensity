import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch
import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

# Import centralized configuration from config package
from config import AudioFeaturesConfig as AudioFrontendConfig

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import torchaudio.functional as F_audio

def init_bn(bn: nn.BatchNorm2d) -> None:
    """
    Initialize BatchNorm2d weights with default values (bias = 0, weight = 1).
    """
    if bn.bias is not None:
        bn.bias.data.fill_(0.)
    if bn.weight is not None:
        bn.weight.data.fill_(1.)


class LogLinearFilterBank(nn.Module):
    """
    GPU-based Linear Filterbank extractor computing Log-Linear Spectrogram.
    Applies linearly spaced triangular filterbanks across the frequency spectrum.
    """
    def __init__(
        self,
        sr: int = 256000,
        n_fft: int = 4096,
        n_filters: int = 128,
        fmin: float = 200.0,
        fmax: Optional[float] = None,
        amin: float = 1e-10,
        freeze_parameters: bool = True
    ) -> None:
        super(LogLinearFilterBank, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.n_filters = n_filters
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else float(sr // 2)
        self.amin = amin

        # Number of STFT frequency bins
        n_freqs = n_fft // 2 + 1

        # Generate linear filterbank matrix of shape [n_freqs, n_filters]
        fbanks = F_audio.linear_fbanks(
            n_freqs=n_freqs,
            f_min=self.fmin,
            f_max=self.fmax,
            n_filter=self.n_filters,
            sample_rate=self.sr
        )
        self.register_buffer("fbanks", fbanks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): STFT Spectrogram [Batch, 1, Time_Steps, Freq_Bins]
        Returns:
            Tensor: Log-Linear Spectrogram [Batch, 1, Time_Steps, n_filters]
        """
        out = torch.matmul(x, self.fbanks)
        out = torch.log(torch.clamp(out, min=self.amin))
        return out


class AudioFrontend(nn.Module):
    """
    GPU-based raw audio waveform preprocessing frontend supporting Linear and Mel Filterbanks.
    """
    def __init__(self, config: Optional[AudioFrontendConfig] = None) -> None:
        """
        Initialize AudioFrontend module.
        """
        super(AudioFrontend, self).__init__()

        # If config is None, load dynamically from centralized TrainConfig
        if config is None:
            from config import TrainConfig
            self.config = TrainConfig.from_json().audio_features
        else:
            self.config = config

        self.mel_bins = self.config.mel_bins
        self.filter_type = getattr(self.config, "filter_type", "linear").lower()

        # 1. Amplitude Spectrogram Extractor (STFT) on GPU using torchlibrosa
        self.spectrogram_extractor = Spectrogram(
            n_fft=self.config.window_size,
            hop_length=self.config.hop_size,
            win_length=self.config.window_size,
            window='hann',
            center=True,
            pad_mode='reflect',
            freeze_parameters=True
        )

        # 2. Filterbank Extractor on GPU (Linear or Mel)
        fmax = min(self.config.fmax, self.config.sample_rate // 2)
        if self.filter_type == "linear":
            self.filter_extractor = LogLinearFilterBank(
                sr=self.config.sample_rate,
                n_fft=self.config.window_size,
                n_filters=self.config.mel_bins,
                fmin=self.config.fmin,
                fmax=fmax,
                amin=1e-10,
                freeze_parameters=True
            )
        else:
            self.filter_extractor = LogmelFilterBank(
                sr=self.config.sample_rate,
                n_fft=self.config.window_size,
                n_mels=self.config.mel_bins,
                fmin=self.config.fmin,
                fmax=fmax,
                ref=1.0,
                amin=1e-10,
                top_db=None,
                freeze_parameters=True
            )
        # Backward compatibility alias
        self.logmel_extractor = self.filter_extractor

        # 3. SpecAugment Spec Augmentation Extractor on GPU using torchlibrosa
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=self.config.time_drop_width,
            time_stripes_num=self.config.time_stripes_num,
            freq_drop_width=self.config.freq_drop_width,
            freq_stripes_num=self.config.freq_stripes_num
        )

        # 4. BatchNorm normalization layer
        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        init_bn(self.bn0)

        logger.info("==================================================")
        logger.info("Initialized AudioFrontend module on GPU:")
        logger.info(f"  - Filter Type:              {self.filter_type.upper()}")
        logger.info(f"  - Sample Rate:              {self.config.sample_rate} Hz")
        logger.info(f"  - Window Size:              {self.config.window_size}")
        logger.info(f"  - Hop Size:                 {self.config.hop_size}")
        logger.info(f"  - Frequency Bins:           {self.config.mel_bins}")
        logger.info(f"  - Fmin/Fmax:                {self.config.fmin} / {fmax} Hz")
        logger.info(f"  - SpecAugment Time Masking: Width={self.config.time_drop_width}, Stripes={self.config.time_stripes_num}")
        logger.info(f"  - SpecAugment Freq Masking: Width={self.config.freq_drop_width}, Stripes={self.config.freq_stripes_num}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass converting raw 1D waveforms into 2D Linear or Mel-spectrograms.

        Args:
            input_tensor (torch.Tensor): Raw waveform tensor [Batch, Num_Samples].

        Returns:
            torch.Tensor: Augmented Spectrogram tensor [Batch, 1, Time_Steps + 2, Frequency_Bins].
        """
        # Step A: Raw 1D Waveform -> STFT 2D Spectrogram [Batch, 1, Time_Steps, Freq_Bins]
        x = self.spectrogram_extractor(input_tensor)
        
        # Step B: Filterbank filtering (Linear or Mel) -> [Batch, 1, Time_Steps, Frequency_Bins]
        x = self.filter_extractor(x)
        
        # Step C: Pad time-steps dimension by 2 rows of zeros for shape alignment
        m = nn.ZeroPad2d((0, 0, 2, 0))
        x = m(x)

        # Step D: Transpose for BatchNorm2d along mel bins axis
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # Result shape: [Batch, 1, Time_Steps + 2, Mel_Bins]

        # Step E: Apply SpecAugment masking during training
        if self.training:
            x = self.spec_augmenter(x)

        return x
