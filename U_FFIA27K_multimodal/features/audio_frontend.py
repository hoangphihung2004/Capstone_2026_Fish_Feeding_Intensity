import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import AudioFeaturesConfig

logger = logging.getLogger(__name__)


def init_bn(bn: nn.BatchNorm2d) -> None:
    if bn.bias is not None:
        bn.bias.data.fill_(0.0)
    if bn.weight is not None:
        bn.weight.data.fill_(1.0)


class AudioFrontend(nn.Module):
    """
    GPU raw-waveform frontend copied in behavior from the audio baseline:
    waveform -> spectrogram -> log-mel -> BatchNorm -> SpecAugment.
    """

    def __init__(self, config: Optional[AudioFeaturesConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else AudioFeaturesConfig()
        self.mel_bins = self.config.mel_bins

        self.spectrogram_extractor = Spectrogram(
            n_fft=self.config.window_size,
            hop_length=self.config.hop_size,
            win_length=self.config.window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        fmax = min(self.config.fmax, self.config.sample_rate // 2)
        self.logmel_extractor = LogmelFilterBank(
            sr=self.config.sample_rate,
            n_fft=self.config.window_size,
            n_mels=self.config.mel_bins,
            fmin=self.config.fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=self.config.time_drop_width,
            time_stripes_num=self.config.time_stripes_num,
            freq_drop_width=self.config.freq_drop_width,
            freq_stripes_num=self.config.freq_stripes_num,
        )
        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        init_bn(self.bn0)

        logger.info("Initialized AudioFrontend: sr=%s, window=%s, hop=%s, mel_bins=%s", self.config.sample_rate, self.config.window_size, self.config.hop_size, self.config.mel_bins)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram_extractor(input_tensor)
        x = self.logmel_extractor(x)
        x = nn.ZeroPad2d((0, 0, 2, 0))(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        return x
