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
import torch.nn.functional as F
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


def init_bn(bn: nn.BatchNorm2d) -> None:
    """
    Initialize BatchNorm2d weights with default values (bias = 0, weight = 1).
    """
    if bn.bias is not None:
        bn.bias.data.fill_(0.)
    if bn.weight is not None:
        bn.weight.data.fill_(1.)


# Lower clipping bound of the adaptive coefficient (Algorithm 1, line 19).
TKEO_ALPHA_MIN = 0.1


def tkeo_adaptive_alpha(
    frames: torch.Tensor,
    alpha_max: float = 0.99,
    beta: float = 0.8,
    clamp_raw: bool = False,
) -> torch.Tensor:
    """
    Per-frame pre-emphasis coefficient from the Teager-Kaiser Energy Operator
    (Algorithm 1, "TKEO-Guided Adaptive Pre-Emphasis", MIWAI 2026).

    Args:
        frames: framed waveform [Batch, Frames, N].
        alpha_max: maximum coefficient.
        beta: temporal smoothing factor across frames.
        clamp_raw: also clip alpha_raw to [0.1, alpha_max] before smoothing
            (multimodal implementation); False follows the paper.

    Returns:
        alpha: [Batch, Frames], each value in [0.1, alpha_max].
    """
    # 1. TKEO: psi[n] = x[n]^2 - x[n-1] x[n+1], edges replicated (lines 6-10)
    psi_inner = frames[..., 1:-1] ** 2 - frames[..., :-2] * frames[..., 2:]
    psi = torch.cat([psi_inner[..., :1], psi_inner, psi_inner[..., -1:]], dim=-1)

    # 2. Control signal C = mean|psi| / mean(x^2), C = 0 for silent frames (lines 12-14)
    e_tkeo = psi.abs().mean(dim=-1)
    e_frame = (frames ** 2).mean(dim=-1)
    ctrl = torch.where(
        e_frame < 1e-10,
        torch.zeros_like(e_frame),
        e_tkeo / e_frame.clamp_min(1e-10),
    )

    # 3. Nonlinear mapping (line 16)
    alpha_raw = alpha_max * (1.0 - torch.exp(-ctrl))
    if clamp_raw:
        alpha_raw = alpha_raw.clamp(TKEO_ALPHA_MIN, alpha_max)

    # 4. Temporal smoothing + clipping, alpha_prev starts at 0 (lines 2, 18-20)
    alpha_prev = torch.zeros_like(alpha_raw[:, 0])
    alphas = []
    for t in range(alpha_raw.size(1)):
        alpha_t = (beta * alpha_prev + (1.0 - beta) * alpha_raw[:, t]).clamp(TKEO_ALPHA_MIN, alpha_max)
        alphas.append(alpha_t)
        alpha_prev = alpha_t
    return torch.stack(alphas, dim=1)


def frame_pre_emphasis(frames: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    y[0] = x[0], y[n] = x[n] - alpha * x[n-1] inside every frame (Algorithm 1, lines 22-25).

    Args:
        frames: [Batch, Frames, N].
        alpha: scalar, or a per-frame tensor broadcastable to [Batch, Frames].
    """
    if isinstance(alpha, torch.Tensor) and alpha.dim() == 2:
        alpha = alpha.unsqueeze(-1)
    previous = torch.cat([torch.zeros_like(frames[..., :1]), frames[..., :-1]], dim=-1)
    return frames - alpha * previous


class AudioFrontend(nn.Module):
    """
    GPU-based raw audio waveform preprocessing frontend utilizing torchlibrosa.
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

        # 2. Logmel Filterbank Extractor on GPU using torchlibrosa
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
            freeze_parameters=True
        )

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

        # 5. Optional frame-level pre-emphasis. torchlibrosa computes framing and
        #    the DFT in one Conv1d, so a pre-emphasized STFT is computed explicitly
        #    (reflect pad + unfold + periodic Hann + rfft, power=2) and then fed to
        #    the same log-mel filterbank. spectrogram_extractor is still built so the
        #    state_dict is identical for every mode, and the Hann buffer is
        #    non-persistent for the same reason.
        self.pre_emphasis = getattr(self.config, "pre_emphasis", "none")
        self.fixed_alpha = float(getattr(self.config, "fixed_alpha", 0.97))
        self.alpha_max = float(getattr(self.config, "alpha_max", 0.99))
        self.beta = float(getattr(self.config, "beta", 0.8))
        self.tkeo_clamp_raw = bool(getattr(self.config, "tkeo_clamp_raw", False))
        if self.pre_emphasis not in ("none", "fixed", "tkeo"):
            raise ValueError(f"Unknown pre_emphasis '{self.pre_emphasis}'. Use 'none', 'fixed' or 'tkeo'.")
        if self.pre_emphasis != "none" and self.config.window_size != self.spectrogram_extractor.stft.n_fft:
            raise ValueError("Pre-emphasis STFT assumes win_length == n_fft.")
        self.register_buffer(
            "stft_window",
            torch.hann_window(self.config.window_size, periodic=True),
            persistent=False,
        )

        logger.info("==================================================")
        logger.info("Initialized AudioFrontend module on GPU:")
        logger.info(f"  - Sample Rate:              {self.config.sample_rate} Hz")
        logger.info(f"  - Window Size:              {self.config.window_size}")
        logger.info(f"  - Hop Size:                 {self.config.hop_size}")
        logger.info(f"  - Mel Bins:                 {self.config.mel_bins}")
        logger.info(f"  - Fmin/Fmax:                {self.config.fmin} / {fmax} Hz")
        logger.info(f"  - SpecAugment Time Masking: Width={self.config.time_drop_width}, Stripes={self.config.time_stripes_num}")
        logger.info(f"  - SpecAugment Freq Masking: Width={self.config.freq_drop_width}, Stripes={self.config.freq_stripes_num}")
        if self.pre_emphasis == "tkeo":
            logger.info(
                f"  - Pre-emphasis:             TKEO adaptive (alpha_max={self.alpha_max}, beta={self.beta}, "
                f"clamp_raw={self.tkeo_clamp_raw})"
            )
        elif self.pre_emphasis == "fixed":
            logger.info(f"  - Pre-emphasis:             fixed (alpha={self.fixed_alpha})")
        else:
            logger.info("  - Pre-emphasis:             none")
        logger.info("==================================================")

    def _pre_emphasized_power_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Power spectrogram with frame-level pre-emphasis, matching the output of
        torchlibrosa.Spectrogram (center=True, reflect pad, periodic Hann, power=2).

        Args:
            waveform: [Batch, Num_Samples].

        Returns:
            torch.Tensor: [Batch, 1, Time_Steps, n_fft // 2 + 1].
        """
        n_fft = self.config.window_size
        pad = n_fft // 2
        padded = F.pad(waveform.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        frames = padded.unfold(dimension=-1, size=n_fft, step=self.config.hop_size)  # [B, T, n_fft]

        if self.pre_emphasis == "tkeo":
            alpha = tkeo_adaptive_alpha(frames, self.alpha_max, self.beta, self.tkeo_clamp_raw)
        else:
            alpha = self.fixed_alpha
        frames = frame_pre_emphasis(frames, alpha)

        spectrum = torch.fft.rfft(frames * self.stft_window.to(frames.dtype), n=n_fft, dim=-1)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        return power.unsqueeze(1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass converting raw 1D waveforms into 2D Mel-spectrograms.

        Args:
            input_tensor (torch.Tensor): Raw waveform tensor [Batch, Num_Samples].

        Returns:
            torch.Tensor: Augmented Mel-spectrogram tensor [Batch, 1, Time_Steps + 2, Mel_Bins].
        """
        # Step A: Raw 1D Waveform -> STFT 2D Spectrogram [Batch, 1, Time_Steps, Freq_Bins]
        if self.pre_emphasis == "none":
            x = self.spectrogram_extractor(input_tensor)
        else:
            x = self._pre_emphasized_power_spectrogram(input_tensor)
        
        # Step B: Logmel filtering -> [Batch, 1, Time_Steps, Mel_Bins]
        x = self.logmel_extractor(x)
        
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
