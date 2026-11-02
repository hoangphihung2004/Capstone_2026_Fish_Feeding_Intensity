"""High-resolution TKEO-STFT frontend (ported from the `final` branch), split so the expensive part can be cached.

TkeoStftExtractor: waveform [B, N] @ 256 kHz -> adaptive TKEO pre-emphasis -> windowed rFFT -> log magnitude -> mean over
    time -> [B, 2049]. Deterministic and parameter-free, so the dataloader computes it once per sample and caches the vector.
TkeoStftFrontend: 1D spectral augmentation (train only) + LayerNorm over the 2049 bins. Runs inside the student.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import StftFeaturesConfig

logger = logging.getLogger(__name__)


class Spectral1DAugmentation(nn.Module):
    """Frequency cutout (filled with the sample's minimum, not 0) + Gaussian jitter on a [B, bins] spectral vector."""

    def __init__(self, cutout_width: int = 24, cutout_prob: float = 0.5, noise_std: float = 0.02) -> None:
        super().__init__()
        self.cutout_width = int(cutout_width)
        self.cutout_prob = float(cutout_prob)
        self.noise_std = float(noise_std)

    def forward(self, spec_vector: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec_vector
        out = spec_vector.clone()
        batch, bins = out.shape
        if self.cutout_width > 0 and self.cutout_prob > 0.0 and bins > self.cutout_width:
            apply = torch.rand(batch, 1, device=out.device) < self.cutout_prob
            start = torch.randint(0, bins - self.cutout_width, (batch, 1), device=out.device)
            index = torch.arange(bins, device=out.device).unsqueeze(0)
            mask = (index >= start) & (index < start + self.cutout_width) & apply
            out = torch.where(mask, out.min(dim=-1, keepdim=True)[0], out)
        if self.noise_std > 0.0:
            out = out + torch.randn_like(out) * self.noise_std
        return out


class TkeoStftExtractor(nn.Module):
    """Raw waveform [B, N] (or [N]) -> mean log-magnitude STFT vector [B, n_fft // 2 + 1]."""

    def __init__(self, config: Optional[StftFeaturesConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else StftFeaturesConfig()
        self.n_fft = self.config.window_size
        self.hop_length = self.config.hop_size
        self.alpha_max = float(self.config.alpha_max)
        self.beta = float(self.config.beta)
        self.use_tkeo = bool(self.config.use_tkeo)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        pad = self.n_fft // 2
        padded = F.pad(waveform, (pad, pad), mode="reflect")
        frames = padded.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)  # [B, T, n_fft]

        if self.use_tkeo and self.alpha_max > 0:
            mid, left, right = frames[:, :, 1:-1], frames[:, :, :-2], frames[:, :, 2:]
            psi = mid ** 2 - left * right
            psi = torch.cat([psi[:, :, :1], psi, psi[:, :, -1:]], dim=-1)
            control = psi.abs().mean(dim=-1, keepdim=True) / ((frames ** 2).mean(dim=-1, keepdim=True) + 1e-10)
            alpha_raw = torch.clamp(self.alpha_max * (1.0 - torch.exp(-control)), min=0.1, max=self.alpha_max)
            if self.beta > 0.0 and frames.size(1) > 1:
                alpha = torch.empty_like(alpha_raw)
                previous = torch.zeros(frames.size(0), 1, 1, device=frames.device, dtype=frames.dtype)
                for step in range(alpha_raw.size(1)):
                    current = torch.clamp(
                        self.beta * previous + (1.0 - self.beta) * alpha_raw[:, step:step + 1], min=0.1, max=self.alpha_max
                    )
                    alpha[:, step:step + 1] = current
                    previous = current
            else:
                alpha = alpha_raw
            previous_frame = torch.cat([torch.zeros_like(frames[:, :, :1]), frames[:, :, :-1]], dim=-1)
            frames = frames - alpha * previous_frame

        window = self.window.to(device=frames.device, dtype=frames.dtype)
        spectrum = torch.fft.rfft(frames * window, n=self.n_fft, dim=-1)
        return torch.log(spectrum.abs() + 1e-8).mean(dim=1)


class TkeoStftFrontend(nn.Module):
    """Student-side part: [optional extraction] -> 1D spectral augmentation (train only) -> LayerNorm over the bins."""

    def __init__(self, config: Optional[StftFeaturesConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else StftFeaturesConfig()
        self.extractor = TkeoStftExtractor(self.config)
        self.augmenter = (
            Spectral1DAugmentation(self.config.cutout_width, self.config.cutout_prob, self.config.noise_std)
            if self.config.use_spectral_aug
            else None
        )
        self.norm = nn.LayerNorm(self.config.stft_bins)
        logger.info(
            "Initialized TKEO-STFT frontend: sr=%s, n_fft=%s, hop=%s, bins=%s, tkeo=%s, spectral_aug=%s",
            self.config.sample_rate, self.config.window_size, self.config.hop_size, self.config.stft_bins,
            self.config.use_tkeo, self.config.use_spectral_aug,
        )

    def forward(self, stft_vector: Optional[torch.Tensor] = None, waveform: Optional[torch.Tensor] = None) -> torch.Tensor:
        if stft_vector is None:
            if waveform is None:
                raise ValueError("Provide either a cached stft_vector or a 256 kHz waveform.")
            stft_vector = self.extractor(waveform)
        if self.augmenter is not None:
            stft_vector = self.augmenter(stft_vector)
        return self.norm(stft_vector)
