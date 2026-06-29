import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch
import torchaudio.transforms as T
from tqdm import tqdm


@dataclass
class Config:
    sr: Optional[int] = 96000
    pre_emphasis: Optional[float] = 0.97
    frame_length: int = 1024
    hop_length: int = 512
    n_fft: int = 1024
    windowing: str = "hamming"
    pad: int = 0
    power: float = 2.0
    normalized: Union[bool, str] = False
    wkwargs: Optional[dict] = None
    center: bool = False
    pad_mode: str = "reflect"
    onesided: bool = True
    num_filters: int = 40
    n_lfcc: int = 13
    f_min: float = 0.0
    f_max: Optional[float] = None
    dct_type: int = 2
    norm: str = "ortho"
    log_lf: bool = True
    use_std: bool = False


class LFCC:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._validate_config()

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        signal, sr = librosa.load(audio_path, sr=self.config.sr, mono=True)
        return signal.astype(np.float32), sr

    def extract(self, audio: Union[str, np.ndarray], sr: Optional[int] = None) -> np.ndarray:
        if isinstance(audio, str):
            signal, sr = self.load_audio(audio)
        else:
            signal = np.asarray(audio, dtype=np.float32)
            signal = self._to_mono(signal)
            if sr is None:
                if self.config.sr is None:
                    raise ValueError("sr must be provided when audio is an array and config.sr is None")
                sr = self.config.sr

        signal = self._apply_pre_emphasis(signal)
        waveform = torch.from_numpy(signal).float().unsqueeze(0)

        lfcc_transform = T.LFCC(
            sample_rate=sr,
            n_filter=self.config.num_filters,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_lfcc=self.config.n_lfcc,
            dct_type=self.config.dct_type,
            norm=self.config.norm,
            log_lf=self.config.log_lf,
            speckwargs={
                "n_fft": self.config.n_fft,
                "win_length": self.config.frame_length,
                "hop_length": self.config.hop_length,
                "pad": self.config.pad,
                "window_fn": self._get_window(),
                "power": self.config.power,
                "normalized": self.config.normalized,
                "wkwargs": self.config.wkwargs,
                "center": self.config.center,
                "pad_mode": self.config.pad_mode,
                "onesided": self.config.onesided,
            },
        )

        with torch.no_grad():
            lfcc = lfcc_transform(waveform)

        lfcc = lfcc.squeeze(0).cpu().numpy().astype(np.float32)
        mean = np.mean(lfcc, axis=1)
        if not self.config.use_std:
            return mean.astype(np.float32)

        std = np.std(lfcc, axis=1)
        return np.concatenate([mean, std]).astype(np.float32)

    def _validate_config(self) -> None:
        if self.config.num_filters <= 0:
            raise ValueError("num_filters must be > 0")
        if self.config.n_lfcc <= 0:
            raise ValueError("n_lfcc must be > 0")
        if self.config.n_lfcc > self.config.num_filters:
            raise ValueError("n_lfcc must be <= num_filters")
        if self.config.frame_length <= 0:
            raise ValueError("frame_length must be > 0")
        if self.config.hop_length <= 0:
            raise ValueError("hop_length must be > 0")
        if self.config.n_fft <= 0:
            raise ValueError("n_fft must be > 0")
        if self.config.power is not None and self.config.power <= 0:
            raise ValueError("power must be > 0 or None")
        if self.config.f_max is not None and self.config.sr is not None:
            if self.config.f_max > self.config.sr / 2:
                raise ValueError("f_max must be <= sr / 2")

    def _to_mono(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            return signal.astype(np.float32)
        if signal.ndim != 2:
            raise ValueError("audio array must be 1D or 2D")
        if signal.shape[0] <= signal.shape[1]:
            signal = signal.mean(axis=0)
        else:
            signal = signal.mean(axis=1)
        return signal.astype(np.float32)

    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            return signal.astype(np.float32)

        coeff = self.config.pre_emphasis
        if coeff is None or coeff == 0:
            return signal.astype(np.float32)

        return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)

    def _get_window(self):
        w = self.config.windowing.lower()

        if w == "hamming":
            return torch.hamming_window

        if w in ["hann", "hanning"]:
            return torch.hann_window

        raise ValueError("windowing must be 'hamming', 'hann', or 'hanning'")


def main(
    dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
    output_root: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA/features",
    config: Optional[Config] = None,
) -> None:
    config = config or Config()
    extractor = LFCC(config)
    config_info = asdict(config)
    config_info["feature_dim"] = config.n_lfcc * (2 if config.use_std else 1)

    config_hash = hashlib.md5(json.dumps(config_info, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    config_name = (
        f"sr_{config.sr}_pre_{config.pre_emphasis}_frame_{config.frame_length}_nfft_{config.n_fft}_"
        f"hop_{config.hop_length}_win_{config.windowing}_filters_{config.num_filters}_"
        f"lfcc_{config.n_lfcc}_std_{config.use_std}_{config_hash}"
    )
    output_dir = Path(output_root) / "lfcc_features" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.txt").write_text(json.dumps(config_info, indent=4), encoding="utf-8")

    split_dir = Path(dataset_dir) / "splits"
    output_csv_path = output_dir / "features.csv"

    with output_csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = None

        for split in ("train", "val", "test"):
            input_csv_path = split_dir / f"{split}.csv"

            with input_csv_path.open("r", encoding="utf-8", newline="") as input_file:
                reader = csv.DictReader(input_file)
                fieldnames = ["feature", "type"] + list(reader.fieldnames or [])

                if writer is None:
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()

                rows = list(reader)
                for row in tqdm(rows, desc=f"Extracting LFCC {split}", unit="file"):
                    feature = extractor.extract(row["audio_path"])
                    writer.writerow({"feature": json.dumps(feature.tolist()), "type": split, **row})


if __name__ == "__main__":
    main()
