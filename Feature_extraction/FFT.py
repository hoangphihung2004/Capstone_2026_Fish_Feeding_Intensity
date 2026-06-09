import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from tqdm import tqdm


@dataclass
class Config:
    sr: Optional[int] = None
    pre_emphasis: float = 0.97
    frame_length: int = 1024
    hop_length: int = 512
    windowing: str = "hamming"
    use_std: bool = False

    @property
    def n_fft(self) -> int:
        return self.frame_length


class FFT:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        signal, sr = librosa.load(audio_path, sr=self.config.sr)
        return signal.astype(np.float32), sr

    def extract(self, audio: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(audio, str):
            signal, _ = self.load_audio(audio)
        else:
            signal = np.asarray(audio, dtype=np.float32)

        signal = self._apply_pre_emphasis(signal)
        frames = self._framing(signal)
        frames = self._windowing(frames)
        magnitude = self._fft_frames(frames)
        log_magnitude = np.log(magnitude + 1e-10)

        mean = np.mean(log_magnitude, axis=0)
        if not self.config.use_std:
            return mean.astype(np.float32)

        std = np.std(log_magnitude, axis=0)
        return np.concatenate([mean, std]).astype(np.float32)

    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            return signal

        coeff = self.config.pre_emphasis
        if coeff is None or coeff == 0:
            return signal

        return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)

    def _framing(self, signal: np.ndarray) -> np.ndarray:
        frame_length = self.config.frame_length
        hop_length = self.config.hop_length
        signal_length = len(signal)

        num_frames = max(1, int(np.ceil((signal_length - frame_length) / hop_length)) + 1)
        pad_length = (num_frames - 1) * hop_length + frame_length
        pad_signal = np.pad(signal, (0, max(0, pad_length - signal_length)), mode="constant")
        indices = np.arange(frame_length)[None, :] + np.arange(num_frames)[:, None] * hop_length
        return pad_signal[indices]

    def _windowing(self, frames: np.ndarray) -> np.ndarray:
        w = self.config.windowing.lower()

        if w == "hamming":
            window = np.hamming(frames.shape[1])
        elif w == "hann":
            window = np.hanning(frames.shape[1])
        else:
            raise ValueError("windowing must be 'hamming' or 'hann'")

        return frames * window

    def _fft_frames(self, frames: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(frames, n=self.config.n_fft, axis=1))


def main(
    dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
    output_root: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA/features",
    config: Optional[Config] = None,
) -> None:
    config = config or Config()
    extractor = FFT(config)
    config_info = asdict(config)
    config_info["n_fft"] = config.n_fft
    config_info["feature_dim"] = (config.n_fft // 2 + 1) * (2 if config.use_std else 1)

    config_hash = hashlib.md5(json.dumps(config_info, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    config_name = (
        f"sr_{config.sr}_pre_{config.pre_emphasis}_frame_{config.frame_length}_"
        f"hop_{config.hop_length}_win_{config.windowing}_std_{config.use_std}_{config_hash}"
    )
    output_dir = Path(output_root) / "fft_features" / config_name
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
                for row in tqdm(rows, desc=f"Extracting FFT {split}", unit="file"):
                    feature = extractor.extract(row["audio_path"])
                    writer.writerow({"feature": json.dumps(feature.tolist()), "type": split, **row})


if __name__ == "__main__":
    main()
