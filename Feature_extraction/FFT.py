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
    sr: Optional[int] = 48000
    pre_emphasis: float = 0.97
    n_fft: Optional[int] = None
    windowing: str = "hamming"

    def __post_init__(self):
        self.n_fft = self.sr


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
        signal = self._fix_length(signal)
        magnitude = self._fft_signal(signal)
        log_magnitude = np.log(magnitude + 1e-10)

        return log_magnitude.astype(np.float32)

    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            return signal

        coeff = self.config.pre_emphasis
        if coeff is None or coeff == 0:
            return signal

        return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)

    def _fix_length(self, signal: np.ndarray) -> np.ndarray:
        if self.config.n_fft is None:
            return signal

        n_fft = self.config.n_fft
        if signal.size >= n_fft:
            return signal[:n_fft]

        return np.pad(signal, (0, n_fft - signal.size), mode="constant").astype(np.float32)

    def _get_window(self, signal_length: int) -> np.ndarray:
        w = self.config.windowing.lower()

        if w == "hamming":
            return np.hamming(signal_length).astype(np.float32)

        if w == "hann":
            return np.hanning(signal_length).astype(np.float32)

        raise ValueError("windowing must be 'hamming' or 'hann'")

    def _fft_signal(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            n_fft = self.config.n_fft or 1
            signal = np.zeros(n_fft, dtype=np.float32)

        window = self._get_window(signal.size)
        signal = signal * window

        return np.abs(np.fft.rfft(signal))


def main(
    dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
    output_root: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA/features",
    config: Optional[Config] = None,
) -> None:
    config = config or Config()
    extractor = FFT(config)
    config_info = asdict(config)
    config_info["feature_dim"] = None if config.n_fft is None else config.n_fft // 2 + 1

    config_hash = hashlib.md5(json.dumps(config_info, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    config_name = f"sr_{config.sr}_pre_{config.pre_emphasis}_nfft_{config.n_fft}_win_{config.windowing}_{config_hash}"
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
