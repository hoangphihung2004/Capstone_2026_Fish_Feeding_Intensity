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
    sr: Optional[int] = 96000
    n_fft: int = 1024
    pre_emphasis: float = 0.97

    @property
    def feature_dim(self) -> int:
        return self.n_fft // 2 + 1


class FFT:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        signal, sr = librosa.load(audio_path, sr=self.config.sr, mono=True)
        return signal.astype(np.float32), sr

    def extract(self, audio: Union[str, np.ndarray], sr: Optional[int] = None) -> np.ndarray:
        if isinstance(audio, str):
            signal, sr = self.load_audio(audio)
        else:
            signal = np.asarray(audio, dtype=np.float32)
            if sr is None:
                if self.config.sr is None:
                    raise ValueError("sr must be provided when audio is an array and config.sr is None")
                sr = self.config.sr

        signal = self._apply_pre_emphasis(signal)

        fft_result = np.fft.rfft(signal, n=self.config.n_fft)
        magnitude = np.abs(fft_result)
        log_magnitude = np.log(np.maximum(magnitude, 1e-8))

        return log_magnitude.astype(np.float32)


    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        coeff = self.config.pre_emphasis
        if coeff is None or coeff == 0:
            return signal

        return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)


def main(
    dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
    output_root: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA/features",
    config: Optional[Config] = None,
) -> None:
    config = config or Config()
    extractor = FFT(config)
    config_info = asdict(config)
    config_info["feature_dim"] = config.feature_dim

    config_hash = hashlib.md5(json.dumps(config_info, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    config_name = (
        f"sr_{config.sr}_nfft_{config.n_fft}_pre_{config.pre_emphasis}_{config_hash}"
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
