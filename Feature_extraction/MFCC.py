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
    num_filters: int = 40
    n_mfcc: int = 13
    use_std: bool = False

    @property
    def n_fft(self) -> int:
        return self.frame_length


class MFCC:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        signal, sr = librosa.load(audio_path, sr=self.config.sr)
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
        window = self._get_window()

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.frame_length,
            window=window,
            n_mels=self.config.num_filters,
        )

        mean = np.mean(mfcc, axis=1)
        if not self.config.use_std:
            return mean.astype(np.float32)

        std = np.std(mfcc, axis=1)
        return np.concatenate([mean, std]).astype(np.float32)

    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            return signal

        coeff = self.config.pre_emphasis
        if coeff is None or coeff == 0:
            return signal

        return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)

    def _get_window(self) -> str:
        w = self.config.windowing.lower()

        if w == "hamming":
            return "hamming"

        if w == "hann":
            return "hann"

        raise ValueError("windowing must be 'hamming' or 'hanning'")


def main(
    dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
    output_root: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA/features",
    config: Optional[Config] = None,
) -> None:
    config = config or Config()
    extractor = MFCC(config)
    config_info = asdict(config)
    config_info["n_fft"] = config.n_fft
    config_info["feature_dim"] = config.n_mfcc * (2 if config.use_std else 1)

    config_hash = hashlib.md5(json.dumps(config_info, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    config_name = (
        f"sr_{config.sr}_pre_{config.pre_emphasis}_frame_{config.frame_length}_"
        f"hop_{config.hop_length}_win_{config.windowing}_filters_{config.num_filters}_"
        f"mfcc_{config.n_mfcc}_std_{config.use_std}_{config_hash}"
    )
    output_dir = Path(output_root) / "mfcc_features" / config_name
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
                for row in tqdm(rows, desc=f"Extracting MFCC {split}", unit="file"):
                    feature = extractor.extract(row["audio_path"])
                    writer.writerow({"feature": json.dumps(feature.tolist()), "type": split, **row})


if __name__ == "__main__":
    main()
