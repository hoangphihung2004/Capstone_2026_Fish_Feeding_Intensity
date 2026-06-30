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
    pre_emphasis: float = 0.97
    frame_length: int = 1024
    hop_length: int = 512
    n_fft: int = 1024
    windowing: str = "hamming"
    n_chroma: int = 12
    use_std: bool = False

    @property
    def feature_dim(self) -> int:
        return self.n_chroma * (2 if self.use_std else 1)


class Chroma:
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
            if sr is None:
                if self.config.sr is None:
                    raise ValueError("sr must be provided when audio is an array and config.sr is None")
                sr = self.config.sr

        signal = self._apply_pre_emphasis(signal)
        signal = self._pad_short_signal(signal)
        window = self._get_window()

        chroma = librosa.feature.chroma_stft(
            y=signal,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.frame_length,
            window=window,
            n_chroma=self.config.n_chroma,
        )

        mean = np.mean(chroma, axis=1)
        if not self.config.use_std:
            return mean.astype(np.float32)

        std = np.std(chroma, axis=1)
        return np.concatenate([mean, std]).astype(np.float32)

    def _validate_config(self) -> None:
        if self.config.sr is not None and self.config.sr <= 0:
            raise ValueError("sr must be > 0")
        if self.config.frame_length <= 0:
            raise ValueError("frame_length must be > 0")
        if self.config.hop_length <= 0:
            raise ValueError("hop_length must be > 0")
        if self.config.n_fft <= 0:
            raise ValueError("n_fft must be > 0")
        if self.config.n_chroma <= 0:
            raise ValueError("n_chroma must be > 0")

    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            return signal

        coeff = self.config.pre_emphasis
        if coeff is None or coeff == 0:
            return signal

        return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)

    def _pad_short_signal(self, signal: np.ndarray) -> np.ndarray:
        if signal.size >= self.config.frame_length:
            return signal.astype(np.float32)

        return np.pad(signal, (0, self.config.frame_length - signal.size), mode="constant").astype(np.float32)

    def _get_window(self) -> str:
        w = self.config.windowing.lower()

        if w == "hamming":
            return "hamming"

        if w in ["hann", "hanning"]:
            return "hann"

        raise ValueError("windowing must be 'hamming', 'hann', or 'hanning'")


ChromaSTFT = Chroma


def main(
    dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
    output_root: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA/features",
    config: Optional[Config] = None,
) -> None:
    config = config or Config()
    extractor = Chroma(config)
    config_info = asdict(config)
    config_info["feature_dim"] = config.feature_dim

    config_hash = hashlib.md5(json.dumps(config_info, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    config_name = (
        f"sr_{config.sr}_pre_{config.pre_emphasis}_frame_{config.frame_length}_nfft_{config.n_fft}_"
        f"hop_{config.hop_length}_win_{config.windowing}_chroma_{config.n_chroma}_"
        f"std_{config.use_std}_{config_hash}"
    )
    output_dir = Path(output_root) / "chroma_features" / config_name
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
                for row in tqdm(rows, desc=f"Extracting Chroma {split}", unit="file"):
                    feature = extractor.extract(row["audio_path"])
                    writer.writerow({"feature": json.dumps(feature.tolist()), "type": split, **row})


if __name__ == "__main__":
    main()
