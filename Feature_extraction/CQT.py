import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    hop_length: int = 512
    windowing: str = "hamming"
    n_bins: int = 126
    bins_per_octave: int = 12
    use_std: bool = False
    num_workers: int = -1

    @property
    def feature_dim(self) -> int:
        return self.n_bins * (2 if self.use_std else 1)


class CQT:
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
        signal = self._pad_empty_signal(signal)
        window = self._get_window()

        cqt = librosa.cqt(
            y=signal,
            sr=sr,
            hop_length=self.config.hop_length,
            n_bins=self.config.n_bins,
            bins_per_octave=self.config.bins_per_octave,
            window=window,
            pad_mode="constant",
            dtype=np.complex64,
        )

        magnitude = np.abs(cqt)
        log_magnitude = librosa.amplitude_to_db(magnitude, ref=1.0, amin=1e-5, top_db=None)

        mean = np.mean(log_magnitude, axis=1)
        if not self.config.use_std:
            return mean.astype(np.float32)

        std = np.std(log_magnitude, axis=1)
        return np.concatenate([mean, std]).astype(np.float32)

    def _validate_config(self) -> None:
        if self.config.sr is not None and self.config.sr <= 0:
            raise ValueError("sr must be > 0")
        if self.config.hop_length <= 0:
            raise ValueError("hop_length must be > 0")
        if self.config.n_bins <= 0:
            raise ValueError("n_bins must be > 0")
        if self.config.bins_per_octave <= 0:
            raise ValueError("bins_per_octave must be > 0")

    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            return signal

        coeff = self.config.pre_emphasis
        if coeff is None or coeff == 0:
            return signal

        return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)

    def _pad_empty_signal(self, signal: np.ndarray) -> np.ndarray:
        if signal.size > 0:
            return signal.astype(np.float32)

        return np.zeros(self.config.hop_length, dtype=np.float32)

    def _get_window(self) -> str:
        w = self.config.windowing.lower()

        if w == "hamming":
            return "hamming"

        if w in ["hann", "hanning"]:
            return "hann"

        raise ValueError("windowing must be 'hamming', 'hann', or 'hanning'")


def process_single_file(row, dataset_dir, extractor_config):
    audio_path = Path(row["audio_path"])
    if not audio_path.is_absolute():
        audio_path = Path(dataset_dir) / audio_path

    if not audio_path.exists():
        return None, f"Missing: {audio_path}"

    try:
        extractor = CQT(extractor_config)
        feature = extractor.extract(str(audio_path))
        return {"feature": json.dumps(feature.tolist()), **row}, None
    except Exception as e:
        return None, f"Error processing {audio_path}: {e}"


def main(
    dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
    output_root: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA/features",
    config: Optional[Config] = None,
) -> None:
    config = config or Config()
    extractor = CQT(config)
    config_info = asdict(config)
    config_info["feature_dim"] = config.feature_dim

    config_hash = hashlib.md5(json.dumps(config_info, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    config_name = (
        f"sr_{config.sr}_pre_{config.pre_emphasis}_hop_{config.hop_length}_"
        f"win_{config.windowing}_bins_{config.n_bins}_bpo_{config.bins_per_octave}_"
        f"std_{config.use_std}_{config_hash}"
    )
    output_dir = Path(output_root) / "cqt_features" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.txt").write_text(json.dumps(config_info, indent=4), encoding="utf-8")

    split_dir = Path(dataset_dir) / "splits"
    output_csv_path = output_dir / "features.csv"

    max_workers = config.num_workers if config.num_workers > 0 else None

    all_tasks = []
    for split in ("train", "val", "test"):
        input_csv_path = split_dir / f"{split}.csv"
        if not input_csv_path.exists():
            continue

        with input_csv_path.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                task_row = dict(row)
                task_row["type"] = split
                all_tasks.append(task_row)

    print(f"\nTotal tasks loaded: {len(all_tasks)} files")
    print(f"Executing parallel CQT extraction using {max_workers or 'all'} CPU cores...")

    total_failed = 0

    with output_csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file, task, dataset_dir, config): task
                for task in all_tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting CQT features", unit="file"):
                result, error_msg = future.result()
                if error_msg:
                    print(f"\n{error_msg}")
                    total_failed += 1
                elif result:
                    if writer is None:
                        fieldnames = ["feature", "type"] + [k for k in result.keys() if k not in ("feature", "type")]
                        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                        writer.writeheader()
                    writer.writerow(result)


if __name__ == "__main__":
    main()
