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
    sr: Optional[int] = None
    pre_emphasis: float = 0.97
    frame_length: int = 4096
    hop_length: int = 2048
    n_fft: int = 4096
    windowing: str = "hamming"
    num_filters: int = 40
    n_mfcc: int = 13
    feature_mode: str = "mean"
    num_workers: int = -1


class MFCC:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._validate_config()

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

        return self._build_feature(mfcc)

    def _build_feature(self, mfcc: np.ndarray) -> np.ndarray:
        mode = self.config.feature_mode.lower()
        mean = np.mean(mfcc, axis=1)

        if mode == "mean":
            feature = mean
        elif mode == "mean_std":
            feature = np.concatenate([mean, np.std(mfcc, axis=1)])
        elif mode == "mean_delta":
            delta1 = librosa.feature.delta(mfcc, order=1, mode="nearest")
            feature = np.concatenate([mean, np.mean(delta1, axis=1)])
        elif mode == "mean_delta_delta":
            delta1 = librosa.feature.delta(mfcc, order=1, mode="nearest")
            delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
            feature = np.concatenate([mean, np.mean(delta1, axis=1), np.mean(delta2, axis=1)])
        elif mode == "mean_delta2":
            delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
            feature = np.concatenate([mean, np.mean(delta2, axis=1)])
        else:
            raise ValueError(
                "feature_mode must be one of: mean, mean_std, mean_delta, "
                "mean_delta_delta, mean_delta2"
            )

        return feature.astype(np.float32)

    def _validate_config(self) -> None:
        valid_modes = {"mean", "mean_std", "mean_delta", "mean_delta_delta", "mean_delta2"}
        if self.config.feature_mode.lower() not in valid_modes:
            raise ValueError(f"feature_mode must be one of: {', '.join(sorted(valid_modes))}")
        if self.config.frame_length <= 0:
            raise ValueError("frame_length must be > 0")
        if self.config.hop_length <= 0:
            raise ValueError("hop_length must be > 0")
        if self.config.n_fft <= 0:
            raise ValueError("n_fft must be > 0")
        if self.config.num_filters <= 0:
            raise ValueError("num_filters must be > 0")
        if self.config.n_mfcc <= 0:
            raise ValueError("n_mfcc must be > 0")

    @property
    def feature_dim(self) -> int:
        mode = self.config.feature_mode.lower()
        multipliers = {
            "mean": 1,
            "mean_std": 2,
            "mean_delta": 2,
            "mean_delta_delta": 3,
            "mean_delta2": 2,
        }
        return self.config.n_mfcc * multipliers[mode]

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


def process_single_file(row, dataset_dir, extractor_config):
    audio_path = Path(row["audio_path"])
    if not audio_path.is_absolute():
        audio_path = Path(dataset_dir) / audio_path

    if not audio_path.exists():
        return None, f"Missing: {audio_path}"

    try:
        extractor = MFCC(extractor_config)
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
    extractor = MFCC(config)
    config_info = asdict(config)
    config_info["feature_dim"] = extractor.feature_dim

    config_hash = hashlib.md5(json.dumps(config_info, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    config_name = (
        f"sr_{config.sr}_pre_{config.pre_emphasis}_frame_{config.frame_length}_nfft_{config.n_fft}_"
        f"hop_{config.hop_length}_win_{config.windowing}_filters_{config.num_filters}_"
        f"mfcc_{config.n_mfcc}_mode_{config.feature_mode}_{config_hash}"
    )
    output_dir = Path(output_root) / "mfcc_pre_features" / config_name
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
    print(f"Executing parallel MFCC extraction using {max_workers or 'all'} CPU cores...")

    total_failed = 0

    with output_csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file, task, dataset_dir, config): task
                for task in all_tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting MFCC features", unit="file"):
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
