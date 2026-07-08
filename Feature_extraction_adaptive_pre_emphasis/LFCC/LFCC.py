import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from logging import config
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
@dataclass
class Config:
    sr: Optional[int] = 44100
    pre_emphasis: Optional[float] = 0.97

    frame_length: int = 704
    hop_length: int = 352
    windowing: str = "hamming"


    num_filters: int = 40
    n_lfcc: int = 13
    f_min: float = 0.0
    f_max: Optional[float] = None
    power: float = 2.0
    wkwargs: Optional[dict] = None


    norm: str = "ortho"
    log_lf: bool = True


    use_deltas: bool = False
    use_double_deltas: bool = False
    delta_win_length: int = 5
    use_std: bool = False

    num_workers: int = -1

    @property
    def n_fft(self) -> int:
        return self.frame_length

def linear_filter_bank(n_freqs: int,
                       sample_rate: int,
                       n_filter: int = 128,
                       f_min: float = 0.0,
                       f_max: Optional[float] = None) -> np.ndarray:
    if f_max is None:
        f_max = float(sample_rate // 2)

    all_freqs = np.linspace(0, sample_rate // 2, n_freqs, dtype=np.float32)
    f_pts = np.linspace(f_min, f_max, n_filter + 2, dtype=np.float32)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]

    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]

    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    return fb


def create_dct_matrix(n_lfcc: int, n_filter: int, norm: Optional[str] = "ortho") -> np.ndarray:
    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')

    n = np.arange(float(n_filter))
    k = np.arange(float(n_lfcc))[:, np.newaxis]
    dct = np.cos(np.pi / float(n_filter) * (n + 0.5) * k)

    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / np.sqrt(2.0)
        dct *= np.sqrt(2.0 / float(n_filter))

    return dct.T


def amplitude_to_db(x: np.ndarray,
                    multiplier: float,
                    amin: float = 1e-10,
                    ref: float = 1.0,
                    top_db: Optional[float] = None) -> np.ndarray:
    db_multiplier = np.log10(np.maximum(ref, amin))
    x_db = multiplier * np.log10(np.maximum(x, amin))
    x_db -= multiplier * db_multiplier

    if top_db is not None:
        max_val = np.max(x_db, axis=(-2, -1), keepdims=True)
        x_db = np.maximum(x_db, max_val - top_db)

    return x_db


def compute_deltas(specgram: np.ndarray, win_length: int = 5) -> np.ndarray:
    if win_length < 3:
        raise ValueError("win_length must be >= 3")

    n = (win_length - 1) // 2
    denom = 2.0 * np.sum(np.arange(1, n + 1) ** 2)

    padded = np.pad(specgram, ((0, 0), (n, n)), mode='edge')

    deltas = np.zeros_like(specgram)
    n_frames = specgram.shape[1]
    for i in range(1, n + 1):
        deltas += i * (padded[:, n + i: n + i + n_frames] - padded[:, n - i: n - i + n_frames])

    return deltas / denom


class LFCCExtractor:
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

        y_flat = signal.reshape(1, -1)

        window = self._get_window()

        stft_kwargs = {
            "y": y_flat[0],
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "win_length": self.config.frame_length,
            "window": window,
        }


        stft_matrix = librosa.stft(**stft_kwargs)
        stft_matrix = np.expand_dims(stft_matrix, axis=0)

        specgram = np.abs(stft_matrix) ** self.config.power

        n_freqs = int(1 + self.config.n_fft // 2)
        fb = linear_filter_bank(n_freqs, sr, self.config.num_filters, self.config.f_min, self.config.f_max)

        specgram_trans = np.swapaxes(specgram, -1, -2)
        specgram_filtered = np.matmul(specgram_trans, fb)
        specgram = np.swapaxes(specgram_filtered, -1, -2)

        if self.config.log_lf:
            log_offset = 1e-6
            specgram = np.log(specgram + log_offset)
        else:
            multiplier = 10.0 if self.config.power == 2.0 else 20.0
            specgram = amplitude_to_db(specgram, multiplier=multiplier, amin=1e-10, ref=1.0, top_db=80.0)

        dct_matrix = create_dct_matrix(self.config.n_lfcc, self.config.num_filters, self.config.norm)
        specgram_trans_dct = np.swapaxes(specgram, -1, -2)
        lfcc_coefs = np.matmul(specgram_trans_dct, dct_matrix)
        lfcc_coefs = np.swapaxes(lfcc_coefs, -1, -2)

        lfcc_static = np.squeeze(lfcc_coefs, axis=0)

        features_list = [lfcc_static]

        if self.config.use_deltas:
            deltas = compute_deltas(lfcc_static, win_length=self.config.delta_win_length)
            features_list.append(deltas)

            if self.config.use_double_deltas:
                double_deltas = compute_deltas(deltas, win_length=self.config.delta_win_length)
                features_list.append(double_deltas)

        features = np.concatenate(features_list, axis=0)

        mean = np.mean(features, axis=1)
        if not self.config.use_std:
            return mean.astype(np.float32)

        std = np.std(features, axis=1)
        return np.concatenate([mean, std]).astype(np.float32)

    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            return signal.astype(np.float32)

        coeff = self.config.pre_emphasis
        if coeff is None or coeff == 0:
            return signal.astype(np.float32)

        return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)

    def _get_window(self) -> str:
        w = self.config.windowing.lower()

        if w == "hamming":
            return "hamming"

        if w == "hann":
            return "hann"

        raise ValueError("windowing must be 'hamming' or 'hann'")

def process_single_file(row, dataset_dir, config):
    audio_path = Path(row["audio_path"])

    if not audio_path.is_absolute():
        audio_path = Path(dataset_dir) / audio_path

    if not audio_path.exists():
        return None, f"Missing: {audio_path}"

    try:
        extractor = LFCCExtractor(config)
        feature = extractor.extract(str(audio_path))

        return {
            "feature": json.dumps(feature.tolist()),
            **row
        }, None

    except Exception as e:
        return None, f"Error processing {audio_path}: {e}"

def main(
        dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
        output_root: str = "D:/Fish_Feeding_Intensity/lfcc_features",
        config: Optional[Config] = None,
) -> None:
    config = config or Config()

    config_info = asdict(config)
    config_info["n_fft"] = config.n_fft

    num_feature_types = 1
    if config.use_deltas:
        num_feature_types += 1
    if config.use_double_deltas:
        num_feature_types += 1

    config_info["feature_dim"] = (
        config.n_lfcc
        * num_feature_types
        * (2 if config.use_std else 1)
    )

    config_hash = hashlib.md5(
        json.dumps(config_info, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    config_name = (
        f"sr_{config.sr}_pre_{config.pre_emphasis}_frame_{config.frame_length}_"
        f"hop_{config.hop_length}_win_{config.windowing}_filters_{config.num_filters}_"
        f"lfcc_{config.n_lfcc}_std_{config.use_std}_deltas_{config.use_deltas}_"
        f"ddeltas_{config.use_double_deltas}_{config_hash}"
    )

    output_dir = Path(output_root) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu config
    (output_dir / "config.txt").write_text(
        json.dumps(config_info, indent=4),
        encoding="utf-8"
    )

    split_dir = Path(dataset_dir) / "splits"
    output_csv_path = output_dir / "features.csv"

    max_workers = config.num_workers if config.num_workers > 0 else None

    all_tasks = []
    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    # Đọc các file csv
    for split in ("train", "val", "test"):
        input_csv_path = split_dir / f"{split}.csv"

        if not input_csv_path.exists():
            continue

        with input_csv_path.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)

            for row in reader:
                task = dict(row)
                task["type"] = split
                all_tasks.append(task)
                split_counts[split] += 1

    # =======================
    # Print information
    # =======================
    print("\n" + "=" * 80)
    print("LFCC FEATURE EXTRACTION")
    print("=" * 80)

    print(f"Dataset directory : {dataset_dir}")
    print(f"Output directory  : {output_dir}")
    print(f"Output CSV        : {output_csv_path}")
    print(f"Config hash       : {config_hash}")
    print()

    print("Configuration")
    print("-" * 80)
    for key, value in config_info.items():
        print(f"{key:<25}: {value}")

    print("\nDataset Statistics")
    print("-" * 80)
    print(f"{'Train':<20}: {split_counts['train']}")
    print(f"{'Validation':<20}: {split_counts['val']}")
    print(f"{'Test':<20}: {split_counts['test']}")
    print(f"{'Total':<20}: {len(all_tasks)}")
    print(f"{'Workers':<20}: {max_workers if max_workers is not None else 'Auto'}")
    print("=" * 80)
    print()

    total_failed = 0

    with output_csv_path.open("w", encoding="utf-8", newline="") as output_file:

        writer = None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            futures = {
                executor.submit(
                    process_single_file,
                    row,
                    dataset_dir,
                    config,
                ): row
                for row in all_tasks
            }

            for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Extracting LFCC features",
                    unit="file"):

                result, err = future.result()

                if err:
                    print(err)
                    total_failed += 1
                    continue

                if writer is None:
                    fieldnames = ["feature", "type"] + [
                        k for k in result.keys()
                        if k not in ("feature", "type")
                    ]
                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=fieldnames,
                    )
                    writer.writeheader()

                writer.writerow(result)


if __name__ == "__main__":
    config = Config()
    main(config=config)
