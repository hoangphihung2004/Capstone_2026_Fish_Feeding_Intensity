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
from TKEO import stft_ape

@dataclass
class Config:
    sr: Optional[int] = None

    # STFT config
    frame_length: int = 4096
    hop_length: int = 2048
    windowing: str = "hamming"
    use_std: bool = False

    # TKEO config
    alpha_max: float = 0.99
    tkeo_smooth_beta: float = 0.8

    # Performance config
    num_workers: int = -1

    @property
    def n_fft(self) -> int:
        return self.frame_length

class STFTExtractor:
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

        window = self._get_window()
        stft = stft_ape(
            y=signal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.frame_length,
            window=window,
            alpha_max=self.config.alpha_max,
            tkeo_smooth_beta=self.config.tkeo_smooth_beta,

        )

        magnitude = np.abs(stft)
        log_magnitude = np.log(magnitude + 1e-10)

        mean = np.mean(log_magnitude, axis=1)
        if not self.config.use_std:
            return mean.astype(np.float32)

        std = np.std(log_magnitude, axis=1)
        return np.concatenate([mean, std]).astype(np.float32)

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
        extractor = STFTExtractor(extractor_config)
        feature = extractor.extract(str(audio_path))
        return {
            "feature": json.dumps(feature.tolist()),
            **row
        }, None
    except Exception as e:
        return None, f"Error processing {audio_path}: {e}"

def main(
        dataset_dir: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA",
        output_root: str = "D:/Fish_Feeding_Intensity/Dataset/U_FFIA/features",
        config: Optional[Config] = None,
) -> None:
    config = config or Config()

    config_info = asdict(config)
    config_info["n_fft"] = config.n_fft
    config_info["feature_dim"] = (config.n_fft // 2 + 1) * (2 if config.use_std else 1)
    config_info["method"] = "Deep_Integration_TKEO_Adaptive_Pre_Emphasis"

    config_hash = hashlib.md5(
        json.dumps(config_info, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    config_name = (
        f"sr_{config.sr}_DeepIn_TKEO_"
        f"amax_{config.alpha_max}_"
        f"beta_{config.tkeo_smooth_beta}_"
        f"frame_{config.frame_length}_"
        f"hop_{config.hop_length}_"
        f"win_{config.windowing}_"
        f"std_{config.use_std}_"
        f"{config_hash}"
    )

    output_dir = Path(output_root) / "stft_features" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.txt").write_text(
        json.dumps(config_info, indent=4),
        encoding="utf-8",
    )

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
    print("Accelerating Deep Integration TKEO loop with Numba JIT compiler...")
    print(f"Executing parallel extraction using {max_workers or 'all'} CPU cores...")

    total_failed = 0

    with output_csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file, task, dataset_dir, config): task
                for task in all_tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting DeepIn TKEO features"):
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
    config = Config(
        sr=None,
        frame_length=4096,
        hop_length=2048,
        windowing="hamming",
        use_std=False,
        alpha_max=0.99,
        tkeo_smooth_beta=0.8,
        num_workers=-1
    )
    main(config=config)
