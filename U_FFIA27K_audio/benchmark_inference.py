"""Measure per-file end-to-end audio inference latency.

The measured region is deliberately limited to one real WAV file at a time:
disk read -> audio preprocessing -> CPU-to-device transfer -> frontend -> model.
Model construction and optional checkpoint loading happen before the warm-up
phase and are not included in the reported inference latency.
"""

import copy
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TrainConfig
from dataset import FishDataSplitter
from dataset.dataloader_melspectrogram import FishVoiceDataLoader
from features import AudioFrontend
from main import build_backbone, validate_backbone_config
from models import AudioModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "inference_benchmark_config.json"


def _resolve_path(value: str, *, field_name: str) -> Path:
    if not str(value).strip():
        raise ValueError(f"'{field_name}' must be set in config/inference_benchmark_config.json.")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    cleaned_state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=True)


def build_test_audio_paths(train_config: TrainConfig, required_samples: int, seed: int) -> list[str]:
    """Build the same test split as the audio pipeline without writing split files."""
    splitter_config = copy.deepcopy(train_config.dataset_splitter)
    splitter_config.save_results = False
    splitter_config.include_video = False
    if splitter_config.evaluation_mode == "cross_validation" and splitter_config.fold_index is None:
        splitter_config.fold_index = 0
        logger.info("No CV fold was selected in train_config; using deterministic fold 00 for benchmarking.")

    _, test_entries, _ = FishDataSplitter(config=splitter_config).split_data()

    audio_paths: list[str] = []
    seen_paths: set[str] = set()
    for entry in test_entries:
        normalized_path = str(Path(entry[0]).resolve())
        if normalized_path not in seen_paths:
            seen_paths.add(normalized_path)
            audio_paths.append(normalized_path)

    if len(audio_paths) < required_samples:
        raise ValueError(
            f"The automatically generated test split contains only {len(audio_paths)} unique WAV files, but "
            f"warmup_samples + timed_samples requires {required_samples}."
        )

    missing_paths = [path for path in audio_paths if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Generated test split contains {len(missing_paths)} missing WAV files. First missing file: {missing_paths[0]}"
        )

    random.Random(seed).shuffle(audio_paths)
    return audio_paths[:required_samples]


def run_one_end_to_end(model: AudioModel, audio_path: str, sample_rate: int, device: torch.device) -> None:
    """Run one complete file-to-logits inference without application-level audio caching."""
    waveform = FishVoiceDataLoader.load_audio(audio_path, sr=sample_rate)
    waveform = waveform.unsqueeze(0).to(device)
    _ = model(waveform)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    model: AudioModel,
    warmup_paths: list[str],
    timed_paths: list[str],
    sample_rate: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logger.info("Running warm-up on %d real WAV samples (excluded from timing).", len(warmup_paths))
        for audio_path in warmup_paths:
            run_one_end_to_end(model, audio_path, sample_rate, device)
        synchronize_if_cuda(device)

        durations_ms: list[float] = []
        logger.info("Measuring end-to-end inference for %d real WAV samples, one sample per batch.", len(timed_paths))
        for audio_path in timed_paths:
            synchronize_if_cuda(device)
            start = time.perf_counter()
            run_one_end_to_end(model, audio_path, sample_rate, device)
            synchronize_if_cuda(device)
            durations_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = sum(durations_ms) / len(durations_ms)
    variance = sum((duration - mean_ms) ** 2 for duration in durations_ms) / (len(durations_ms) - 1)
    return mean_ms, variance**0.5


def main() -> None:
    benchmark_config = load_json(DEFAULT_CONFIG_PATH)
    warmup_samples = int(benchmark_config.get("warmup_samples", 100))
    timed_samples = int(benchmark_config.get("timed_samples", 1000))
    if warmup_samples <= 0 or timed_samples <= 1:
        raise ValueError("warmup_samples must be positive and timed_samples must be greater than one.")

    train_config_path = _resolve_path(benchmark_config.get("train_config_path", ""), field_name="train_config_path")
    checkpoint_value = str(benchmark_config.get("checkpoint_path", "")).strip()
    checkpoint_path = _resolve_path(checkpoint_value, field_name="checkpoint_path") if checkpoint_value else None
    output_path = _resolve_path(benchmark_config.get("output_path", ""), field_name="output_path")

    requested_device = str(benchmark_config.get("device", "cuda")).lower()
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    train_config = TrainConfig.from_json(str(train_config_path))
    validate_backbone_config(train_config)

    frontend = AudioFrontend(config=train_config.audio_features)
    model = AudioModel(frontend=frontend, backbone=build_backbone(train_config)).to(device)
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, device)
    else:
        logger.info("No checkpoint configured; benchmarking the initialized architecture with random weights.")

    paths = build_test_audio_paths(
        train_config=train_config,
        required_samples=warmup_samples + timed_samples,
        seed=int(benchmark_config.get("seed", 42)),
    )
    mean_ms, std_ms = benchmark(
        model=model,
        warmup_paths=paths[:warmup_samples],
        timed_paths=paths[warmup_samples:],
        sample_rate=train_config.audio_features.sample_rate,
        device=device,
    )

    result = {
        "samples": timed_samples,
        "end_to_end_inference_ms_mean": mean_ms,
        "end_to_end_inference_ms_std": std_ms,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"End-to-end inference time: {mean_ms:.3f} +/- {std_ms:.3f} ms/sample (n={timed_samples})")


if __name__ == "__main__":
    main()
