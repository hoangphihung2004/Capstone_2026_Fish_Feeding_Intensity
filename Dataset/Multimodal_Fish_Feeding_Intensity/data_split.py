"""
Create train/validation/test splits for Multimodal Fish Feeding Intensity.

Target distribution:
  Strong: train=1927, val=241, test=241, total=2409
  Weak  : train=1881, val=236, test=236, total=2353
  None  : train=1861, val=233, test=233, total=2327

The script writes:
  splits/train.csv, splits/val.csv, splits/test.csv
  splits/train.jsonl, splits/val.jsonl, splits/test.jsonl
  splits/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "splits"
DEFAULT_SEED = 42

CLASS_TO_LABEL = {
    "none": 0,
    "weak": 1,
    "strong": 2
}

SPLIT_QUOTAS = {
    "strong": {"train": 1927, "val": 241, "test": 241},
    "weak": {"train": 1881, "val": 236, "test": 236},
    "none": {"train": 1861, "val": 233, "test": 233},
}

Sample = Dict[str, str | int]


def natural_key(path: Path) -> Tuple[int | str, ...]:
    """Sort filenames naturally, e.g. audio_2.wav before audio_10.wav."""
    parts: List[int | str] = []
    current = ""
    for char in path.name:
        if char.isdigit():
            if current and not current[-1].isdigit():
                parts.append(current.lower())
                current = ""
            current += char
        else:
            if current and current[-1].isdigit():
                parts.append(int(current))
                current = ""
            current += char
    if current:
        parts.append(int(current) if current.isdigit() else current.lower())
    return tuple(parts)


def canonical_class_name(folder_name: str) -> str:
    return folder_name.lower()


def sample_id_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def expected_paths(dataset_root: Path, class_folder: str, sample_id: str) -> tuple[Path, Path, Path]:
    audio_path = dataset_root / "Audio" / class_folder / f"audio_{sample_id}.wav"
    image_path = dataset_root / "Image" / class_folder / f"image_{sample_id}.jpg"
    wave_path = dataset_root / "Wave" / class_folder / f"wave_{sample_id}.csv"
    return audio_path, image_path, wave_path


def collect_multimodal_samples(dataset_root: Path) -> Dict[str, List[Sample]]:
    dataset_root = dataset_root.resolve()
    audio_root = dataset_root / "Audio"
    image_root = dataset_root / "Image"
    wave_root = dataset_root / "Wave"

    for root in (audio_root, image_root, wave_root):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing required directory: {root}")

    samples_by_class: Dict[str, List[Sample]] = {class_name: [] for class_name in CLASS_TO_LABEL}
    missing_pairs: List[str] = []

    for class_folder in ("None", "Strong", "Weak"):
        class_name = canonical_class_name(class_folder)
        label = CLASS_TO_LABEL[class_name]
        audio_files = sorted((audio_root / class_folder).glob("audio_*.wav"), key=natural_key)

        for audio_path in audio_files:
            sample_id = sample_id_from_path(audio_path)
            expected_audio, image_path, wave_path = expected_paths(dataset_root, class_folder, sample_id)

            if audio_path != expected_audio or not image_path.exists() or not wave_path.exists():
                missing_pairs.append(
                    f"sample_id={sample_id}, class={class_name}, "
                    f"audio={audio_path.exists()}, image={image_path.exists()}, wave={wave_path.exists()}"
                )
                continue

            samples_by_class[class_name].append(
                {
                    "audio_path": str(audio_path),
                    "image_path": str(image_path),
                    "wave_path": str(wave_path),
                    "label": label,
                    "class_name": class_name,
                    "sample_id": sample_id,
                }
            )

    if missing_pairs:
        preview = "\n".join(missing_pairs[:10])
        raise FileNotFoundError(
            f"Found {len(missing_pairs)} incomplete multimodal samples. First issues:\n{preview}"
        )

    return samples_by_class


def validate_quotas(samples_by_class: Dict[str, List[Sample]]) -> None:
    for class_name, quotas in SPLIT_QUOTAS.items():
        expected_total = sum(quotas.values())
        actual_total = len(samples_by_class[class_name])
        if actual_total != expected_total:
            raise ValueError(
                f"Class '{class_name}' has {actual_total} samples, "
                f"but split quotas require {expected_total}."
            )


def split_by_quota(samples_by_class: Dict[str, List[Sample]], seed: int) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}

    for class_name in ("strong", "weak", "none"):
        class_samples = list(samples_by_class[class_name])
        rng.shuffle(class_samples)

        train_count = SPLIT_QUOTAS[class_name]["train"]
        val_count = SPLIT_QUOTAS[class_name]["val"]
        test_count = SPLIT_QUOTAS[class_name]["test"]

        train_end = train_count
        val_end = train_end + val_count
        test_end = val_end + test_count

        splits["train"].extend(class_samples[:train_end])
        splits["val"].extend(class_samples[train_end:val_end])
        splits["test"].extend(class_samples[val_end:test_end])

    for split_samples in splits.values():
        rng.shuffle(split_samples)

    return splits


def write_jsonl(samples: Iterable[Sample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_csv(samples: Iterable[Sample], path: Path) -> None:
    rows = list(samples)
    fieldnames = ["audio_path", "image_path", "wave_path", "label", "class_name", "sample_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    splits: Dict[str, List[Sample]],
    samples_by_class: Dict[str, List[Sample]],
    dataset_root: Path,
    output_dir: Path,
    seed: int,
) -> Dict:
    summary = {
        "label_map": CLASS_TO_LABEL,
        "target_split_quotas": SPLIT_QUOTAS,
        "total_by_class": {name: len(samples) for name, samples in samples_by_class.items()},
        "splits": {},
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "seed": seed,
    }

    for split_name, samples in splits.items():
        counts = {name: 0 for name in CLASS_TO_LABEL}
        for sample in samples:
            counts[str(sample["class_name"])] += 1
        summary["splits"][split_name] = {"total": len(samples), "by_class": counts}

    return summary


def save_splits(splits: Dict[str, List[Sample]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        samples = splits[split_name]
        write_csv(samples, output_dir / f"{split_name}.csv")
        write_jsonl(samples, output_dir / f"{split_name}.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 8:1:1 train/val/test splits for Multimodal Fish Feeding Intensity."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    samples_by_class = collect_multimodal_samples(dataset_root)
    validate_quotas(samples_by_class)
    splits = split_by_quota(samples_by_class, seed=args.seed)

    save_splits(splits, output_dir)

    summary = summarize(
        splits=splits,
        samples_by_class=samples_by_class,
        dataset_root=dataset_root,
        output_dir=output_dir,
        seed=args.seed,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
