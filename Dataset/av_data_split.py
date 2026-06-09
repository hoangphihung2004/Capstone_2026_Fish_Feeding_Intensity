"""
Split U-FFIA multimodal dataset with same rule used in original U-FFIA source.

Original logic:
- collect samples per class: none, strong, medium, weak
- shuffle each class list with np.random.RandomState(seed)
- first N per class -> test
- next N per class -> val
- rest -> train
- label map: none=0, strong=1, medium=2, weak=3

This script keeps audio/video paired by relative path + shared id.
Example pair:
  video/2022_6_13/AM_100/none/13_video_1.mp4
  audio/2022_6_13/AM_100/none/13_audio_1.wav
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


CLASS_TO_LABEL = {
    "none": 0,
    "strong": 1,
    "medium": 2,
    "weak": 3,
}

LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

DEFAULT_DATASET_ROOT = Path(r"D:\Fish_Feeding_Intensity\Dataset\U_FFIA")
DEFAULT_OUTPUT_DIR = Path(r"D:\Fish_Feeding_Intensity\Dataset\U_FFIA\splits")
DEFAULT_SEED = 25
DEFAULT_TEST_SAMPLE_PER_CLASS = 700


Sample = Dict[str, str | int]


def natural_key(path: Path) -> Tuple:
    """Sort paths stably so seeded shuffle matches across machines."""
    parts: List[int | str] = []
    for token in str(path).replace("\\", "/").split("/"):
        current = ""
        for char in token:
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


def audio_path_from_video(video_path: Path, dataset_root: Path) -> Path:
    """Map video path to matching audio path using U-FFIA naming convention."""
    rel = video_path.relative_to(dataset_root / "video")
    audio_name = video_path.name.replace("_video_", "_audio_").replace(".mp4", ".wav")
    return dataset_root / "audio" / rel.parent / audio_name


def collect_multimodal_samples(dataset_root: Path) -> Dict[str, List[Sample]]:
    """Collect paired audio/video samples grouped by class."""
    dataset_root = dataset_root.resolve()
    video_root = dataset_root / "video"
    audio_root = dataset_root / "audio"

    if not video_root.is_dir():
        raise FileNotFoundError(f"Missing video directory: {video_root}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing audio directory: {audio_root}")

    samples_by_class: Dict[str, List[Sample]] = {name: [] for name in CLASS_TO_LABEL}
    missing_audio: List[Tuple[Path, Path]] = []

    for class_name, label in CLASS_TO_LABEL.items():
        video_files = sorted(video_root.glob(f"*/*/{class_name}/*.mp4"), key=natural_key)
        for video_path in video_files:
            audio_path = audio_path_from_video(video_path, dataset_root)
            if not audio_path.exists():
                missing_audio.append((video_path, audio_path))
                continue

            samples_by_class[class_name].append(
                {
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "label": label,
                    "class_name": class_name,
                    "date": video_path.parts[-4],
                    "session": video_path.parts[-3],
                    "sample_id": video_path.stem.split("_video_")[-1],
                }
            )

    if missing_audio:
        preview = "\n".join(
            f"video={video}\nexpected_audio={audio}" for video, audio in missing_audio[:10]
        )
        raise FileNotFoundError(
            f"Found {len(missing_audio)} video files without matching audio. First missing pairs:\n{preview}"
        )

    return samples_by_class


def split_like_original(
    samples_by_class: Dict[str, List[Sample]],
    seed: int = DEFAULT_SEED,
    test_sample_per_class: int = DEFAULT_TEST_SAMPLE_PER_CLASS,
) -> Dict[str, List[Sample]]:
    """Apply original U-FFIA split rule: test first N, val next N, train rest."""
    random_state = np.random.RandomState(seed)

    splits: Dict[str, List[Sample]] = {"train": [], "test": [], "val": []}

    # Keep same class order as original source: strong, medium, weak, none.
    for class_name in ("strong", "medium", "weak", "none"):
        class_samples = list(samples_by_class[class_name])
        random_state.shuffle(class_samples)

        test_samples = class_samples[:test_sample_per_class]
        val_samples = class_samples[test_sample_per_class : 2 * test_sample_per_class]
        train_samples = class_samples[2 * test_sample_per_class :]

        splits["train"].extend(train_samples)
        splits["test"].extend(test_samples)
        splits["val"].extend(val_samples)

    random_state.shuffle(splits["train"])

    return splits


def write_jsonl(samples: Iterable[Sample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_csv(samples: Iterable[Sample], path: Path) -> None:
    rows = list(samples)
    fieldnames = ["video_path", "audio_path", "label", "class_name", "date", "session", "sample_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(splits: Dict[str, List[Sample]], samples_by_class: Dict[str, List[Sample]]) -> Dict:
    summary = {
        "label_map": CLASS_TO_LABEL,
        "total_by_class": {name: len(samples) for name, samples in samples_by_class.items()},
        "splits": {},
    }

    for split_name, samples in splits.items():
        counts = {name: 0 for name in CLASS_TO_LABEL}
        for sample in samples:
            counts[str(sample["class_name"])] += 1
        summary["splits"][split_name] = {"total": len(samples), "by_class": counts}

    return summary


def save_splits(splits: Dict[str, List[Sample]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        write_jsonl(samples, output_dir / f"{split_name}.jsonl")
        write_csv(samples, output_dir / f"{split_name}.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create U-FFIA train/test/val splits for paired audio-video data."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-sample-per-class", type=int, default=DEFAULT_TEST_SAMPLE_PER_CLASS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples_by_class = collect_multimodal_samples(args.dataset_root)
    splits = split_like_original(
        samples_by_class,
        seed=args.seed,
        test_sample_per_class=args.test_sample_per_class,
    )

    save_splits(splits, args.output_dir)

    summary = summarize(splits, samples_by_class)
    summary.update(
        {
            "dataset_root": str(args.dataset_root),
            "output_dir": str(args.output_dir),
            "seed": args.seed,
            "test_sample_per_class": args.test_sample_per_class,
        }
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
