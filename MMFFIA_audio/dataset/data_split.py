import csv
import glob
import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from config import SplitterConfig


LABEL_TO_CLASS = {0: "none", 1: "strong", 2: "weak"}
CLASS_TO_LABEL = {class_name: label for label, class_name in LABEL_TO_CLASS.items()}
LABEL_ORDER = [0, 1, 2]
VALID_SPLIT_STRATEGIES = {"random_sample", "time_series"}
VALID_EVALUATION_MODES = {"holdout", "cross_validation"}
MODALITY_DIRS = {
    "audio": "Audio",
    "image": "Image",
    "wave": "Wave",
}
FILE_PREFIX = {
    "audio": "audio",
    "image": "image",
    "wave": "wave",
}
FILE_EXTENSIONS = {
    "audio": [".wav"],
    "image": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    "wave": [".csv"],
}


def _stable_path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _extract_sample_id(path: str) -> str:
    stem = Path(path).stem
    numbers = re.findall(r"\d+", stem)
    return numbers[-1] if numbers else stem


def _sample_id_sort_key(path: str) -> Tuple[int, str]:
    sample_id = _extract_sample_id(path)
    return (int(sample_id), sample_id) if sample_id.isdigit() else (10**18, sample_id)


def _sample_identity_key(path: str) -> str:
    path_obj = Path(path)
    class_name = path_obj.parent.name.lower()
    return f"{class_name}/{_extract_sample_id(str(path_obj))}"


def _case_insensitive_child(parent: Path, child_name: str) -> Optional[Path]:
    if not parent.exists() or not parent.is_dir():
        return None
    direct = parent / child_name
    if direct.is_dir():
        return direct
    expected = child_name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == expected:
            return child
    return None


def _resolve_modality_root(dataset_path: str, modality_key: str, required: bool = True) -> Optional[Path]:
    root = Path(dataset_path)
    canonical = MODALITY_DIRS[modality_key]
    if root.name.lower() == canonical.lower():
        return root
    modality_root = _case_insensitive_child(root, canonical)
    if modality_root is not None:
        return modality_root
    if required:
        raise FileNotFoundError(
            f"MMFFIA dataset requires '{canonical}' inside dataset_path='{dataset_path}'."
        )
    return None


def _class_dir(root: Path, class_name: str) -> Optional[Path]:
    return _case_insensitive_child(root, class_name)


class BaseDataSplitter(ABC):
    """
    Shared split/export logic for MMFFIA. The dataset contains paired
    Audio/Image/Wave samples, and sample ids in filenames define temporal order.
    """

    def __init__(self, config: SplitterConfig) -> None:
        self.config = config
        self.dataset_path = config.dataset_path
        self.seed = config.seed
        self.test_sample_per_class = config.test_sample_per_class
        self.save_results = config.save_results
        self.include_video = config.include_video
        self.split_strategy = getattr(config, "split_strategy", "random_sample")
        self.evaluation_mode = getattr(config, "evaluation_mode", "holdout")
        self.num_folds = int(getattr(config, "num_folds", 5))
        self.fold_index = getattr(config, "fold_index", None)
        self.cv_val_ratio = float(getattr(config, "cv_val_ratio", 0.2))

        if self.split_strategy not in VALID_SPLIT_STRATEGIES:
            raise ValueError(
                f"Invalid split_strategy='{self.split_strategy}'. "
                f"Expected one of {sorted(VALID_SPLIT_STRATEGIES)}."
            )
        if self.evaluation_mode not in VALID_EVALUATION_MODES:
            raise ValueError(
                f"Invalid evaluation_mode='{self.evaluation_mode}'. "
                f"Expected one of {sorted(VALID_EVALUATION_MODES)}."
            )
        if self.evaluation_mode == "cross_validation":
            if self.fold_index is None:
                raise ValueError("fold_index must be set when evaluation_mode='cross_validation'.")
            if not 0 <= int(self.fold_index) < self.num_folds:
                raise ValueError(f"fold_index={self.fold_index} is outside [0, {self.num_folds}).")

        self.audio_path = str(_resolve_modality_root(self.dataset_path, "audio", required=True))
        self.image_path = str(_resolve_modality_root(self.dataset_path, "image", required=True))
        self.wave_path = str(_resolve_modality_root(self.dataset_path, "wave", required=True))
        self.num_classes = len(LABEL_ORDER)

    @abstractmethod
    def get_file_list(self, split_name: str) -> List[str]:
        pass

    @abstractmethod
    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        pass

    @abstractmethod
    def _format_samples(self, split_name: str, data_list: List[List]) -> List[Dict[str, Any]]:
        pass

    @staticmethod
    def _write_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_csv(samples: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

    @staticmethod
    def _primary_path_from_entry(item: List[Any]) -> str:
        return str(item[0])

    @staticmethod
    def _label_from_entry(item: List[Any]) -> int:
        return int(item[-1])

    @staticmethod
    def _temporal_path_key(path: str) -> Tuple[int, str]:
        return _sample_id_sort_key(path)

    def _entry_sample_key(self, item: List[Any]) -> str:
        return _sample_identity_key(self._primary_path_from_entry(item))

    def _entry_temporal_key(self, item: List[Any]) -> Tuple[int, str]:
        return self._temporal_path_key(self._primary_path_from_entry(item))

    def _count_labels(self, data_list: List[List]) -> Counter:
        counts = Counter()
        for item in data_list:
            counts[self._label_from_entry(item)] += 1
        return counts

    @staticmethod
    def _sample_key_set(data_list: List[List], sample_key_fn) -> set:
        return {sample_key_fn(item) for item in data_list}

    def _time_series_expected_splits(self, all_entries: List[List]) -> Tuple[set, set, set]:
        expected_train = set()
        expected_test = set()
        expected_val = set()

        for label in LABEL_ORDER:
            class_entries = [item for item in all_entries if self._label_from_entry(item) == label]
            class_entries = sorted(class_entries, key=self._entry_temporal_key)
            required = 2 * self.test_sample_per_class
            if len(class_entries) < required:
                raise ValueError(
                    f"Class '{LABEL_TO_CLASS[label]}' has only {len(class_entries)} samples, "
                    f"but at least {required} are required for validation and test splits."
                )

            train_entries = class_entries[:-required]
            val_entries = class_entries[-required:-self.test_sample_per_class]
            test_entries = class_entries[-self.test_sample_per_class:]

            expected_train.update(self._entry_sample_key(item) for item in train_entries)
            expected_val.update(self._entry_sample_key(item) for item in val_entries)
            expected_test.update(self._entry_sample_key(item) for item in test_entries)

        return expected_train, expected_test, expected_val

    def _ordered_class_entries(self, entries: List[List], label: int, seed_offset: int = 0) -> List[List]:
        class_entries = [item for item in entries if self._label_from_entry(item) == label]
        if self.split_strategy == "time_series":
            return sorted(class_entries, key=self._entry_temporal_key)

        class_entries = sorted(class_entries, key=self._entry_sample_key)
        rng = np.random.RandomState(self.seed + seed_offset + label)
        rng.shuffle(class_entries)
        return class_entries

    @staticmethod
    def _split_into_folds(items: List[List], num_folds: int) -> List[List[List]]:
        indexes = np.array_split(np.arange(len(items)), num_folds)
        return [[items[int(idx)] for idx in fold_indexes] for fold_indexes in indexes]

    def _cross_validation_expected_splits(self, all_entries: List[List]) -> Tuple[set, set, set]:
        fold_index = int(self.fold_index)
        train_entries: List[List] = []
        test_entries: List[List] = []
        val_entries: List[List] = []

        for label in LABEL_ORDER:
            class_entries = self._ordered_class_entries(entries=all_entries, label=label, seed_offset=0)
            outer_folds = self._split_into_folds(class_entries, self.num_folds)
            test_class_entries = list(outer_folds[fold_index])
            dev_class_entries = [
                item
                for idx, fold_entries in enumerate(outer_folds)
                if idx != fold_index
                for item in fold_entries
            ]

            if self.split_strategy == "time_series":
                dev_class_entries = sorted(dev_class_entries, key=self._entry_temporal_key)
            else:
                dev_class_entries = sorted(dev_class_entries, key=self._entry_sample_key)
                rng = np.random.RandomState(self.seed + 10000 + fold_index + label)
                rng.shuffle(dev_class_entries)

            val_count = int(round(len(dev_class_entries) * self.cv_val_ratio))
            if val_count <= 0 or val_count >= len(dev_class_entries):
                raise ValueError(
                    f"Invalid cv_val_ratio={self.cv_val_ratio} for class '{LABEL_TO_CLASS[label]}' "
                    f"with {len(dev_class_entries)} development samples."
                )

            if self.split_strategy == "time_series":
                train_class_entries = dev_class_entries[:-val_count]
                val_class_entries = dev_class_entries[-val_count:]
            else:
                val_class_entries = dev_class_entries[:val_count]
                train_class_entries = dev_class_entries[val_count:]

            train_entries.extend(train_class_entries)
            val_entries.extend(val_class_entries)
            test_entries.extend(test_class_entries)

        return (
            {self._entry_sample_key(item) for item in train_entries},
            {self._entry_sample_key(item) for item in test_entries},
            {self._entry_sample_key(item) for item in val_entries},
        )

    def _split_entries_cross_validation(self, all_entries: List[List]) -> Tuple[List[List], List[List], List[List]]:
        expected_train, expected_test, expected_val = self._cross_validation_expected_splits(all_entries)
        train_dict = [item for item in all_entries if self._entry_sample_key(item) in expected_train]
        test_dict = [item for item in all_entries if self._entry_sample_key(item) in expected_test]
        val_dict = [item for item in all_entries if self._entry_sample_key(item) in expected_val]

        self._validate_split_policy(train_dict, test_dict, val_dict)
        return train_dict, test_dict, val_dict

    def _validate_split_policy(self, train_dict: List[List], test_dict: List[List], val_dict: List[List]) -> None:
        splits = {"train": train_dict, "test": test_dict, "val": val_dict}

        seen_samples: Dict[str, str] = {}
        for split_name, data_list in splits.items():
            for item in data_list:
                sample_key = self._entry_sample_key(item)
                if sample_key in seen_samples:
                    raise ValueError(
                        f"Duplicate sample detected across splits: '{sample_key}' "
                        f"appears in both {seen_samples[sample_key]} and {split_name}."
                    )
                seen_samples[sample_key] = split_name

        counts_by_split = {name: self._count_labels(data) for name, data in splits.items()}
        total_counts = Counter()
        for counts in counts_by_split.values():
            total_counts.update(counts)

        if self.evaluation_mode == "holdout":
            for split_name in ["test", "val"]:
                for label in LABEL_ORDER:
                    actual = counts_by_split[split_name][label]
                    if actual != self.test_sample_per_class:
                        raise ValueError(
                            f"{split_name} split class '{LABEL_TO_CLASS[label]}' has {actual} samples; "
                            f"expected exactly {self.test_sample_per_class}."
                        )

            for label in LABEL_ORDER:
                expected_train = total_counts[label] - (2 * self.test_sample_per_class)
                actual_train = counts_by_split["train"][label]
                if actual_train != expected_train:
                    raise ValueError(
                        f"train split class '{LABEL_TO_CLASS[label]}' has {actual_train} samples; "
                        f"expected exactly {expected_train}."
                    )

        if self.evaluation_mode == "holdout" and self.split_strategy == "time_series":
            expected_train, expected_test, expected_val = self._time_series_expected_splits(
                train_dict + test_dict + val_dict
            )
            actual_train = self._sample_key_set(train_dict, self._entry_sample_key)
            actual_test = self._sample_key_set(test_dict, self._entry_sample_key)
            actual_val = self._sample_key_set(val_dict, self._entry_sample_key)
            if actual_train != expected_train or actual_test != expected_test or actual_val != expected_val:
                raise ValueError("Existing split files do not match the configured time_series split strategy.")

        if self.evaluation_mode == "cross_validation":
            expected_train, expected_test, expected_val = self._cross_validation_expected_splits(
                train_dict + test_dict + val_dict
            )
            actual_train = self._sample_key_set(train_dict, self._entry_sample_key)
            actual_test = self._sample_key_set(test_dict, self._entry_sample_key)
            actual_val = self._sample_key_set(val_dict, self._entry_sample_key)
            if actual_train != expected_train or actual_test != expected_test or actual_val != expected_val:
                raise ValueError("Existing split files do not match the configured cross-validation fold.")

        logger.info(f"Split validation passed for mode='{self.evaluation_mode}', strategy='{self.split_strategy}':")
        for split_name in ["train", "test", "val"]:
            counts = counts_by_split[split_name]
            details = ", ".join(f"{LABEL_TO_CLASS[label]}={counts[label]}" for label in LABEL_ORDER)
            logger.info(f"  - {split_name}: {details}")

    def _save_splits(self, train_dict: List[List], test_dict: List[List], val_dict: List[List], output_dir: Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        splits = {
            "train": train_dict,
            "test": test_dict,
            "val": val_dict,
        }
        formatted_splits: Dict[str, List[Dict[str, Any]]] = {}
        fieldnames = ["audio_path", "image_path", "wave_path", "label", "class_name", "sample_id"]

        for split_name, data_list in splits.items():
            samples = self._format_samples(split_name, data_list)
            formatted_splits[split_name] = samples
            self._write_jsonl(samples, output_path / f"{split_name}.jsonl")
            self._write_csv(samples, output_path / f"{split_name}.csv", fieldnames=fieldnames)
            logger.info(f"Successfully saved split files to {output_path}")

        label_map = {class_name: label for label, class_name in LABEL_TO_CLASS.items()}
        summary: Dict[str, Any] = {
            "label_map": label_map,
            "splits": {},
        }

        for split_name, samples in formatted_splits.items():
            counts = {class_name: 0 for class_name in CLASS_TO_LABEL}
            for sample in samples:
                counts[str(sample["class_name"])] += 1
            summary["splits"][split_name] = {
                "total": len(samples),
                "by_class": counts,
            }

        summary.update({
            "dataset_root": self.dataset_path,
            "audio_root": self.audio_path,
            "image_root": self.image_path,
            "wave_root": self.wave_path,
            "output_dir": str(output_path),
            "seed": self.seed,
            "test_sample_per_class": self.test_sample_per_class,
            "split_strategy": self.split_strategy,
            "evaluation_mode": self.evaluation_mode,
            "num_folds": self.num_folds,
            "fold_index": self.fold_index,
            "cv_val_ratio": self.cv_val_ratio,
        })

        with (output_path / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved split summary file to {output_path}")


class FishDataSplitter(BaseDataSplitter):
    """
    MMFFIA audio splitter. Audio is the training anchor; paired Image and Wave
    paths are resolved and stored so every split is synchronized across modalities.
    """

    def get_file_list(self, split_name: str) -> List[str]:
        class_root = _class_dir(Path(self.audio_path), split_name)
        if class_root is None:
            return []
        files: List[str] = []
        for ext in FILE_EXTENSIONS["audio"]:
            files.extend(glob.glob(str(class_root / f"*{ext}")))
        return sorted(files, key=_sample_id_sort_key)

    def _resolve_paired_path(self, audio_path: str, modality_key: str) -> str:
        audio = Path(audio_path)
        class_name = audio.parent.name
        sample_id = _extract_sample_id(str(audio))
        root = Path(getattr(self, f"{modality_key}_path"))
        class_root = _class_dir(root, class_name)
        if class_root is None:
            return ""

        prefix = FILE_PREFIX[modality_key]
        for ext in FILE_EXTENSIONS[modality_key]:
            candidate = class_root / f"{prefix}_{sample_id}{ext}"
            if candidate.exists():
                return str(candidate)

        glob_pattern = str(class_root / f"*{sample_id}*")
        candidates = [
            candidate
            for candidate in glob.glob(glob_pattern)
            if Path(candidate).suffix.lower() in FILE_EXTENSIONS[modality_key]
        ]
        return sorted(candidates, key=_stable_path_key)[0] if candidates else ""

    def _resolve_audio_path_from_row(self, row: Dict[str, str]) -> str:
        raw_audio_path = row.get("audio_path", "")
        if raw_audio_path and os.path.exists(raw_audio_path):
            return raw_audio_path

        class_name = row.get("class_name", "")
        sample_id = row.get("sample_id", "")
        if not class_name or not sample_id:
            source_path = raw_audio_path or row.get("image_path", "") or row.get("wave_path", "")
            class_name = Path(source_path).parent.name
            sample_id = _extract_sample_id(source_path)

        class_root = _class_dir(Path(self.audio_path), class_name)
        if class_root is None:
            return raw_audio_path

        for ext in FILE_EXTENSIONS["audio"]:
            candidate = class_root / f"audio_{sample_id}{ext}"
            if candidate.exists():
                return str(candidate)
        return raw_audio_path

    def _make_multimodal_entry(self, audio_path: str, label: int) -> List[Any]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio file while creating split: '{audio_path}'")

        image_path = self._resolve_paired_path(audio_path, "image")
        wave_path = self._resolve_paired_path(audio_path, "wave")
        if not image_path:
            raise FileNotFoundError(f"Missing paired image file for audio sample: '{audio_path}'")
        if not wave_path:
            raise FileNotFoundError(f"Missing paired wave CSV file for audio sample: '{audio_path}'")
        return [audio_path, image_path, wave_path, label]

    def _validate_multimodal_entries(self, split_name: str, data_list: List[List]) -> None:
        for idx, item in enumerate(data_list):
            if len(item) not in [2, 4]:
                raise ValueError(
                    f"{split_name} split entry #{idx} must be [audio_path, label] or "
                    f"[audio_path, image_path, wave_path, label], got: {item}"
                )
            audio_path = item[0]
            if not os.path.exists(audio_path):
                raise FileNotFoundError(
                    f"{split_name} split entry #{idx} has missing audio_path: '{audio_path}'"
                )
            if len(item) == 4:
                image_path, wave_path = item[1], item[2]
                expected_image = self._resolve_paired_path(audio_path, "image")
                expected_wave = self._resolve_paired_path(audio_path, "wave")
                if not image_path or not os.path.exists(image_path):
                    raise FileNotFoundError(
                        f"{split_name} split entry #{idx} has missing image_path: '{image_path}'"
                    )
                if not wave_path or not os.path.exists(wave_path):
                    raise FileNotFoundError(
                        f"{split_name} split entry #{idx} has missing wave_path: '{wave_path}'"
                    )
                if os.path.normpath(image_path) != os.path.normpath(expected_image):
                    raise ValueError(
                        f"{split_name} split entry #{idx} has mismatched image_path.\n"
                        f"  audio_path:    {audio_path}\n"
                        f"  image_path:    {image_path}\n"
                        f"  expected_image:{expected_image}"
                    )
                if os.path.normpath(wave_path) != os.path.normpath(expected_wave):
                    raise ValueError(
                        f"{split_name} split entry #{idx} has mismatched wave_path.\n"
                        f"  audio_path:  {audio_path}\n"
                        f"  wave_path:   {wave_path}\n"
                        f"  expected_wave:{expected_wave}"
                    )

    def _splits_dir(self) -> Path:
        dataset_path = Path(self.dataset_path)
        local_splits_dir = dataset_path / "splits"
        if self.evaluation_mode == "cross_validation":
            return local_splits_dir / "cv" / f"fold_{int(self.fold_index):02d}"
        return local_splits_dir

    def _load_existing_splits(self, splits_dir: Path) -> Optional[Tuple[List[List], List[List], List[List]]]:
        train_csv = splits_dir / "train.csv"
        test_csv = splits_dir / "test.csv"
        val_csv = splits_dir / "val.csv"
        if not (train_csv.exists() and test_csv.exists() and val_csv.exists()):
            return None

        logger.info("==================================================")
        logger.info(f"Fallback mode enabled: found existing split files at '{splits_dir}'.")
        logger.info("Loading existing dataset splits instead of recomputing them...")
        logger.info(f"Base audio search path: '{self.audio_path}'")
        logger.info("==================================================")

        need_rewrite_files = False

        def load_and_fix_csv(csv_path: Path, split_name: str) -> List[List]:
            nonlocal need_rewrite_files
            loaded_data: List[List] = []
            fixed_count = 0
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = int(row["label"])
                    audio_path = self._resolve_audio_path_from_row(row)
                    if not os.path.exists(audio_path):
                        raise FileNotFoundError(
                            f"{split_name} split has a missing audio file that could not be corrected: '{audio_path}'"
                        )

                    image_path = self._resolve_paired_path(audio_path, "image")
                    wave_path = self._resolve_paired_path(audio_path, "wave")
                    if not image_path or not wave_path:
                        raise FileNotFoundError(
                            f"{split_name} split cannot resolve paired Image/Wave files for audio_path='{audio_path}'"
                        )

                    if (
                        os.path.normpath(row.get("audio_path", "")) != os.path.normpath(audio_path)
                        or os.path.normpath(row.get("image_path", row.get("video_path", ""))) != os.path.normpath(image_path)
                        or os.path.normpath(row.get("wave_path", "")) != os.path.normpath(wave_path)
                    ):
                        fixed_count += 1
                        need_rewrite_files = True

                    loaded_data.append([audio_path, image_path, wave_path, label])

            if fixed_count > 0:
                logger.info(f"{split_name} split: corrected or filled {fixed_count} MMFFIA multimodal paths.")
            return loaded_data

        try:
            train_dict = load_and_fix_csv(train_csv, "Train")
            test_dict = load_and_fix_csv(test_csv, "Test")
            val_dict = load_and_fix_csv(val_csv, "Validation")

            self._validate_multimodal_entries("Train", train_dict)
            self._validate_multimodal_entries("Test", test_dict)
            self._validate_multimodal_entries("Validation", val_dict)
            self._validate_split_policy(train_dict, test_dict, val_dict)

            if need_rewrite_files:
                logger.info("Updating split files on disk to synchronize corrected paths...")
                self._save_splits(train_dict, test_dict, val_dict, splits_dir)

            logger.info("Successfully loaded dataset splits from files:")
            logger.info(f"- Train samples: {len(train_dict)}")
            logger.info(f"- Test samples:  {len(test_dict)}")
            logger.info(f"- Val samples:   {len(val_dict)}")
            logger.info("==================================================")
            return train_dict, test_dict, val_dict
        except Exception as exc:
            logger.warning(f"Failed to load or auto-correct existing splits: {exc}. Falling back to standard splitting.")
            return None

    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        splits_dir = self._splits_dir()
        existing_splits = self._load_existing_splits(splits_dir)
        if existing_splits is not None:
            return existing_splits

        logger.info("==================================================")
        logger.info("Starting MMFFIA audio dataset splitting...")
        logger.info(f"Dataset root directory: '{self.dataset_path}'")
        logger.info(f"Audio root: '{self.audio_path}'")
        logger.info(f"Image root: '{self.image_path}'")
        logger.info(f"Wave root:  '{self.wave_path}'")
        logger.info(f"Random seed: {self.seed}")
        logger.info(f"Test/Val samples per class: {self.test_sample_per_class}")
        logger.info(f"Evaluation mode: {self.evaluation_mode}")
        logger.info(f"Split strategy: {self.split_strategy}")
        if self.evaluation_mode == "cross_validation":
            logger.info(f"CV folds: {self.num_folds} | Fold index: {self.fold_index} | CV val ratio: {self.cv_val_ratio}")
        logger.info(f"Save split results: {self.save_results}")
        logger.info("Audio is the training anchor; Image/Wave paths are paired by class and numeric sample id.")
        logger.info("==================================================")

        class_lists: Dict[int, List[str]] = {}
        for label in LABEL_ORDER:
            class_name = LABEL_TO_CLASS[label]
            files = self.get_file_list(class_name)
            class_lists[label] = files
            logger.info(f"Class '{class_name}': Found {len(files)} audio files.")

        def build_entries(paths_by_label: Dict[int, List[str]]) -> List[List]:
            entries: List[List] = []
            for label in LABEL_ORDER:
                for audio_path in paths_by_label[label]:
                    if self.include_video:
                        entries.append(self._make_multimodal_entry(audio_path, label))
                    else:
                        entries.append([audio_path, label])
            return entries

        if self.evaluation_mode == "cross_validation":
            logger.info("Building stratified cross-validation fold split...")
            train_dict, test_dict, val_dict = self._split_entries_cross_validation(build_entries(class_lists))
            if self.save_results:
                self._save_splits(train_dict, test_dict, val_dict, splits_dir)
            return train_dict, test_dict, val_dict

        random_state = np.random.RandomState(self.seed)
        ordered_lists: Dict[int, List[str]] = {}
        for label, file_list in class_lists.items():
            if self.split_strategy == "time_series":
                ordered_lists[label] = sorted(file_list, key=_sample_id_sort_key)
            else:
                shuffled = list(sorted(file_list, key=_sample_identity_key))
                random_state.shuffle(shuffled)
                ordered_lists[label] = shuffled

        train_by_label: Dict[int, List[str]] = {}
        test_by_label: Dict[int, List[str]] = {}
        val_by_label: Dict[int, List[str]] = {}

        for label in LABEL_ORDER:
            file_list = ordered_lists[label]
            required = 2 * self.test_sample_per_class
            if len(file_list) < required:
                logger.warning(
                    f"Class '{LABEL_TO_CLASS[label]}' has only {len(file_list)} files, "
                    f"but at least {required} samples are required for Test and Val splits."
                )

            if self.split_strategy == "time_series":
                train_by_label[label] = file_list[:-required]
                val_by_label[label] = file_list[-required:-self.test_sample_per_class]
                test_by_label[label] = file_list[-self.test_sample_per_class:]
            else:
                test_by_label[label] = file_list[:self.test_sample_per_class]
                val_by_label[label] = file_list[self.test_sample_per_class:required]
                train_by_label[label] = file_list[required:]

        logger.info("Per-class split details:")
        for label in LABEL_ORDER:
            logger.info(
                f"       - class '{LABEL_TO_CLASS[label]}': "
                f"Train={len(train_by_label[label])}, Test={len(test_by_label[label])}, Val={len(val_by_label[label])}"
            )

        train_dict = build_entries(train_by_label)
        test_dict = build_entries(test_by_label)
        val_dict = build_entries(val_by_label)

        if self.split_strategy == "random_sample":
            logger.info("Applying final shuffle to the train split...")
            random_state.shuffle(train_dict)

        self._validate_split_policy(train_dict, test_dict, val_dict)

        logger.info("==================================================")
        logger.info("MMFFIA audio dataset splitting completed successfully!")
        logger.info(f"- Train samples: {len(train_dict)}")
        logger.info(f"- Test samples:  {len(test_dict)}")
        logger.info(f"- Val samples:   {len(val_dict)}")
        if len(train_dict) > 0:
            logger.info(f"  * First generated train sample: {train_dict[0]}")
        if len(test_dict) > 0:
            logger.info(f"  * First generated test sample:  {test_dict[0]}")
        if len(val_dict) > 0:
            logger.info(f"  * First generated val sample:   {val_dict[0]}")
        logger.info("==================================================")

        if self.save_results:
            self._save_splits(train_dict, test_dict, val_dict, splits_dir)

        return train_dict, test_dict, val_dict

    def _format_samples(self, split_name: str, data_list: List[List]) -> List[Dict[str, Any]]:
        samples = []
        for item in data_list:
            if len(item) == 4:
                audio_path, image_path, wave_path, label = item
            else:
                audio_path, label = item
                image_path = self._resolve_paired_path(audio_path, "image")
                wave_path = self._resolve_paired_path(audio_path, "wave")

            samples.append({
                "audio_path": str(audio_path),
                "image_path": str(image_path),
                "wave_path": str(wave_path),
                "label": int(label),
                "class_name": LABEL_TO_CLASS.get(int(label), ""),
                "sample_id": _extract_sample_id(str(audio_path)),
            })
        return samples


if __name__ == "__main__":
    pass
