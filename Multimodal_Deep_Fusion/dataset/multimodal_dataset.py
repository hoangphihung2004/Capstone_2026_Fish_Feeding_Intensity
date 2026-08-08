from __future__ import annotations

import concurrent.futures
import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DatasetConfig, SplitterConfig
from dataset.audio_loader import FishVoiceDataLoader
from dataset.data_split import FishDataSplitter
from dataset.video_loader import _decode_center_image, _decode_center_image_cv2
from transforms import VideoTransform

logger = logging.getLogger(__name__)


LABEL_COLUMNS = ("label", "label_id", "target")
AUDIO_COLUMNS = ("audio_path", "wav_path", "input_path", "path")
VIDEO_COLUMNS = ("video_path", "mp4_path", "image_path")


def resolve_num_workers(num_workers: int) -> int:
    if num_workers != -1:
        return max(0, int(num_workers))
    max_cpu = os.cpu_count()
    if max_cpu is None or max_cpu <= 0:
        return 0
    if max_cpu == 2:
        return max_cpu // 2
    return (max_cpu // 2) + 1


def _first_value(row: Dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _sample_key_from_paths(audio_path: str, video_path: str) -> str:
    path = audio_path or video_path
    normalized = str(path).replace("\\", "/").lower()
    stem = Path(normalized).stem.replace("_audio_", "_sample_").replace("_video_", "_sample_")
    parts = list(Path(normalized).parts)
    if len(parts) >= 4:
        return "/".join([parts[-4], parts[-3], parts[-2], stem])
    return stem


def _normalize_entry(entry: List[Any] | Tuple[Any, ...] | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(entry, dict):
        audio_path = _first_value(entry, AUDIO_COLUMNS)
        video_path = _first_value(entry, VIDEO_COLUMNS)
        label_value = _first_value(entry, LABEL_COLUMNS)
        sample_key = str(entry.get("sample_key") or _sample_key_from_paths(audio_path, video_path))
    elif len(entry) >= 3:
        audio_path = str(entry[0])
        video_path = str(entry[1])
        label_value = entry[2]
        sample_key = _sample_key_from_paths(audio_path, video_path)
    else:
        raise ValueError(f"Expected multimodal entry [audio_path, video_path, label], got: {entry}")

    if not audio_path:
        raise ValueError(f"Missing audio path in split entry: {entry}")
    if not video_path:
        raise ValueError(f"Missing video path in split entry: {entry}")
    if label_value in (None, ""):
        raise ValueError(f"Missing label in split entry: {entry}")

    label = int(label_value)
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    return {
        "audio_path": audio_path,
        "video_path": video_path,
        "label": label,
        "sample_key": sample_key,
    }


def load_split_csv(path: str | Path) -> List[Dict[str, Any]]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")

    rows: List[Dict[str, Any]] = []
    with split_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_normalize_entry(row))
    if not rows:
        raise ValueError(f"Split CSV is empty: {split_path}")
    return rows


def _split_dir_for_mode(dataset_cfg: DatasetConfig, mode: str, fold_index: Optional[int] = None) -> Path:
    if mode == "holdout":
        if not dataset_cfg.holdout_splits_dir:
            raise ValueError("dataset.holdout_splits_dir must be set when use_existing_splits=true.")
        return Path(dataset_cfg.holdout_splits_dir)
    if mode == "cross_validation":
        if not dataset_cfg.cross_validation_splits_dir:
            raise ValueError("dataset.cross_validation_splits_dir must be set when use_existing_splits=true.")
        if fold_index is None:
            raise ValueError("fold_index must be set for cross_validation split loading.")
        return Path(dataset_cfg.cross_validation_splits_dir) / f"fold_{fold_index:02d}" / "splits"
    raise ValueError(f"Unsupported split mode: {mode}")


def load_splits(dataset_cfg: DatasetConfig, mode: str, fold_index: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    if dataset_cfg.use_existing_splits:
        split_dir = _split_dir_for_mode(dataset_cfg, mode=mode, fold_index=fold_index)
        splits = {
            "train": load_split_csv(split_dir / "train.csv"),
            "val": load_split_csv(split_dir / "val.csv"),
            "test": load_split_csv(split_dir / "test.csv"),
        }
        _validate_splits(splits)
        logger.info("Loaded existing split files from: %s", split_dir)
        return splits

    splitter_cfg = SplitterConfig(
        dataset_path=dataset_cfg.dataset_path,
        seed=dataset_cfg.seed,
        test_sample_per_class=dataset_cfg.test_sample_per_class,
        save_results=False,
        include_video=True,
        split_strategy=dataset_cfg.split_strategy,
        evaluation_mode=mode,
        num_folds=dataset_cfg.num_folds,
        fold_index=fold_index,
        cv_val_ratio=dataset_cfg.cv_val_ratio,
    )
    splitter = FishDataSplitter(splitter_cfg)
    train_raw, test_raw, val_raw = splitter.split_data()
    splits = {
        "train": [_normalize_entry(item) for item in train_raw],
        "val": [_normalize_entry(item) for item in val_raw],
        "test": [_normalize_entry(item) for item in test_raw],
    }
    _validate_splits(splits)
    return splits


def _validate_splits(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    seen: Dict[str, str] = {}
    for split_name, entries in splits.items():
        for entry in entries:
            key = entry["sample_key"]
            if key in seen:
                raise ValueError(f"Sample key '{key}' appears in both {seen[key]} and {split_name}.")
            seen[key] = split_name
    for split_name, entries in splits.items():
        counts = {label: 0 for label in range(4)}
        for entry in entries:
            counts[int(entry["label"])] = counts.get(int(entry["label"]), 0) + 1
        logger.info(
            "Split %s: total=%d, none=%d, strong=%d, medium=%d, weak=%d",
            split_name,
            len(entries),
            counts.get(0, 0),
            counts.get(1, 0),
            counts.get(2, 0),
            counts.get(3, 0),
        )


def save_split_files(splits: Dict[str, List[Dict[str, Any]]], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_path", "audio_path", "label", "sample_key"]
    for split_name, entries in splits.items():
        with (output_path / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "video_path": entry["video_path"],
                        "audio_path": entry["audio_path"],
                        "label": entry["label"],
                        "sample_key": entry["sample_key"],
                    }
                )


class MultimodalFishDataset(Dataset):
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        split: str,
        sample_rate: int,
        image_size: int,
        cache_audio: bool,
        cache_video: bool,
        video_decode_backend: str,
        num_workers: int,
    ) -> None:
        self.entries = entries
        self.split = split
        self.sample_rate = sample_rate
        self.image_size = image_size
        self.cache_audio = cache_audio
        self.cache_video = cache_video
        self.video_decode_backend = video_decode_backend
        self.preload_workers = max(1, num_workers)
        self.audio_cache: Optional[List[np.ndarray]] = None
        self.video_cache: Optional[List[Dict[str, Any]]] = None
        self.transform = VideoTransform.get_transforms(image_size=image_size)[split]

        if cache_audio:
            self._preload_audio()
        if cache_video:
            self._preload_video()

    def __len__(self) -> int:
        return len(self.entries)

    def _preload_audio(self) -> None:
        logger.info("Preloading %s audio samples to RAM (%d samples)...", self.split, len(self.entries))

        def load_one(index_entry: Tuple[int, Dict[str, Any]]) -> Tuple[int, np.ndarray]:
            index, entry = index_entry
            waveform = FishVoiceDataLoader.load_audio(entry["audio_path"], sr=self.sample_rate)
            waveform = waveform.squeeze(0) if waveform.ndim > 1 else waveform
            return index, waveform.numpy()

        cache: List[Optional[np.ndarray]] = [None] * len(self.entries)
        total_bytes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
            futures = {executor.submit(load_one, item): item[0] for item in enumerate(self.entries)}
            iterator = concurrent.futures.as_completed(futures)
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc=f"Preloading {self.split} audio to RAM")
            except ImportError:
                pass
            for future in iterator:
                index, waveform_np = future.result()
                cache[index] = waveform_np
                total_bytes += waveform_np.nbytes

        self.audio_cache = [item for item in cache if item is not None]
        logger.info("Cached %s audio to RAM: %.1f MB", self.split, total_bytes / (1024**2))

    def _decode_video(self, video_path: str, label: int) -> Dict[str, Any]:
        if self.video_decode_backend == "decord":
            try:
                return _decode_center_image(video_path=video_path, label=label, image_size=self.image_size)
            except Exception as exc:
                logger.warning("Decord failed for '%s', falling back to OpenCV. Error: %s", video_path, exc)
        return _decode_center_image_cv2(video_path=video_path, label=label, image_size=self.image_size)

    def _preload_video(self) -> None:
        logger.info("Preloading %s center-frame video samples to RAM (%d samples)...", self.split, len(self.entries))

        def load_one(index_entry: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
            index, entry = index_entry
            return index, self._decode_video(entry["video_path"], int(entry["label"]))

        cache: List[Optional[Dict[str, Any]]] = [None] * len(self.entries)
        total_bytes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
            futures = {executor.submit(load_one, item): item[0] for item in enumerate(self.entries)}
            iterator = concurrent.futures.as_completed(futures)
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc=f"Preloading {self.split} video to RAM")
            except ImportError:
                pass
            for future in iterator:
                index, sample = future.result()
                cache[index] = sample
                total_bytes += sample["image_form"].nbytes

        self.video_cache = [item for item in cache if item is not None]
        logger.info("Cached %s video frames to RAM: %.1f MB", self.split, total_bytes / (1024**2))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        if self.audio_cache is not None:
            waveform_np = self.audio_cache[index]
        else:
            waveform = FishVoiceDataLoader.load_audio(entry["audio_path"], sr=self.sample_rate)
            waveform = waveform.squeeze(0) if waveform.ndim > 1 else waveform
            waveform_np = waveform.numpy()

        if self.video_cache is not None:
            video_sample = self.video_cache[index]
        else:
            video_sample = self._decode_video(entry["video_path"], int(entry["label"]))

        image = self.transform(video_sample["image_form"])
        return {
            "audio_path": entry["audio_path"],
            "video_path": entry["video_path"],
            "sample_key": entry["sample_key"],
            "waveform": waveform_np,
            "video_form": image,
            "label": int(entry["label"]),
        }


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    waveforms = torch.FloatTensor(np.array([item["waveform"] for item in batch]))
    videos = torch.stack([item["video_form"] for item in batch])
    labels = torch.LongTensor([item["label"] for item in batch])
    return {
        "audio_path": [item["audio_path"] for item in batch],
        "video_path": [item["video_path"] for item in batch],
        "sample_key": [item["sample_key"] for item in batch],
        "waveform": waveforms,
        "video_form": videos,
        "target": labels,
    }


def create_dataloaders(
    splits: Dict[str, List[Dict[str, Any]]],
    dataset_cfg: DatasetConfig,
    sample_rate: int,
    image_size: int,
) -> Dict[str, DataLoader]:
    workers = resolve_num_workers(dataset_cfg.num_workers)
    loaders: Dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        dataset = MultimodalFishDataset(
            entries=splits[split_name],
            split=split_name,
            sample_rate=sample_rate,
            image_size=image_size,
            cache_audio=dataset_cfg.cache_audio,
            cache_video=dataset_cfg.cache_video and dataset_cfg.video_cache_mode == "ram",
            video_decode_backend=dataset_cfg.video_decode_backend,
            num_workers=workers,
        )
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": dataset_cfg.batch_size,
            "shuffle": split_name == "train",
            "drop_last": False,
            "num_workers": workers,
            "collate_fn": multimodal_collate_fn,
            "pin_memory": True,
            "persistent_workers": workers > 0,
        }
        if workers > 0 and dataset_cfg.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = dataset_cfg.prefetch_factor
        loaders[split_name] = DataLoader(**dataloader_kwargs)
    return loaders
