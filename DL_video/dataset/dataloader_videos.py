import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import SplitterConfig, FishDataSplitter
from transforms import VideoTransform

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_image_cache_subdir(image_size: int) -> str:
    return f"single_frame_size_{image_size}"


def _resolve_image_cache_dir(video_cache_dir: Optional[str], image_size: int) -> Optional[str]:
    if not video_cache_dir:
        return None

    cache_root = os.path.normpath(video_cache_dir)
    cache_subdir = _get_image_cache_subdir(image_size=image_size)

    if os.path.basename(cache_root) == cache_subdir:
        return cache_root

    return os.path.join(cache_root, cache_subdir)


def _get_image_cache_path(image_cache_dir: Optional[str], split: str, index: int) -> Optional[str]:
    if not image_cache_dir:
        return None
    return os.path.join(image_cache_dir, split, f"{index}.pkl")


def _load_image_from_disk_cache(
    cache_path: Optional[str],
    video_path: str,
    image_size: int,
    label: Any
) -> Optional[Dict[str, Any]]:
    if cache_path is None or not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as f:
            sample = pickle.load(f)
    except Exception as exc:
        logger.warning(f"Could not read image cache '{cache_path}'. It will be regenerated. Error: {exc}")
        return None

    image_form = sample.get("image_form")
    meta = sample.get("_cache_meta", {})

    expected_shape = (3, image_size, image_size)
    if sample.get("video_name") != video_path:
        return None
    if meta.get("image_size") != image_size:
        return None
    if not isinstance(image_form, np.ndarray):
        return None
    if image_form.shape != expected_shape or image_form.dtype != np.uint8:
        return None

    return {
        "video_name": video_path,
        "image_form": image_form,
        "target": label,
    }


def _save_image_to_disk_cache(
    cache_path: Optional[str],
    sample: Dict[str, Any],
    image_size: int
) -> None:
    if cache_path is None:
        return

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    payload = {
        "video_name": sample["video_name"],
        "image_form": sample["image_form"],
        "target": sample["target"],
        "_cache_meta": {
            "image_size": image_size,
            "format": "uint8_CHW",
            "frame_policy": "center",
        },
    }

    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _decode_center_image(video_path: str, label: Any, image_size: int) -> Dict[str, Any]:
    from decord import VideoReader, cpu, gpu

    try:
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=gpu(0))
    except Exception:
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=cpu(0))

    full_vid_length = len(vr)

    if full_vid_length == 0:
        image_uint8 = np.zeros((3, image_size, image_size), dtype=np.uint8)
    else:
        frame_index = full_vid_length // 2
        image = vr.get_batch([frame_index]).asnumpy()[0]  # [H, W, C] RGB
        image_uint8 = image.transpose(2, 0, 1).astype(np.uint8)

    return {
        "video_name": video_path,
        "image_form": image_uint8,
        "target": label,
    }


def _decode_center_image_cv2(video_path: str, label: Any, image_size: int) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file: '{video_path}'")
        image_uint8 = np.zeros((3, image_size, image_size), dtype=np.uint8)
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(frame_count // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            image_uint8 = np.zeros((3, image_size, image_size), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            image_uint8 = frame.transpose(2, 0, 1).astype(np.uint8)
        cap.release()

    return {
        "video_name": video_path,
        "image_form": image_uint8,
        "target": label,
    }


def _load_or_create_image_sample(
    index: int,
    video_path: str,
    label: Any,
    image_size: int,
    disk_cache_video: bool,
    image_cache_dir: Optional[str],
    split: str
) -> Dict[str, Any]:
    cache_path = _get_image_cache_path(image_cache_dir, split, index) if disk_cache_video else None

    cached_sample = _load_image_from_disk_cache(
        cache_path=cache_path,
        video_path=video_path,
        image_size=image_size,
        label=label
    )
    if cached_sample is not None:
        return cached_sample

    try:
        sample = _decode_center_image(video_path=video_path, label=label, image_size=image_size)
    except Exception as exc:
        logger.warning(f"Decord failed for '{video_path}', falling back to OpenCV. Error: {exc}")
        sample = _decode_center_image_cv2(video_path=video_path, label=label, image_size=image_size)

    _save_image_to_disk_cache(
        cache_path=cache_path,
        sample=sample,
        image_size=image_size
    )
    return sample


class FishVideoDataLoader:
    """
    DataLoader manager for single-frame fish feeding image classification.
    Each video contributes one center RGB frame.
    """
    def __init__(
        self,
        batch_size: int = 50,
        dataloader_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        disk_cache_video: bool = False,
        video_cache_dir: Optional[str] = None,
        image_size: int = 224,
        splitter_config: Optional[SplitterConfig] = None
    ) -> None:
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.prefetch_factor = prefetch_factor
        self.disk_cache_video = disk_cache_video
        self.image_size = image_size
        self.video_cache_root = video_cache_dir
        self.image_cache_dir = _resolve_image_cache_dir(
            video_cache_dir=video_cache_dir,
            image_size=image_size
        )

        self.splitter_config = splitter_config if splitter_config is not None else SplitterConfig()
        self.splitter = FishDataSplitter(config=self.splitter_config)
        self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()

        logger.info("==================================================")
        logger.info("Initializing FishVideoDataLoader (Single Frame Image Pipeline):")
        logger.info(f"  - Batch Size:               {self.batch_size}")
        logger.info(f"  - DataLoader Workers:       {self.dataloader_workers}")
        if self.dataloader_workers <= 0:
            prefetch_log = "disabled"
        elif self.prefetch_factor is None:
            prefetch_log = "PyTorch default"
        else:
            prefetch_log = self.prefetch_factor
        logger.info(f"  - DataLoader Prefetch:      {prefetch_log}")
        logger.info(f"  - Disk PKL Caching:         {self.disk_cache_video}")
        logger.info(f"  - Disk PKL Cache Root:      '{self.video_cache_root if self.disk_cache_video else 'disabled'}'")
        logger.info(f"  - Disk PKL Cache Dir:       '{self.image_cache_dir if self.disk_cache_video else 'disabled'}'")
        logger.info(f"  - Image Resolution:         {self.image_size}x{self.image_size}")
        logger.info("  - Frame Policy:             center frame")
        logger.info("==================================================")

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        video_names = [data['video_name'] for data in batch]
        targets = [data['target'] for data in batch]

        images = torch.stack([data['video_form'] for data in batch])
        targets_tensor = torch.FloatTensor(np.array(targets))

        return {
            'video_name': video_names,
            'video_form': images,
            'target': targets_tensor
        }

    class _InnerDataset(Dataset):
        def __init__(self, parent: 'FishVideoDataLoader', split: str) -> None:
            self.parent = parent
            self.split = split
            self.disk_cache_video = parent.disk_cache_video

            if self.split == 'train':
                self.data_dict = parent.train_dict
            elif self.split == 'test':
                self.data_dict = parent.test_dict
            elif self.split == 'val':
                self.data_dict = parent.val_dict
            else:
                raise ValueError(f"Invalid split value '{self.split}'. Must be one of ['train', 'test', 'val'].")

            data_transform = VideoTransform.get_transforms(image_size=parent.image_size)
            self.transform = data_transform[self.split]
            logger.info(f"Initialized '{self.split}' image transformation pipeline.")

        def __len__(self) -> int:
            return len(self.data_dict)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            item = self.data_dict[index]
            if len(item) < 3:
                raise ValueError("Video training requires split entries in [audio_path, video_path, label] format.")

            video_name = item[1]
            target_val = item[2]
            if not video_name:
                raise ValueError(f"Missing video path for split='{self.split}', index={index}.")

            sample = _load_or_create_image_sample(
                index=index,
                video_path=video_name,
                label=target_val,
                image_size=self.parent.image_size,
                disk_cache_video=self.disk_cache_video,
                image_cache_dir=self.parent.image_cache_dir,
                split=self.split
            )

            image = self.transform(sample['image_form'])
            target = np.eye(4)[sample['target']] if isinstance(sample['target'], (int, np.integer)) else sample['target']

            return {
                'video_name': video_name,
                'video_form': image,
                'target': target
            }

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = False,
        drop_last: bool = False
    ) -> DataLoader:
        dataset = self._InnerDataset(parent=self, split=split)

        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "num_workers": self.dataloader_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": True,
            "persistent_workers": self.dataloader_workers > 0,
        }
        if self.dataloader_workers > 0 and self.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**dataloader_kwargs)
