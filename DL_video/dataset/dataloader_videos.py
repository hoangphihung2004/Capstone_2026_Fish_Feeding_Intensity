import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import SplitterConfig, FishDataSplitter
from transforms import VideoTransform

# Force stdout/stderr to use UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_video_cache_subdir(image_size: int, frames_count: int) -> str:
    return f"frames_{frames_count}_size_{image_size}"


def _resolve_video_cache_dir(
    video_cache_dir: Optional[str],
    image_size: int,
    frames_count: int
) -> Optional[str]:
    if not video_cache_dir:
        return None

    cache_root = os.path.normpath(video_cache_dir)
    cache_subdir = _get_video_cache_subdir(
        image_size=image_size,
        frames_count=frames_count
    )

    if os.path.basename(cache_root) == cache_subdir:
        return cache_root

    return os.path.join(cache_root, cache_subdir)


def _get_video_cache_path(video_cache_dir: Optional[str], split: str, index: int) -> Optional[str]:
    if not video_cache_dir:
        return None
    return os.path.join(video_cache_dir, split, f"{index}.pkl")


def _load_video_from_disk_cache(
    cache_path: Optional[str],
    video_path: str,
    image_size: int,
    frames_count: int,
    label: Any
) -> Optional[Dict[str, Any]]:
    """
    Load one cached sample from disk if the file matches the active dataset/config.
    Returns None when cache is missing, stale, or invalid.
    """
    if cache_path is None or not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as f:
            sample = pickle.load(f)
    except Exception as exc:
        logger.warning(f"Could not read video cache '{cache_path}'. It will be regenerated. Error: {exc}")
        return None

    video_form = sample.get("video_form")
    meta = sample.get("_cache_meta", {})

    expected_shape = (frames_count, 3, image_size, image_size)
    if sample.get("video_name") != video_path:
        return None
    if meta.get("image_size") != image_size or meta.get("frames_count") != frames_count:
        return None
    if not isinstance(video_form, np.ndarray):
        return None
    if video_form.shape != expected_shape or video_form.dtype != np.uint8:
        return None

    return {
        "video_name": video_path,
        "video_form": video_form,
        "target": label
    }


def _save_video_to_disk_cache(
    cache_path: Optional[str],
    sample: Dict[str, Any],
    image_size: int,
    frames_count: int
) -> None:
    if cache_path is None:
        return

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    payload = {
        "video_name": sample["video_name"],
        "video_form": sample["video_form"],
        "target": sample["target"],
        "_cache_meta": {
            "image_size": image_size,
            "frames_count": frames_count,
            "format": "uint8_TCHW",
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


def _decode_video_sample(video_path: str, label: Any, image_size: int, frames_count: int) -> Dict[str, Any]:
    import numpy as np
    import torch
    from decord import VideoReader, cpu, gpu

    # Attempt GPU decode first (following original author's approach), fallback to CPU
    try:
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=gpu(0))
    except Exception:
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=cpu(0))

    full_vid_length = len(vr)

    if full_vid_length == 0:
        vf_uint8 = np.zeros((frames_count, 3, image_size, image_size), dtype=np.uint8)
    else:
        # Segment-based sampling: divide video into frames_count equal segments,
        # pick 1 random frame per segment (identical logic to original cv2 implementation)
        actual_count = min(frames_count, full_vid_length)
        segment_width = full_vid_length / actual_count
        selected_indices = []
        for i in range(actual_count):
            start = int(i * segment_width)
            end = max(start, int((i + 1) * segment_width) - 1)
            end = min(end, full_vid_length - 1)
            if start == end:
                val = start
            else:
                val = int(torch.randint(low=start, high=end + 1, size=(1,)).item())
            selected_indices.append(val)
        selected_indices = sorted(selected_indices)

        # Pad if fewer frames than required
        while len(selected_indices) < frames_count:
            selected_indices.append(selected_indices[-1])

        # decord get_batch: returns NDArray shape [T, H, W, C] in RGB
        video_frames = vr.get_batch(selected_indices).asnumpy()  # [T, H, W, C]

        # Convert to [T, C, H, W] uint8 matching original format
        vf_uint8 = video_frames.transpose(0, 3, 1, 2).astype(np.uint8)

    return {
        "video_name": video_path,
        "video_form": vf_uint8,
        "target": label
    }


def _load_or_create_video_sample(
    index: int,
    video_path: str,
    label: Any,
    image_size: int,
    frames_count: int,
    disk_cache_video: bool,
    video_cache_dir: Optional[str],
    split: str
) -> Dict[str, Any]:
    cache_path = _get_video_cache_path(video_cache_dir, split, index) if disk_cache_video else None

    cached_sample = _load_video_from_disk_cache(
        cache_path=cache_path,
        video_path=video_path,
        image_size=image_size,
        frames_count=frames_count,
        label=label
    )
    if cached_sample is not None:
        return cached_sample

    sample = _decode_video_sample(
        video_path=video_path,
        label=label,
        image_size=image_size,
        frames_count=frames_count
    )
    _save_video_to_disk_cache(
        cache_path=cache_path,
        sample=sample,
        image_size=image_size,
        frames_count=frames_count
    )
    return sample


class FishVideoDataLoader:
    """
    Unified manager class for the fish video raw dataset (FishVideoDataLoader).
    Supports optional per-sample disk .pkl caching without preloading the dataset into RAM.
    Disk cache stores decoded/resized uint8 clips before tensor normalization.
    """
    def __init__(
        self,
        batch_size: int = 50,
        dataloader_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        disk_cache_video: bool = False,
        video_cache_dir: Optional[str] = None,
        image_size: int = 224,
        frames_count: int = 4,
        splitter_config: Optional[SplitterConfig] = None
    ) -> None:
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.prefetch_factor = prefetch_factor
        self.disk_cache_video = disk_cache_video
        self.image_size = image_size
        self.frames_count = frames_count
        self.video_cache_root = video_cache_dir
        self.video_cache_dir = _resolve_video_cache_dir(
            video_cache_dir=video_cache_dir,
            image_size=image_size,
            frames_count=frames_count
        )

        # 1. Load splitter configurations and initialize the data splitter
        if splitter_config is None:
            self.splitter_config = SplitterConfig()
        else:
            self.splitter_config = splitter_config
        self.splitter = FishDataSplitter(config=self.splitter_config)

        # 2. Split dataset into train, val, and test partitions
        self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()

        logger.info("==================================================")
        logger.info("Initializing FishVideoDataLoader (Video Cache Pipeline):")
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
        logger.info(f"  - Disk PKL Cache Dir:       '{self.video_cache_dir if self.disk_cache_video else 'disabled'}'")
        logger.info(f"  - Image Resolution:         {self.image_size}x{self.image_size}")
        logger.info(f"  - Frames per Video:         {self.frames_count} (Segment-based Sampling)")
        logger.info("==================================================")

    @staticmethod
    def extract_video_frames_segment_based(video_path: str, image_size: int = 224, frames_count: int = 4) -> np.ndarray:
        """
        Extract frames_count from a raw video file using Segment-based Sampling.
        Divides video into frames_count equal time intervals and picks 1 random frame per interval.
        Guarantees no consecutive frames and uniform temporal coverage.

        Returns:
            np.ndarray: Array of shape [frames_count, 3, image_size, image_size] in uint8 RGB format.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error: Could not open video file: '{video_path}'")
            return np.zeros((frames_count, 3, image_size, image_size), dtype=np.uint8)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        num_frames = len(frames)
        if num_frames == 0:
            logger.warning(f"Warning: Video is empty: '{video_path}'")
            return np.zeros((frames_count, 3, image_size, image_size), dtype=np.uint8)

        actual_frames_count = frames_count
        if actual_frames_count > num_frames:
            logger.warning(f"Warning: Requested frames ({frames_count}) exceeds video frames ({num_frames}) for '{video_path}'. Capping to {num_frames}.")
            actual_frames_count = num_frames

        # Segment-based sampling
        segment_width = num_frames / actual_frames_count
        Y = []
        for i in range(actual_frames_count):
            start = int(i * segment_width)
            end = int((i + 1) * segment_width) - 1
            end = max(start, end)
            end = min(end, num_frames - 1)
            if start >= num_frames:
                start = num_frames - 1

            if start == end:
                val = start
            else:
                val = int(torch.randint(low=start, high=end + 1, size=(1,)).item())
            Y.append(val)
        Y = sorted(Y)

        selected_frames = [frames[y] for y in Y]

        while len(selected_frames) < frames_count:
            selected_frames.append(selected_frames[-1])

        stacked = np.stack([f.transpose(2, 0, 1) for f in selected_frames]).astype(np.uint8)
        return stacked

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate single video sample frames into batch tensors.
        Permutes tensor dimensions from [Batch, Frames, Channels, Height, Width]
        to [Batch, Channels, Frames, Height, Width] as expected by 3D Convolutional networks.
        """
        video_names = [data['video_name'] for data in batch]
        targets = [data['target'] for data in batch]
        
        vf = torch.stack([data['video_form'] for data in batch])
        targets_tensor = torch.FloatTensor(np.array(targets))
        
        vf = vf.permute(0, 2, 1, 3, 4)

        return {
            'video_name': video_names,
            'video_form': vf,
            'target': targets_tensor
        }

    class _InnerDataset(Dataset):
        """
        Internal PyTorch Dataset wrapper matching the standard PyTorch API.
        """
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

            # Load video transform dictionary from transforms/video_transform.py
            data_transform = VideoTransform.get_transforms(image_size=parent.image_size)
            self.transform = data_transform[self.split]
            logger.info(f"Initialized '{self.split}' transformation pipeline.")

        def __len__(self) -> int:
            return len(self.data_dict)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            item = self.data_dict[index]
            video_name = item[1]
            target_val = item[2]

            if self.disk_cache_video:
                sample = _load_or_create_video_sample(
                    index=index,
                    video_path=video_name,
                    label=target_val,
                    image_size=self.parent.image_size,
                    frames_count=self.parent.frames_count,
                    disk_cache_video=True,
                    video_cache_dir=self.parent.video_cache_dir,
                    split=self.split
                )
                vf_raw = sample['video_form']
                target_val = sample['target']
            else:
                vf_raw = FishVideoDataLoader.extract_video_frames_segment_based(
                    video_path=video_name,
                    image_size=self.parent.image_size,
                    frames_count=self.parent.frames_count
                )

            vf = self.transform(vf_raw)

            if isinstance(target_val, (int, np.integer)):
                target = np.eye(4)[target_val]
            else:
                target = target_val

            return {
                'video_name': video_name,
                'video_form': vf,
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
    