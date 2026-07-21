import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def _load_video_worker(args: Tuple) -> Tuple[int, Dict[str, Any]]:
    """
    Module-level worker function for ProcessPoolExecutor.
    Must be a top-level function (not nested) to be picklable across processes.
    Uses decord GPU decode (following original author's pattern) for maximum throughput.
    Falls back to decord CPU decode if GPU context is unavailable.

    Args:
        args: Tuple of (idx, video_path, label, image_size, frames_count)

    Returns:
        Tuple of (idx, sample_dict)
    """
    idx, video_path, label, image_size, frames_count = args

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

    return idx, {
        'video_name': video_path,
        'video_form': vf_uint8,
        'target': label
    }


class FishVideoDataLoader:
    """
    Unified manager class for the fish video raw dataset (FishVideoDataLoader).
    Preloads extracted frames directly into System RAM at startup using Segment-based Sampling,
    completely bypassing disk pickle creation and eliminating GPU starvation.
    """
    def __init__(
        self,
        batch_size: int = 50,
        num_workers: int = -1,
        cache_video: bool = True,
        image_size: int = 224,
        frames_count: int = 4,
        splitter_config: Optional[SplitterConfig] = None
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_video = cache_video
        self.image_size = image_size
        self.frames_count = frames_count

        # Auto-detect optimal CPU worker threads if set to -1
        if self.num_workers == -1:
            max_cpu = os.cpu_count()
            if max_cpu is None or max_cpu <= 0:
                self.num_workers = 0
            elif max_cpu == 2:
                self.num_workers = max_cpu // 2
            else:
                self.num_workers = (max_cpu // 2) + 1

        # 1. Load splitter configurations and initialize the data splitter
        if splitter_config is None:
            self.splitter_config = SplitterConfig()
        else:
            self.splitter_config = splitter_config
        self.splitter = FishDataSplitter(config=self.splitter_config)

        # 2. Split dataset into train, val, and test partitions
        self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()

        logger.info("==================================================")
        logger.info("Initializing FishVideoDataLoader (Direct In-RAM Pipeline):")
        logger.info(f"  - Batch Size:               {self.batch_size}")
        logger.info(f"  - Num Workers:              {self.num_workers} (Auto-calculated from {os.cpu_count()} CPU cores)")
        logger.info(f"  - Direct RAM Caching:       {self.cache_video}")
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
        Internal PyTorch Dataset wrapper matching standard API with built-in Direct System RAM caching.
        """
        def __init__(self, parent: 'FishVideoDataLoader', split: str) -> None:
            self.parent = parent
            self.split = split
            self.cache_video = parent.cache_video
            self.video_cache = None

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

            if self.cache_video:
                self._preload_video_to_ram()

        def _preload_video_to_ram(self) -> None:
            # Reuse num_workers already calculated in FishVideoDataLoader.__init__
            max_workers = self.parent.num_workers

            logger.info(f"Starting direct MP4 -> RAM preload for split '{self.split}' ({len(self.data_dict)} samples)...")
            logger.info(f"Using ProcessPoolExecutor with {max_workers} workers ({os.cpu_count()} CPU cores detected).")

            # Build picklable args list for module-level worker function
            args_list = [
                (i, self.data_dict[i][1], self.data_dict[i][2],
                 self.parent.image_size, self.parent.frames_count)
                for i in range(len(self.data_dict))
            ]

            cache = {}

            # Safe check for tqdm library import
            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                has_tqdm = False

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_load_video_worker, args): args[0] for args in args_list}

                if has_tqdm:
                    pbar = tqdm(total=len(self.data_dict), desc=f"Preloading {self.split} to RAM")
                    for future in as_completed(futures):
                        try:
                            idx, result = future.result()
                            cache[idx] = result
                        except Exception as e:
                            idx = futures[future]
                            logger.error(f"Error loading video index {idx}: {e}")
                        pbar.update(1)
                    pbar.close()
                else:
                    for future in as_completed(futures):
                        try:
                            idx, result = future.result()
                            cache[idx] = result
                        except Exception as e:
                            idx = futures[future]
                            logger.error(f"Error loading video index {idx}: {e}")

            self.video_cache = cache
            logger.info(f"Successfully cached {len(cache)} video samples in System RAM for split '{self.split}'.")

        def __len__(self) -> int:
            return len(self.data_dict)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            if self.video_cache is not None:
                sample = self.video_cache[index]
                video_name = sample['video_name']
                vf_raw = sample['video_form']
                target_val = sample['target']
            else:
                item = self.data_dict[index]
                video_name = item[1]
                target_val = item[2]
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

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=1 if self.num_workers > 0 else None

        )
