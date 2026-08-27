import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import concurrent.futures

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DEFAULT_IMAGE_CACHE_ROOT, VALID_CACHE_MODES
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


def _get_image_cache_subdir(image_size: int, frame_policy: str) -> str:
    return f"single_frame_size_{image_size}_{frame_policy}"


def _resolve_image_cache_dir(video_cache_dir: Optional[str], image_size: int, frame_policy: str) -> str:
    cache_root = os.path.normpath(video_cache_dir or DEFAULT_IMAGE_CACHE_ROOT)
    cache_subdir = _get_image_cache_subdir(image_size=image_size, frame_policy=frame_policy)

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
    image_size: int,
    frame_policy: str
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
            "frame_policy": frame_policy,
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


def _get_multi_frame_indices(full_vid_length: int, frame_policy: str) -> list:
    if full_vid_length == 0:
        return [0]
    
    q = full_vid_length // 4
    c = full_vid_length // 2
    tq = 3 * full_vid_length // 4
    e = full_vid_length - 1 if full_vid_length > 0 else 0

    if frame_policy == "quarter_end":
        return [q, e]
    elif frame_policy == "quarter_center_end":
        return [q, c, e]
    elif frame_policy == "quarter_center_three_quarters_end":
        return [q, c, tq, e]
    else:
        raise ValueError(f"Unknown multi-channel frame policy: {frame_policy}")

def _decode_image(video_path: str, label: Any, image_size: int, frame_policy: str, split: str) -> Dict[str, Any]:
    from decord import VideoReader, cpu, gpu

    try:
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=gpu(0))
    except Exception:
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=cpu(0))

    full_vid_length = len(vr)
    
    if frame_policy == "quarter_end":
        channels = 6
    elif frame_policy == "quarter_center_end":
        channels = 9
    elif frame_policy == "quarter_center_three_quarters_end":
        channels = 12
    else:
        raise ValueError(f"Unknown multi-channel frame policy: {frame_policy}")

    if full_vid_length == 0:
        image_uint8 = np.zeros((channels, image_size, image_size), dtype=np.uint8)
    else:
        indices = _get_multi_frame_indices(full_vid_length, frame_policy)
        batch = vr.get_batch(indices).asnumpy()  # [N, H, W, 3] RGB
        frames = [batch[i] for i in range(len(indices))]
        image = np.concatenate(frames, axis=-1)  # [H, W, Channels]
        image_uint8 = image.transpose(2, 0, 1).astype(np.uint8)

    return {
        'video_name': video_path,
        'image_form': image_uint8,
        'target': label
    }

def _decode_image_cv2(video_path: str, label: Any, image_size: int, frame_policy: str, split: str) -> Dict[str, Any]:
    import cv2
    import numpy as np
    
    if frame_policy == "quarter_end":
        channels = 6
    elif frame_policy == "quarter_center_end":
        channels = 9
    elif frame_policy == "quarter_center_three_quarters_end":
        channels = 12
    else:
        raise ValueError(f"Unknown multi-channel frame policy: {frame_policy}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file: '{video_path}'")
        image_uint8 = np.zeros((channels, image_size, image_size), dtype=np.uint8)
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            image_uint8 = np.zeros((channels, image_size, image_size), dtype=np.uint8)
        else:
            indices = _get_multi_frame_indices(frame_count, frame_policy)
            frames = []
            valid = True
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    valid = False
                    break
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (image_size, image_size))
                frames.append(frame)
            
            if not valid:
                image_uint8 = np.zeros((channels, image_size, image_size), dtype=np.uint8)
            else:
                image = np.concatenate(frames, axis=-1)
                image_uint8 = image.transpose(2, 0, 1).astype(np.uint8)
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
    use_disk_cache: bool,
    image_cache_dir: Optional[str],
    split: str,
    frame_policy: str
) -> Dict[str, Any]:
    cache_path = _get_image_cache_path(image_cache_dir, split, index) if use_disk_cache else None

    cached_sample = _load_image_from_disk_cache(
        cache_path=cache_path,
        video_path=video_path,
        image_size=image_size,
        label=label
    )
    if cached_sample is not None:
        return cached_sample

    try:
        sample = _decode_image(video_path=video_path, label=label, image_size=image_size, frame_policy=frame_policy, split=split)
    except Exception as exc:
        logger.warning(f"Decord failed for '{video_path}', falling back to OpenCV. Error: {exc}")
        sample = _decode_image_cv2(video_path=video_path, label=label, image_size=image_size, frame_policy=frame_policy, split=split)

    _save_image_to_disk_cache(
        cache_path=cache_path,
        sample=sample,
        image_size=image_size,
        frame_policy=frame_policy
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
        dataloader_workers: int = -1,
        prefetch_factor: Optional[int] = None,
        cache_mode: str = "disk",
        image_size: int = 224,
        frame_policy: str = "center",
        splitter_config: Optional[SplitterConfig] = None
    ) -> None:
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.prefetch_factor = prefetch_factor
        self.cache_mode = cache_mode.lower()
        if self.cache_mode not in VALID_CACHE_MODES:
            raise ValueError(f"Invalid cache_mode='{cache_mode}'. Expected one of {sorted(VALID_CACHE_MODES)}.")
        self.image_size = image_size
        self.frame_policy = frame_policy
        self.image_cache_root = DEFAULT_IMAGE_CACHE_ROOT
        self.image_cache_dir = _resolve_image_cache_dir(video_cache_dir=None, image_size=image_size, frame_policy=frame_policy)

        if self.dataloader_workers == -1:
            max_cpu = os.cpu_count()
            if max_cpu is None or max_cpu <= 0:
                self.dataloader_workers = 0
            elif max_cpu == 2:
                self.dataloader_workers = max_cpu // 2
            else:
                self.dataloader_workers = (max_cpu // 2) + 1

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
        logger.info(f"  - Cache Mode:               {self.cache_mode}")
        logger.info(f"  - Disk PKL Cache Root:      '{self.image_cache_root if self.cache_mode == 'disk' else 'disabled'}'")
        logger.info(f"  - Disk PKL Cache Dir:       '{self.image_cache_dir if self.cache_mode == 'disk' else 'disabled'}'")
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
            self.cache_mode = parent.cache_mode
            self.image_cache = None
            self.cache_size_mb = 0.0

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

            if self.cache_mode == "ram":
                self._preload_images_to_ram()

        def _preload_images_to_ram(self) -> None:
            """
            Preload decoded uint8 center-frame images directly into system RAM.
            Transform/augmentation is still applied lazily in __getitem__.
            """
            preload_workers = self.parent.dataloader_workers
            if preload_workers <= 0:
                preload_workers = 1

            logger.info(f"Starting MP4 -> RAM image preload for split '{self.split}' ({len(self.data_dict)} samples)...")
            logger.info(f"Using ThreadPoolExecutor with {preload_workers} workers for RAM preload.")

            def load_single_image(index_and_item: tuple) -> tuple:
                index, item = index_and_item
                if len(item) < 3:
                    raise ValueError("Video training requires split entries in [audio_path, video_path, label] format.")

                video_name = item[1]
                target_val = item[2]
                if not video_name:
                    raise ValueError(f"Missing video path for split='{self.split}', index={index}.")

                try:
                    sample = _decode_image(video_path=video_name, label=target_val, image_size=self.parent.image_size, frame_policy=self.parent.frame_policy, split=self.split)
                except Exception as exc:
                    logger.warning(f"Decord failed for '{video_name}', falling back to OpenCV. Error: {exc}")
                    sample = _decode_image_cv2(video_path=video_name, label=target_val, image_size=self.parent.image_size, frame_policy=self.parent.frame_policy, split=self.split)

                return index, sample

            cache = [None] * len(self.data_dict)
            total_bytes = 0

            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                has_tqdm = False

            indexed_items = list(enumerate(self.data_dict))
            with concurrent.futures.ThreadPoolExecutor(max_workers=preload_workers) as executor:
                futures = {
                    executor.submit(load_single_image, indexed_item): indexed_item[0]
                    for indexed_item in indexed_items
                }

                iterator = concurrent.futures.as_completed(futures)
                if has_tqdm:
                    pbar = tqdm(total=len(futures), desc=f"Preloading {self.split} images to RAM")
                    for future in iterator:
                        idx = futures[future]
                        try:
                            result_idx, sample = future.result()
                            cache[result_idx] = sample
                            total_bytes += sample["image_form"].nbytes
                        except Exception as exc:
                            logger.error(f"Error preloading image index {idx}: {exc}")
                            item = self.data_dict[idx]
                            cache[idx] = _decode_image_cv2(video_path=item[1], label=item[2], image_size=self.parent.image_size, frame_policy=self.parent.frame_policy, split=self.split)
                            total_bytes += cache[idx]["image_form"].nbytes
                        pbar.update(1)
                    pbar.close()
                else:
                    for future in iterator:
                        idx = futures[future]
                        try:
                            result_idx, sample = future.result()
                            cache[result_idx] = sample
                            total_bytes += sample["image_form"].nbytes
                        except Exception as exc:
                            logger.error(f"Error preloading image index {idx}: {exc}")
                            item = self.data_dict[idx]
                            cache[idx] = _decode_image_cv2(video_path=item[1], label=item[2], image_size=self.parent.image_size, frame_policy=self.parent.frame_policy, split=self.split)
                            total_bytes += cache[idx]["image_form"].nbytes

            self.image_cache = cache
            self.cache_size_mb = total_bytes / (1024 ** 2)
            logger.info(
                f"Cached {self.split} split to RAM: {len(cache)} images, {self.cache_size_mb:.1f} MB"
            )

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

            if self.image_cache is not None:
                sample = self.image_cache[index]
            elif self.cache_mode == "disk":
                sample = _load_or_create_image_sample(
                    index=index,
                    video_path=video_name,
                    label=target_val,
                    image_size=self.parent.image_size,
                    use_disk_cache=True,
                    image_cache_dir=self.parent.image_cache_dir,
                    split=self.split,
                    frame_policy=self.parent.frame_policy
                )
            else:
                try:
                    sample = _decode_image(video_path=video_name, label=target_val, image_size=self.parent.image_size, frame_policy=self.parent.frame_policy, split=self.split)
                except Exception as exc:
                    logger.warning(f"Decord failed for '{video_name}', falling back to OpenCV. Error: {exc}")
                    sample = _decode_image_cv2(video_path=video_name, label=target_val, image_size=self.parent.image_size, frame_policy=self.parent.frame_policy, split=self.split)

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
