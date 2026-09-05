import concurrent.futures
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DEFAULT_IMAGE_CACHE_ROOT, VALID_CACHE_MODES, SplitterConfig
from dataset import FishDataSplitter
from transforms import VideoTransform

logger = logging.getLogger(__name__)

VIDEO_FRAME_POLICY = "first_last"
VIDEO_CHANNELS = 6


def _resolve_workers(value: int) -> int:
    if value >= 0:
        return value
    max_cpu = os.cpu_count()
    if max_cpu is None or max_cpu <= 0:
        return 0
    if max_cpu == 2:
        return max_cpu // 2
    return (max_cpu // 2) + 1


def _first_last_cache_dir(image_size: int, cache_root: Optional[str] = None) -> str:
    return os.path.join(os.path.normpath(cache_root or DEFAULT_IMAGE_CACHE_ROOT), f"{VIDEO_FRAME_POLICY}_size_{image_size}")


def _cache_path(cache_dir: Optional[str], split: str, index: int) -> Optional[str]:
    if not cache_dir:
        return None
    return os.path.join(cache_dir, split, f"{index}.pkl")


def _load_video_cache(cache_path: Optional[str], video_path: str, image_size: int, label: int) -> Optional[Dict[str, Any]]:
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
    if sample.get("video_name") != video_path:
        return None
    if sample.get("target") != label:
        return None
    if meta.get("image_size") != image_size or meta.get("frame_policy") != VIDEO_FRAME_POLICY:
        return None
    if not isinstance(video_form, np.ndarray) or video_form.shape != (VIDEO_CHANNELS, image_size, image_size) or video_form.dtype != np.uint8:
        return None
    return {"video_name": video_path, "video_form": video_form, "target": label}


def _save_video_cache(cache_path: Optional[str], sample: Dict[str, Any], image_size: int) -> None:
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "video_name": sample["video_name"],
        "video_form": sample["video_form"],
        "target": int(sample["target"]),
        "_cache_meta": {
            "image_size": image_size,
            "format": "uint8_CHW",
            "frame_policy": VIDEO_FRAME_POLICY,
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


def _first_last_indices(full_vid_length: int) -> List[int]:
    if full_vid_length <= 0:
        return [0, 0]
    return [0, full_vid_length - 1]


def decode_first_last_video(video_path: str, label: int, image_size: int) -> Dict[str, Any]:
    from decord import VideoReader, cpu, gpu

    try:
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=gpu(0))
    except Exception:
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=cpu(0))

    if len(vr) <= 0:
        video_uint8 = np.zeros((VIDEO_CHANNELS, image_size, image_size), dtype=np.uint8)
    else:
        batch = vr.get_batch(_first_last_indices(len(vr))).asnumpy()
        image = np.concatenate([batch[0], batch[1]], axis=-1)
        video_uint8 = image.transpose(2, 0, 1).astype(np.uint8)

    return {"video_name": video_path, "video_form": video_uint8, "target": int(label)}


def decode_first_last_video_cv2(video_path: str, label: int, image_size: int) -> Dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: '{video_path}'")
        video_uint8 = np.zeros((VIDEO_CHANNELS, image_size, image_size), dtype=np.uint8)
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for frame_index in _first_last_indices(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                frames = []
                break
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (image_size, image_size))
            frames.append(frame)
        cap.release()
        if len(frames) != 2:
            video_uint8 = np.zeros((VIDEO_CHANNELS, image_size, image_size), dtype=np.uint8)
        else:
            video_uint8 = np.concatenate(frames, axis=-1).transpose(2, 0, 1).astype(np.uint8)

    return {"video_name": video_path, "video_form": video_uint8, "target": int(label)}


def load_audio(path: str, sr: int = 64000) -> torch.Tensor:
    import torchaudio

    try:
        waveform, original_sr = torchaudio.load(path)
    except ImportError:
        # New torchaudio releases require TorchCodec; PCM WAV can use SoundFile.
        import soundfile as sf

        samples, original_sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sr != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sr)(waveform)

    y = waveform.squeeze(0).to(torch.float32)
    target_length = sr * 2
    if y.numel() > target_length:
        y = y[:target_length]
    elif y.numel() < target_length:
        y = torch.nn.functional.pad(y, (0, target_length - y.numel()))
    return y


class FishMultimodalDataLoader:
    """
    DataLoader manager for paired audio-video fish feeding intensity samples.

    Audio follows the raw waveform RAM-cache baseline. Video is fixed to the
    best 1_49_channels policy: first RGB frame + last RGB frame => 6 channels.
    """

    def __init__(
        self,
        sample_rate: int = 64000,
        batch_size: int = 128,
        dataloader_workers: int = -1,
        prefetch_factor: Optional[int] = None,
        cache_audio: bool = True,
        cache_video_mode: str = "ram",
        image_size: int = 224,
        splitter_config: Optional[SplitterConfig] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.dataloader_workers = _resolve_workers(dataloader_workers)
        self.prefetch_factor = prefetch_factor
        self.cache_audio = cache_audio
        self.cache_video_mode = cache_video_mode.lower()
        if self.cache_video_mode not in VALID_CACHE_MODES:
            raise ValueError(f"Invalid cache_video_mode='{cache_video_mode}'. Expected {sorted(VALID_CACHE_MODES)}.")
        self.image_size = image_size
        self.video_cache_dir = _first_last_cache_dir(image_size=image_size)

        self.splitter_config = splitter_config if splitter_config is not None else SplitterConfig()
        self.splitter_config.include_video = True
        self.splitter = FishDataSplitter(config=self.splitter_config)
        self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()

        logger.info("==================================================")
        logger.info("Initializing FishMultimodalDataLoader:")
        logger.info(f"  - Sample Rate:              {self.sample_rate} Hz")
        logger.info(f"  - Batch Size:               {self.batch_size}")
        logger.info(f"  - DataLoader Workers:       {self.dataloader_workers}")
        logger.info(f"  - Cache audio in RAM:       {self.cache_audio}")
        logger.info(f"  - Cache video mode:         {self.cache_video_mode}")
        logger.info(f"  - Video input policy:       {VIDEO_FRAME_POLICY}")
        logger.info(f"  - Image Resolution:         {self.image_size}x{self.image_size}")
        logger.info("==================================================")

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "audio_name": [item["audio_name"] for item in batch],
            "video_name": [item["video_name"] for item in batch],
            "waveform": torch.tensor(np.stack([item["waveform"] for item in batch]), dtype=torch.float32),
            "video_form": torch.stack([item["video_form"] for item in batch]),
            "target": torch.tensor(np.stack([item["target"] for item in batch]), dtype=torch.float32),
        }

    class _InnerDataset(Dataset):
        def __init__(self, parent: "FishMultimodalDataLoader", split: str) -> None:
            self.parent = parent
            self.split = split
            self.data_dict = {
                "train": parent.train_dict,
                "test": parent.test_dict,
                "val": parent.val_dict,
            }.get(split)
            if self.data_dict is None:
                raise ValueError(f"Invalid split value '{split}'. Must be one of ['train', 'test', 'val'].")
            self.transform = VideoTransform.get_transforms(parent.image_size)[split]
            self.waveform_cache: List[np.ndarray] | None = None
            self.video_cache: List[Dict[str, Any]] | None = None
            if parent.cache_audio:
                self._preload_audio()
            if parent.cache_video_mode == "ram":
                self._preload_video()

        def _preload_audio(self) -> None:
            logger.info(f"Preloading audio split '{self.split}' to RAM ({len(self.data_dict)} samples)...")
            workers = max(self.parent.dataloader_workers, 1)

            def load_one(index: int) -> tuple[int, np.ndarray]:
                audio_path = self.data_dict[index][0]
                return index, load_audio(audio_path, sr=self.parent.sample_rate).numpy()

            cache: List[np.ndarray | None] = [None] * len(self.data_dict)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(load_one, index): index for index in range(len(self.data_dict))}
                completed = concurrent.futures.as_completed(futures)
                for future in tqdm(
                    completed,
                    total=len(futures),
                    desc=f"Preloading {self.split} audio to RAM",
                    unit="file",
                ):
                    index, waveform = future.result()
                    cache[index] = waveform
            if any(item is None for item in cache):
                raise RuntimeError(f"Audio RAM preload failed for split '{self.split}'.")
            self.waveform_cache = [item for item in cache if item is not None]

        def _decode_video_row(self, index: int) -> Dict[str, Any]:
            _, video_path, label = self.data_dict[index]
            try:
                return decode_first_last_video(video_path=video_path, label=label, image_size=self.parent.image_size)
            except Exception as exc:
                logger.warning(f"Decord failed for '{video_path}', falling back to OpenCV. Error: {exc}")
                return decode_first_last_video_cv2(video_path=video_path, label=label, image_size=self.parent.image_size)

        def _preload_video(self) -> None:
            logger.info(f"Preloading first-last video split '{self.split}' to RAM ({len(self.data_dict)} samples)...")
            workers = max(self.parent.dataloader_workers, 1)
            cache: List[Dict[str, Any] | None] = [None] * len(self.data_dict)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._decode_video_row, index): index for index in range(len(self.data_dict))}
                completed = concurrent.futures.as_completed(futures)
                for future in tqdm(
                    completed,
                    total=len(futures),
                    desc=f"Preloading {self.split} video to RAM",
                    unit="file",
                ):
                    cache[futures[future]] = future.result()
            if any(item is None for item in cache):
                raise RuntimeError(f"Video RAM preload failed for split '{self.split}'.")
            self.video_cache = [item for item in cache if item is not None]

        def __len__(self) -> int:
            return len(self.data_dict)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            audio_path, video_path, label = self.data_dict[index]
            if self.waveform_cache is not None:
                waveform = self.waveform_cache[index]
            else:
                waveform = load_audio(audio_path, sr=self.parent.sample_rate).numpy()

            if self.video_cache is not None:
                video_sample = self.video_cache[index]
            elif self.parent.cache_video_mode == "disk":
                cache_path = _cache_path(self.parent.video_cache_dir, self.split, index)
                video_sample = _load_video_cache(cache_path, video_path, self.parent.image_size, int(label))
                if video_sample is None:
                    video_sample = self._decode_video_row(index)
                    _save_video_cache(cache_path, video_sample, self.parent.image_size)
            else:
                video_sample = self._decode_video_row(index)

            return {
                "audio_name": audio_path,
                "video_name": video_path,
                "waveform": waveform,
                "video_form": self.transform(video_sample["video_form"]),
                "target": np.eye(4)[int(label)],
            }

    def get_dataloader(self, split: str, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        dataset = self._InnerDataset(parent=self, split=split)
        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "num_workers": self.dataloader_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": self.dataloader_workers > 0,
        }
        if self.dataloader_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)
