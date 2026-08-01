import os
import sys
import pickle
import logging
import hashlib
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
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
from transforms import ClipVideoTransform, VideoTransform

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _resolve_worker_count(value: int) -> int:
    if value != -1:
        return value

    max_cpu = os.cpu_count()
    if max_cpu is None or max_cpu <= 0:
        return 0
    if max_cpu == 2:
        return max_cpu // 2
    return (max_cpu // 2) + 1


def _normalized_video_key(video_path: str) -> str:
    return os.path.normcase(os.path.normpath(str(video_path)))


def _stable_video_seed(global_seed: int, video_path: str) -> int:
    seed_material = f"{int(global_seed)}|{_normalized_video_key(video_path)}".encode("utf-8")
    digest = hashlib.sha256(seed_material).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _sample_segment_indices(total_frames: int, frames: int, seed: int) -> np.ndarray:
    if frames < 1:
        raise ValueError(f"frames must be positive, got {frames}.")
    if total_frames < frames:
        raise ValueError(
            f"Cannot sample {frames} non-empty temporal segments from a video with {total_frames} frames."
        )

    boundaries = np.linspace(0, total_frames, num=frames + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = np.empty(frames, dtype=np.int64)
    for segment_index in range(frames):
        start = int(boundaries[segment_index])
        stop = int(boundaries[segment_index + 1])
        if stop <= start:
            raise ValueError(
                f"Temporal segment {segment_index} is empty for total_frames={total_frames}, frames={frames}."
            )
        indices[segment_index] = rng.integers(start, stop)

    indices.sort()
    return indices


def _get_video_frame_count(video_path: str) -> int:
    try:
        from decord import VideoReader, cpu
        frame_count = len(VideoReader(video_path, ctx=cpu(0)))
    except Exception as decord_exc:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV could not open video after Decord failed: {decord_exc}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    if frame_count <= 0:
        raise ValueError(f"Video has no readable frames: '{video_path}'.")
    return frame_count


def _scan_video_frame_counts(video_paths: List[str], workers: int) -> Dict[str, int]:
    frame_counts: Dict[str, int] = {}
    scan_workers = max(1, workers)
    logger.info(f"Scanning frame counts for {len(video_paths)} videos with {scan_workers} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=scan_workers) as executor:
        futures = {
            executor.submit(_get_video_frame_count, video_path): video_path
            for video_path in video_paths
        }
        try:
            from tqdm import tqdm
            iterator = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Scanning video frame counts",
            )
        except ImportError:
            iterator = concurrent.futures.as_completed(futures)

        for future in iterator:
            video_path = futures[future]
            try:
                frame_counts[_normalized_video_key(video_path)] = int(future.result())
            except Exception as exc:
                raise RuntimeError(f"Failed to read frame count for '{video_path}': {exc}") from exc

    return frame_counts


def _resolve_effective_frames(
    configured_frames: int,
    minimum_required_frames: int,
    frame_counts: Dict[str, int],
) -> Tuple[int, int, str]:
    if not frame_counts:
        raise ValueError("Cannot resolve effective frames because the video dataset is empty.")

    shortest_path, minimum_video_frames = min(frame_counts.items(), key=lambda item: item[1])
    if minimum_video_frames < minimum_required_frames:
        raise ValueError(
            f"S3D requires at least {minimum_required_frames} frames, but video "
            f"'{shortest_path}' contains only {minimum_video_frames} frames."
        )

    requested_frames = max(configured_frames, minimum_required_frames)
    if configured_frames < minimum_required_frames:
        logger.warning(
            f"S3D requires at least {minimum_required_frames} frames. "
            f"Configured frames={configured_frames} has been adjusted to {minimum_required_frames}."
        )

    effective_frames = min(requested_frames, minimum_video_frames)
    if requested_frames > minimum_video_frames:
        logger.warning(
            f"Configured frames={requested_frames} exceeds the shortest video length="
            f"{minimum_video_frames}. Using frames={effective_frames} for all videos."
        )

    return effective_frames, minimum_video_frames, shortest_path


def _decode_clip_decord(video_path: str, indices: np.ndarray, image_size: int) -> np.ndarray:
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, width=image_size, height=image_size, ctx=cpu(0))
    if len(vr) <= int(indices[-1]):
        raise ValueError(
            f"Video frame count changed while decoding '{video_path}': "
            f"requested index {int(indices[-1])}, available frames {len(vr)}."
        )

    clip = vr.get_batch(indices.tolist()).asnumpy()
    expected_shape = (len(indices), image_size, image_size, 3)
    if clip.shape != expected_shape:
        raise ValueError(f"Decoded clip has shape {clip.shape}, expected {expected_shape} for '{video_path}'.")
    return np.ascontiguousarray(clip, dtype=np.uint8)


def _decode_clip_cv2(video_path: str, indices: np.ndarray, image_size: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: '{video_path}'.")

    frames: List[np.ndarray] = []
    try:
        for frame_index in indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()
            if not success:
                raise RuntimeError(f"OpenCV could not decode frame {frame_index} from '{video_path}'.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame.astype(np.uint8, copy=False))
    finally:
        cap.release()

    return np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)


def _decode_video_clip(
    video_path: str,
    total_frames: int,
    frames: int,
    image_size: int,
    global_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = _sample_segment_indices(
        total_frames=total_frames,
        frames=frames,
        seed=_stable_video_seed(global_seed=global_seed, video_path=video_path),
    )
    try:
        clip = _decode_clip_decord(video_path=video_path, indices=indices, image_size=image_size)
    except Exception as exc:
        logger.warning(f"Decord failed for '{video_path}', falling back to OpenCV. Error: {exc}")
        clip = _decode_clip_cv2(video_path=video_path, indices=indices, image_size=image_size)

    clip.setflags(write=False)
    return clip, indices


class VideoRAMCache:
    """One read-only uint8 clip cache shared by every split and CV fold."""

    def __init__(self) -> None:
        self.clips: Dict[str, np.ndarray] = {}
        self.frame_counts: Dict[str, int] = {}
        self.sampled_indices: Dict[str, np.ndarray] = {}
        self.effective_frames: Optional[int] = None
        self.minimum_video_frames: Optional[int] = None
        self.total_bytes = 0
        self._settings: Optional[Tuple[int, int, int, int]] = None

    def prepare(
        self,
        video_paths: List[str],
        configured_frames: int,
        minimum_required_frames: int,
        image_size: int,
        global_seed: int,
        preload_workers: int,
    ) -> int:
        unique_paths = sorted({_normalized_video_key(path): path for path in video_paths}.values())
        settings = (configured_frames, minimum_required_frames, image_size, global_seed)

        if self.clips:
            if self._settings != settings:
                raise ValueError(
                    f"Shared RAM cache was prepared with settings={self._settings}, "
                    f"but the current fold requested settings={settings}."
                )
            missing_paths = [path for path in unique_paths if _normalized_video_key(path) not in self.clips]
            if missing_paths:
                raise ValueError(
                    f"Shared RAM cache is missing {len(missing_paths)} videos; first missing path: '{missing_paths[0]}'."
                )
            logger.info(
                f"Reusing shared RAM cache: {len(self.clips)} clips, "
                f"{self.total_bytes / (1024 ** 3):.2f} GiB."
            )
            return int(self.effective_frames)

        self.frame_counts = _scan_video_frame_counts(unique_paths, workers=preload_workers)
        effective_frames, minimum_video_frames, shortest_path = _resolve_effective_frames(
            configured_frames=configured_frames,
            minimum_required_frames=minimum_required_frames,
            frame_counts=self.frame_counts,
        )
        self.effective_frames = effective_frames
        self.minimum_video_frames = minimum_video_frames
        self._settings = settings

        logger.info("==================================================")
        logger.info("Preparing shared S3D RAM cache:")
        logger.info(f"  - Videos:                    {len(unique_paths)}")
        logger.info(f"  - Configured Frames:         {configured_frames}")
        logger.info(f"  - Minimum S3D Frames:        {minimum_required_frames}")
        logger.info(f"  - Shortest Video Frames:     {minimum_video_frames}")
        logger.info(f"  - Shortest Video Path:       '{shortest_path}'")
        logger.info(f"  - Effective Frames:          {effective_frames}")
        logger.info(f"  - Sampling:                  one seeded random frame per temporal segment")
        logger.info(f"  - Global Seed:               {global_seed}")
        logger.info(f"  - Preload Workers:           {max(1, preload_workers)}")
        logger.info(f"  - Cache Format:              uint8 [T, H, W, C]")
        estimated_bytes = len(unique_paths) * effective_frames * image_size * image_size * 3
        logger.info(f"  - Estimated Cache Size:      {estimated_bytes / (1024 ** 3):.2f} GiB")
        logger.info("==================================================")

        preload_threads = max(1, preload_workers)

        def load_one(video_path: str) -> Tuple[str, np.ndarray, np.ndarray]:
            key = _normalized_video_key(video_path)
            clip, indices = _decode_video_clip(
                video_path=video_path,
                total_frames=self.frame_counts[key],
                frames=effective_frames,
                image_size=image_size,
                global_seed=global_seed,
            )
            return key, clip, indices

        with concurrent.futures.ThreadPoolExecutor(max_workers=preload_threads) as executor:
            futures = {executor.submit(load_one, path): path for path in unique_paths}
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Preloading S3D clips to RAM",
                )
            except ImportError:
                iterator = concurrent.futures.as_completed(futures)

            for future in iterator:
                video_path = futures[future]
                try:
                    key, clip, indices = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to preload video '{video_path}': {exc}") from exc
                self.clips[key] = clip
                self.sampled_indices[key] = indices
                self.total_bytes += clip.nbytes

        logger.info(
            f"Shared RAM cache ready: {len(self.clips)} clips, "
            f"{self.total_bytes / (1024 ** 3):.2f} GiB."
        )
        for video_path in unique_paths[:3]:
            key = _normalized_video_key(video_path)
            logger.info(f"Sampled indices for '{video_path}': {self.sampled_indices[key].tolist()}")

        return effective_frames

    def get(self, video_path: str) -> np.ndarray:
        key = _normalized_video_key(video_path)
        if key not in self.clips:
            raise KeyError(f"Video is not present in shared RAM cache: '{video_path}'.")
        return self.clips[key]


def _get_image_cache_subdir(image_size: int) -> str:
    return f"single_frame_size_{image_size}"


def _resolve_image_cache_dir(video_cache_dir: Optional[str], image_size: int) -> str:
    cache_root = os.path.normpath(video_cache_dir or DEFAULT_IMAGE_CACHE_ROOT)
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
    use_disk_cache: bool,
    image_cache_dir: Optional[str],
    split: str
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


def _get_clip_cache_dir(image_size: int, frames: int) -> str:
    return os.path.join(DEFAULT_IMAGE_CACHE_ROOT, f"clip_frames_{frames}_size_{image_size}")


def _get_clip_cache_path(cache_dir: str, video_path: str) -> str:
    path_hash = hashlib.sha256(_normalized_video_key(video_path).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{path_hash}.pkl")


def _load_clip_from_disk_cache(
    cache_path: str,
    video_path: str,
    frames: int,
    image_size: int,
    global_seed: int,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as cache_file:
            payload = pickle.load(cache_file)
    except Exception as exc:
        logger.warning(f"Could not read clip cache '{cache_path}'. It will be regenerated. Error: {exc}")
        return None

    clip = payload.get("video_form")
    metadata = payload.get("_cache_meta", {})
    expected_shape = (frames, image_size, image_size, 3)
    if payload.get("video_name") != video_path:
        return None
    if metadata.get("frames") != frames or metadata.get("image_size") != image_size:
        return None
    if metadata.get("global_seed") != global_seed:
        return None
    if not isinstance(clip, np.ndarray) or clip.shape != expected_shape or clip.dtype != np.uint8:
        return None

    clip.setflags(write=False)
    return {
        "video_name": video_path,
        "video_form": clip,
        "target": payload["target"],
    }


def _save_clip_to_disk_cache(
    cache_path: str,
    sample: Dict[str, Any],
    frames: int,
    image_size: int,
    global_seed: int,
) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "video_name": sample["video_name"],
        "video_form": sample["video_form"],
        "target": sample["target"],
        "_cache_meta": {
            "frames": frames,
            "image_size": image_size,
            "global_seed": global_seed,
            "format": "uint8_THWC",
            "frame_policy": "seeded_random_per_temporal_segment",
        },
    }
    temporary_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "wb") as cache_file:
            pickle.dump(payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, cache_path)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def _load_or_create_clip_sample(
    video_path: str,
    label: Any,
    total_frames: int,
    frames: int,
    image_size: int,
    global_seed: int,
    use_disk_cache: bool,
) -> Dict[str, Any]:
    cache_path = None
    if use_disk_cache:
        cache_dir = _get_clip_cache_dir(image_size=image_size, frames=frames)
        cache_path = _get_clip_cache_path(cache_dir=cache_dir, video_path=video_path)
        cached = _load_clip_from_disk_cache(
            cache_path=cache_path,
            video_path=video_path,
            frames=frames,
            image_size=image_size,
            global_seed=global_seed,
        )
        if cached is not None:
            cached["target"] = label
            return cached

    clip, _ = _decode_video_clip(
        video_path=video_path,
        total_frames=total_frames,
        frames=frames,
        image_size=image_size,
        global_seed=global_seed,
    )
    sample = {
        "video_name": video_path,
        "video_form": clip,
        "target": label,
    }
    if cache_path is not None:
        _save_clip_to_disk_cache(
            cache_path=cache_path,
            sample=sample,
            frames=frames,
            image_size=image_size,
            global_seed=global_seed,
        )
    return sample


class FishVideoDataLoader:
    """DataLoader manager supporting both center-frame image and S3D clip inputs."""

    def __init__(
        self,
        batch_size: int = 8,
        preload_workers: int = -1,
        dataloader_workers: int = -1,
        cache_mode: str = "ram",
        image_size: int = 224,
        frames: int = 20,
        clip_mode: bool = False,
        minimum_required_frames: int = 1,
        shared_ram_cache: Optional[VideoRAMCache] = None,
        splitter_config: Optional[SplitterConfig] = None,
    ) -> None:
        self.batch_size = batch_size
        self.preload_workers = _resolve_worker_count(preload_workers)
        self.dataloader_workers = _resolve_worker_count(dataloader_workers)
        self.cache_mode = cache_mode.lower()
        if self.cache_mode not in VALID_CACHE_MODES:
            raise ValueError(f"Invalid cache_mode='{cache_mode}'. Expected one of {sorted(VALID_CACHE_MODES)}.")

        self.image_size = image_size
        self.configured_frames = frames
        self.clip_mode = clip_mode
        self.minimum_required_frames = minimum_required_frames
        self.image_cache_root = DEFAULT_IMAGE_CACHE_ROOT
        self.image_cache_dir = _resolve_image_cache_dir(video_cache_dir=None, image_size=image_size)
        self.shared_ram_cache = shared_ram_cache
        self.frame_counts: Dict[str, int] = {}
        self.effective_frames = 1

        self.splitter_config = splitter_config if splitter_config is not None else SplitterConfig()
        self.splitter = FishDataSplitter(config=self.splitter_config)
        self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()

        all_entries = self.train_dict + self.val_dict + self.test_dict
        all_video_paths = [item[1] for item in all_entries if len(item) >= 3 and item[1]]

        if self.clip_mode:
            if self.cache_mode == "ram":
                if self.shared_ram_cache is None:
                    self.shared_ram_cache = VideoRAMCache()
                self.effective_frames = self.shared_ram_cache.prepare(
                    video_paths=all_video_paths,
                    configured_frames=self.configured_frames,
                    minimum_required_frames=self.minimum_required_frames,
                    image_size=self.image_size,
                    global_seed=self.splitter_config.seed,
                    preload_workers=self.preload_workers,
                )
                self.frame_counts = self.shared_ram_cache.frame_counts
            else:
                unique_paths = sorted({_normalized_video_key(path): path for path in all_video_paths}.values())
                self.frame_counts = _scan_video_frame_counts(unique_paths, workers=self.preload_workers)
                self.effective_frames, _, _ = _resolve_effective_frames(
                    configured_frames=self.configured_frames,
                    minimum_required_frames=self.minimum_required_frames,
                    frame_counts=self.frame_counts,
                )

        start_method = multiprocessing.get_context().get_start_method()
        logger.info("==================================================")
        logger.info("Initializing FishVideoDataLoader:")
        logger.info(f"  - Pipeline:                 {'S3D clip' if self.clip_mode else 'single-frame image'}")
        logger.info(f"  - Batch Size:               {self.batch_size}")
        logger.info(f"  - Preload Workers:          {self.preload_workers}")
        logger.info(f"  - DataLoader Workers:       {self.dataloader_workers}")
        logger.info(f"  - DataLoader Prefetch:      {'disabled' if self.dataloader_workers <= 0 else 'PyTorch default'}")
        logger.info(f"  - Multiprocessing Method:   {start_method}")
        logger.info(f"  - Cache Mode:               {self.cache_mode}")
        logger.info(f"  - Image Resolution:         {self.image_size}x{self.image_size}")
        logger.info(f"  - Configured Frames:        {self.configured_frames if self.clip_mode else 1}")
        logger.info(f"  - Effective Frames:         {self.effective_frames}")
        logger.info(
            "  - Frame Policy:             "
            + ("one seeded random frame per temporal segment" if self.clip_mode else "center frame")
        )
        logger.info("==================================================")

        if (
            self.clip_mode
            and self.cache_mode == "ram"
            and self.dataloader_workers > 0
            and start_method != "fork"
        ):
            logger.warning(
                f"DataLoader start method is '{start_method}', not 'fork'. A large RAM cache may be copied "
                "into worker processes. Use a shared-memory backend or set dataloader_workers=0 if RAM grows."
            )

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        video_names = [data["video_name"] for data in batch]
        videos = torch.stack([data["video_form"] for data in batch])
        targets_tensor = torch.as_tensor(np.asarray([data["target"] for data in batch]), dtype=torch.float32)
        return {
            "video_name": video_names,
            "video_form": videos,
            "target": targets_tensor,
        }

    class _InnerDataset(Dataset):
        def __init__(self, parent: "FishVideoDataLoader", split: str) -> None:
            self.parent = parent
            self.split = split
            self.cache_mode = parent.cache_mode
            self.image_cache = None

            if split == "train":
                self.data_dict = parent.train_dict
            elif split == "test":
                self.data_dict = parent.test_dict
            elif split == "val":
                self.data_dict = parent.val_dict
            else:
                raise ValueError(f"Invalid split value '{split}'. Must be one of ['train', 'test', 'val'].")

            if parent.clip_mode:
                transforms_by_split = ClipVideoTransform.get_transforms()
                logger.info(f"Initialized '{split}' S3D clip transformation pipeline.")
            else:
                transforms_by_split = VideoTransform.get_transforms(image_size=parent.image_size)
                logger.info(f"Initialized '{split}' image transformation pipeline.")
            self.transform = transforms_by_split[split]

            if self.cache_mode == "ram" and not parent.clip_mode:
                self._preload_images_to_ram()

        def _preload_images_to_ram(self) -> None:
            preload_workers = max(1, self.parent.preload_workers)
            logger.info(f"Starting center-frame RAM preload for '{self.split}' with {preload_workers} threads...")

            def load_single_image(index_and_item: tuple) -> tuple:
                index, item = index_and_item
                if len(item) < 3 or not item[1]:
                    raise ValueError(f"Missing video path for split='{self.split}', index={index}.")
                try:
                    sample = _decode_center_image(item[1], item[2], self.parent.image_size)
                except Exception as exc:
                    logger.warning(f"Decord failed for '{item[1]}', falling back to OpenCV. Error: {exc}")
                    sample = _decode_center_image_cv2(item[1], item[2], self.parent.image_size)
                sample["image_form"].setflags(write=False)
                return index, sample

            cache = [None] * len(self.data_dict)
            total_bytes = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=preload_workers) as executor:
                futures = {
                    executor.submit(load_single_image, indexed_item): indexed_item[0]
                    for indexed_item in enumerate(self.data_dict)
                }
                try:
                    from tqdm import tqdm
                    iterator = tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"Preloading {self.split} images to RAM",
                    )
                except ImportError:
                    iterator = concurrent.futures.as_completed(futures)

                for future in iterator:
                    index = futures[future]
                    try:
                        result_index, sample = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to preload center frame for split='{self.split}', index={index}: {exc}"
                        ) from exc
                    cache[result_index] = sample
                    total_bytes += sample["image_form"].nbytes

            self.image_cache = cache
            logger.info(
                f"Cached {self.split} split to RAM: {len(cache)} images, "
                f"{total_bytes / (1024 ** 2):.1f} MB."
            )

        def __len__(self) -> int:
            return len(self.data_dict)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            item = self.data_dict[index]
            if len(item) < 3:
                raise ValueError("Video training requires [audio_path, video_path, label] split entries.")

            video_name = item[1]
            target_value = item[2]
            if not video_name:
                raise ValueError(f"Missing video path for split='{self.split}', index={index}.")

            if self.parent.clip_mode:
                if self.cache_mode == "ram":
                    if self.parent.shared_ram_cache is None:
                        raise RuntimeError("S3D RAM cache was not initialized.")
                    raw_video = self.parent.shared_ram_cache.get(video_name)
                else:
                    key = _normalized_video_key(video_name)
                    raw_sample = _load_or_create_clip_sample(
                        video_path=video_name,
                        label=target_value,
                        total_frames=self.parent.frame_counts[key],
                        frames=self.parent.effective_frames,
                        image_size=self.parent.image_size,
                        global_seed=self.parent.splitter_config.seed,
                        use_disk_cache=self.cache_mode == "disk",
                    )
                    raw_video = raw_sample["video_form"]
                video_tensor = self.transform(raw_video)
            else:
                if self.image_cache is not None:
                    sample = self.image_cache[index]
                elif self.cache_mode == "disk":
                    sample = _load_or_create_image_sample(
                        index=index,
                        video_path=video_name,
                        label=target_value,
                        image_size=self.parent.image_size,
                        use_disk_cache=True,
                        image_cache_dir=self.parent.image_cache_dir,
                        split=self.split,
                    )
                else:
                    try:
                        sample = _decode_center_image(video_name, target_value, self.parent.image_size)
                    except Exception as exc:
                        logger.warning(f"Decord failed for '{video_name}', falling back to OpenCV. Error: {exc}")
                        sample = _decode_center_image_cv2(video_name, target_value, self.parent.image_size)
                video_tensor = self.transform(sample["image_form"])

            target = np.eye(4)[target_value] if isinstance(target_value, (int, np.integer)) else target_value
            return {
                "video_name": video_name,
                "video_form": video_tensor,
                "target": target,
            }

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        dataset = self._InnerDataset(parent=self, split=split)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.dataloader_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.dataloader_workers > 0,
        )

    @staticmethod
    def shutdown_dataloader(data_loader: DataLoader) -> None:
        iterator = getattr(data_loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            data_loader._iterator = None
