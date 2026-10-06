"""Validated cache for preprocessed inputs, never for model predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class InputTensorCache:
    """Disk cache for exact audio/video tensors used by TensorRT batch-1 inference.

    A cache hit still runs TensorRT.  The metadata fingerprint prevents reuse when
    the preprocessing contract, checkpoint, or source sample identity changes.
    """

    def __init__(self, root: str | Path, enabled: bool) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sample_index: int) -> Path:
        return self.root / f"sample_{int(sample_index):05d}.npz"

    def load(self, sample_index: int, fingerprint: str) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.enabled:
            return None
        path = self._path(sample_index)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = str(archive["metadata"].item())
                if metadata != fingerprint:
                    return None
                audio = np.asarray(archive["audio_features"], dtype=np.float32)
                video = np.asarray(archive["video_form"], dtype=np.float32)
        except (OSError, ValueError, KeyError):
            return None
        if audio.shape != (1, 1, 128, 128) or video.shape != (1, 6, 224, 224):
            return None
        return np.ascontiguousarray(audio), np.ascontiguousarray(video)

    def save(self, sample_index: int, fingerprint: str, audio: np.ndarray, video: np.ndarray) -> None:
        if not self.enabled:
            return
        path = self._path(sample_index)
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    metadata=np.array(fingerprint),
                    audio_features=np.asarray(audio, dtype=np.float32),
                    video_form=np.asarray(video, dtype=np.float32),
                )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    @staticmethod
    def make_fingerprint(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
