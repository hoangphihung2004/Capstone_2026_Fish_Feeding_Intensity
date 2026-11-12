"""
model.py - THE ONLY FILE YOU NEED TO EDIT.

The server (app.py) calls these functions for every 2.0 s window and measures
the latency of each stage itself:

    load(weights_dir)                          once, at startup
    preprocess_audio(waveform)       -> a      timed as latency.audio_ms
    preprocess_video(frame_first, frame_last) -> v   timed as latency.video_ms
    infer(a, v)                      -> 4 logits      timed as latency.gpu_ms

Class order is fixed: [none, strong, medium, weak].

Inputs:
    waveform     np.float32 [128000]        2.0 s of mono audio at 64 kHz
    frame_first  np.uint8   [H, W, 3] BGR   first frame of the window (native resolution, e.g. 2560x1440)
    frame_last   np.uint8   [H, W, 3] BGR   last frame of the window

If your model only uses one modality, return None from the other preprocess
function and ignore that argument in infer().

This file currently contains a DEMO model that returns random scores so the
dashboard can be tested before the real model is deployed. Replace it.
See examples/model_aquaffia_reference.py for a complete TensorRT example.
"""

import cv2
import numpy as np

# Shown by GET /api/status. Fill in with your model's details.
INFO = {
    "name": "Demo model (placeholder)",
    "runtime": "Demo - random scores, replace model.py",
    "audio_frontend_device": "cpu",
    "video_frontend_device": "cpu",
    "fold": "",
    "accuracy_baseline": "",
}

# Set to True if infer() already returns probabilities instead of logits.
OUTPUT_IS_PROBABILITY = False


def load(weights_dir: str) -> None:
    """Load engines / checkpoints from weights_dir (mounted from ./weights)."""
    pass


def preprocess_audio(waveform: np.ndarray):
    """waveform: float32 [128000] at 64 kHz."""
    return waveform


def preprocess_video(frame_first: np.ndarray, frame_last: np.ndarray):
    """frame_first / frame_last: uint8 BGR frames at native resolution."""
    f0 = cv2.resize(frame_first, (224, 224), interpolation=cv2.INTER_LINEAR)
    f1 = cv2.resize(frame_last, (224, 224), interpolation=cv2.INTER_LINEAR)
    return np.stack([f0, f1], axis=0)


def infer(audio_input, video_input) -> np.ndarray:
    """Return 4 logits (or probabilities if OUTPUT_IS_PROBABILITY) in order [none, strong, medium, weak]."""
    seed = int(abs(float(np.sum(audio_input[:4000]))) * 1e6) % (2**32)
    return np.random.default_rng(seed).normal(0.0, 2.0, size=4).astype(np.float32)
