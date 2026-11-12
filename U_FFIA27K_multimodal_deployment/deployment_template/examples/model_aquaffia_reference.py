"""
Reference model: Deployment 1 (AquaFFIA, fold 00) run through this template.

Used to verify that the template feeds exactly the same input as Deployment 1:
for the same step it must return the same prediction. It is also a complete
example of a three-engine TensorRT pipeline (audio frontend, video frontend,
multimodal core).

Run with MODEL_MODULE=examples.model_aquaffia_reference and the AquaFFIA
engines in the weights directory:
    audio_frontend_fp32.engine, video_frontend_fp32.engine,
    multimodal_core_fold_00_fp32.engine
"""

import os

import cv2
import numpy as np

from trt_helper import TRTModule

INFO = {
    "name": "AquaFFIA reference (Deployment 1 model)",
    "runtime": "NVIDIA TensorRT FP32 (Ampere GPU)",
    "audio_frontend_device": "gpu (cuda-python)",
    "video_frontend_device": "gpu (cuda-python)",
    "fold": "Fold 00",
    "accuracy_baseline": "97.10%",
}
OUTPUT_IS_PROBABILITY = False

_audio = _video = _core = None


def load(weights_dir: str) -> None:
    global _audio, _video, _core
    _audio = TRTModule(os.path.join(weights_dir, "audio_frontend_fp32.engine"))
    _video = TRTModule(os.path.join(weights_dir, "video_frontend_fp32.engine"))
    _core = TRTModule(os.path.join(weights_dir, "multimodal_core_fold_00_fp32.engine"))


def preprocess_audio(waveform: np.ndarray):
    return _audio.infer(waveform.reshape(1, -1))                 # -> [1, 1, 128, 128] log-mel


def preprocess_video(frame_first: np.ndarray, frame_last: np.ndarray):
    f0, f1 = frame_first, frame_last
    if f0.shape[:2] != (224, 224):
        f0 = cv2.resize(f0, (224, 224), interpolation=cv2.INTER_LINEAR)
    if f1.shape[:2] != (224, 224):
        f1 = cv2.resize(f1, (224, 224), interpolation=cv2.INTER_LINEAR)
    pair = np.stack([f0, f1], axis=0)[None]                      # [1, 2, 224, 224, 3]
    return _video.infer(pair)                                    # -> [1, 6, 224, 224]


def infer(audio_input, video_input) -> np.ndarray:
    return _core.infer(audio_features=audio_input, video_form=video_input)   # -> [1, 4] logits
