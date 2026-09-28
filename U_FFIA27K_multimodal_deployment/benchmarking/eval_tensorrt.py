"""
Multimodal Fish Feeding Intensity TensorRT Evaluation & Benchmarking

Module: benchmarking.eval_tensorrt
Responsibilities:
- Evaluate multimodal classification accuracy on NVIDIA Jetson Orin Nano / edge devices.
- Direct comparison with published baseline (Accuracy: 97.10%, mAP: 0.9947).
- Compute 4x4 Confusion Matrix, Precision, Recall, and F1-Score for all classes:
  0: none, 1: strong, 2: medium, 3: weak.
- High-precision latency profiling: Audio Preprocessing, Video Decoding, GPU Inference, and Total Latency (ms).
- Throughput (FPS) calculation.
- Dual-runtime support:
  * Primary: TensorRT Engine (FP16) via PyCUDA / TensorRT on NVIDIA Jetson / Linux Docker.
  * Fallback: ONNX Runtime (CPU/CUDA) or PyTorch Reference for PC development/validation.
- Outputs: Terminal ASCII summary report, JSON metrics log, and confusion matrix export.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Root directory of deployment package
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
import cv2
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

try:
    import decord
    decord.bridge.set_bridge("torch")
except Exception:
    pass

from features.audio_frontend import AudioFrontend

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EvalTensorRT")

CLASS_NAMES = ["none", "strong", "medium", "weak"]
LABEL_MAP = {0: "none", 1: "strong", 2: "medium", 3: "weak"}

# Check TensorRT availability
TRT_AVAILABLE = False
TRT_BACKEND = None
try:
    import tensorrt as trt
    try:
        from cuda.bindings import runtime as cudart
        TRT_AVAILABLE = True
        TRT_BACKEND = "cuda-python"
    except Exception:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            TRT_AVAILABLE = True
            TRT_BACKEND = "pycuda"
        except Exception:
            TRT_AVAILABLE = False
except Exception:
    TRT_AVAILABLE = False

# Check ONNX Runtime availability
ORT_AVAILABLE = False
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Fish Feeding Intensity TensorRT Benchmarking")
    default_engine = os.path.join(project_root, "weights", "multimodal_core_fold_00_fp32.engine")
    if not os.path.exists(default_engine):
        fallback_fp16 = os.path.join(project_root, "weights", "multimodal_core_fold_00_fp16.engine")
        if os.path.exists(fallback_fp16):
            default_engine = fallback_fp16

    parser.add_argument(
        "--engine",
        type=str,
        default=default_engine,
        help="Path to compiled TensorRT engine file (default: FP32 engine)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=os.path.join(project_root, "weights", "multimodal_core_fold_00_sim.onnx"),
        help="Path to fallback ONNX model file",
    )
    default_audio_engine = os.path.join(project_root, "weights", "audio_frontend_fp32.engine")
    parser.add_argument(
        "--audio_engine",
        type=str,
        default=default_audio_engine,
        help="Path to compiled TensorRT AudioFrontend engine file (GPU)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(project_root, "checkpoint", "multimodal_model", "fold_00", "multimodal_best.pt"),
        help="Path to PyTorch checkpoint (for AudioFrontend weights fallback)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(project_root, "samples", "manifest.csv"),
        help="Path to samples manifest.csv",
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default=os.path.join(project_root, "samples"),
        help="Root directory of sample audio and video files",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (-1 for full test set)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="If num_samples > 0, sample equally from all 4 classes for balanced evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "benchmarking", "results"),
        help="Directory to save benchmark metrics and confusion matrix",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before timing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Must be 1 on Jetson for representative batch-1 latency. Higher values are coerced to 1 to avoid CPU oversubscription.",
    )
    return parser.parse_args()


# ==============================================================================
# 1. TENSORRT INFERENCE WRAPPER
# ==============================================================================
class TensorRTInference:
    """Wrapper for NVIDIA TensorRT execution context on Jetson Orin Nano."""

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("Neither cuda-python nor PyCUDA is installed in this environment.")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.backend = TRT_BACKEND

        # Input 0: audio_features [1, 1, 128, 128] float32
        self.audio_shape = (1, 1, 128, 128)
        self.audio_nbytes = int(np.prod(self.audio_shape) * 4)

        # Input 1: video_form [1, 6, 224, 224] float32
        self.video_shape = (1, 6, 224, 224)
        self.video_nbytes = int(np.prod(self.video_shape) * 4)

        # Output: clipwise_output [1, 4] float32
        self.out_shape = (1, 4)
        self.out_nbytes = int(np.prod(self.out_shape) * 4)
        self.h_out = np.zeros(self.out_shape, dtype=np.float32)

        if self.backend == "cuda-python":
            _, self.d_audio = cudart.cudaMalloc(self.audio_nbytes)
            _, self.d_video = cudart.cudaMalloc(self.video_nbytes)
            _, self.d_out = cudart.cudaMalloc(self.out_nbytes)
            _, self.stream = cudart.cudaStreamCreate()

            # Set tensor addresses for TensorRT 10.x API
            self.context.set_tensor_address("audio_features", int(self.d_audio))
            self.context.set_tensor_address("video_form", int(self.d_video))
            self.context.set_tensor_address("clipwise_output", int(self.d_out))
        else:
            self.stream = cuda.Stream()
            self.h_audio = cuda.pagelocked_empty(self.audio_shape, dtype=np.float32)
            self.h_video = cuda.pagelocked_empty(self.video_shape, dtype=np.float32)
            self.h_out = cuda.pagelocked_empty(self.out_shape, dtype=np.float32)

            self.d_audio = cuda.mem_alloc(self.audio_nbytes)
            self.d_video = cuda.mem_alloc(self.video_nbytes)
            self.d_out = cuda.mem_alloc(self.out_nbytes)

            self.bindings = [int(self.d_audio), int(self.d_video), int(self.d_out)]

        logger.info(f"Loaded TensorRT Engine successfully ({self.backend}): {engine_path}")

    def infer(self, audio_features: np.ndarray, video_form: np.ndarray) -> np.ndarray:
        audio_features = np.ascontiguousarray(audio_features, dtype=np.float32)
        video_form = np.ascontiguousarray(video_form, dtype=np.float32)

        if self.backend == "cuda-python":
            cudart.cudaMemcpy(self.d_audio, audio_features.ctypes.data, self.audio_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            cudart.cudaMemcpy(self.d_video, video_form.ctypes.data, self.video_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            cudart.cudaMemcpy(self.h_out.ctypes.data, self.d_out, self.out_nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            return self.h_out.copy()
        else:
            np.copyto(self.h_audio, audio_features)
            np.copyto(self.h_video, video_form)

            cuda.memcpy_htod_async(self.d_audio, self.h_audio, self.stream)
            cuda.memcpy_htod_async(self.d_video, self.h_video, self.stream)

            if hasattr(self.context, "execute_async_v2"):
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            else:
                self.context.execute_v2(bindings=self.bindings)

            cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()
            return np.copy(self.h_out)


class AudioFrontendTRT:
    """Wrapper for GPU-accelerated AudioFrontend Log-Mel Spectrogram TensorRT Engine."""

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("Neither cuda-python nor PyCUDA is installed in this environment.")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"AudioFrontend TensorRT engine not found at: {engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize AudioFrontend TensorRT engine from: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.backend = TRT_BACKEND

        # Input: waveform [1, 128000] float32
        self.input_shape = (1, 128000)
        self.input_nbytes = int(np.prod(self.input_shape) * 4)

        # Output: audio_features [1, 1, 128, 128] float32
        self.out_shape = (1, 1, 128, 128)
        self.out_nbytes = int(np.prod(self.out_shape) * 4)
        self.h_out = np.zeros(self.out_shape, dtype=np.float32)

        if self.backend == "cuda-python":
            _, self.d_in = cudart.cudaMalloc(self.input_nbytes)
            _, self.d_out = cudart.cudaMalloc(self.out_nbytes)
            _, self.stream = cudart.cudaStreamCreate()

            # Set tensor addresses for TensorRT 10.x API
            self.context.set_tensor_address("waveform", int(self.d_in))
            self.context.set_tensor_address("audio_features", int(self.d_out))
        else:
            self.stream = cuda.Stream()
            self.h_in = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
            self.h_out = cuda.pagelocked_empty(self.out_shape, dtype=np.float32)

            self.d_in = cuda.mem_alloc(self.input_nbytes)
            self.d_out = cuda.mem_alloc(self.out_nbytes)

            self.bindings = [int(self.d_in), int(self.d_out)]

        logger.info(f"Loaded AudioFrontend TensorRT Engine successfully ({self.backend}): {engine_path}")

    def __call__(self, waveform: Any) -> np.ndarray:
        return self.infer(waveform)

    def infer(self, waveform: Any) -> np.ndarray:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)

        if self.backend == "cuda-python":
            cudart.cudaMemcpy(self.d_in, waveform.ctypes.data, self.input_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            cudart.cudaMemcpy(self.h_out.ctypes.data, self.d_out, self.out_nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            return self.h_out.copy()
        else:
            np.copyto(self.h_in, waveform)
            cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)

            if hasattr(self.context, "execute_async_v2"):
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            else:
                self.context.execute_v2(bindings=self.bindings)

            cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()
            return np.copy(self.h_out)


# ==============================================================================
# 2. ONNX RUNTIME FALLBACK WRAPPER
# ==============================================================================
class ONNXRuntimeInference:
    """Fallback runner using ONNX Runtime for testing on PC without TensorRT."""

    def __init__(self, onnx_path: str):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed in this environment.")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            if "Unsupported model IR version" in str(e):
                import onnx
                logger.warning(f"Downgrading model IR version for runtime compatibility: {onnx_path}")
                m = onnx.load(onnx_path)
                m.ir_version = 9
                fixed_path = onnx_path.replace(".onnx", "_ir9.onnx")
                onnx.save(m, fixed_path)
                self.session = ort.InferenceSession(fixed_path, providers=providers)
            else:
                raise e
        logger.info(f"Loaded ONNX Model fallback with providers {providers}: {onnx_path}")

    def infer(self, audio_features: np.ndarray, video_form: np.ndarray) -> np.ndarray:
        inputs = {
            "audio_features": audio_features.astype(np.float32),
            "video_form": video_form.astype(np.float32),
        }
        outputs = self.session.run(["clipwise_output"], inputs)
        return outputs[0]


# ==============================================================================
# 3. PREPROCESSING HELPERS
# ==============================================================================
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=torch.float32).view(6, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=torch.float32).view(6, 1, 1)


def load_audio_frontend(checkpoint_path: str) -> torch.nn.Module:
    """Load AudioFrontend module with trained BatchNorm weights and JIT freeze for speed."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    frontend = AudioFrontend()

    frontend_sd = {}
    for k, v in state_dict.items():
        if k.startswith("frontend.") and not k.endswith("total_ops") and not k.endswith("total_params"):
            frontend_sd[k.replace("frontend.", "")] = v

    missing, unexpected = frontend.load_state_dict(frontend_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"AudioFrontend checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    frontend.eval()

    # Accelerate AudioFrontend via TorchScript JIT freeze
    dummy = torch.zeros((1, 128000), dtype=torch.float32)
    with torch.no_grad():
        frontend = torch.jit.freeze(torch.jit.trace(frontend, dummy))
    logger.info("AudioFrontend JIT traced and frozen for accelerated CPU inference.")
    return frontend


_RESAMPLER_CACHE: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}


def preprocess_audio(
    audio_path: str,
    frontend: Any,
    target_sr: int = 64000,
) -> Tuple[np.ndarray, float]:
    """Load WAV audio, resample to 64kHz, and compute Log-Mel Spectrogram (1, 1, 128, 128)."""
    t0 = time.perf_counter()
    samples, original_sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if samples.shape[1] == 1:
        waveform = torch.as_tensor(samples.T)
    else:
        waveform = torch.as_tensor(samples.T).mean(dim=0, keepdim=True)

    if original_sr != target_sr:
        key = (original_sr, target_sr)
        if key not in _RESAMPLER_CACHE:
            _RESAMPLER_CACHE[key] = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = _RESAMPLER_CACHE[key](waveform)

    y = waveform.squeeze(0).to(torch.float32)
    target_len = target_sr * 2
    if y.numel() > target_len:
        y = y[:target_len]
    elif y.numel() < target_len:
        y = torch.nn.functional.pad(y, (0, target_len - y.numel()))

    if hasattr(frontend, "infer"):
        audio_feature_np = frontend.infer(y.unsqueeze(0))
    else:
        with torch.no_grad():
            feat = frontend(y.unsqueeze(0))
        audio_feature_np = feat.detach().cpu().numpy()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return audio_feature_np, elapsed_ms


def preprocess_video(video_path: str, image_size: int = 224) -> Tuple[np.ndarray, float]:
    """Decode first and last video frames and apply ImageNet normalization (1, 6, 224, 224)."""
    t0 = time.perf_counter()
    normed_tensor = None

    # Try decord first for fast zero-copy frame extraction matching training pipeline
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=cpu(0), num_threads=2)
        if len(vr) > 0:
            batch = vr.get_batch([0, len(vr) - 1])  # PyTorch uint8 tensor [2, 224, 224, 3]
            img = torch.cat([batch[0], batch[1]], dim=-1).permute(2, 0, 1).float().mul_(1.0 / 255.0)
            img.sub_(_IMAGENET_MEAN).div_(_IMAGENET_STD)
            normed_tensor = img
    except Exception:
        pass

    # OpenCV fallback
    if normed_tensor is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            normed_tensor = torch.zeros((6, image_size, image_size), dtype=torch.float32)
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = [0, max(0, frame_count - 1)]
            frames = []
            for f_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ok, frame = cap.read()
                if ok:
                    frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (image_size, image_size), interpolation=cv2.INTER_AREA)
                    frames.append(frame)
            cap.release()
            if len(frames) == 2:
                video_uint8 = np.concatenate(frames, axis=-1).transpose(2, 0, 1).astype(np.uint8)
                img = torch.from_numpy(video_uint8).float().mul_(1.0 / 255.0)
                img.sub_(_IMAGENET_MEAN).div_(_IMAGENET_STD)
                normed_tensor = img
            else:
                normed_tensor = torch.zeros((6, image_size, image_size), dtype=torch.float32)

    video_np = normed_tensor.unsqueeze(0).numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return video_np, elapsed_ms


# ==============================================================================
# 4. METRICS & CONFUSION MATRIX EVALUATION
# ==============================================================================
def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Compute overall accuracy, class-level metrics, and 4x4 confusion matrix."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    total = len(y_true_arr)

    overall_acc = float(np.mean(y_true_arr == y_pred_arr))

    # Build 4x4 confusion matrix
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
        if 0 <= t < 4 and 0 <= p < 4:
            cm[t, p] += 1

    class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for c in range(4):
        tp = cm[c, c]
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        class_metrics[CLASS_NAMES[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": round(overall_acc, 6),
        "total_samples": total,
        "correct_samples": int(np.sum(y_true_arr == y_pred_arr)),
        "confusion_matrix": cm.tolist(),
        "class_metrics": class_metrics,
        "macro_avg": {
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1_score": round(macro_f1, 4),
        },
    }


def print_ascii_dashboard(
    metrics: Dict[str, Any],
    latency_stats: Dict[str, float],
    runtime_name: str,
    audio_frontend_info: str = "GPU TensorRT",
    baseline_acc: float = 0.971006,
):
    """Print clean, professional ASCII summary dashboard in the Terminal."""
    cm = np.array(metrics["confusion_matrix"])
    acc = metrics["accuracy"]
    acc_diff = (acc - baseline_acc) * 100.0

    print("\n" + "=" * 76)
    print("      MULTIMODAL FISH FEEDING INTENSITY - EDGE EVALUATION REPORT      ")
    print("=" * 76)
    print(f" Runtime Engine:      {runtime_name}")
    print(f" Audio Frontend:      {audio_frontend_info}")
    print(" Target Hardware:     NVIDIA Jetson Orin Nano")
    print(f" Total Evaluated:     {metrics['total_samples']:,} samples")
    print(f" Accuracy Achieved:   {acc * 100.0:.2f}% ({metrics['correct_samples']:,} / {metrics['total_samples']:,})")
    print(f" Published Baseline:  {baseline_acc * 100.0:.2f}% (Deviation: {acc_diff:+.2f}%)")
    print("-" * 76)
    print(" 1. LATENCY BREAKDOWN & THROUGHPUT PROFILING:")
    print(f"    - Audio Preprocessing (Log-Mel STFT):  {latency_stats['audio_ms']:>6.2f} ms")
    print(f"    - Video Preprocessing (Decord/CV2):     {latency_stats['video_ms']:>6.2f} ms")
    print(f"    - Model Inference (TensorRT/GPU):      {latency_stats['gpu_ms']:>6.2f} ms")
    print(f"    - End-to-End Pipeline Latency:         {latency_stats['total_ms']:>6.2f} ms")
    print(f"    - System Throughput (Real-time FPS):   {latency_stats['fps']:>6.2f} FPS")
    print("-" * 76)
    print(" 2. CLASSIFICATION METRICS PER CLASS:")
    print(f"    {'Class':<10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    for cname in CLASS_NAMES:
        cm_data = metrics["class_metrics"][cname]
        print(
            f"    {cname:<10} "
            f"{cm_data['precision'] * 100.0:>11.2f}% "
            f"{cm_data['recall'] * 100.0:>11.2f}% "
            f"{cm_data['f1_score'] * 100.0:>11.2f}% "
            f"{cm_data['support']:>12,}"
        )
    macro = metrics["macro_avg"]
    print(
        f"    {'Macro Avg':<10} "
        f"{macro['precision'] * 100.0:>11.2f}% "
        f"{macro['recall'] * 100.0:>11.2f}% "
        f"{macro['f1_score'] * 100.0:>11.2f}% "
        f"{metrics['total_samples']:>12,}"
    )
    print("-" * 76)
    print(" 3. CONFUSION MATRIX (4x4):")
    header_title = "Actual \\ Pred"
    print(f"    {header_title:<14} {'none':>10} {'strong':>10} {'medium':>10} {'weak':>10}")
    for idx, cname in enumerate(CLASS_NAMES):
        row_str = f"    {cname:<14}"
        for col in range(4):
            row_str += f"{cm[idx, col]:>10,}"
        print(row_str)
    print("=" * 76 + "\n")


# ==============================================================================
# 5. MAIN BENCHMARK DRIVER
# ==============================================================================
def main():
    args = parse_args()
    if args.workers != 1:
        logger.warning(
            "workers=%d was requested, but CPU STFT and video decode oversubscribe Orin CPU cores. "
            "Using workers=1 for valid batch-1 latency.",
            args.workers,
        )
        args.workers = 1
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize Inference Engine (TensorRT with ORN fallback)
    runtime_name = "Unknown"
    engine_runner = None

    if TRT_AVAILABLE and os.path.exists(args.engine):
        try:
            engine_runner = TensorRTInference(args.engine)
            prec = "FP32" if "fp32" in args.engine.lower() else "FP16"
            runtime_name = f"NVIDIA TensorRT {prec} (Ampere GPU - {engine_runner.backend})"
        except Exception as exc:
            logger.warning(f"Failed to initialize TensorRT: {exc}. Trying ONNX fallback.")

    if engine_runner is None:
        if ORT_AVAILABLE and os.path.exists(args.onnx):
            engine_runner = ONNXRuntimeInference(args.onnx)
            runtime_name = "ONNX Runtime (Fallback)"
        else:
            raise RuntimeError(
                f"No working inference runtime found. Neither TensorRT ({args.engine}) nor ONNX ({args.onnx}) could be loaded."
            )

    # 2. Load AudioFrontend (Prefer GPU TensorRT engine if available)
    frontend = None
    audio_frontend_device = "cpu"
    if args.audio_engine and TRT_AVAILABLE and os.path.exists(args.audio_engine):
        try:
            frontend = AudioFrontendTRT(args.audio_engine)
            audio_frontend_device = f"GPU TensorRT ({frontend.backend})"
            logger.info(f"Loaded GPU AudioFrontend TensorRT Engine: {args.audio_engine}")
        except Exception as exc:
            logger.warning(f"Failed to load AudioFrontend TensorRT Engine: {exc}. Falling back to PyTorch CPU.")
            frontend = None

    if frontend is None:
        logger.info("Loading PyTorch AudioFrontend fallback from checkpoint...")
        frontend = load_audio_frontend(args.checkpoint)
        audio_frontend_device = "CPU (TorchScript JIT)"
        logger.info(f"AudioFrontend device: {audio_frontend_device}")

    # 3. Load Manifest
    logger.info(f"Loading samples manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)

    if args.num_samples > 0:
        if args.stratified:
            logger.info(f"Applying stratified sampling: {args.num_samples} total across 4 classes...")
            per_class = max(1, args.num_samples // 4)
            sampled_dfs = []
            for cname in CLASS_NAMES:
                cdf = df[df["class_name"] == cname]
                sample_count = min(len(cdf), per_class)
                sampled_dfs.append(cdf.sample(n=sample_count, random_state=args.seed))
            df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
            logger.info(f"Stratified sample prepared: {len(df)} samples ({dict(df['class_name'].value_counts())})")
        else:
            logger.info(f"Evaluating subset of {args.num_samples} samples (out of {len(df)})")
            df = df.head(args.num_samples)
    else:
        logger.info(f"Evaluating FULL test set: {len(df):,} samples")

    # 4. Warmup
    logger.info(f"Warming up inference engines for {args.warmup} iterations...")
    dummy_raw_audio = torch.zeros((1, 128000), dtype=torch.float32)
    dummy_audio = np.zeros((1, 1, 128, 128), dtype=np.float32)
    dummy_video = np.zeros((1, 6, 224, 224), dtype=np.float32)
    for _ in range(args.warmup):
        if hasattr(frontend, "infer"):
            frontend.infer(dummy_raw_audio)
        else:
            with torch.no_grad():
                frontend(dummy_raw_audio)
        engine_runner.infer(dummy_audio, dummy_video)

    # 5. Benchmark Loop
    y_true: List[int] = []
    y_pred: List[int] = []

    audio_times = []
    video_times = []
    gpu_times = []
    total_times = []

    samples_dir = Path(args.samples_dir)
    total_samples = len(df)
    logger.info("==========================================================")
    logger.info(f"Starting Benchmark Execution (Workers: {args.workers})...")
    logger.info(f"Engine: {runtime_name}")
    logger.info("==========================================================")

    start_eval_time = time.perf_counter()

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        import gc

        def preprocess_worker(row_tuple):
            idx, row = row_tuple
            v_file = samples_dir / row["video_path"]
            a_file = samples_dir / row["audio_path"]
            lbl = int(row["label"])
            a_feat, t_a = preprocess_audio(str(a_file), frontend)
            v_feat, t_v = preprocess_video(str(v_file))
            return idx, lbl, a_feat, v_feat, t_a, t_v

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Bounded sliding window to prevent RAM exhaustion (OOM on Edge devices)
            max_in_flight = max(8, args.workers * 4)
            futures_queue = deque()
            row_iter = df.iterrows()

            for _ in range(max_in_flight):
                try:
                    r = next(row_iter)
                    futures_queue.append(executor.submit(preprocess_worker, r))
                except StopIteration:
                    break

            while futures_queue:
                fut = futures_queue.popleft()
                idx, label, audio_feat, video_feat, t_audio, t_video = fut.result()

                # Keep queue filled with next item
                try:
                    r = next(row_iter)
                    futures_queue.append(executor.submit(preprocess_worker, r))
                except StopIteration:
                    pass

                t_gpu_0 = time.perf_counter()
                logits = engine_runner.infer(audio_feat, video_feat)
                t_gpu = (time.perf_counter() - t_gpu_0) * 1000.0

                pred_class = int(np.argmax(logits, axis=1)[0])
                y_true.append(label)
                y_pred.append(pred_class)

                audio_times.append(t_audio)
                video_times.append(t_video)
                gpu_times.append(t_gpu)
                total_times.append(t_audio + t_video + t_gpu)

                if len(y_true) % 500 == 0 or len(y_true) == total_samples:
                    gc.collect()
                    current_acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100.0
                    wall_fps = len(y_true) / (time.perf_counter() - start_eval_time)
                    logger.info(
                        f"Progress: [{len(y_true):>5}/{total_samples:>5}] | "
                        f"Acc: {current_acc:>6.2f}% | "
                        f"Real Throughput: {wall_fps:>5.1f} FPS"
                    )
    else:
        for idx, row in df.iterrows():
            t_sample_start = time.perf_counter()

            v_file = samples_dir / row["video_path"]
            a_file = samples_dir / row["audio_path"]
            label = int(row["label"])

            # Preprocessing
            audio_feat, t_audio = preprocess_audio(str(a_file), frontend)
            video_feat, t_video = preprocess_video(str(v_file))

            # Model Inference
            t_gpu_0 = time.perf_counter()
            logits = engine_runner.infer(audio_feat, video_feat)
            t_gpu = (time.perf_counter() - t_gpu_0) * 1000.0

            t_total = (time.perf_counter() - t_sample_start) * 1000.0

            pred_class = int(np.argmax(logits, axis=1)[0])

            y_true.append(label)
            y_pred.append(pred_class)

            audio_times.append(t_audio)
            video_times.append(t_video)
            gpu_times.append(t_gpu)
            total_times.append(t_total)

            if (idx + 1) % 500 == 0 or (idx + 1) == total_samples:
                current_acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100.0
                avg_lat = np.mean(total_times)
                logger.info(
                    f"Progress: [{idx + 1:>5}/{total_samples:>5}] | "
                    f"Acc: {current_acc:>6.2f}% | "
                    f"Latency: {avg_lat:>5.2f} ms ({1000.0 / avg_lat:>5.1f} FPS)"
                )

    total_eval_duration = time.perf_counter() - start_eval_time
    wall_clock_fps = total_samples / max(0.001, total_eval_duration)

    # 6. Compute Results
    metrics = calculate_metrics(y_true, y_pred)
    latency_stats = {
        "audio_ms": round(float(np.mean(audio_times)), 2),
        "video_ms": round(float(np.mean(video_times)), 2),
        "gpu_ms": round(float(np.mean(gpu_times)), 2),
        "total_ms": round(float(np.mean(total_times)), 2),
        "fps": round(float(wall_clock_fps if args.workers > 1 else (1000.0 / np.mean(total_times))), 2),
        "total_duration_s": round(total_eval_duration, 2),
    }

    # 7. Print Terminal ASCII Dashboard
    print_ascii_dashboard(metrics, latency_stats, runtime_name, audio_frontend_info=audio_frontend_device)

    # 8. Save Metrics to JSON
    summary_payload = {
        "runtime": runtime_name,
        "evaluation_metrics": metrics,
        "latency_profiling": latency_stats,
        "device_info": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "audio_frontend_device": audio_frontend_device,
        },
    }

    results_json = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info(f"Detailed benchmark metrics saved to: {results_json}")

    # 9. Save Confusion Matrix to CSV
    cm_csv = os.path.join(args.output_dir, "confusion_matrix.csv")
    with open(cm_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted"] + CLASS_NAMES)
        for idx, cname in enumerate(CLASS_NAMES):
            writer.writerow([cname] + metrics["confusion_matrix"][idx])
    logger.info(f"Confusion Matrix CSV saved to: {cm_csv}")

    # 10. Generate and save Confusion Matrix Plot (PNG)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            title=f"Confusion Matrix (Acc: {metrics['accuracy']*100:.2f}%)\n{runtime_name}",
            ylabel="Actual Label",
            xlabel="Predicted Label",
        )
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold" if i == j else "normal",
                )
        fig.tight_layout()
        cm_png = os.path.join(args.output_dir, "confusion_matrix.png")
        fig.savefig(cm_png)
        plt.close(fig)
        logger.info(f"Confusion Matrix Plot saved to: {cm_png}")
    except Exception as exc:
        logger.warning(f"Could not generate confusion matrix plot: {exc}")


if __name__ == "__main__":
    main()
