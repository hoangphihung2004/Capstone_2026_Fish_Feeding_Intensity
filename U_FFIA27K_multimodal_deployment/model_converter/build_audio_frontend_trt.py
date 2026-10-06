"""
AudioFrontend TensorRT Engine Builder for NVIDIA Jetson Orin Nano

Module: model_converter.build_audio_frontend_trt
Responsibilities:
- Export PyTorch AudioFrontend to ONNX graph (waveform [1, 128000] -> audio_features [1, 1, 128, 128]).
- Fix the infinite upper-bound constant in torchlibrosa LogmelFilterBank Clip operator
  (Assertion failed: std::isfinite(beta) in TensorRT Activation Clip).
- Build high-precision FP32 TensorRT engine to prevent FP16 log10(amin) underflow.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from features.audio_frontend import AudioFrontend


def export_and_fix_onnx(checkpoint_path: str, raw_onnx_path: str, fixed_onnx_path: str):
    import onnx
    from onnx import numpy_helper

    print(f">> 1. Loading AudioFrontend weights from: {checkpoint_path}")
    frontend = AudioFrontend().eval()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        fe_sd = {
            k.replace("frontend.", ""): v
            for k, v in sd.items()
            if k.startswith("frontend.") and not k.endswith("total_ops") and not k.endswith("total_params")
        }
        frontend.load_state_dict(fe_sd, strict=False)
        print("   Weights successfully loaded into AudioFrontend.")

    print(f">> 2. Exporting AudioFrontend to ONNX: {raw_onnx_path}")
    dummy = torch.randn(1, 128000, dtype=torch.float32)
    os.makedirs(os.path.dirname(raw_onnx_path), exist_ok=True)
    torch.onnx.export(
        frontend,
        dummy,
        raw_onnx_path,
        input_names=["waveform"],
        output_names=["audio_features"],
        opset_version=17,
        do_constant_folding=True,
    )
    print("   Raw ONNX export complete.")

    print(f">> 3. Fixing infinite Clip constants for TensorRT compatibility...")
    model = onnx.load(raw_onnx_path)
    fixed_count = 0

    for node in model.graph.node:
        if node.op_type == "Clip":
            if len(node.input) >= 3:
                max_name = node.input[2]
                for init in model.graph.initializer:
                    if init.name == max_name:
                        arr = numpy_helper.to_array(init)
                        if not np.isfinite(arr).all():
                            new_arr = np.where(np.isinf(arr), 1e30, arr).astype(arr.dtype)
                            init.CopyFrom(numpy_helper.from_array(new_arr, name=init.name))
                            fixed_count += 1
                for n in model.graph.node:
                    if n.op_type == "Constant" and n.output[0] == max_name:
                        for attr in n.attribute:
                            if attr.name == "value":
                                arr = numpy_helper.to_array(attr.t)
                                if not np.isfinite(arr).all():
                                    new_arr = np.where(np.isinf(arr), 1e30, arr).astype(arr.dtype)
                                    attr.t.CopyFrom(numpy_helper.from_array(new_arr))
                                    fixed_count += 1

    print(f"   Fixed {fixed_count} infinite Clip constants.")
    onnx.save(model, fixed_onnx_path)
    print(f"   Saved patched ONNX to: {fixed_onnx_path}")


def build_tensorrt_engine(fixed_onnx_path: str, engine_path: str):
    import subprocess
    print(f">> 4. Building TensorRT FP32 Engine: {engine_path}")
    trtexec_bin = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_bin):
        trtexec_bin = "trtexec"

    cmd = [
        trtexec_bin,
        f"--onnx={fixed_onnx_path}",
        f"--saveEngine={engine_path}",
        "--memPoolSize=workspace:256MiB",
        "--avgRuns=10",
        "--duration=3",
    ]
    print(f"   Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f">> AudioFrontend TensorRT engine compiled successfully at: {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Build AudioFrontend TensorRT Engine")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(project_root, "checkpoint", "multimodal_model", "fold_00", "multimodal_best.pt"),
        help="Path to multimodal_best.pt checkpoint",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=os.path.join(project_root, "weights"),
        help="Directory to save ONNX and engine files",
    )
    args = parser.parse_args()

    raw_onnx = os.path.join(args.weights_dir, "audio_frontend_raw.onnx")
    fixed_onnx = os.path.join(args.weights_dir, "audio_frontend_fixed.onnx")
    engine = os.path.join(args.weights_dir, "audio_frontend_fp32.engine")

    export_and_fix_onnx(args.checkpoint, raw_onnx, fixed_onnx)
    try:
        build_tensorrt_engine(fixed_onnx, engine)
    except Exception as exc:
        print(f"Note: trtexec execution failed or is not available on this host: {exc}")
        print(f"To compile on Jetson, run: trtexec --onnx={fixed_onnx} --saveEngine={engine}")


if __name__ == "__main__":
    main()
