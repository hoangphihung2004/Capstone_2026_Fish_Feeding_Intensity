"""
Export Multimodal Fish Feeding Intensity Model to ONNX format.

Module: deployment.model_converter
Responsibilities:
- Load PyTorch checkpoint (.pt)
- Export Multimodal Core Architecture to ONNX (opset 18)
- Optimize graph with onnxslim
- Numerically validate predictions against PyTorch
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Root directory of the repository (U_FFIA27K_multimodal_deployment)
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from models.multimodal_model import MultimodalModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelConverter")


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch Multimodal Model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(project_root, "checkpoint", "multimodal_model", "fold_00", "multimodal_best.pt"),
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "weights"),
        help="Directory to save exported ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Fixed batch size for real-time edge inference (default: 1)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str) -> torch.nn.Module:
    """Load model architecture and weights from checkpoint, filtering profiler hooks."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Clean state dict from any thop/profiler keys (total_ops, total_params)
    cleaned_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }

    full_model = MultimodalModel(audio_config=None, pretrained_video=False)
    missing, unexpected = full_model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys during load: {len(missing)} (expected if frontend weights are frozen/separate)")
    if unexpected:
        logger.warning(f"Unexpected keys ignored: {len(unexpected)}")

    core_model = full_model.architecture.eval()
    total_params = sum(p.numel() for p in core_model.parameters())
    logger.info(f"Successfully loaded Core Architecture. Parameters: {total_params:,}")
    return core_model


def export_to_onnx(
    model: torch.nn.Module,
    output_dir: str,
    batch_size: int = 1,
    opset: int = 18,
) -> Tuple[str, str]:
    """Export PyTorch core model to ONNX and run graph optimization via onnxslim."""
    os.makedirs(output_dir, exist_ok=True)
    raw_onnx_path = os.path.join(output_dir, "multimodal_core.onnx")
    sim_onnx_path = os.path.join(output_dir, "multimodal_core_sim.onnx")

    dummy_audio = torch.randn(batch_size, 1, 128, 128, dtype=torch.float32)
    dummy_video = torch.randn(batch_size, 6, 224, 224, dtype=torch.float32)

    logger.info(f"Exporting to ONNX (opset {opset}) -> {raw_onnx_path}")
    torch.onnx.export(
        model,
        (dummy_audio, dummy_video),
        raw_onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["audio_features", "video_form"],
        output_names=["clipwise_output"],
        dynamic_axes=None,  # Fixed batch size for optimal TensorRT tactic selection
    )

    import onnx
    onnx_model = onnx.load(raw_onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph structural validation passed!")

    # Graph simplification with onnxslim if available
    active_path = raw_onnx_path
    try:
        import onnxslim
        logger.info("Simplifying ONNX model with onnxslim...")
        slimmed_model = onnxslim.slim(raw_onnx_path)
        onnx.save(slimmed_model, sim_onnx_path)
        active_path = sim_onnx_path
        logger.info(f"Optimized ONNX saved to: {sim_onnx_path} ({os.path.getsize(sim_onnx_path) / (1024*1024):.2f} MB)")
    except Exception as exc:
        logger.warning(f"onnxslim optimization skipped ({exc}). Using standard ONNX.")

    return raw_onnx_path, active_path


def verify_numerical_consistency(
    model: torch.nn.Module,
    onnx_path: str,
    batch_size: int = 1,
) -> bool:
    """Verify PyTorch and ONNX Runtime outputs match within tolerance on test data."""
    import onnxruntime as ort

    logger.info(f"Verifying numerical accuracy against ONNX Runtime: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    np.random.seed(42)
    test_audio = np.random.randn(batch_size, 1, 128, 128).astype(np.float32)
    test_video = np.random.randn(batch_size, 6, 224, 224).astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.from_numpy(test_audio), torch.from_numpy(test_video))["clipwise_output"].numpy()

    ort_inputs = {
        "audio_features": test_audio,
        "video_form": test_video,
    }
    ort_out = session.run(["clipwise_output"], ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    pt_class = int(np.argmax(pt_out, axis=1)[0])
    ort_class = int(np.argmax(ort_out, axis=1)[0])

    pt_probs = torch.softmax(torch.from_numpy(pt_out), dim=-1).numpy()[0]
    ort_probs = torch.softmax(torch.from_numpy(ort_out), dim=-1).numpy()[0]
    max_prob_diff = np.max(np.abs(pt_probs - ort_probs))

    logger.info(f"PyTorch logits:        {pt_out[0]}")
    logger.info(f"ONNX Runtime logits:   {ort_out[0]}")
    logger.info(f"Max logit difference:  {max_diff:.6e} (relative: {max_diff / (np.max(np.abs(pt_out)) + 1e-6):.6e})")
    logger.info(f"Max prob difference:   {max_prob_diff:.6e}")
    logger.info(f"Argmax Match: PyTorch class {pt_class} == ONNX class {ort_class}")

    assert pt_class == ort_class, f"Argmax mismatch! PyTorch={pt_class}, ONNX={ort_class}"
    assert np.allclose(pt_out, ort_out, rtol=1e-3, atol=5e-2), f"Logit difference exceeded tolerance!"
    assert max_prob_diff < 1e-4, f"Probability difference {max_prob_diff} exceeded tolerance!"
    logger.info("Numerical consistency check: PASSED (100% classification and probability match)")
    return True


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    raw_path, active_path = export_to_onnx(model, args.output_dir, args.batch_size, args.opset)
    verify_numerical_consistency(model, active_path, args.batch_size)
    logger.info("Model conversion complete: ONNX model ready for TensorRT compilation.")


if __name__ == "__main__":
    main()
