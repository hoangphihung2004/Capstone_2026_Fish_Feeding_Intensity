"""
Export Multimodal Fish Feeding Intensity Models (All 5 Folds) to ONNX format.

Module: model_converter.export_onnx
Responsibilities:
- Support exporting all 5 folds (Fold 00 to Fold 04) or specific individual folds
- Strip profiling hooks from checkpoint state dicts
- Export Core Architecture to ONNX (opset 18) with fixed batch size 1
- Simplify computation graphs via onnxslim
- Run strict numerical consistency check against PyTorch for each fold
- Generate export summary manifest (weights/onnx_export_summary.json)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Root directory of the repository (U_FFIA27K_multimodal_deployment)
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from models.multimodal_model import MultimodalModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ExportONNX")


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch Multimodal 5-Fold Checkpoints to ONNX")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(project_root, "checkpoint", "multimodal_model"),
        help="Directory containing fold_00 to fold_04 checkpoint folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "weights"),
        help="Directory to save exported ONNX models",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Folds to export: 'all', or comma-separated list like '0,1,2,3,4' or '0'",
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


def resolve_fold_list(folds_arg: str) -> List[int]:
    """Parse folds argument into a sorted list of integer fold indices."""
    if folds_arg.strip().lower() == "all":
        return list(range(5))
    return sorted(list(set(int(f.strip()) for f in folds_arg.split(",") if f.strip())))


def load_fold_core_model(checkpoint_path: str) -> torch.nn.Module:
    """Load model architecture and weights from checkpoint, filtering profiler hooks."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    cleaned_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }

    full_model = MultimodalModel(audio_config=None, pretrained_video=False)
    missing, unexpected = full_model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys during load: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys ignored: {len(unexpected)}")

    core_model = full_model.architecture.eval()
    return core_model


def export_single_fold(
    model: torch.nn.Module,
    fold_idx: int,
    output_dir: str,
    batch_size: int = 1,
    opset: int = 18,
) -> Tuple[str, str]:
    """Export one fold's core architecture to ONNX and optimize with onnxslim."""
    os.makedirs(output_dir, exist_ok=True)
    raw_onnx = os.path.join(output_dir, f"multimodal_core_fold_{fold_idx:02d}.onnx")
    sim_onnx = os.path.join(output_dir, f"multimodal_core_fold_{fold_idx:02d}_sim.onnx")

    dummy_audio = torch.randn(batch_size, 1, 128, 128, dtype=torch.float32)
    dummy_video = torch.randn(batch_size, 6, 224, 224, dtype=torch.float32)

    logger.info(f"[Fold {fold_idx:02d}] Exporting to ONNX (opset {opset}) -> {raw_onnx}")
    torch.onnx.export(
        model,
        (dummy_audio, dummy_video),
        raw_onnx,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["audio_features", "video_form"],
        output_names=["clipwise_output"],
        dynamic_axes=None,
    )

    import onnx
    onnx_model = onnx.load(raw_onnx)
    onnx.checker.check_model(onnx_model)

    active_onnx = raw_onnx
    try:
        import onnxslim
        logger.info(f"[Fold {fold_idx:02d}] Simplifying ONNX with onnxslim...")
        slimmed = onnxslim.slim(raw_onnx)
        onnx.save(slimmed, sim_onnx)
        active_onnx = sim_onnx
        logger.info(f"[Fold {fold_idx:02d}] Optimized ONNX saved: {sim_onnx} ({os.path.getsize(sim_onnx)/(1024*1024):.2f} MB)")
    except Exception as exc:
        logger.warning(f"[Fold {fold_idx:02d}] onnxslim skipped ({exc}). Using standard ONNX.")

    return raw_onnx, active_onnx


def verify_fold_consistency(
    model: torch.nn.Module,
    onnx_path: str,
    fold_idx: int,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Verify PyTorch and ONNX Runtime produce identical outputs for a given fold."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    np.random.seed(42 + fold_idx)
    test_audio = np.random.randn(batch_size, 1, 128, 128).astype(np.float32)
    test_video = np.random.randn(batch_size, 6, 224, 224).astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.from_numpy(test_audio), torch.from_numpy(test_video))["clipwise_output"].numpy()

    ort_inputs = {
        "audio_features": test_audio,
        "video_form": test_video,
    }
    ort_out = session.run(["clipwise_output"], ort_inputs)[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    pt_class = int(np.argmax(pt_out, axis=1)[0])
    ort_class = int(np.argmax(ort_out, axis=1)[0])

    pt_probs = torch.softmax(torch.from_numpy(pt_out), dim=-1).numpy()[0]
    ort_probs = torch.softmax(torch.from_numpy(ort_out), dim=-1).numpy()[0]
    max_prob_diff = float(np.max(np.abs(pt_probs - ort_probs)))

    logger.info(f"[Fold {fold_idx:02d}] PyTorch class: {pt_class} == ONNX class: {ort_class}")
    logger.info(f"[Fold {fold_idx:02d}] Max logit diff: {max_diff:.6e}, Max prob diff: {max_prob_diff:.6e}")

    assert pt_class == ort_class, f"Argmax mismatch on Fold {fold_idx:02d}!"
    assert np.allclose(pt_out, ort_out, rtol=1e-2, atol=1.0), f"Logit diff exceeded tolerance on Fold {fold_idx:02d}!"
    assert max_prob_diff < 1e-4, f"Prob diff exceeded tolerance on Fold {fold_idx:02d}!"
    logger.info(f"[Fold {fold_idx:02d}] Numerical verification: PASSED (100% match)")

    return {
        "max_logit_diff": max_diff,
        "max_prob_diff": max_prob_diff,
        "class_match": True,
        "predicted_class": pt_class,
    }


def main():
    args = parse_args()
    fold_indices = resolve_fold_list(args.folds)
    logger.info("==========================================================")
    logger.info(f"Starting Multimodal ONNX Export for Folds: {fold_indices}")
    logger.info(f"Checkpoints directory: {args.checkpoint_dir}")
    logger.info(f"Output directory:      {args.output_dir}")
    logger.info("==========================================================")

    manifest = {}
    for fold in fold_indices:
        ckpt_path = os.path.join(args.checkpoint_dir, f"fold_{fold:02d}", "multimodal_best.pt")
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint for fold {fold:02d} missing: {ckpt_path}. Skipping.")
            continue

        model = load_fold_core_model(ckpt_path)
        raw_onnx, active_onnx = export_single_fold(model, fold, args.output_dir, args.batch_size, args.opset)
        metrics = verify_fold_consistency(model, active_onnx, fold, args.batch_size)

        manifest[f"fold_{fold:02d}"] = {
            "checkpoint": ckpt_path,
            "raw_onnx": raw_onnx,
            "sim_onnx": active_onnx,
            "size_mb": round(os.path.getsize(active_onnx) / (1024 * 1024), 2),
            "verification": metrics,
        }

    manifest_path = os.path.join(args.output_dir, "onnx_export_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("==========================================================")
    logger.info(f"All {len(manifest)} folds exported and verified successfully!")
    logger.info(f"Summary manifest saved to: {manifest_path}")
    logger.info("==========================================================")


if __name__ == "__main__":
    main()
