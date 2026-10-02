import os
import sys
import json
import csv
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from config import TrainConfig, VideoFeaturesConfig, SplitterConfig
from dataset.data_split import FishDataSplitter, LABEL_TO_CLASS, _sample_identity_key
from dataset.dataloader_videos import _decode_image, _decode_image_cv2
from transforms import VideoTransform
from models import (
    ConvNeXtTiny,
    DenseNet121,
    EfficientNetB0,
    MobileNetV2,
    ResNet18,
    ResNet50,
    SwinTiny,
    VideoModel,
    ViTBase16,
    MobileViTXXS,
    MobileViTXS,
    MobileViTv2_050,
    MobileViTv2_075,
)

MODEL_REGISTRY = {
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0,
    "ResNet18": ResNet18,
    "ResNet50": ResNet50,
    "DenseNet121": DenseNet121,
    "SwinTiny": SwinTiny,
    "ViTBase16": ViTBase16,
    "ConvNeXtTiny": ConvNeXtTiny,
    "MobileViTXXS": MobileViTXXS,
    "MobileViTXS": MobileViTXS,
    "MobileViTv2_050": MobileViTv2_050,
    "MobileViTv2_075": MobileViTv2_075,
}


def get_target_channels(frame_policy: str) -> int:
    """Calculate input channel count based on frame policy."""
    if frame_policy in ["6_channels", "1_49_channels", "25_49_channels"]:
        return 6
    elif frame_policy == "9_channels":
        return 9
    elif frame_policy == "12_channels":
        return 12
    else:
        return 3


def build_backbone(model_name: str, pretrained: bool = True) -> nn.Module:
    """Instantiate video backbone module from MODEL_REGISTRY."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown video backbone '{model_name}'. Available backbones: {available}.")

    backbone_cls = MODEL_REGISTRY[model_name]
    return backbone_cls(classes_num=4, pretrained=pretrained)


def decode_video_sample(
    video_path: str,
    image_size: int = 224,
    frame_policy: str = "1_49_channels"
) -> np.ndarray:
    """
    Decode multi-frame video sample to uint8 numpy array [Channels, Height, Width].
    Falls back gracefully from Decord to OpenCV.
    """
    try:
        sample = _decode_image(
            video_path=video_path,
            label=0,
            image_size=image_size,
            frame_policy=frame_policy,
            split="test"
        )
    except Exception as exc:
        sample = _decode_image_cv2(
            video_path=video_path,
            label=0,
            image_size=image_size,
            frame_policy=frame_policy,
            split="test"
        )
    return sample["image_form"]


class PredictionVideoDataset(Dataset):
    """
    PyTorch Dataset wrapper for video multi-channels inference and Late Fusion export.
    Decodes multi-frame representations and applies normalization.
    """
    def __init__(
        self,
        data_list: List[List[Any]],
        image_size: int = 224,
        frame_policy: str = "1_49_channels"
    ) -> None:
        self.data_list = data_list
        self.image_size = image_size
        self.frame_policy = frame_policy
        self.transform = VideoTransform(image_size=image_size, split="test")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data_list[idx]
        audio_path = str(item[0]) if len(item) > 2 else ""
        video_path = str(item[1]) if len(item) > 2 else str(item[0])
        label = int(item[-1])

        image_uint8 = decode_video_sample(
            video_path=video_path,
            image_size=self.image_size,
            frame_policy=self.frame_policy
        )
        image_tensor = self.transform(image_uint8)

        return {
            "audio_path": audio_path,
            "video_path": video_path,
            "label": label,
            "video_form": image_tensor,
        }


def collate_prediction_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate image tensors and metadata for batched inference."""
    audio_paths = [item["audio_path"] for item in batch]
    video_paths = [item["video_path"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    images = torch.stack([item["video_form"] for item in batch], dim=0)

    return {
        "audio_path": audio_paths,
        "video_path": video_paths,
        "label": labels,
        "video_form": images,
    }


def parse_sample_metadata(video_path: str, audio_path: str = "") -> Dict[str, str]:
    """Extract sample_key, date, session, and sample_id from video/audio path."""
    primary_path = str(audio_path or video_path).replace("\\", "/")
    parts = Path(primary_path).parts
    date_part = parts[-4] if len(parts) >= 4 else ""
    session_part = parts[-3] if len(parts) >= 3 else ""
    stem = Path(primary_path).stem
    if "_video_" in stem:
        sample_id = stem.split("_video_")[-1]
    elif "_audio_" in stem:
        sample_id = stem.split("_audio_")[-1]
    else:
        sample_id = stem

    sample_key = _sample_identity_key(primary_path)

    return {
        "sample_key": sample_key,
        "date": date_part,
        "session": session_part,
        "sample_id": sample_id,
    }


def load_prediction_config(config_path: str) -> Dict[str, Any]:
    """Load configuration JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: '{config_path}'")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_test_split_consistency(
    current_split_dict: List[List[Any]],
    reference_csv_path: Path,
    split_name: str = "test",
    is_explicit: bool = False
) -> bool:
    """
    Verify that current_split_dict matches reference_csv (e.g. <checkpoint_dir>/splits/test.csv).
    Compares:
    1. Total sample count
    2. Sample identity keys (date/session/class/sample_id)
    3. Ground truth labels and sample ordering
    """
    if not reference_csv_path.exists():
        if is_explicit:
            raise FileNotFoundError(
                f"Explicitly requested reference split CSV does not exist: '{reference_csv_path}'"
            )
        logger.info(
            f"No reference {split_name}.csv found at '{reference_csv_path}'. "
            "Proceeding with newly generated native split."
        )
        return False

    logger.info("==================================================")
    logger.info(f"Verifying {split_name} split consistency against historical training split:")
    logger.info(f"  Reference file: '{reference_csv_path}'")
    logger.info("==================================================")

    with reference_csv_path.open("r", encoding="utf-8") as f:
        ref_rows = list(csv.DictReader(f))

    if len(current_split_dict) != len(ref_rows):
        raise ValueError(
            f"Sample count mismatch on '{split_name}' split!\n"
            f"  - Current split:   {len(current_split_dict)} samples\n"
            f"  - Reference split: {len(ref_rows)} samples ({reference_csv_path})\n"
            "Please ensure fold_index, seed, and split_strategy match the training configuration."
        )

    ref_keys = [_sample_identity_key(r.get("video_path") or r.get("audio_path")) for r in ref_rows]
    ref_labels = [int(r["label"]) for r in ref_rows]

    mismatches = []
    order_identical = True

    for idx, item in enumerate(current_split_dict):
        # In video data_split: item is [audio_path, video_path, label]
        curr_primary = item[1] if (len(item) > 2 and item[1]) else item[0]
        curr_key = _sample_identity_key(curr_primary)
        curr_label = int(item[-1])

        ref_k = ref_keys[idx]
        ref_l = ref_labels[idx]

        if curr_key != ref_k or curr_label != ref_l:
            order_identical = False
            mismatches.append({
                "index": idx,
                "current_key": curr_key,
                "current_label": curr_label,
                "ref_key": ref_k,
                "ref_label": ref_l,
            })
            if len(mismatches) >= 5:
                break

    if not order_identical:
        curr_key_set = {_sample_identity_key(item[1] if (len(item) > 2 and item[1]) else item[0]) for item in current_split_dict}
        ref_key_set = set(ref_keys)
        missing_in_current = ref_key_set - curr_key_set
        extra_in_current = curr_key_set - ref_key_set

        if missing_in_current or extra_in_current:
            raise ValueError(
                f"Split verification failed for '{split_name}'!\n"
                f"  - Samples missing in current split: {len(missing_in_current)}\n"
                f"  - Unexpected samples in current split: {len(extra_in_current)}\n"
                f"  - Example mismatch: {mismatches[0]}\n"
                f"Reference file: '{reference_csv_path}'"
            )
        else:
            logger.warning(
                f"All {len(current_split_dict)} samples match reference '{reference_csv_path}', "
                "but the ordering within the split is different."
            )
    else:
        logger.info("==================================================")
        logger.info(
            f"VERIFICATION SUCCESSFUL: All {len(current_split_dict)} {split_name} samples, "
            "exact ordering, and ground truth labels match 100% identically with:\n"
            f"  '{reference_csv_path}'"
        )
        logger.info("==================================================")

    return True


def export_predictions(pred_cfg: Dict[str, Any]) -> str:
    """
    Main routine:
    1. Parse settings (weights, fold_index, model backbone, video features).
    2. Split data using codebase native FishDataSplitter (strictly on chosen fold test split).
    3. Verify split consistency against <checkpoint_dir>/splits/test.csv.
    4. Load VideoModel and checkpoint weights.
    5. Run batch inference and extract logits, probabilities, and metadata.
    6. Save results to a single CSV for Late Fusion.
    """
    # 1. Resolve configuration parameters
    train_config_path = pred_cfg.get("train_config_path", "config/train_config.json")
    if Path(train_config_path).exists():
        base_train_cfg = TrainConfig.from_json(train_config_path)
        logger.info(f"Loaded base training configuration from: '{train_config_path}'")
    else:
        logger.warning(f"train_config_path '{train_config_path}' not found. Using default TrainConfig.")
        base_train_cfg = TrainConfig()

    # Model configuration
    model_info = pred_cfg.get("model", {})
    backbone_name = model_info.get("backbone", base_train_cfg.model.backbone)
    pretrained = model_info.get("pretrained", base_train_cfg.model.pretrained)

    # Video features configuration
    vf_info = pred_cfg.get("video_features", {})
    image_size = int(vf_info.get("image_size", base_train_cfg.video_features.image_size))
    frame_policy = vf_info.get("frame_policy", "1_49_channels")

    # Fold selection & Data Splitter configuration
    fold_index = pred_cfg.get("fold_index", 0)
    splitter_cfg_data = base_train_cfg.dataset_splitter.model_dump()

    # Optional dataset_path override
    dataset_path_override = pred_cfg.get("dataset_path", "").strip()
    if dataset_path_override:
        splitter_cfg_data["dataset_path"] = dataset_path_override

    splitter_cfg_data["evaluation_mode"] = "cross_validation"
    splitter_cfg_data["fold_index"] = fold_index
    splitter_cfg_data["save_results"] = False  # Avoid overwriting dataset splits directory
    splitter_cfg_data["include_video"] = True  # Ensure video_path is resolved

    splitter_config = SplitterConfig(**splitter_cfg_data)

    # Checkpoint path
    checkpoint_path = pred_cfg.get("checkpoint_path", "")
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be specified in prediction_config.json!")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: '{checkpoint_path}'")

    # Auto-detect fold_train_config.json next to checkpoint if train_config_path is default
    ckpt_fold_config = Path(checkpoint_path).parent / "fold_train_config.json"
    if train_config_path == "config/train_config.json" and ckpt_fold_config.exists():
        try:
            with ckpt_fold_config.open("r", encoding="utf-8") as f:
                detected_cfg = json.load(f)
            logger.info(f"Auto-detected fold training config at checkpoint location: '{ckpt_fold_config}'")
            if "video_features" in detected_cfg and "video_features" not in pred_cfg:
                image_size = int(detected_cfg["video_features"].get("image_size", image_size))
                frame_policy = detected_cfg["video_features"].get("frame_policy", frame_policy)
                logger.info(f"Synchronized video_features with fold_train_config.json: size={image_size}, policy='{frame_policy}'.")
            if "model" in detected_cfg and "model" not in pred_cfg:
                backbone_name = detected_cfg["model"].get("backbone", backbone_name)
                pretrained = detected_cfg["model"].get("pretrained", pretrained)
        except Exception as e:
            logger.warning(f"Could not load checkpoint-adjacent fold_train_config.json: {e}")

    target_channels = get_target_channels(frame_policy)

    # Output CSV path
    output_csv_path = pred_cfg.get("output_csv_path", f"outputs/predictions_test_fold{fold_index:02d}.csv")
    output_csv = Path(output_csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Batch size and device
    batch_size = int(pred_cfg.get("batch_size", 64))
    requested_device = pred_cfg.get("device", "auto").lower()
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    target_split = pred_cfg.get("split", "test").lower()

    logger.info("==================================================")
    logger.info("Starting Video Prediction Export for Late Fusion:")
    logger.info(f"  - Model Backbone:       {backbone_name}")
    logger.info(f"  - Pretrained:           {pretrained}")
    logger.info(f"  - Target Channels:      {target_channels} ({frame_policy})")
    logger.info(f"  - Image Resolution:     {image_size}x{image_size}")
    logger.info(f"  - Checkpoint Weight:    '{checkpoint_path}'")
    logger.info(f"  - Fold Index:           {fold_index}")
    logger.info(f"  - Target Split:         '{target_split}'")
    logger.info(f"  - Dataset Path:         '{splitter_config.dataset_path}'")
    logger.info(f"  - Output CSV Path:      '{output_csv}'")
    logger.info(f"  - Batch Size:           {batch_size}")
    logger.info(f"  - Device:               {device}")
    logger.info("==================================================")

    # 2. Split dataset using native codebase FishDataSplitter
    logger.info("Initializing FishDataSplitter to reproduce exact fold splits...")
    splitter = FishDataSplitter(config=splitter_config)
    train_dict, test_dict, val_dict = splitter.split_data()

    if target_split == "test":
        target_dict = test_dict
    elif target_split == "val":
        target_dict = val_dict
    elif target_split == "train":
        target_dict = train_dict
    else:
        raise ValueError(f"Unknown split '{target_split}'. Expected 'test', 'val', or 'train'.")

    logger.info(f"Selected split '{target_split}' with {len(target_dict)} samples.")

    # 2.1 Verify split consistency against training split (splits/test.csv)
    ref_split_override = pred_cfg.get("reference_test_csv", "").strip()
    is_explicit = bool(ref_split_override)
    if is_explicit:
        ref_test_csv = Path(ref_split_override)
    else:
        # Auto-detect splits/test.csv at same level as checkpoint_path
        ref_test_csv = Path(checkpoint_path).parent / "splits" / f"{target_split}.csv"

    verify_test_split_consistency(
        current_split_dict=target_dict,
        reference_csv_path=ref_test_csv,
        split_name=target_split,
        is_explicit=is_explicit
    )

    # Optional sample limit for testing
    limit = pred_cfg.get("limit")
    if limit is not None and int(limit) > 0:
        target_dict = target_dict[:int(limit)]
        logger.info(f"Sample limit applied: evaluating only first {len(target_dict)} samples.")

    # 3. Assemble VideoModel and load checkpoint weights
    logger.info(f"Building VideoModel (backbone='{backbone_name}', target_channels={target_channels}) and loading checkpoint...")
    backbone = build_backbone(model_name=backbone_name, pretrained=pretrained)
    model = VideoModel(backbone=backbone, target_channels=target_channels).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Strip thop buffers (total_ops, total_params) and module. prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith("total_ops") or k.endswith("total_params"):
            continue
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    logger.info(f"Successfully loaded checkpoint weights (epoch={checkpoint.get('epoch', 'N/A')}).")

    # 4. Prepare DataLoader
    dataset = PredictionVideoDataset(
        target_dict,
        image_size=image_size,
        frame_policy=frame_policy
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_prediction_batch,
        num_workers=0,  # 0 ensures smooth cross-platform execution
    )

    # 5. Run inference and accumulate predictions
    logger.info("Running batched inference...")
    records: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    start_time = time.perf_counter()

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Exporting predictions ({target_split})")
        for batch in pbar:
            images = batch["video_form"].to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                logits_tensor = outputs.get("clipwise_output", outputs)
            else:
                logits_tensor = outputs
            if logits_tensor.dim() == 3 and logits_tensor.size(2) == 1:
                logits_tensor = logits_tensor.squeeze(2)

            probs_tensor = torch.softmax(logits_tensor, dim=-1)  # [B, 4]

            batch_logits = logits_tensor.cpu().numpy()
            batch_probs = probs_tensor.cpu().numpy()
            batch_audio_paths = batch["audio_path"]
            batch_video_paths = batch["video_path"]
            batch_labels = batch["label"].cpu().numpy()

            for i in range(len(batch_video_paths)):
                video_p = batch_video_paths[i]
                audio_p = batch_audio_paths[i]
                gt_id = int(batch_labels[i])
                gt_name = LABEL_TO_CLASS.get(gt_id, f"class_{gt_id}")

                logits = batch_logits[i]
                probs = batch_probs[i]
                pred_id = int(np.argmax(probs))
                pred_name = LABEL_TO_CLASS.get(pred_id, f"class_{pred_id}")
                conf = float(probs[pred_id])
                is_correct = int(pred_id == gt_id)

                meta = parse_sample_metadata(video_path=video_p, audio_path=audio_p)

                row = {
                    "sample_key": meta["sample_key"],
                    "date": meta["date"],
                    "session": meta["session"],
                    "sample_id": meta["sample_id"],
                    "audio_path": audio_p,
                    "video_path": video_p,
                    "ground_truth_id": gt_id,
                    "ground_truth_name": gt_name,
                    "pred_class_id": pred_id,
                    "pred_class_name": pred_name,
                    "confidence": round(conf, 6),
                    "is_correct": is_correct,
                    # Logits
                    "logit_none": round(float(logits[0]), 6),
                    "logit_strong": round(float(logits[1]), 6),
                    "logit_medium": round(float(logits[2]), 6),
                    "logit_weak": round(float(logits[3]), 6),
                    # Probabilities
                    "prob_none": round(float(probs[0]), 6),
                    "prob_strong": round(float(probs[1]), 6),
                    "prob_medium": round(float(probs[2]), 6),
                    "prob_weak": round(float(probs[3]), 6),
                }

                records.append(row)
                y_true.append(gt_id)
                y_pred.append(pred_id)

    total_time = time.perf_counter() - start_time
    total_samples = len(records)
    latency_ms = (total_time / total_samples) * 1000 if total_samples > 0 else 0.0

    # 6. Save records to CSV
    fieldnames = [
        "sample_key",
        "date",
        "session",
        "sample_id",
        "audio_path",
        "video_path",
        "ground_truth_id",
        "ground_truth_name",
        "pred_class_id",
        "pred_class_name",
        "confidence",
        "is_correct",
        "logit_none",
        "logit_strong",
        "logit_medium",
        "logit_weak",
        "prob_none",
        "prob_strong",
        "prob_medium",
        "prob_weak",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Successfully exported {len(records)} prediction rows to: '{output_csv}'")

    # 7. Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    clf_report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2, 3],
        target_names=["none (0)", "strong (1)", "medium (2)", "weak (3)"],
        digits=4,
        zero_division=0
    )

    # Save summary JSON alongside CSV
    summary_json_path = output_csv.with_name(f"{output_csv.stem}_summary.json")
    summary_data = {
        "model_backbone": backbone_name,
        "frame_policy": frame_policy,
        "target_channels": target_channels,
        "image_size": image_size,
        "checkpoint_path": checkpoint_path,
        "fold_index": fold_index,
        "split": target_split,
        "total_samples": total_samples,
        "elapsed_time_seconds": round(total_time, 2),
        "latency_ms_per_sample": round(latency_ms, 3),
        "accuracy": round(float(acc), 6),
        "precision_weighted": round(float(prec_weighted), 6),
        "recall_weighted": round(float(rec_weighted), 6),
        "f1_weighted": round(float(f1_weighted), 6),
        "precision_macro": round(float(prec_macro), 6),
        "recall_macro": round(float(rec_macro), 6),
        "f1_macro": round(float(f1_macro), 6),
        "confusion_matrix": cm.tolist(),
        "output_csv_path": str(output_csv),
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved evaluation summary JSON to: '{summary_json_path}'")
    logger.info("==================================================")
    logger.info("EVALUATION RESULTS ON TEST SET:")
    logger.info(f"  - Total Test Samples:    {total_samples}")
    logger.info(f"  - Accuracy:              {acc * 100:.2f}% ({acc:.6f})")
    logger.info(f"  - F1-Score (Weighted):   {f1_weighted:.6f}")
    logger.info(f"  - F1-Score (Macro):      {f1_macro:.6f}")
    logger.info(f"  - Latency:               {latency_ms:.2f} ms/sample")
    logger.info("Detailed Classification Report:\n" + clf_report)
    logger.info("Confusion Matrix (rows: True, cols: Pred):\n" + str(cm))
    logger.info("==================================================")

    return str(output_csv)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with optional overrides."""
    parser = argparse.ArgumentParser(
        description="Export video prediction logits, probabilities, and metadata for Late Fusion."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/prediction_config.json",
        help="Path to prediction configuration JSON file (default: config/prediction_config.json)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Cross-validation fold index override (e.g. 0, 1, 2, 3, 4)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt) override"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Video backbone name override (e.g. MobileNetV2, EfficientNetB0, SwinTiny)"
    )
    parser.add_argument(
        "--frame_policy",
        type=str,
        default=None,
        help="Frame policy override (e.g. 1_49_channels)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Target image size override (e.g. 224)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset root override"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file override"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Inference batch size override"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda', 'cpu', 'auto')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for rapid testing"
    )
    parser.add_argument(
        "--reference_csv",
        type=str,
        default=None,
        help="Path to reference test.csv from training split to verify against"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Load configuration JSON
    cfg = load_prediction_config(args.config)

    # Apply command-line overrides if supplied
    if args.fold is not None:
        cfg["fold_index"] = args.fold
    if args.checkpoint is not None:
        cfg["checkpoint_path"] = args.checkpoint
    if args.model is not None:
        if "model" not in cfg or not isinstance(cfg["model"], dict):
            cfg["model"] = {}
        cfg["model"]["backbone"] = args.model
    if args.frame_policy is not None:
        if "video_features" not in cfg or not isinstance(cfg["video_features"], dict):
            cfg["video_features"] = {}
        cfg["video_features"]["frame_policy"] = args.frame_policy
    if args.image_size is not None:
        if "video_features" not in cfg or not isinstance(cfg["video_features"], dict):
            cfg["video_features"] = {}
        cfg["video_features"]["image_size"] = args.image_size
    if args.dataset is not None:
        cfg["dataset_path"] = args.dataset
    if args.output is not None:
        cfg["output_csv_path"] = args.output
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.reference_csv is not None:
        cfg["reference_test_csv"] = args.reference_csv

    export_predictions(cfg)


if __name__ == "__main__":
    main()
