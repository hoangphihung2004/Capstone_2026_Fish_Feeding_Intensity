import csv
import copy
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ArtifactUploadConfig, TrainConfig
from dataset import FishDataSplitter
from dataset.dataloader_melspectrogram import FishVoiceDataLoader
from features import AudioFrontend
from main import artifact_repo_path, artifact_zip_path, build_backbone, safe_filename_part, zip_directory
from models import AudioModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_NAME = {0: "none", 1: "strong", 2: "medium", 3: "weak"}
SPLIT_ORDER = ("train", "val", "test")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_feature_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "feature_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing feature config: {config_path}")
    return load_json(config_path)


def load_artifact_upload_config(feature_cfg: Dict[str, Any]) -> ArtifactUploadConfig:
    config_path = PROJECT_ROOT / feature_cfg.get(
        "artifact_upload_config_path", "config/artifact_upload_config.json"
    )
    if not config_path.exists():
        logger.info(f"Artifact upload config not found. Skipping upload: {config_path}")
        return ArtifactUploadConfig(enabled=False)
    return ArtifactUploadConfig.from_json(str(config_path))


def build_feature_artifact_filename(config: TrainConfig, timestamp: str, suffix: str = ".zip") -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    filename_parts = [
        config.model.backbone,
        config.dataset_splitter.evaluation_mode,
        config.dataset_splitter.split_strategy,
        timestamp,
    ]
    safe_stem = "_".join(safe_filename_part(part) for part in filename_parts)
    return f"Features_{safe_stem}{suffix}"


def upload_features_artifact_if_enabled(upload_config: ArtifactUploadConfig, config: TrainConfig) -> None:
    if not upload_config.enabled:
        logger.info("Feature artifact upload is disabled. Skipping Hugging Face upload.")
        return

    if not upload_config.repo_id:
        raise ValueError("artifact_upload.repo_id must be set when artifact upload is enabled.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
        except Exception:
            token = None
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable must be set or Hugging Face CLI login must be completed when artifact upload is enabled."
        )

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for feature artifact upload. Install it with requirements.txt.") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(upload_config.path_in_repo).suffix or ".zip"
    artifact_filename = build_feature_artifact_filename(config, timestamp, suffix=suffix)
    output_zip_path = artifact_zip_path(upload_config.zip_path, artifact_filename)
    artifact_zip = zip_directory(upload_config.source_dir, str(output_zip_path))

    if upload_config.create_repo:
        create_repo(
            repo_id=upload_config.repo_id,
            repo_type=upload_config.repo_type,
            token=token,
            exist_ok=True,
        )

    path_in_repo = artifact_repo_path(upload_config.path_in_repo, artifact_filename)
    logger.info("==================================================")
    logger.info(f"Uploading feature artifact to Hugging Face repo: '{upload_config.repo_id}'")
    logger.info(f"Repo type:                                    '{upload_config.repo_type}'")
    logger.info(f"Path in repo:                                 '{path_in_repo}'")
    logger.info("==================================================")

    upload_file(
        path_or_fileobj=str(artifact_zip),
        path_in_repo=path_in_repo,
        repo_id=upload_config.repo_id,
        repo_type=upload_config.repo_type,
        token=token,
    )
    logger.info("Successfully uploaded feature artifact to Hugging Face.")


def make_train_config(feature_cfg: Dict[str, Any], mode: str, fold_index: int | None = None) -> TrainConfig:
    train_config_path = PROJECT_ROOT / feature_cfg.get("train_config_path", "config/train_config.json")
    config = TrainConfig.from_json(str(train_config_path))
    config.batch_size = int(feature_cfg.get("batch_size", config.batch_size))
    config.cache_audio = bool(feature_cfg.get("cache_audio", False))
    config.dataset_splitter.seed = int(feature_cfg.get("seed", config.dataset_splitter.seed))
    config.dataset_splitter.test_sample_per_class = int(
        feature_cfg.get("test_sample_per_class", config.dataset_splitter.test_sample_per_class)
    )
    config.dataset_splitter.split_strategy = feature_cfg.get(
        "split_strategy", config.dataset_splitter.split_strategy
    )
    config.dataset_splitter.num_folds = int(feature_cfg.get("num_folds", config.dataset_splitter.num_folds))
    config.dataset_splitter.cv_val_ratio = float(
        feature_cfg.get("cv_val_ratio", config.dataset_splitter.cv_val_ratio)
    )
    config.dataset_splitter.evaluation_mode = mode
    config.dataset_splitter.fold_index = fold_index
    config.dataset_splitter.include_video = True
    config.dataset_splitter.save_results = True
    return config


def sample_key_from_path(path: str) -> str:
    normalized = Path(str(path).replace("\\", "/"))
    parts = normalized.parts
    lowered = [p.lower() for p in parts]
    anchor = None
    for name in ("audio", "video"):
        if name in lowered:
            anchor = lowered.index(name)
            break
    if anchor is not None and anchor + 1 < len(parts):
        rel_parts = list(parts[anchor + 1 :])
    else:
        rel_parts = list(parts[-3:])
    rel_parts[-1] = Path(rel_parts[-1]).stem
    return "/".join(rel_parts)


def entry_paths(entry: List[Any]) -> Tuple[str, str, int]:
    if len(entry) >= 3:
        return str(entry[0]), str(entry[1]), int(entry[2])
    return str(entry[0]), "", int(entry[-1])


def entry_to_base_row(entry: List[Any]) -> Dict[str, Any]:
    audio_path, video_path, label = entry_paths(entry)
    return {
        "sample_key": sample_key_from_path(audio_path),
        "label": LABEL_TO_NAME[label],
        "label_id": label,
        "input_path": audio_path,
        "pair_path": video_path,
    }


def unique_entries(entries: Iterable[List[Any]]) -> List[List[Any]]:
    by_key: Dict[str, List[Any]] = {}
    for entry in entries:
        row = entry_to_base_row(entry)
        by_key[row["sample_key"]] = entry
    return [by_key[key] for key in sorted(by_key)]


def split_entries(feature_cfg: Dict[str, Any], mode: str, fold_index: int | None = None) -> Tuple[List, List, List, TrainConfig]:
    config = make_train_config(feature_cfg, mode=mode, fold_index=fold_index)
    splitter = FishDataSplitter(config=config.dataset_splitter)
    train_entries, test_entries, val_entries = splitter.split_data()
    return train_entries, val_entries, test_entries, config


class AudioFeatureDataset(Dataset):
    def __init__(self, entries: List[List[Any]], sample_rate: int) -> None:
        self.entries = entries
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = entry_to_base_row(self.entries[index])
        waveform = FishVoiceDataLoader.load_audio(row["input_path"], sr=self.sample_rate)
        return {**row, "waveform": waveform.numpy()}


def collate_audio(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "waveform": torch.tensor(np.stack([item["waveform"] for item in batch]), dtype=torch.float32),
        "rows": [{k: item[k] for k in ("sample_key", "label", "label_id", "input_path", "pair_path")} for item in batch],
    }


class ClassifierInputHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.features: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        feature = inputs[0]
        self.features = feature.detach()

    def close(self) -> None:
        self.handle.remove()


def build_model(config: TrainConfig, device: torch.device) -> AudioModel:
    frontend = AudioFrontend(config.audio_features)
    backbone = build_backbone(config)
    model = AudioModel(frontend=frontend, backbone=backbone).to(device)
    return model


def default_checkpoint_path(config: TrainConfig, model: AudioModel) -> Path:
    model_name = model.backbone.get_name() if hasattr(model.backbone, "get_name") else model.backbone.__class__.__name__
    return PROJECT_ROOT / (config.ckpt_dir if config.ckpt_dir else "checkpoint") / model_name / "audio_best.pt"


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(f"Missing checkpoint keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        logger.warning(f"Unexpected checkpoint keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    logger.info(f"Loaded checkpoint: {checkpoint_path}")


def extract_all_features(
    model: AudioModel,
    entries: List[List[Any]],
    config: TrainConfig,
    feature_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, List[Dict[str, Any]], str, int]:
    dataset = AudioFeatureDataset(entries=entries, sample_rate=config.audio_features.sample_rate)
    loader = DataLoader(
        dataset,
        batch_size=int(feature_cfg.get("batch_size", config.batch_size)),
        shuffle=False,
        num_workers=int(feature_cfg.get("num_workers", 0)),
        collate_fn=collate_audio,
        pin_memory=device.type == "cuda",
    )
    target_module = model.backbone.fc_audioset
    hook = ClassifierInputHook(target_module)
    model.eval()
    arrays: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    index = 0
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            hook.features = None
            _ = model(waveform)
            if hook.features is None:
                raise RuntimeError("Feature hook did not capture classifier input.")
            features = hook.features
            if features.dim() > 2:
                features = torch.flatten(features, start_dim=1)
            feature_np = features.detach().cpu().numpy().astype(np.float32)
            arrays.append(feature_np)
            for row in batch["rows"]:
                metadata.append(
                    {
                        "feature_index": index,
                        "modality": "audio",
                        **row,
                    }
                )
                index += 1
    hook.close()
    features_np = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 0), dtype=np.float32)
    feature_name = "classifier_input_after_fc1_before_fc_audioset"
    feature_dim = int(features_np.shape[1]) if features_np.ndim == 2 and features_np.shape[0] else 0
    return features_np, metadata, feature_name, feature_dim


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_hash(rows: List[Dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['split']}|{row['sample_key']}|{row['label']}|{row['feature_index']}" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_split(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {split: 0 for split in SPLIT_ORDER}
    class_counts: Dict[str, Dict[str, int]] = {split: {} for split in SPLIT_ORDER}
    for row in rows:
        split = row["split"]
        counts[split] += 1
        class_counts[split][row["label"]] = class_counts[split].get(row["label"], 0) + 1
    return {"counts": counts, "class_counts": class_counts}


def write_split_metadata(
    output_dir: Path,
    mode: str,
    fold: int | None,
    split_entries_map: Dict[str, List[List[Any]]],
    all_index: Dict[str, Dict[str, Any]],
    config: TrainConfig,
) -> None:
    rows: List[Dict[str, Any]] = []
    split_id = f"{mode}_fold_{fold}" if fold is not None else mode
    for split in SPLIT_ORDER:
        for entry in split_entries_map[split]:
            base = entry_to_base_row(entry)
            sample_key = base["sample_key"]
            if sample_key not in all_index:
                raise KeyError(f"Split sample is missing from all features: {sample_key}")
            feature_index = all_index[sample_key]["feature_index"]
            rows.append(
                {
                    "feature_index": feature_index,
                    "split_id": split_id,
                    "mode": mode,
                    "fold": "" if fold is None else fold,
                    "split": split,
                    "sample_key": sample_key,
                    "label": base["label"],
                    "label_id": base["label_id"],
                    "input_path": base["input_path"],
                    "pair_path": base["pair_path"],
                }
            )

    if len({(row["split"], row["sample_key"]) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate split/sample_key rows detected for {split_id}.")

    fieldnames = [
        "feature_index",
        "split_id",
        "mode",
        "fold",
        "split",
        "sample_key",
        "label",
        "label_id",
        "input_path",
        "pair_path",
    ]
    write_csv(output_dir / "metadata.csv", rows, fieldnames)
    summary = summarize_split(rows)
    write_json(
        output_dir / "split_info.json",
        {
            "split_id": split_id,
            "mode": mode,
            "fold": fold,
            "seed": config.dataset_splitter.seed,
            "split_strategy": config.dataset_splitter.split_strategy,
            "test_sample_per_class": config.dataset_splitter.test_sample_per_class,
            "num_folds": config.dataset_splitter.num_folds,
            "cv_val_ratio": config.dataset_splitter.cv_val_ratio,
            "split_hash": split_hash(rows),
            **summary,
        },
    )


def main() -> None:
    feature_cfg = load_feature_config()
    base_config = make_train_config(feature_cfg, mode="holdout")
    requested_device = str(feature_cfg.get("device", "cuda"))
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    output_root = PROJECT_ROOT / feature_cfg.get("output_dir", "outputs")

    holdout_train, holdout_val, holdout_test, holdout_config = split_entries(feature_cfg, mode="holdout")
    all_entries = unique_entries(holdout_train + holdout_val + holdout_test)
    logger.info(f"Unique samples for all-feature extraction: {len(all_entries)}")

    model = build_model(base_config, device)
    ckpt_value = str(feature_cfg.get("checkpoint_path", "")).strip()
    checkpoint_path = Path(ckpt_value) if ckpt_value else default_checkpoint_path(base_config, model)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    load_checkpoint(model, checkpoint_path, device)

    features, all_metadata, feature_name, feature_dim = extract_all_features(
        model=model,
        entries=all_entries,
        config=base_config,
        feature_cfg=feature_cfg,
        device=device,
    )

    all_dir = output_root / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    np.save(all_dir / "features.npy", features)
    write_csv(
        all_dir / "metadata.csv",
        all_metadata,
        ["feature_index", "modality", "sample_key", "label", "label_id", "input_path", "pair_path"],
    )
    write_json(
        all_dir / "feature_info.json",
        {
            "modality": "audio",
            "backbone": base_config.model.backbone,
            "checkpoint_path": str(checkpoint_path),
            "feature_name": feature_name,
            "feature_dim": feature_dim,
            "num_samples": int(features.shape[0]),
            "feature_shape": list(features.shape),
            "dataset_path": base_config.dataset_splitter.dataset_path,
        },
    )

    all_index = {row["sample_key"]: row for row in all_metadata}
    write_split_metadata(
        output_root / "holdout",
        mode="holdout",
        fold=None,
        split_entries_map={"train": holdout_train, "val": holdout_val, "test": holdout_test},
        all_index=all_index,
        config=holdout_config,
    )

    cv_root = output_root / "cross_validation"
    for fold_index in range(base_config.dataset_splitter.num_folds):
        train_entries, val_entries, test_entries, fold_config = split_entries(
            feature_cfg, mode="cross_validation", fold_index=fold_index
        )
        write_split_metadata(
            cv_root / f"fold_{fold_index}",
            mode="cross_validation",
            fold=fold_index,
            split_entries_map={"train": train_entries, "val": val_entries, "test": test_entries},
            all_index=all_index,
            config=fold_config,
        )

    logger.info(f"Feature extraction completed successfully. Output: {output_root}")
    artifact_upload_config = load_artifact_upload_config(feature_cfg)
    upload_features_artifact_if_enabled(artifact_upload_config, base_config)


if __name__ == "__main__":
    main()
