import logging
import argparse
import copy
import csv
import json
import os
import random
import sys
import zipfile
from datetime import datetime
from pathlib import Path

project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import torch.optim as optim

from config import ArtifactUploadConfig, TrainConfig
from dataset import FishMultimodalDataLoader
from models import MultimodalModel
from tasks import MultimodalTrainer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def safe_filename_part(value: str) -> str:
    safe_chars = []
    for char in str(value):
        if char.isascii() and (char.isalnum() or char in ("-", "_", ".")):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe_value = "".join(safe_chars).strip("._-")
    return safe_value or "unknown"


def build_artifact_filename(config: TrainConfig, timestamp: str, suffix: str = ".zip") -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    filename_parts = [
        "U_FFIA27K_multimodal",
        config.model.audio_backbone,
        config.model.video_backbone,
        config.dataset_splitter.evaluation_mode,
        config.dataset_splitter.split_strategy,
        "first_last",
        timestamp,
    ]
    return f"{'_'.join(safe_filename_part(part) for part in filename_parts)}{suffix}"


def artifact_zip_path(zip_path: str, artifact_filename: str) -> Path:
    return Path(zip_path).with_name(artifact_filename)


def artifact_repo_path(path_in_repo: str, artifact_filename: str) -> str:
    path = Path(path_in_repo)
    if str(path.parent) == ".":
        return artifact_filename
    return str(path.parent / artifact_filename).replace("\\", "/")


def zip_directory(source_dir: str, zip_path: str) -> Path:
    source_path = Path(source_dir).resolve()
    output_path = Path(zip_path).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Artifact source_dir does not exist or is not a directory: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    logger.info("==================================================")
    logger.info(f"Creating artifact zip from: '{source_path}'")
    logger.info(f"Artifact zip path:          '{output_path}'")
    logger.info("==================================================")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in source_path.rglob("*"):
            if not file_path.is_file():
                continue
            resolved_file = file_path.resolve()
            if resolved_file == output_path:
                continue
            zip_file.write(resolved_file, arcname=resolved_file.relative_to(source_path))

    logger.info(f"Successfully created artifact zip: '{output_path}'")
    return output_path


def upload_artifact_if_enabled(upload_config: ArtifactUploadConfig, config: TrainConfig) -> None:
    if not upload_config.enabled:
        logger.info("Artifact upload is disabled. Skipping Hugging Face upload.")
        return
    if not upload_config.repo_id:
        raise ValueError("artifact_upload.repo_id must be set when artifact_upload.enabled is true.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except ImportError:
            token = None
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable must be set or Hugging Face CLI login must be completed when artifact upload is enabled."
        )

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for artifact upload. Install it with requirements.txt.") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(upload_config.path_in_repo).suffix or ".zip"
    artifact_filename = build_artifact_filename(config, timestamp, suffix=suffix)
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
    logger.info(f"Uploading artifact to Hugging Face repo: '{upload_config.repo_id}'")
    logger.info(f"Repo type:                            '{upload_config.repo_type}'")
    logger.info(f"Path in repo:                         '{path_in_repo}'")
    logger.info("==================================================")

    upload_file(
        path_or_fileobj=str(artifact_zip),
        path_in_repo=path_in_repo,
        repo_id=upload_config.repo_id,
        repo_type=upload_config.repo_type,
        token=token,
    )
    logger.info("Successfully uploaded artifact to Hugging Face.")


def build_model(config: TrainConfig) -> MultimodalModel:
    model = MultimodalModel(config.audio_features, pretrained_video=config.model.pretrained_video)
    num_params = model.architecture_num_params()
    if num_params >= 5_000_000:
        raise ValueError(f"Architecture exceeds parameter budget: {num_params:,}")
    logger.info("Architecture parameters (frontend excluded): %s", f"{num_params:,}")
    return model


def log_model_complexity(model: MultimodalModel, config: TrainConfig, device: torch.device) -> None:
    """Log parameter counts and THOP MAC/FLOP estimates before training."""
    try:
        from thop import profile
    except ImportError:
        logger.warning("THOP is not installed; skipping FLOPs profiling.")
        return

    was_training = model.training
    model.eval()
    try:
        audio_features = torch.zeros(1, 1, config.audio_features.mel_bins, config.audio_features.mel_bins, device=device)
        video = torch.zeros(1, 6, config.video_features.image_size, config.video_features.image_size, device=device)
        waveform = torch.zeros(1, config.audio_features.sample_rate * 2, device=device)
        with torch.no_grad():
            architecture_macs, _ = profile(model.architecture, inputs=(audio_features, video), verbose=False)
            total_macs, _ = profile(model, inputs=(waveform, video), verbose=False)

        architecture_params = sum(parameter.numel() for parameter in model.architecture.parameters())
        frontend_params = sum(parameter.numel() for parameter in model.frontend.parameters())
        total_params = sum(parameter.numel() for parameter in model.parameters())
        logger.info("==================================================")
        logger.info("Model Complexity (single sample; FLOPs = 2 x MACs):")
        logger.info("  - Architecture parameters (frontend excluded): %s", f"{architecture_params:,}")
        logger.info("  - Frontend parameters:                         %s", f"{frontend_params:,}")
        logger.info("  - Total model parameters:                      %s", f"{total_params:,}")
        logger.info("  - Architecture MACs:                           %.3f G", architecture_macs / 1e9)
        logger.info("  - Architecture FLOPs:                          %.3f G", 2.0 * architecture_macs / 1e9)
        logger.info("  - Total model MACs:                            %.3f G", total_macs / 1e9)
        logger.info("  - Total model FLOPs:                           %.3f G", 2.0 * total_macs / 1e9)
        logger.info("==================================================")
    except Exception as exc:
        logger.warning("FLOPs profiling failed; training can continue without it: %s", exc)
    finally:
        model.train(was_training)


def model_cv_dir(base_ckpt_dir: str, model_name: str) -> str:
    base_dir = base_ckpt_dir if base_ckpt_dir else "checkpoint"
    if Path(base_dir).name == model_name:
        return base_dir
    return os.path.join(base_dir, model_name)


def write_runtime_config(config: TrainConfig, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config.model_dump(), file, indent=2)
    return str(path)


def read_single_summary_row(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Summary file is empty: {summary_path}")
    return rows[-1]


def aggregate_cv_summaries(model_dir: str, num_folds: int) -> None:
    import statistics

    output_dir = Path(model_dir)
    fold_rows = []
    for fold_index in range(num_folds):
        summary_path = output_dir / f"fold_{fold_index:02d}" / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing fold summary: {summary_path}")
        fold_rows.append({"fold": str(fold_index), **read_single_summary_row(summary_path)})

    metric_columns = [key for key in fold_rows[0] if key != "fold"]
    numeric_columns = []
    for key in metric_columns:
        try:
            [float(row[key]) for row in fold_rows]
            numeric_columns.append(key)
        except (TypeError, ValueError):
            pass

    mean_row, std_row = {"fold": "mean"}, {"fold": "std"}
    for key in metric_columns:
        if key in numeric_columns:
            values = [float(row[key]) for row in fold_rows]
            mean_row[key] = f"{statistics.mean(values):.6f}"
            std_row[key] = f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}"
        else:
            mean_row[key] = std_row[key] = ""

    with (output_dir / "cv_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["fold"] + metric_columns)
        writer.writeheader()
        writer.writerows(fold_rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as file:
        json.dump({"num_folds": num_folds, "folds": fold_rows, "mean": mean_row, "std": std_row}, file, indent=2)


def aggregate_cv_summaries_if_ready(model_dir: str, num_folds: int) -> None:
    if all((Path(model_dir) / f"fold_{index:02d}" / "summary.csv").exists() for index in range(num_folds)):
        aggregate_cv_summaries(model_dir, num_folds)


def selected_cv_fold_indices(config: TrainConfig) -> list[int]:
    if config.dataset_splitter.fold_index is None:
        return list(range(config.dataset_splitter.num_folds))
    fold_index = int(config.dataset_splitter.fold_index)
    if not 0 <= fold_index < config.dataset_splitter.num_folds:
        raise ValueError(f"Invalid fold_index={fold_index}")
    return [fold_index]


def run_multimodal_training(config: TrainConfig, train_config_path: str, device: torch.device, fold_index: int = None) -> str:
    seed = config.dataset_splitter.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(config).to(device)
    log_model_complexity(model, config, device)
    loader_manager = FishMultimodalDataLoader(
        sample_rate=config.audio_features.sample_rate,
        batch_size=config.batch_size,
        dataloader_workers=config.dataloader_workers,
        prefetch_factor=config.prefetch_factor,
        cache_audio=config.cache_audio,
        cache_video_mode=config.cache_video_mode,
        image_size=config.video_features.image_size,
        splitter_config=config.dataset_splitter,
    )
    train_loader = loader_manager.get_dataloader("train", shuffle=True, drop_last=True)
    val_loader = loader_manager.get_dataloader("val", shuffle=False, drop_last=False)
    test_loader = loader_manager.get_dataloader("test", shuffle=False, drop_last=False)

    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=config,
        train_config_path=train_config_path,
        split_saver=loader_manager,
    )
    trainer.train(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, max_epoch=config.epochs)
    if config.dataset_splitter.evaluation_mode == "cross_validation":
        return config.ckpt_dir
    return os.path.join(config.ckpt_dir if config.ckpt_dir else "checkpoint", "multimodal_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first_last audio-video model.")
    parser.add_argument("--config", default=str(Path(project_root) / "config/train_config.json"))
    parser.add_argument("--upload-config", default=str(Path(project_root) / "config/artifact_upload_config.json"))
    args = parser.parse_args()
    train_config_path = str(Path(args.config).resolve())
    artifact_upload_config_path = str(Path(args.upload_config).resolve())
    config = TrainConfig.from_json(train_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("==================================================")
    logger.info("Launching Multimodal Audio-Video Pipeline:")
    logger.info(f"  - Device:                  {device}")
    logger.info(f"  - Epochs:                  {config.epochs}")
    logger.info(f"  - Batch Size:              {config.batch_size}")
    logger.info(f"  - Monitor:                 {config.monitor}")
    logger.info(f"  - Early Stopping:          {config.early_stopping}")
    if config.early_stopping:
        logger.info(f"  - Early Stopping Patience: {config.patience} epochs")
    logger.info(f"  - Audio Backbone:          {config.model.audio_backbone}")
    logger.info(f"  - Video Backbone:          {config.model.video_backbone}")
    logger.info("  - Video Input Policy:      first_last")
    logger.info(f"  - Image Size:              {config.video_features.image_size}")
    logger.info(f"  - Audio Cache RAM:         {config.cache_audio}")
    logger.info(f"  - Video Cache Mode:        {config.cache_video_mode}")
    logger.info("==================================================")

    splitter = config.dataset_splitter
    if splitter.evaluation_mode == "cross_validation":
        model_dir = None
        fold_indices = selected_cv_fold_indices(config)
        for fold in fold_indices:
            fold_config = copy.deepcopy(config)
            fold_config.dataset_splitter.fold_index = fold
            fold_dir = os.path.join(model_cv_dir(config.ckpt_dir, "multimodal_model"), f"fold_{fold:02d}")
            fold_config.ckpt_dir = fold_dir
            runtime_config = write_runtime_config(fold_config, os.path.join(fold_dir, "fold_train_config.json"))
            run_multimodal_training(fold_config, runtime_config, device, fold_index=fold)
            model_dir = str(Path(fold_dir).parent)
        if model_dir:
            if len(fold_indices) == splitter.num_folds:
                aggregate_cv_summaries(model_dir, splitter.num_folds)
            else:
                aggregate_cv_summaries_if_ready(model_dir, splitter.num_folds)
    else:
        run_multimodal_training(config=config, train_config_path=train_config_path, device=device)
    artifact_upload_config = ArtifactUploadConfig.from_json(artifact_upload_config_path)
    upload_artifact_if_enabled(artifact_upload_config, config)


if __name__ == "__main__":
    main()
