import os
import sys
from pathlib import Path
from datetime import datetime
import zipfile

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch
import torch.optim as optim

# Import refactored OOP components
from config import ArtifactUploadConfig, TrainConfig
from dataset import FishVideoDataLoader
from models import MobileNetV2, VideoModel
from tasks import VideoTrainer

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


def timestamped_repo_path(path_in_repo: str) -> str:
    path = Path(path_in_repo)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{path.stem}_{timestamp}{path.suffix}"
    parent = path.parent
    if str(parent) == ".":
        return filename
    return str(parent / filename).replace("\\", "/")


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


def upload_artifact_if_enabled(upload_config: ArtifactUploadConfig) -> None:
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
        raise EnvironmentError("HF_TOKEN environment variable must be set or Hugging Face CLI login must be completed when artifact upload is enabled.")

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for artifact upload. Install it with requirements.txt.") from exc

    artifact_zip = zip_directory(upload_config.source_dir, upload_config.zip_path)

    if upload_config.create_repo:
        create_repo(
            repo_id=upload_config.repo_id,
            repo_type=upload_config.repo_type,
            token=token,
            exist_ok=True,
        )

    logger.info("==================================================")
    logger.info(f"Uploading artifact to Hugging Face repo: '{upload_config.repo_id}'")
    logger.info(f"Repo type:                            '{upload_config.repo_type}'")
    path_in_repo = timestamped_repo_path(upload_config.path_in_repo)
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


def main():
    # 1. Main configuration file path (override this when running other configs)
    train_config_path = 'config/train_config.json'
    artifact_upload_config_path = 'config/artifact_upload_config.json'

    # Load unified training configurations
    config = TrainConfig.from_json(train_config_path)

    logger.info("==================================================")
    logger.info("Launching Single-Frame Image Classification Training:")
    logger.info(f"  - Device Name:              {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"  - Max Epochs:               {config.epochs}")
    logger.info(f"  - Batch Size:               {config.batch_size}")
    logger.info(f"  - Learning Rate (LR):       {config.learning_rate}")
    logger.info(f"  - Checkpoint Directory:     '{config.ckpt_dir}'")
    logger.info(f"  - Monitor Metric:           '{config.monitor}'")
    logger.info("==================================================")

    # 2. Hardware device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training will run on device: '{device}'")

    # 3. Initialize FishVideoDataLoader and fetch splits
    logger.info("Initializing DataLoaders...")
    loader_manager = FishVideoDataLoader(
        batch_size=config.batch_size,
        dataloader_workers=config.dataloader_workers,
        prefetch_factor=config.prefetch_factor,
        cache_mode=config.cache_mode,
        image_size=config.video_features.image_size,
        splitter_config=config.dataset_splitter
    )

    train_loader = loader_manager.get_dataloader(split='train', shuffle=True, drop_last=False)
    val_loader = loader_manager.get_dataloader(split='val', shuffle=False)
    test_loader = loader_manager.get_dataloader(split='test', shuffle=False)

    # 4. Construct unified VideoModel
    logger.info("Assembling neural network model layers...")
    backbone = MobileNetV2(classes_num=4, pretrained=True)
    model = VideoModel(backbone=backbone)
    model = model.to(device)

    # 5. Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 6. Instantiate VideoTrainer
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        optimizer=optimizer,
        train_config_path=train_config_path
    )

    # 7. Start Training & Evaluation process
    trainer.fit()

    artifact_upload_config = ArtifactUploadConfig.from_json(artifact_upload_config_path)
    upload_artifact_if_enabled(artifact_upload_config)

    logger.info("Training pipeline finished successfully!")


if __name__ == '__main__':
    main()
