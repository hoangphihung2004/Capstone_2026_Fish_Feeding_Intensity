import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch
import torch.optim as optim

# Import refactored OOP components
from config import TrainConfig
from dataset import FishVideoDataLoader
from models import S3D, VideoModel
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


def main():
    # 1. Main configuration file path (override this when running other configs)
    train_config_path = 'config/train_config.json'

    # Load unified training configurations
    config = TrainConfig.from_json(train_config_path)

    logger.info("==================================================")
    logger.info("Launching Video Pipeline Training (Centralized Config):")
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
        preload_workers=config.preload_workers,
        dataloader_workers=config.dataloader_workers,
        prefetch_factor=config.prefetch_factor,
        cache_video=config.cache_video,
        image_size=config.video_features.image_size,
        frames_count=config.video_features.frames,
        splitter_config=config.dataset_splitter
    )

    train_loader = loader_manager.get_dataloader(split='train', shuffle=True, drop_last=True)
    val_loader = loader_manager.get_dataloader(split='val', shuffle=False)
    test_loader = loader_manager.get_dataloader(split='test', shuffle=False)

    # 4. Construct unified VideoModel
    logger.info("Assembling neural network model layers...")
    backbone = S3D(classes_num=4)
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
        optimizer=optimizer
    )

    # 7. Start Training & Evaluation process
    trainer.fit()

    logger.info("Training pipeline finished successfully!")


if __name__ == '__main__':
    main()
