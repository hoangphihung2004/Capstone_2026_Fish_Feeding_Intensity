import os
import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from config import VideoTrainConfig
from dataset import FishVideoDataLoader
from models import S3D, VideoModel
from tasks import VideoTrainer

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("==================================================")
    logger.info("Launching Video Pipeline Training (DL_video)")
    logger.info("==================================================")

    # 1. Load unified video configuration
    config = VideoTrainConfig.from_json('config/train_config.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Execution Device: '{device}' ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Instantiate FishVideoDataLoader
    logger.info("Initializing DataLoaders...")
    data_manager = FishVideoDataLoader(
        batch_size=config.batch_size,
        num_workers=-1,
        cache_video=config.cache_video,
        image_size=config.video_features.image_size,
        frames_count=config.video_features.frames,
        splitter_config=config.dataset_splitter
    )

    train_loader = data_manager.get_dataloader(split='train', shuffle=True)
    val_loader = data_manager.get_dataloader(split='val', shuffle=False)
    test_loader = data_manager.get_dataloader(split='test', shuffle=False)

    logger.info(f"Dataset Split Sizes -> Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 3. Instantiate Video Backbone & Unified VideoModel Wrapper
    logger.info("Assembling Video Backbone and Unified VideoModel Wrapper...")
    backbone = S3D(classes_num=4)
    model = VideoModel(backbone=backbone)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters Summary -> Total: {total_params:,} | Trainable: {trainable_params:,}")

    # 4. Instantiate and run VideoTrainer
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    logger.info("Starting Video Training Pipeline...")
    trainer.fit()
    logger.info("==================================================")
    logger.info("Video Training Pipeline Finished Successfully!")
    logger.info("==================================================")


if __name__ == '__main__':
    main()
