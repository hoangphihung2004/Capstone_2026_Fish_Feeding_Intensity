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
from models import S3D
from tasks import VideoTrainer

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("==================================================")
    logger.info("Launching Video Pipeline Training Sub-Project (DL_video)")
    logger.info("==================================================")

    # 1. Load modular video configuration
    config = VideoTrainConfig.load_modular(
        common_path='config/video_common.json',
        aug_path='config/video_transform.json',
        model_path='config/models/s3d.json'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Instantiate FishVideoDataLoader
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

    # 3. Instantiate Video Model (e.g. S3D)
    model_classes = config.model_config.get("classes_num", 4)
    model = S3D(classes_num=model_classes)
    model = model.to(device)

    # Apply PyTorch 2.0 compiler optimization if supported
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            logger.info("Applying torch.compile() model acceleration for Blackwell GPU...")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Could not compile model with torch.compile: {e}")

    # 4. Instantiate and run VideoTrainer
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    trainer.fit()


if __name__ == '__main__':
    main()
