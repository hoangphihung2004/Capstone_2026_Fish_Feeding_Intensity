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
from dataset.dataloader_melspectrogram import FishVoiceDataLoader
from features import AudioFrontend
from models import Cnn14MobileV2, AudioModel
from tasks import AudioTrainer

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
    logger.info("Launching Audio Pipeline Training (Centralized Config):")
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

    # 3. Initialize FishVoiceDataLoader and fetch splits
    logger.info("Initializing DataLoaders...")
    loader_manager = FishVoiceDataLoader(
        sample_rate=config.audio_features.sample_rate,
        batch_size=config.batch_size,
        num_workers=-1,
        cache_audio=config.cache_audio,
        splitter_config=config.dataset_splitter
    )
    
    train_loader = loader_manager.get_dataloader('train', shuffle=True)
    val_loader = loader_manager.get_dataloader('val', shuffle=False)
    test_loader = loader_manager.get_dataloader('test', shuffle=False)

    # 4. Construct unified AudioModel
    logger.info("Assembling neural network model layers...")
    frontend = AudioFrontend(config=config.audio_features)
    backbone = Cnn14MobileV2(classes_num=4)
    
    model = AudioModel(frontend=frontend, backbone=backbone)
    model = model.to(device)

    # 5. Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 6. Instantiate AudioTrainer
    trainer = AudioTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        ckpt_dir=config.ckpt_dir,
        monitor=config.monitor,
        early_stopping=config.early_stopping,
        patience=config.patience,
        delta=config.delta,
        train_config_path=train_config_path
    )

    # 7. Start Training & Evaluation process
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        max_epoch=config.epochs
    )

    logger.info("Training pipeline finished successfully!")


if __name__ == '__main__':
    main()
