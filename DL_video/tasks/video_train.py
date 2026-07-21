import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import VideoTrainConfig
from dataset import FishVideoDataLoader
from utils import EarlyStopping, HistoryLogger, VideoEvaluator

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoTrainer:
    """
    Unified Trainer class for Fish Video Intensity Classification.
    Identical OOP structure with AudioTrainer: supports AMP bfloat16 mixed precision,
    HistoryLogger, EarlyStopping, and automatic experiment checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        config: VideoTrainConfig,
        device: torch.device,
        optimizer: Optional[optim.Optimizer] = None
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.ckpt_dir = config.ckpt_dir

        os.makedirs(self.ckpt_dir, exist_ok=True)
        config_dest = os.path.join(self.ckpt_dir, 'video_config.json')
        with open(config_dest, 'w', encoding='utf-8') as f:
            json.dump(self.config.model_dump(), f, indent=2)

        self.criterion = nn.CrossEntropyLoss()
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Early stopping & History logger
        self.early_stopping = config.early_stopping
        if self.early_stopping:
            self.early_stopper = EarlyStopping(patience=config.patience, delta=config.delta, verbose=True)
        else:
            self.early_stopper = None

        self.history_logger = HistoryLogger(log_dir=self.ckpt_dir)

        # Mixed precision settings
        self.use_amp = torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        logger.info("==================================================")
        logger.info("Initialized VideoTrainer:")
        logger.info(f"  - Optimizer:                   {self.optimizer.__class__.__name__} (LR={self.config.learning_rate})")
        logger.info(f"  - Loss Function:               {self.criterion.__class__.__name__}")
        logger.info(f"  - AMP Precision:               AMP={self.use_amp} ({self.amp_dtype})")
        logger.info(f"  - Checkpoint Dir:              '{self.ckpt_dir}'")
        logger.info(f"  - Monitor Metric:              '{self.config.monitor}'")
        logger.info(f"  - Early Stopping:              {self.early_stopping} (Patience={self.config.patience})")
        logger.info("==================================================")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}/{self.config.epochs:03d} [Train]")
        for batch in pbar:
            inputs = batch['video_form'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs.get('clipwise_output', outputs)
                    
                    if targets.dim() > 1:
                        target_labels = targets.argmax(dim=1)
                    else:
                        target_labels = targets
                    loss = self.criterion(outputs, target_labels)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs.get('clipwise_output', outputs)

                if targets.dim() > 1:
                    target_labels = targets.argmax(dim=1)
                else:
                    target_labels = targets
                loss = self.criterion(outputs, target_labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target_labels).sum().item()
            total += inputs.size(0)

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct / total:.4f}"})

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, dataloader: Any, split_name: str = "Val") -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            inputs = batch['video_form'].to(self.device)
            targets = batch['target'].to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs.get('clipwise_output', outputs)
                    if targets.dim() > 1:
                        target_labels = targets.argmax(dim=1)
                    else:
                        target_labels = targets
                    loss = self.criterion(outputs, target_labels)
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs.get('clipwise_output', outputs)
                if targets.dim() > 1:
                    target_labels = targets.argmax(dim=1)
                else:
                    target_labels = targets
                loss = self.criterion(outputs, target_labels)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target_labels).sum().item()
            total += inputs.size(0)

        eval_loss = total_loss / total
        eval_acc = correct / total
        logger.info(f"[{split_name} Evaluation] Loss: {eval_loss:.4f} | Accuracy: {eval_acc:.4f}")
        return eval_loss, eval_acc

    def fit(self) -> None:
        best_metric = -1.0 if self.config.monitor == 'accuracy' else float('inf')

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader, split_name="Val")

            # Log history to CSV & dictionary
            self.history_logger.log(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc
            )

            current_metric = val_acc if self.config.monitor == 'accuracy' else val_loss
            is_better = (current_metric > best_metric) if self.config.monitor == 'accuracy' else (current_metric < best_metric)

            if is_better:
                best_metric = current_metric
                save_path = os.path.join(self.ckpt_dir, "best_video_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, save_path)
                logger.info(f"--> Improved {self.config.monitor}! Saved Best Model to '{save_path}' (Val Acc: {val_acc:.4f})")

            # Early stopping check
            if self.early_stopping and self.early_stopper is not None:
                self.early_stopper(val_loss if self.config.monitor == 'loss' else val_acc)
                if self.early_stopper.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}!")
                    break

        logger.info("==================================================")
        logger.info("Evaluating Best Model on Test Dataset...")
        best_ckpt = os.path.join(self.ckpt_dir, "best_video_model.pt")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        evaluator = VideoEvaluator(model=self.model)
        test_metrics = evaluator.evaluate(self.test_loader)
        logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f} | mAP: {np.mean(test_metrics['average_precision']):.4f}")
        logger.info(f"Detailed Classification Report:{test_metrics['message']}")
        logger.info("==================================================")
