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

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoTrainer:
    """
    Unified Trainer class for Fish Video Intensity Classification.
    Supports AMP bfloat16 mixed precision on NVIDIA Blackwell GPUs,
    early stopping, history logging, and automatic checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        config: VideoTrainConfig,
        device: torch.device
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.ckpt_dir = config.ckpt_dir

        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.config.save_merged_checkpoint_config(self.ckpt_dir)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # Mixed precision scaler for CUDA
        self.use_amp = torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        logger.info(f"Initialized VideoTrainer (AMP={self.use_amp}, dtype={self.amp_dtype}, device={self.device})")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs} [Train]")
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
        logger.info(f"[{split_name}] Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f}")
        return eval_loss, eval_acc

    def fit(self) -> None:
        best_metric = -1.0 if self.config.monitor == 'accuracy' else float('inf')
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader, split_name="Val")

            current_metric = val_acc if self.config.monitor == 'accuracy' else val_loss
            is_better = (current_metric > best_metric) if self.config.monitor == 'accuracy' else (current_metric < best_metric)

            if is_better:
                best_metric = current_metric
                patience_counter = 0
                save_path = os.path.join(self.ckpt_dir, "best_video_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, save_path)
                logger.info(f"--> Saved Best Model to '{save_path}' (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if self.config.early_stopping and patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}!")
                    break

        logger.info("Evaluating Best Model on Test Dataset...")
        best_ckpt = os.path.join(self.ckpt_dir, "best_video_model.pt")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc = self.evaluate(self.test_loader, split_name="Test")
        logger.info(f"Final Test Result -> Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
