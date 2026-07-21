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
from utils import EarlyStopping, HistoryLogger, VideoEvaluator, InferenceTimer

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
    HistoryLogger, EarlyStopping, InferenceTimer, and automatic experiment checkpointing.
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
        # Determine model name for the checkpoint subfolder (matching AudioTrainer)
        model_name = "video_model"
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'get_name'):
            model_name = model.backbone.get_name()

        base_dir = config.ckpt_dir if config.ckpt_dir else "video_checkpoint"
        if not base_dir.rstrip('/\\').endswith(model_name):
            self.ckpt_dir = os.path.join(base_dir, model_name)
        else:
            self.ckpt_dir = base_dir

        os.makedirs(self.ckpt_dir, exist_ok=True)
        config_dest = os.path.join(self.ckpt_dir, 'train_config.json')
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
        self.evaluator = VideoEvaluator(model=self.model)

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

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        train_preds = []
        train_targets = []

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
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = total_loss / len(self.train_loader.dataset)
        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

        target_acc_labels = np.argmax(train_targets, axis=1) if train_targets.ndim > 1 else train_targets
        pred_acc_labels = np.argmax(train_preds, axis=1)
        train_acc = np.mean(target_acc_labels == pred_acc_labels)

        from sklearn import metrics as sklearn_metrics
        train_mAP = np.mean(sklearn_metrics.average_precision_score(train_targets, train_preds, average=None))

        return epoch_loss, train_acc, train_mAP

    def fit(self) -> None:
        best_val_metric = -1.0 if self.config.monitor == 'accuracy' else float('inf')
        best_epoch = 0
        best_val_statistics = None
        train_start_time = time.perf_counter()

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc, train_mAP = self.train_epoch(epoch)
            
            # Evaluate on validation set
            self.model.eval()
            val_statistics = self.evaluator.evaluate(self.val_loader)
            val_acc = val_statistics['accuracy']
            val_mAP = np.mean(val_statistics['average_precision'])

            # Compute validation loss
            val_loss_sum = 0.0
            with torch.no_grad():
                for val_batch in self.val_loader:
                    val_inputs = val_batch['video_form'].to(self.device)
                    val_targets = val_batch['target'].to(self.device)
                    val_outputs = self.model(val_inputs)
                    if isinstance(val_outputs, dict):
                        val_outputs = val_outputs.get('clipwise_output', val_outputs)
                    val_targ_labels = val_targets.argmax(dim=1) if val_targets.dim() > 1 else val_targets
                    val_loss_sum += self.criterion(val_outputs, val_targ_labels).item() * val_inputs.size(0)
            val_loss = val_loss_sum / len(self.val_loader.dataset)

            logger.info(
                f"Epoch {epoch:03d}: "
                f"Train Loss = {train_loss:.5f} | Train Acc = {train_acc:.4f} | Train mAP = {train_mAP:.4f} | "
                f"Val Loss = {val_loss:.5f} | Val Acc = {val_acc:.4f} | Val mAP = {val_mAP:.4f}"
            )

            is_best = False
            current_metric = val_acc if self.config.monitor == 'accuracy' else val_loss
            if self.config.monitor == 'accuracy':
                if val_acc > best_val_metric:
                    best_val_metric = val_acc
                    is_best = True
            else:
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    is_best = True

            if is_best:
                best_epoch = epoch
                best_val_statistics = val_statistics
                save_path = os.path.join(self.ckpt_dir, "video_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, save_path)
                logger.info(f"--> Saved Best Model to '{save_path}' (Val Acc: {val_acc:.4f} | Val mAP: {val_mAP:.4f})")

            # Log epoch metrics to CSV
            self.history_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_mAP=train_mAP,
                val_loss=val_loss,
                val_statistics=val_statistics,
                is_best=is_best
            )

            # Early stopping check
            if self.early_stopping and self.early_stopper is not None:
                score = val_acc if self.config.monitor == 'accuracy' else -val_loss
                if self.early_stopper.step(score):
                    logger.info(f"Early stopping triggered at epoch {epoch}!")
                    break

        training_time = time.perf_counter() - train_start_time

        # Generate learning curve plots
        try:
            self.history_logger.plot_history()
        except Exception as e:
            logger.warning(f"Warning: Failed to generate learning curves plot: {e}")

        # Final evaluation on Test dataset
        logger.info("==================================================")
        logger.info("Training complete. Starting evaluation on Test split...")
        best_ckpt = os.path.join(self.ckpt_dir, "video_best.pt")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Reloaded best checkpoint model from Epoch {checkpoint['epoch']}...")

        test_statistics = self.evaluator.evaluate(self.test_loader)
        test_mAP = np.mean(test_statistics['average_precision'])
        test_acc = test_statistics['accuracy']
        logger.info(f"TEST Results -> Accuracy: {test_acc:.4f} | mAP: {test_mAP:.4f}")
        logger.info(f"Detailed Classification Report:{test_statistics['message']}")

        # Measure model inference latency
        logger.info("Measuring model Inference Latency on device...")
        timer = InferenceTimer(model=self.model, device=self.device)
        img_size = self.config.video_features.image_size
        frames = self.config.video_features.frames
        latency_ms = timer.measure_latency_per_sample(
            input_shape=(1, 3, frames, img_size, img_size),
            warm_up_steps=10,
            num_steps=50
        )

        # Save performance and timing summary to summary.csv
        if best_val_statistics is not None:
            self.history_logger.save_summary(training_time, latency_ms, best_val_statistics, test_statistics)

        logger.info("==================================================")
