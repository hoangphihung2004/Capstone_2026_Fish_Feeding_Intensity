import logging
import os
import shutil
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics as sklearn_metrics
from tqdm import tqdm

from config import TrainConfig
from utils import ClipCELoss, EarlyStopping, HistoryLogger, InferenceTimer, MultimodalEvaluator
from models.multimodal_model import HEAD_KEYS

logger = logging.getLogger(__name__)


class MultimodalTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, config: TrainConfig, train_config_path: str = "config/train_config.json", split_saver: Any = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.loss_fn = ClipCELoss(
            audio_weight=config.audio_loss_weight,
            video_weight=config.video_loss_weight,
            multimodal_weight=config.multimodal_loss_weight,
        )
        self.evaluator = MultimodalEvaluator(model=self.model, loss_fn=self.loss_fn)
        self.early_stopper = EarlyStopping(patience=config.patience, delta=config.delta, verbose=True) if config.early_stopping else None
        base_dir = config.ckpt_dir if config.ckpt_dir else "checkpoint"
        base_path = Path(base_dir)
        if base_path.name.startswith("fold_") and base_path.parent.name == "multimodal_model":
            self.ckpt_dir = base_dir
        elif not base_dir.rstrip("/\\").endswith("multimodal_model"):
            self.ckpt_dir = os.path.join(base_dir, "multimodal_model")
        else:
            self.ckpt_dir = base_dir
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.history_logger = HistoryLogger(log_dir=self.ckpt_dir)

        # Keep the same experiment backup convention as the baseline trainers.
        try:
            if os.path.exists(train_config_path):
                shutil.copy(train_config_path, os.path.join(self.ckpt_dir, "train_config.json"))
            else:
                with open(os.path.join(self.ckpt_dir, "train_config.json"), "w", encoding="utf-8") as file:
                    json.dump(config.model_dump(), file, indent=2)
            with open(os.path.join(self.ckpt_dir, "splitter_config.json"), "w", encoding="utf-8") as file:
                json.dump(config.dataset_splitter.model_dump(), file, indent=2)
            if split_saver is not None:
                split_saver.save_splits(Path(self.ckpt_dir) / "splits")
        except Exception as exc:
            logger.warning("Warning: Failed to backup configurations or splits: %s", exc)

    def _save_checkpoint(self, path: str, epoch: int, metric_val: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ave_precision": metric_val,
            },
            path,
        )
        logger.info(f"Saved best multimodal checkpoint to: '{path}'")

    def train(self, train_loader: Any, val_loader: Any, test_loader: Any, max_epoch: int) -> None:
        for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
            if len(loader) == 0:
                raise ValueError(f"{name} loader is empty; check split sizes and batch_size/drop_last.")
        if max_epoch < 1:
            raise ValueError("max_epoch must be positive.")
        best_metric = -float("inf") if self.config.monitor == "accuracy" else float("inf")
        best_epoch = -1
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_val_mAP = 0.0
        best_val_statistics = None
        start = time.perf_counter()
        for epoch in range(max_epoch):
            self.model.train()
            total_loss = 0.0
            head_preds = {head: [] for head in HEAD_KEYS}
            targets = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epoch}"):
                waveform = batch["waveform"].to(self.device)
                video_form = batch["video_form"].to(self.device)
                target = batch["target"].to(self.device)
                output = self.model(waveform=waveform, video_form=video_form)
                losses = self.loss_fn.components(output, {"target": target})
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                for head, output_key in HEAD_KEYS.items():
                    head_preds[head].append(output[output_key].detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())

            train_loss = total_loss / len(train_loader)
            train_targets = np.concatenate(targets, axis=0)
            if train_targets.ndim == 1:
                train_targets = np.eye(4)[train_targets.astype(int)]
            train_head_acc = {}
            for head in HEAD_KEYS:
                predictions = np.concatenate(head_preds[head], axis=0)
                train_head_acc[head] = float(
                    np.mean(np.argmax(train_targets, axis=1) == np.argmax(predictions, axis=1))
                )
            train_acc = train_head_acc["multimodal"]
            train_preds = np.concatenate(head_preds["multimodal"], axis=0)
            train_mAP = np.mean(sklearn_metrics.average_precision_score(train_targets, train_preds, average=None))

            val_statistics = self.evaluator.evaluate(val_loader)
            val_acc = float(val_statistics["accuracy"])
            val_mAP = float(np.mean(val_statistics["average_precision"]))
            val_loss = val_statistics["loss"]
            val_head_acc = {
                head: float(statistics["accuracy"])
                for head, statistics in val_statistics["heads"].items()
            }
            if not np.isfinite(val_loss):
                raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}.")
            is_best = (val_acc > best_metric) if self.config.monitor == "accuracy" else (val_loss < best_metric)
            if is_best:
                best_metric = val_acc if self.config.monitor == "accuracy" else val_loss
                best_epoch = epoch
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_mAP = val_mAP
                best_val_statistics = val_statistics
                self._save_checkpoint(os.path.join(self.ckpt_dir, "multimodal_best.pt"), epoch, best_metric)
            self.history_logger.log_epoch(epoch, train_loss, train_acc, train_mAP, val_loss, val_statistics, is_best)
            if self.config.monitor == "accuracy":
                early_stop_score = val_acc
            else:
                early_stop_score = -val_loss
            should_stop = False
            if self.early_stopper is not None:
                should_stop = self.early_stopper.step(early_stop_score)
            logger.info(
                "Epoch %03d/%03d | Train Loss (3 heads): %.5f | Val Loss (3 heads): %.5f | "
                "Train Acc A/V/F: %.4f/%.4f/%.4f | Val Acc A/V/F: %.4f/%.4f/%.4f",
                epoch + 1,
                max_epoch,
                train_loss,
                val_loss,
                train_head_acc["audio"],
                train_head_acc["video"],
                train_head_acc["multimodal"],
                val_head_acc["audio"],
                val_head_acc["video"],
                val_head_acc["multimodal"],
            )
            early_suffix = ""
            if self.early_stopper is not None:
                early_suffix = f" (no improvement for {self.early_stopper.counter}/{self.early_stopper.patience} epochs)"
            logger.info(
                "Best Epoch: %03d | Val Loss: %.5f | Val Accuracy (Fusion): %.4f | Val mAP (Fusion): %.4f%s",
                best_epoch + 1,
                best_val_loss,
                best_val_acc,
                best_val_mAP,
                early_suffix,
            )
            if should_stop:
                logger.warning(
                    "Early stopping triggered at epoch %d/%d after %d/%d epochs without improvement.",
                    epoch + 1,
                    max_epoch,
                    self.early_stopper.counter,
                    self.early_stopper.patience,
                )
                break

        training_time = time.perf_counter() - start
        best_path = os.path.join(self.ckpt_dir, "multimodal_best.pt")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            best_val_statistics = self.evaluator.evaluate(val_loader)
            test_statistics = self.evaluator.evaluate(test_loader)
            self.history_logger.save_confusion_matrices("val", best_val_statistics)
            self.history_logger.save_confusion_matrices("test", test_statistics)
            timer = InferenceTimer(self.model, self.device)
            latency = timer.measure_latency_per_sample(
                waveform_shape=(1, self.config.audio_features.sample_rate * 2),
                video_shape=(1, 6, self.config.video_features.image_size, self.config.video_features.image_size),
            )
            if best_val_statistics is not None:
                self.history_logger.save_summary(training_time, latency, best_val_statistics, test_statistics)
            for head, stats in test_statistics["heads"].items():
                logger.info("TEST %s -> Accuracy: %.4f | mAP: %.4f", head, stats["accuracy"], np.nanmean(stats["average_precision"]))
            return None
        raise RuntimeError("Training did not produce a best checkpoint.")
