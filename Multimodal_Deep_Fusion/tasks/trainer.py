from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm

from config import TrainConfig
from dataset import save_split_files
from models import MultimodalDeepFusionModel
from utils.metrics import (
    compute_metrics,
    save_confusion_outputs,
    save_history_csv,
    save_history_plot,
    save_metrics_csv,
)

logger = logging.getLogger(__name__)


def _optimizer(parameters, cfg: TrainConfig):
    params = [param for param in parameters if param.requires_grad]
    if cfg.optimizer == "adam":
        return Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return SGD(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, momentum=0.9)
    return AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def _score(metrics: Dict[str, float], monitor: str) -> float:
    if monitor == "loss":
        return -metrics["loss"]
    return metrics[monitor]


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _gpu_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "GPU memory: unavailable"
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    return f"GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"


class MultimodalTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        loaders: Dict[str, torch.utils.data.DataLoader],
        output_dir: str | Path,
        splits: Dict[str, List[Dict]],
        run_name: str,
    ) -> None:
        self.cfg = cfg
        self.loaders = loaders
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.splits = splits
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
        self.model = MultimodalDeepFusionModel(cfg).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = _optimizer(self.model.parameters(), cfg)
        self.history: List[Dict[str, float]] = []

        with (self.output_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)
        save_split_files(splits, self.output_dir / "splits")
        self._log_initial_state()

    def _log_initial_state(self) -> None:
        total_params, trainable_params = _count_parameters(self.model)
        logger.info("==================================================")
        logger.info("Initialized multimodal trainer: %s", self.run_name)
        logger.info("  - Device:                   %s", self.device)
        logger.info("  - Output directory:         %s", self.output_dir)
        logger.info("  - Epochs:                   %s", self.cfg.epochs)
        logger.info("  - Batch size:               %s", self.cfg.batch_size)
        logger.info("  - Optimizer:                %s", self.cfg.optimizer)
        logger.info("  - Learning rate:            %s", self.cfg.learning_rate)
        logger.info("  - Weight decay:             %s", self.cfg.weight_decay)
        logger.info("  - Monitor:                  %s", self.cfg.monitor)
        logger.info("  - Early stopping:           %s", self.cfg.early_stopping)
        logger.info("  - Total parameters:         %s", f"{total_params:,}")
        logger.info("  - Trainable parameters:     %s", f"{trainable_params:,}")
        for split_name in ("train", "val", "test"):
            logger.info("  - %-5s samples:             %s", split_name, len(self.loaders[split_name].dataset))
        logger.info("  - %s", _gpu_memory_summary(self.device))
        logger.info("==================================================")

    def _run_epoch(self, split: str, train: bool, epoch: int | None = None) -> Dict[str, float | List[int]]:
        self.model.train(train)
        total_loss = 0.0
        all_true: List[int] = []
        all_pred: List[int] = []
        loader = self.loaders[split]
        if epoch is None:
            desc = f"{self.run_name} {split}"
        else:
            desc = f"{self.run_name} epoch {epoch:03d}/{self.cfg.epochs:03d} {split}"
        iterator = tqdm(loader, desc=desc, unit="batch")
        for batch in iterator:
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            video_form = batch["video_form"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                output = self.model(waveform=waveform, video_form=video_form)
                logits = output["clipwise_output"]
                loss = self.criterion(logits, target)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

            pred = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * target.size(0)
            all_true.extend(target.detach().cpu().numpy().astype(int).tolist())
            all_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
            acc = float(np.mean(np.asarray(all_true) == np.asarray(all_pred)))
            running_loss = total_loss / max(1, len(all_true))
            iterator.set_postfix(batch_loss=f"{loss.item():.4f}", loss=f"{running_loss:.4f}", acc=f"{acc:.4f}")

        metrics = compute_metrics(all_true, all_pred)
        metrics["loss"] = total_loss / max(1, len(all_true))
        metrics["y_true"] = all_true
        metrics["y_pred"] = all_pred
        logger.info(
            "%s %s completed | loss=%.4f | acc=%.4f | f1_macro=%.4f",
            self.run_name,
            split,
            metrics["loss"],
            metrics["accuracy"],
            metrics["f1_macro"],
        )
        return metrics

    def fit(self) -> Dict[str, float]:
        best_score = -float("inf")
        best_val_accuracy = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        bad_epochs = 0
        best_path = self.output_dir / "checkpoint" / "multimodal_best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("==================================================")
        logger.info("Starting training loop for %s", self.run_name)
        logger.info("Checkpoint path: %s", best_path)
        logger.info("==================================================")

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_epoch("train", train=True, epoch=epoch)
            val_metrics = self._run_epoch("val", train=False, epoch=epoch)
            row = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_f1_macro": float(train_metrics["f1_macro"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_f1_macro": float(val_metrics["f1_macro"]),
            }
            self.history.append(row)
            current_score = _score(val_metrics, self.cfg.monitor)
            improved = current_score > best_score + self.cfg.delta
            if improved:
                best_val_accuracy = float(val_metrics["accuracy"])
                best_val_f1 = float(val_metrics["f1_macro"])
                bad_epochs_for_log = 0
            else:
                bad_epochs_for_log = bad_epochs + 1
            logger.info(
                (
                    "Epoch %03d/%03d summary | train_loss=%.4f | train_acc=%.4f | "
                    "val_loss=%.4f | val_acc=%.4f | val_f1=%.4f | "
                    "best_val_acc=%.4f | best_val_f1=%.4f | bad_epochs=%d | %s"
                ),
                epoch,
                self.cfg.epochs,
                row["train_loss"],
                row["train_accuracy"],
                row["val_loss"],
                row["val_accuracy"],
                row["val_f1_macro"],
                best_val_accuracy,
                best_val_f1,
                bad_epochs_for_log,
                _gpu_memory_summary(self.device),
            )

            if improved:
                best_score = current_score
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.cfg.model_dump(),
                        "val_metrics": {k: v for k, v in val_metrics.items() if not k.startswith("y_")},
                    },
                    best_path,
                )
                logger.info(
                    "Saved new best checkpoint | epoch=%d | monitor=%s | score=%.6f | path=%s",
                    epoch,
                    self.cfg.monitor,
                    best_score,
                    best_path,
                )
            else:
                bad_epochs += 1
                if self.cfg.early_stopping and bad_epochs >= self.cfg.patience:
                    logger.info("Early stopping at epoch %d. Best epoch: %d", epoch, best_epoch)
                    break

        save_history_csv(self.history, self.output_dir / "history.csv")
        save_history_plot(self.history, self.output_dir / "history.png")
        logger.info("Saved training history to %s", self.output_dir)
        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info("Loaded best checkpoint from epoch %d for final test evaluation.", int(checkpoint["epoch"]))
        test_metrics = self._run_epoch("test", train=False)
        result = {
            "best_epoch": float(best_epoch),
            "best_val_score": float(best_score),
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_precision_macro": float(test_metrics["precision_macro"]),
            "test_recall_macro": float(test_metrics["recall_macro"]),
            "test_f1_macro": float(test_metrics["f1_macro"]),
            "test_precision_weighted": float(test_metrics["precision_weighted"]),
            "test_recall_weighted": float(test_metrics["recall_weighted"]),
            "test_f1_weighted": float(test_metrics["f1_weighted"]),
        }
        save_metrics_csv(result, self.output_dir / "result.csv")
        save_confusion_outputs(test_metrics["y_true"], test_metrics["y_pred"], self.output_dir)
        logger.info(
            "Final test result | loss=%.4f | acc=%.4f | f1_macro=%.4f | output=%s",
            result["test_loss"],
            result["test_accuracy"],
            result["test_f1_macro"],
            self.output_dir,
        )
        return result


def save_cv_summary(fold_results: List[Dict[str, float]], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not fold_results:
        return
    fieldnames = sorted(fold_results[0].keys())
    with (output_path / "fold_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + fieldnames)
        writer.writeheader()
        for index, row in enumerate(fold_results):
            writer.writerow({"fold": index, **row})

    summary_rows = []
    for key in fieldnames:
        values = np.asarray([row[key] for row in fold_results], dtype=float)
        summary_rows.append({"metric": key, "mean": float(values.mean()), "std": float(values.std(ddof=1))})
    with (output_path / "summary_mean_std.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Saved cross-validation summary to %s", output_path / "summary_mean_std.csv")
