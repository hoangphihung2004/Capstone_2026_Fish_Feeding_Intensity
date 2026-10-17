import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sklearn_metrics
from tqdm import tqdm

from config import TrainConfig
from models.multimodal_model import HEAD_KEYS as TEACHER_HEAD_KEYS
from models.student_model import STUDENT_HEAD_KEYS
from utils import ClipCELoss, EarlyStopping, HintonKDLoss, HistoryLogger, InferenceTimer, MultimodalEvaluator

logger = logging.getLogger(__name__)

MODEL_DIR_NAME = "kd_audio_student"


class KDTrainer:
    """Offline Hinton distillation: frozen multimodal teacher -> audio-only student."""

    def __init__(self, student: nn.Module, teacher: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, config: TrainConfig, lr_scheduler: Any = None, train_config_path: str = "config/train_config.json", split_saver: Any = None) -> None:
        self.model = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.config = config
        self.loss_fn = HintonKDLoss(temperature=config.kd.temperature, alpha=config.kd.alpha)
        self.evaluator = MultimodalEvaluator(
            model=self.model,
            loss_fn=ClipCELoss(head="student"),
            head_keys=STUDENT_HEAD_KEYS,
            use_video=False,
        )
        self.teacher_evaluator = MultimodalEvaluator(
            model=self.teacher,
            loss_fn=ClipCELoss(head="multimodal"),
            head_keys=TEACHER_HEAD_KEYS,
            use_video=True,
        )
        self.early_stopper = EarlyStopping(patience=config.patience, delta=config.delta, verbose=True) if config.early_stopping else None
        base_dir = config.ckpt_dir if config.ckpt_dir else "checkpoint"
        base_path = Path(base_dir)
        if base_path.name.startswith("fold_") and base_path.parent.name == MODEL_DIR_NAME:
            self.ckpt_dir = base_dir
        elif not base_dir.rstrip("/\\").endswith(MODEL_DIR_NAME):
            self.ckpt_dir = os.path.join(base_dir, MODEL_DIR_NAME)
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
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "best_metric": metric_val,
                "monitor": self.config.monitor,
                "kd": self.config.kd.model_dump(),
            },
            path,
        )
        logger.info(f"Saved best student checkpoint to: '{path}'")

    def _evaluate_teacher(self, val_loader: Any, test_loader: Any) -> None:
        """Log the frozen teacher's scores on this fold as the KD reference ceiling."""
        reference = {"teacher_checkpoint": self.config.kd.teacher_checkpoint}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            statistics = self.teacher_evaluator.evaluate(loader)
            if split == "val":
                # The teacher is frozen and val is unaugmented/unshuffled, so these values are reused every epoch.
                self.teacher_val_logits = torch.from_numpy(statistics["logits"]).float()
                self.teacher_val_names = statistics["audio_name"]
                self.teacher_val_loss = float(statistics["loss"])
                self.teacher_val_acc = float(statistics["accuracy"])
            reference[split] = {
                "accuracy": float(statistics["accuracy"]),
                "mAP": float(np.mean(statistics["average_precision"])),
                "f1_weighted": float(statistics["f1_weighted"]),
                "loss": float(statistics["loss"]),
            }
            logger.info(
                "TEACHER %s -> Accuracy: %.4f | mAP: %.4f | F1 (weighted): %.4f",
                split.upper(),
                reference[split]["accuracy"],
                reference[split]["mAP"],
                reference[split]["f1_weighted"],
            )
        path = os.path.join(self.ckpt_dir, "teacher_reference.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(reference, file, indent=2)
        logger.info(f"Saved teacher reference scores to: '{path}'")

    def _val_kd_losses(self, val_statistics: dict) -> tuple[float, float]:
        """Return (val_kd_loss, val_kl_loss) with val_kd_loss = CE + alpha * T^2 * KL."""
        if not np.array_equal(val_statistics["audio_name"], self.teacher_val_names):
            raise RuntimeError("Validation sample order changed between teacher and student evaluation.")
        student_logits = torch.from_numpy(val_statistics["logits"]).float()
        target = torch.from_numpy(val_statistics["target"])
        terms = self.loss_fn.components(student_logits, self.teacher_val_logits, target)
        alpha = self.loss_fn.alpha
        val_kd_loss = terms["ce"] + alpha * terms["kl"]
        return float(val_kd_loss), float(terms["kl"])

    def train(self, train_loader: Any, val_loader: Any, test_loader: Any, max_epoch: int) -> None:
        for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
            if len(loader) == 0:
                raise ValueError(f"{name} loader is empty; check split sizes and batch_size/drop_last.")
        if max_epoch < 1:
            raise ValueError("max_epoch must be positive.")
        self._evaluate_teacher(val_loader, test_loader)
        best_metric = -float("inf") if self.config.monitor == "accuracy" else float("inf")
        best_epoch = -1
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_val_mAP = 0.0
        start = time.perf_counter()
        for epoch in range(max_epoch):
            self.model.train()
            self.teacher.eval()
            learning_rate = self.optimizer.param_groups[0]["lr"]
            alpha = self.loss_fn.alpha
            total_kd = total_ce = total_kl = teacher_total_ce = 0.0
            teacher_correct = teacher_seen = 0
            predictions = []
            targets = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epoch}"):
                waveform = batch["waveform"].to(self.device)
                video_form = batch["video_form"].to(self.device)
                target = batch["target"].to(self.device)
                with torch.no_grad():
                    teacher_logits = self.teacher(waveform=waveform, video_form=video_form)["clipwise_output"]
                student_logits = self.model(waveform=waveform)["clipwise_output"]
                terms = self.loss_fn.components(student_logits, teacher_logits, target)
                # KD objective: CE + alpha * T^2 * KL.
                loss = terms["ce"] + alpha * terms["kl"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                total_kd += loss.item()
                total_ce += terms["ce"].item()
                total_kl += terms["kl"].item()
                with torch.no_grad():
                    target_index = target.argmax(dim=1) if target.dim() > 1 else target.long()
                    teacher_total_ce += F.cross_entropy(teacher_logits, target_index).item()
                    teacher_correct += int((teacher_logits.argmax(dim=1) == target_index).sum().item())
                    teacher_seen += int(target_index.numel())
                predictions.append(student_logits.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())

            num_batches = len(train_loader)
            train_loss = total_ce / num_batches
            train_kd_loss = total_kd / num_batches
            train_kl_loss = total_kl / num_batches
            teacher_metrics = {
                "train_loss": teacher_total_ce / num_batches,
                "train_accuracy": teacher_correct / max(teacher_seen, 1),
                "val_loss": self.teacher_val_loss,
                "val_accuracy": self.teacher_val_acc,
            }
            train_targets = np.concatenate(targets, axis=0)
            if train_targets.ndim == 1:
                train_targets = np.eye(4)[train_targets.astype(int)]
            train_preds = np.concatenate(predictions, axis=0)
            train_acc = float(np.mean(np.argmax(train_targets, axis=1) == np.argmax(train_preds, axis=1)))
            train_mAP = np.mean(sklearn_metrics.average_precision_score(train_targets, train_preds, average=None))

            # The LR scheduler follows val_kd_loss, the objective being optimized: hard-label val CE
            # stalls under KD while accuracy still improves. Model selection still uses config.monitor.
            val_statistics = self.evaluator.evaluate(val_loader)
            val_acc = float(val_statistics["accuracy"])
            val_mAP = float(np.mean(val_statistics["average_precision"]))
            val_loss = val_statistics["loss"]
            val_kd_loss, val_kl_loss = self._val_kd_losses(val_statistics)
            if not np.isfinite(val_loss) or not np.isfinite(val_kd_loss):
                raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}.")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_kd_loss)
                next_learning_rate = self.optimizer.param_groups[0]["lr"]
                if next_learning_rate < learning_rate:
                    logger.info(
                        "LR scheduler reduced learning rate after val_kd_loss plateau: %.2e -> %.2e",
                        learning_rate,
                        next_learning_rate,
                    )
            is_best = (val_acc > best_metric) if self.config.monitor == "accuracy" else (val_loss < best_metric)
            if is_best:
                best_metric = val_acc if self.config.monitor == "accuracy" else val_loss
                best_epoch = epoch
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_mAP = val_mAP
                self._save_checkpoint(os.path.join(self.ckpt_dir, "student_best.pt"), epoch, best_metric)
            self.history_logger.log_epoch(
                epoch, learning_rate, train_loss, train_kd_loss, train_kl_loss, train_acc, train_mAP,
                val_loss, val_kd_loss, val_kl_loss, val_statistics, teacher_metrics, is_best,
            )
            early_stop_score = val_acc if self.config.monitor == "accuracy" else -val_loss
            should_stop = False
            if self.early_stopper is not None:
                should_stop = self.early_stopper.step(early_stop_score)
            logger.info(
                "Epoch %03d/%03d | Train CE Loss: %.5f | Train KD Loss: %.5f (T^2*KL %.5f) | "
                "Val CE Loss: %.5f | Val KD Loss: %.5f (T^2*KL %.5f) | "
                "Train Accuracy (Student): %.4f | Val Accuracy (Student): %.4f",
                epoch + 1,
                max_epoch,
                train_loss,
                train_kd_loss,
                train_kl_loss,
                val_loss,
                val_kd_loss,
                val_kl_loss,
                train_acc,
                val_acc,
            )
            logger.info(
                "Teacher (frozen) | Train CE Loss: %.5f | Train Accuracy: %.4f | Val CE Loss: %.5f | Val Accuracy: %.4f",
                teacher_metrics["train_loss"],
                teacher_metrics["train_accuracy"],
                teacher_metrics["val_loss"],
                teacher_metrics["val_accuracy"],
            )
            logger.info("Learning Rate: %.2e", learning_rate)
            early_suffix = ""
            if self.early_stopper is not None:
                early_suffix = f" (no improvement for {self.early_stopper.counter}/{self.early_stopper.patience} epochs)"
            logger.info(
                "Best Epoch: %03d | Val Loss: %.5f | Val Accuracy (Student): %.4f | Val mAP (Student): %.4f%s",
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
        best_path = os.path.join(self.ckpt_dir, "student_best.pt")
        if not os.path.exists(best_path):
            raise RuntimeError("Training did not produce a best checkpoint.")
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        best_val_statistics = self.evaluator.evaluate(val_loader)
        test_statistics = self.evaluator.evaluate(test_loader)
        self.history_logger.save_confusion_matrices("val", best_val_statistics)
        self.history_logger.save_confusion_matrices("test", test_statistics)
        self.history_logger.save_classification_reports("val", best_val_statistics)
        self.history_logger.save_classification_reports("test", test_statistics)
        timer = InferenceTimer(self.model, self.device)
        latency = timer.measure_latency_per_sample(waveform_shape=(1, self.config.audio_features.sample_rate * 2))
        self.history_logger.save_summary(training_time, latency, best_val_statistics, test_statistics)
        for head, stats in test_statistics["heads"].items():
            logger.info("TEST %s -> Accuracy: %.4f | mAP: %.4f", head, stats["accuracy"], np.nanmean(stats["average_precision"]))
