from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from config import TrainConfig
from dataset import save_split_files
from models import build_model
from models.teacher_registry import build_audio_teacher, build_video_teacher
from utils.distillation import FocalLoss, compute_multimodal_distillation_loss
from utils.metrics import (
    CLASS_NAMES,
    save_confusion_outputs,
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


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def _format_param_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value:,} ({value / 1_000_000:.3f}M)"
    if value >= 1_000:
        return f"{value:,} ({value / 1_000:.3f}K)"
    return f"{value:,}"


def _one_hot(labels: List[int], num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[np.asarray(labels, dtype=int)]


def _statistics_from_outputs(y_true: List[int], logits: List[List[float]], num_classes: int) -> Dict[str, Any]:
    target = _one_hot(y_true, num_classes)
    output = np.asarray(logits, dtype=np.float32)
    pred = np.argmax(output, axis=1)
    true = np.asarray(y_true, dtype=int)

    average_precision = sklearn_metrics.average_precision_score(target, output, average=None)
    auc = sklearn_metrics.roc_auc_score(target, output, average=None)
    acc = accuracy_score(true, pred)
    cm = confusion_matrix(true, pred, labels=list(range(num_classes)))
    message = "\n" + classification_report(true, pred, digits=4, zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true, pred, average="weighted", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true, pred, average="macro", zero_division=0
    )
    return {
        "average_precision": average_precision,
        "accuracy": acc,
        "auc": auc,
        "message": message,
        "confu_matrix": cm,
        "prec_weighted": prec_weighted,
        "rec_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "prec_macro": prec_macro,
        "rec_macro": rec_macro,
        "f1_macro": f1_macro,
        "y_true": true,
        "y_pred": pred,
    }


class EarlyStopping:
    def __init__(self, patience: int = 30, delta: float = 0.0, verbose: bool = True) -> None:
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        logger.info("Early Stopping state has been reset.")

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info("Early Stopping: %d/%d epochs without improvement.", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning(
                    "Early Stopping triggered: Metric did not improve by delta=%s for %d consecutive epochs.",
                    self.delta,
                    self.patience,
                )
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


class MultimodalHistoryLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv_path = self.log_dir / "history.csv"
        self._write_history_header()

    def _history_headers(self) -> list[str]:
        return [
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_mAP",
            "val_loss",
            "val_accuracy",
            "val_mAP",
            "val_auc_class_none",
            "val_auc_class_strong",
            "val_auc_class_medium",
            "val_auc_class_weak",
            "val_ap_class_none",
            "val_ap_class_strong",
            "val_ap_class_medium",
            "val_ap_class_weak",
            "cm_none_none",
            "cm_none_strong",
            "cm_none_medium",
            "cm_none_weak",
            "cm_strong_none",
            "cm_strong_strong",
            "cm_strong_medium",
            "cm_strong_weak",
            "cm_medium_none",
            "cm_medium_strong",
            "cm_medium_medium",
            "cm_medium_weak",
            "cm_weak_none",
            "cm_weak_strong",
            "cm_weak_medium",
            "cm_weak_weak",
        ]

    def _write_history_header(self) -> None:
        with self.history_csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self._history_headers())

    def _save_confusion_matrix_csv(self, path: str | Path, matrix: np.ndarray) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Actual\\Predicted"] + CLASS_NAMES)
            for index, label in enumerate(CLASS_NAMES):
                writer.writerow([label] + list(matrix[index]))

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_mAP: float,
        val_loss: float,
        val_statistics: Dict[str, Any],
        is_best: bool = False,
    ) -> None:
        val_acc = float(np.mean(val_statistics["accuracy"]))
        val_mAP = float(np.mean(val_statistics["average_precision"]))
        val_auc = val_statistics["auc"]
        val_ap = val_statistics["average_precision"]
        cm_flat = list(val_statistics["confu_matrix"].flatten())
        row_data = [
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{train_mAP:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{val_mAP:.6f}",
            f"{val_auc[0]:.6f}",
            f"{val_auc[1]:.6f}",
            f"{val_auc[2]:.6f}",
            f"{val_auc[3]:.6f}",
            f"{val_ap[0]:.6f}",
            f"{val_ap[1]:.6f}",
            f"{val_ap[2]:.6f}",
            f"{val_ap[3]:.6f}",
        ] + [int(value) for value in cm_flat]

        with self.history_csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row_data)

        if is_best:
            self._save_confusion_matrix_csv(self.log_dir / "confusion_matrix_best.csv", val_statistics["confu_matrix"])

    def save_summary(
        self,
        training_time: float,
        inference_time_ms: float,
        val_statistics: Dict[str, Any],
        test_statistics: Dict[str, Any],
    ) -> None:
        summary_csv_path = self.log_dir / "summary.csv"
        val_mAP = float(np.mean(val_statistics["average_precision"]))
        test_mAP = float(np.mean(test_statistics["average_precision"]))
        headers = [
            "Training Time (s)",
            "Inference Time (ms/sample)",
            "Precision Val (Weighted)",
            "Recall Val (Weighted)",
            "F1-score Val (Weighted)",
            "Accuracy Val",
            "mAP Val",
            "Precision Val (Macro)",
            "Recall Val (Macro)",
            "F1-score Val (Macro)",
            "Precision Test (Weighted)",
            "Recall Test (Weighted)",
            "F1-score Test (Weighted)",
            "Accuracy Test",
            "mAP Test",
            "Precision Test (Macro)",
            "Recall Test (Macro)",
            "F1-score Test (Macro)",
        ]
        row_data = [
            f"{training_time:.2f}",
            f"{inference_time_ms:.3f}",
            f"{val_statistics['prec_weighted']:.6f}",
            f"{val_statistics['rec_weighted']:.6f}",
            f"{val_statistics['f1_weighted']:.6f}",
            f"{val_statistics['accuracy']:.6f}",
            f"{val_mAP:.6f}",
            f"{val_statistics['prec_macro']:.6f}",
            f"{val_statistics['rec_macro']:.6f}",
            f"{val_statistics['f1_macro']:.6f}",
            f"{test_statistics['prec_weighted']:.6f}",
            f"{test_statistics['rec_weighted']:.6f}",
            f"{test_statistics['f1_weighted']:.6f}",
            f"{test_statistics['accuracy']:.6f}",
            f"{test_mAP:.6f}",
            f"{test_statistics['prec_macro']:.6f}",
            f"{test_statistics['rec_macro']:.6f}",
            f"{test_statistics['f1_macro']:.6f}",
        ]
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row_data)
        logger.info("Successfully exported Summary Report to: '%s'", summary_csv_path)

    def plot_history(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        import matplotlib.pyplot as plt

        epochs, train_losses, val_losses = [], [], []
        train_accs, val_accs = [], []
        train_maps, val_maps = [], []
        with self.history_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))
                train_accs.append(float(row["train_accuracy"]))
                val_accs.append(float(row["val_accuracy"]))
                train_maps.append(float(row["train_mAP"]))
                val_maps.append(float(row["val_mAP"]))

        if not epochs:
            logger.warning("Warning: No epoch data found to plot.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Fish Feeding Intensity Model Learning History", fontsize=16, fontweight="bold", y=0.98)
        axes[0].plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2, linestyle="--")
        axes[0].plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2)
        axes[0].set_title("Loss Curves", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=10)
        axes[0].set_ylabel("Loss", fontsize=10)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(frameon=True)

        axes[1].plot(epochs, train_accs, label="Train Acc", color="#2ca02c", linewidth=2, linestyle="--")
        axes[1].plot(epochs, val_accs, label="Val Acc", color="#d62728", linewidth=2)
        axes[1].set_title("Accuracy Curves", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=10)
        axes[1].set_ylabel("Accuracy", fontsize=10)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(frameon=True)

        axes[2].plot(epochs, train_maps, label="Train mAP", color="#9467bd", linewidth=2, linestyle="--")
        axes[2].plot(epochs, val_maps, label="Val mAP", color="#8c564b", linewidth=2)
        axes[2].set_title("Mean Average Precision (mAP)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Epoch", fontsize=10)
        axes[2].set_ylabel("mAP", fontsize=10)
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend(frameon=True)

        plt.tight_layout()
        plot_path = self.log_dir / "learning_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Successfully generated and saved learning curves to: '%s'", plot_path)


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
        self.model = build_model(cfg).to(self.device)
        self.audio_teacher = build_audio_teacher(cfg)
        self.video_teacher = build_video_teacher(cfg)
        if self.audio_teacher is not None:
            self.audio_teacher = self.audio_teacher.to(self.device)
        if self.video_teacher is not None:
            self.video_teacher = self.video_teacher.to(self.device)

        train_samples = splits.get("train", [])
        class_counts = {}
        for s in train_samples:
            lbl = int(s.get("label", s.get("target", 0)))
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
        num_classes = cfg.num_classes
        total_train = len(train_samples) if len(train_samples) > 0 else 1
        weights = []
        for c in range(num_classes):
            cnt = class_counts.get(c, 1)
            w = total_train / (num_classes * cnt)
            if c == 2:  # Priority boost for medium class
                w *= 1.2
            weights.append(w)
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        loss_type = getattr(cfg, "loss_type", "weighted_cross_entropy")
        focal_gamma = getattr(cfg, "focal_gamma", 2.0)
        label_smoothing = float(getattr(cfg, "label_smoothing", 0.0))

        if loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            logger.info("  - Loss function: Standard unweighted CrossEntropyLoss (label_smoothing=%.2f)", label_smoothing)
        elif loss_type == "focal_loss":
            self.criterion = FocalLoss(alpha=self.class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
            logger.info("  - Loss function: Weighted FocalLoss (gamma=%.1f, label_smoothing=%.2f) with class weights: %s", focal_gamma, label_smoothing, [round(w, 4) for w in weights])
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=label_smoothing)
            logger.info("  - Loss function: Dynamic Weighted CrossEntropyLoss (label_smoothing=%.2f): %s", label_smoothing, [round(w, 4) for w in weights])

        self.optimizer = _optimizer(self._optimizer_parameters(), cfg)

        # 1. LR Scheduler setup (Cosine Annealing + Warmup)
        self.scheduler_cfg = getattr(cfg, "scheduler", None)
        self.scheduler = None
        if self.scheduler_cfg and getattr(self.scheduler_cfg, "enabled", False):
            if getattr(self.scheduler_cfg, "type", "cosine_warmup") == "cosine_warmup":
                warmup_epochs = getattr(self.scheduler_cfg, "warmup_epochs", 10)
                eta_min = getattr(self.scheduler_cfg, "eta_min", 1e-6)
                total_epochs = cfg.epochs
                base_lr = cfg.learning_rate

                def lr_lambda(epoch: int) -> float:
                    if epoch < warmup_epochs:
                        return (epoch + 1) / float(warmup_epochs)
                    progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    min_ratio = eta_min / base_lr
                    return min_ratio + (1.0 - min_ratio) * cosine_decay

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
                logger.info("  - LR Scheduler: CosineAnnealingLR with %d warmup epochs (eta_min=%.1e)", warmup_epochs, eta_min)

        # 2. SWA setup (Stochastic Weight Averaging)
        self.swa_cfg = getattr(cfg, "swa", None)
        self.swa_model = None
        self.swa_scheduler = None
        if self.swa_cfg and getattr(self.swa_cfg, "enabled", False):
            from torch.optim.swa_utils import AveragedModel, SWALR
            self.swa_model = AveragedModel(self.model)
            swa_lr = getattr(self.swa_cfg, "lr", 1e-5)
            anneal_epochs = getattr(self.swa_cfg, "anneal_epochs", 10)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=swa_lr, anneal_epochs=anneal_epochs)
            logger.info("  - SWA: Stochastic Weight Averaging enabled starting at epoch %d (swa_lr=%.1e)", self.swa_cfg.start_epoch, swa_lr)



        self.model_param_counts = _count_parameters(self.model)
        audio_frontend = getattr(self.model, "audio_frontend", None)
        self.audio_frontend_param_counts = _count_parameters(audio_frontend) if audio_frontend is not None else None
        self.learnable_params_excluding_audio_frontend = self.model_param_counts["trainable"]
        if self.audio_frontend_param_counts is not None:
            self.learnable_params_excluding_audio_frontend -= self.audio_frontend_param_counts["trainable"]
        self.audio_teacher_param_counts = _count_parameters(self.audio_teacher) if self.audio_teacher is not None else None
        self.video_teacher_param_counts = _count_parameters(self.video_teacher) if self.video_teacher is not None else None
        self.history_logger = MultimodalHistoryLogger(log_dir=self.output_dir)
        self.early_stopping = cfg.early_stopping
        if self.early_stopping:
            self.early_stopper = EarlyStopping(patience=cfg.patience, delta=cfg.delta, verbose=True)
        else:
            self.early_stopper = None

        with (self.output_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)
        with (self.output_dir / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)
        with (self.output_dir / "splitter_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.dataset.model_dump(), f, indent=2, ensure_ascii=False)
        with (self.output_dir / "distillation_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.distillation.model_dump(), f, indent=2, ensure_ascii=False)
        with (self.output_dir / "run_info.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": self.run_name,
                    "model": cfg.model.name,
                    "evaluation_mode": cfg.evaluation_mode,
                    "seed": cfg.seed,
                    "device": str(self.device),
                    "distillation_enabled": cfg.distillation.enabled,
                    "distillation_mode": cfg.distillation.mode if cfg.distillation.enabled else "disabled",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        self._save_model_artifacts()
        save_split_files(splits, self.output_dir / "splits")
        self._log_initial_state()
        self._run_model_sanity_check()

    def _optimizer_parameters(self):
        params = list(self.model.parameters())
        if self.cfg.distillation.enabled and self.cfg.distillation.mode == "online":
            if self.audio_teacher is not None:
                params.extend(self.audio_teacher.parameters())
            if self.video_teacher is not None:
                params.extend(self.video_teacher.parameters())
        return params

    def _save_model_artifacts(self) -> None:
        with (self.output_dir / "model_architecture.txt").open("w", encoding="utf-8") as f:
            f.write(str(self.model))
            f.write("\n")

        summary = {
            "model_name": self.cfg.model.name,
            "device": str(self.device),
            "input_shapes": {
                "waveform": [2, self.cfg.audio_features.sample_rate * 2],
                "video_form": [
                    2,
                    3,
                    self.cfg.video_features.image_size,
                    self.cfg.video_features.image_size,
                ],
            },
            "parameters": self.model_param_counts,
            "learnable_parameters_excluding_audio_frontend": int(self.learnable_params_excluding_audio_frontend),
            "audio_frontend_parameters": self.audio_frontend_param_counts,
            "distillation": {
                "enabled": self.cfg.distillation.enabled,
                "mode": self.cfg.distillation.mode if self.cfg.distillation.enabled else "disabled",
                "audio_teacher_parameters": self.audio_teacher_param_counts,
                "video_teacher_parameters": self.video_teacher_param_counts,
            },
        }
        with (self.output_dir / "model_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _log_initial_state(self) -> None:
        logger.info("==================================================")
        logger.info("MultimodalTrainer successfully initialized:")
        logger.info("  - Model:                        %s", self.cfg.model.name)
        logger.info(
            "  - Total Module Parameters:      %s",
            _format_param_count(self.model_param_counts["total"]),
        )
        logger.info(
            "  - Trainable Parameters:         %s",
            _format_param_count(self.model_param_counts["trainable"]),
        )
        logger.info(
            "  - Learnable Model Parameters:   %s",
            _format_param_count(self.learnable_params_excluding_audio_frontend),
        )
        logger.info(
            "  - Non-trainable Parameters:     %s",
            _format_param_count(self.model_param_counts["non_trainable"]),
        )
        if self.audio_frontend_param_counts is not None:
            logger.info(
                "  - Audio Frontend Parameters:    %s total | %s trainable",
                _format_param_count(self.audio_frontend_param_counts["total"]),
                _format_param_count(self.audio_frontend_param_counts["trainable"]),
            )
        logger.info("  - Monitor Metric:               '%s'", self.cfg.monitor)
        logger.info("  - Early Stopping Enabled:       %s", self.cfg.early_stopping)
        if self.cfg.early_stopping:
            logger.info("    * Patience:                   %s epochs", self.cfg.patience)
            logger.info("    * Delta:                      %s", self.cfg.delta)
        logger.info("  - Distillation Enabled:         %s", self.cfg.distillation.enabled)
        if self.cfg.distillation.enabled:
            logger.info("    * Mode:                       %s", self.cfg.distillation.mode)
            logger.info("    * Audio Teacher:              %s", self.cfg.distillation.audio_teacher.name)
            logger.info("    * Video Teacher:              %s", self.cfg.distillation.video_teacher.name)
        logger.info("==================================================")

    def _run_model_sanity_check(self) -> None:
        logger.info("==================================================")
        logger.info("Running model sanity check before training...")
        was_training = self.model.training
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        waveform = torch.zeros(2, self.cfg.audio_features.sample_rate * 2, device=self.device)
        video_form = torch.zeros(
            2,
            3,
            self.cfg.video_features.image_size,
            self.cfg.video_features.image_size,
            device=self.device,
        )
        target = torch.zeros(2, dtype=torch.long, device=self.device)

        output = self.model(waveform=waveform, video_form=video_form)
        if "clipwise_output" not in output:
            raise KeyError("Model forward output must include key 'clipwise_output'.")
        logits = output["clipwise_output"]
        expected_shape = (2, self.cfg.num_classes)
        if tuple(logits.shape) != expected_shape:
            raise ValueError(f"Model logits shape mismatch: got {tuple(logits.shape)}, expected {expected_shape}.")
        loss, _ = self._compute_loss(output, target, waveform, video_form, train=True)
        if not loss.requires_grad:
            raise RuntimeError("Sanity check loss does not require gradients.")
        loss.backward()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train(was_training)

        logger.info("  - Forward output shape:         %s", list(logits.shape))
        logger.info("  - Backward check:               passed")
        logger.info("Model sanity check completed successfully.")
        logger.info("==================================================")

    def _compute_loss(
        self,
        student_output: dict[str, torch.Tensor] | torch.Tensor,
        target: torch.Tensor,
        waveform: torch.Tensor,
        video_form: torch.Tensor,
        train: bool,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(student_output, torch.Tensor):
            logits = student_output
            student_dict = {"clipwise_output": logits, "logits_fused": logits}
        else:
            student_dict = student_output
            logits = student_dict.get("logits_fused", student_dict["clipwise_output"])

        dist_cfg = self.cfg.distillation
        is_cutoff = False
        if dist_cfg.enabled and getattr(dist_cfg, "cutoff_enabled", False) and epoch is not None:
            cutoff_epoch = getattr(dist_cfg, "cutoff_epoch", 140)
            if epoch >= cutoff_epoch:
                is_cutoff = True

        if not dist_cfg.enabled or is_cutoff or (self.audio_teacher is None and self.video_teacher is None):
            ce_loss = self.criterion(logits, target)
            return ce_loss, {"ce_loss": float(ce_loss.detach().item()), "audio_kd_loss": 0.0, "video_kd_loss": 0.0}

        # Distillation enabled with dual teachers
        with torch.no_grad():
            t_audio_out = self.audio_teacher(waveform) if self.audio_teacher is not None else {"logits": logits}
            t_video_out = self.video_teacher(video_form) if self.video_teacher is not None else {"logits": logits}

        alpha_logit = getattr(self.cfg.distillation, "alpha_logit", 1.0)
        beta_feature = getattr(self.cfg.distillation, "beta_feature", 2.0)
        temperature = getattr(self.cfg.distillation.audio_teacher, "temperature", 4.0)

        total_loss, loss_stats = compute_multimodal_distillation_loss(
            student_outputs=student_dict,
            teacher_audio_outputs=t_audio_out,
            teacher_video_outputs=t_video_out,
            targets=target,
            temperature=temperature,
            alpha_logit=alpha_logit,
            beta_feature=beta_feature,
            class_weights=self.class_weights,
            task_loss_fn=self.criterion,
        )

        return total_loss, loss_stats

    def _run_epoch(self, split: str, train: bool, epoch: int | None = None) -> Dict[str, float | List[int]]:
        self.model.train(train)
        if self.audio_teacher is not None:
            self.audio_teacher.eval()
        if self.video_teacher is not None:
            self.video_teacher.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        all_true: List[int] = []
        all_pred: List[int] = []
        all_logits: List[List[float]] = []
        loader = self.loaders[split]
        desc = f"Epoch {epoch}/{self.cfg.epochs}" if train and epoch is not None else "Running model evaluation..."
        iterator = tqdm(loader, desc=desc, unit="batch")

        for batch in iterator:
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            video_form = batch["video_form"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                output = self.model(waveform=waveform, video_form=video_form)
                logits = output["clipwise_output"] if isinstance(output, dict) else output
                loss, loss_parts = self._compute_loss(output, target, waveform, video_form, train, epoch=epoch)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

            pred = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * target.size(0)
            total_ce_loss += loss_parts.get("ce_loss", float(loss.item())) * target.size(0)
            all_true.extend(target.detach().cpu().numpy().astype(int).tolist())
            all_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
            all_logits.extend(torch.softmax(logits.detach(), dim=1).cpu().numpy().astype(float).tolist())
            if train:
                iterator.set_postfix({"Loss": f"{loss.item():.4f}"})

        metrics = _statistics_from_outputs(all_true, all_logits, self.cfg.num_classes)
        metrics["loss"] = total_loss / max(1, len(all_true))
        metrics["ce_loss"] = total_ce_loss / max(1, len(all_true))
        return metrics

    def fit(self) -> Dict[str, float]:
        best_score = -float("inf")
        best_acc = 0.0
        best_mAP = 0.0
        best_loss = float("inf")
        best_epoch = 0
        best_val_statistics = None
        best_path = self.output_dir / "checkpoint" / "model_best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        train_start_time = time.perf_counter()

        if self.early_stopping and self.early_stopper is not None:
            self.early_stopper.reset()

        logger.info("Starting training pipeline (Monitor metric: %s)...", self.cfg.monitor)

        for epoch in range(self.cfg.epochs):
            train_metrics = self._run_epoch("train", train=True, epoch=epoch)
            val_metrics = self._run_epoch("val", train=False, epoch=epoch)
            row = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_mAP": float(np.mean(train_metrics["average_precision"])),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_mAP": float(np.mean(val_metrics["average_precision"])),
            }
            current_score = _score(val_metrics, self.cfg.monitor)
            improved = current_score > best_score + self.cfg.delta
            if improved:
                best_acc = float(val_metrics["accuracy"])
                best_mAP = float(np.mean(val_metrics["average_precision"]))
                best_loss = float(val_metrics["loss"])
                best_val_statistics = val_metrics

            logger.info(
                (
                    "Epoch %d: Train Loss = %.5f | Train Acc = %.4f | Train mAP = %.4f | "
                    "Val Loss = %.5f | Val Acc = %.4f | Val mAP = %.4f"
                ),
                epoch,
                row["train_loss"],
                row["train_accuracy"],
                row["train_mAP"],
                row["val_loss"],
                row["val_accuracy"],
                row["val_mAP"],
            )

            if improved:
                best_score = current_score
                best_epoch = epoch
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
                    "Saved best model checkpoint to: '%s' (Monitor value = %.5f)",
                    best_path,
                    row["val_loss"] if self.cfg.monitor == "loss" else row["val_accuracy"],
                )
            self.history_logger.log_epoch(
                epoch=epoch,
                train_loss=row["train_loss"],
                train_acc=row["train_accuracy"],
                train_mAP=row["train_mAP"],
                val_loss=row["val_loss"],
                val_statistics=val_metrics,
                is_best=improved,
            )

            # Stepping LR Scheduler / SWA
            if self.swa_model is not None and self.swa_cfg and getattr(self.swa_cfg, "enabled", False) and epoch >= getattr(self.swa_cfg, "start_epoch", 150):
                self.swa_model.update_parameters(self.model)
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()
            elif self.scheduler is not None:
                self.scheduler.step()

            if self.early_stopping and self.early_stopper is not None:
                if self.early_stopper.step(current_score):
                    logger.info("Early stopping triggered at epoch %d!", epoch)
                    break

        training_time = time.perf_counter() - train_start_time

        # Final SWA Evaluation and Checkpoint Saving
        if self.swa_model is not None and self.swa_cfg and getattr(self.swa_cfg, "enabled", False):
            logger.info("==================================================")
            logger.info("Updating BatchNorm statistics for SWA model...")
            from torch.optim.swa_utils import update_bn
            update_bn(self.loaders["train"], self.swa_model, device=self.device)

            logger.info("Running final evaluation on SWA model...")
            original_model = self.model
            self.model = self.swa_model.module
            swa_val_metrics = self._run_epoch("val", train=False)
            swa_test_metrics = self._run_epoch("test", train=False)
            self.model = original_model

            swa_path = self.output_dir / "checkpoint" / "model_swa.pt"
            torch.save(
                {
                    "model_state_dict": self.swa_model.module.state_dict(),
                    "config": self.cfg.model_dump(),
                    "val_metrics": {k: v for k, v in swa_val_metrics.items() if not k.startswith("y_")},
                    "test_metrics": {k: v for k, v in swa_test_metrics.items() if not k.startswith("y_")},
                },
                swa_path,
            )
            logger.info("SWA Model Val Acc = %.4f | Test Acc = %.4f (Saved to '%s')", swa_val_metrics["accuracy"], swa_test_metrics["accuracy"], swa_path)
            logger.info("==================================================")

        try:
            self.history_logger.plot_history()
        except Exception as exc:
            logger.warning("Warning: Failed to generate learning curves plot: %s", str(exc))


        logger.info("==================================================")
        logger.info("Training complete. Starting evaluation on Test split...")
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info("Reloaded best checkpoint model from Epoch %d...", int(checkpoint["epoch"]))
        test_metrics = self._run_epoch("test", train=False)
        test_mAP = float(np.mean(test_metrics["average_precision"]))
        test_acc = float(np.mean(test_metrics["accuracy"]))
        logger.info("TEST Results -> Accuracy: %.4f | mAP: %.4f", test_acc, test_mAP)
        logger.info("Detailed Classification Report:\n%s", test_metrics["message"])
        logger.info("Measuring model Inference Latency on device...")
        latency_ms = self._measure_latency_per_sample(warm_up_steps=10, num_steps=50)
        result = {
            "best_epoch": float(best_epoch),
            "best_val_score": float(best_score),
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": test_acc,
            "test_mAP": test_mAP,
            "test_precision_macro": float(test_metrics["prec_macro"]),
            "test_recall_macro": float(test_metrics["rec_macro"]),
            "test_f1_macro": float(test_metrics["f1_macro"]),
            "test_precision_weighted": float(test_metrics["prec_weighted"]),
            "test_recall_weighted": float(test_metrics["rec_weighted"]),
            "test_f1_weighted": float(test_metrics["f1_weighted"]),
        }
        save_metrics_csv(result, self.output_dir / "result.csv")
        save_confusion_outputs(test_metrics["y_true"].tolist(), test_metrics["y_pred"].tolist(), self.output_dir)
        if best_val_statistics is not None:
            self.history_logger.save_summary(training_time, latency_ms, best_val_statistics, test_metrics)
        logger.info(
            "Final test result | loss=%.4f | acc=%.4f | f1_macro=%.4f | output=%s",
            result["test_loss"],
            result["test_accuracy"],
            result["test_f1_macro"],
            self.output_dir,
        )
        return result

    def _measure_latency_per_sample(self, warm_up_steps: int = 10, num_steps: int = 50) -> float:
        self.model.eval()
        waveform = torch.zeros(1, self.cfg.audio_features.sample_rate * 2, device=self.device)
        video = torch.zeros(
            1,
            3,
            self.cfg.video_features.image_size,
            self.cfg.video_features.image_size,
            device=self.device,
        )
        with torch.no_grad():
            for _ in range(warm_up_steps):
                _ = self.model(waveform=waveform, video_form=video)
            _sync_device(self.device)
            start = time.perf_counter()
            for _ in range(num_steps):
                _ = self.model(waveform=waveform, video_form=video)
            _sync_device(self.device)
        latency_ms = ((time.perf_counter() - start) / max(1, num_steps)) * 1000.0
        logger.info("Inference latency: %.3f ms/sample", latency_ms)
        return latency_ms


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
