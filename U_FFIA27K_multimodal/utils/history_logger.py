import csv
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class HistoryLogger:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_csv_path = os.path.join(self.log_dir, "history.csv")
        with open(self.history_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "learning_rate",
                    "train_loss",
                    "train_kd_loss",
                    "train_kl_loss",
                    "train_feat_loss",
                    *[f"train_feat_s{stage}_loss" for stage in range(1, 6)],
                    "train_accuracy",
                    "train_mAP",
                    "val_loss",
                    "val_kd_loss",
                    "val_kl_loss",
                    "val_feat_loss",
                    *[f"val_feat_s{stage}_loss" for stage in range(1, 6)],
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
                    "teacher_train_loss",
                    "teacher_train_accuracy",
                    "teacher_val_loss",
                    "teacher_val_accuracy",
                ]
            )

    def log_epoch(self, epoch: int, learning_rate: float, train_loss: float, train_kd_loss: float, train_kl_loss: float, train_feat_loss: float, train_stage_losses: list, train_acc: float, train_mAP: float, val_loss: float, val_kd_loss: float, val_kl_loss: float, val_feat_loss: float, val_stage_losses: list, val_statistics: dict, teacher_metrics: dict, is_best: bool = False) -> None:
        """*_loss = CE on hard labels, *_kd_loss = CE + alpha * T^2 * KL + beta * feat, *_kl_loss = T^2 * KL, *_feat_loss = L_feat (weighted stage MSE), *_feat_s{k}_loss = MSE of stage k (logged for all five stages).

        teacher_metrics holds the frozen teacher's train/val CE loss and accuracy for this epoch.
        """
        val_auc = val_statistics["auc"]
        val_ap = val_statistics["average_precision"]
        row = [
            epoch,
            f"{learning_rate:.10f}",
            f"{train_loss:.6f}",
            f"{train_kd_loss:.6f}",
            f"{train_kl_loss:.6f}",
            f"{train_feat_loss:.6f}",
            *[f"{value:.6f}" for value in train_stage_losses],
            f"{train_acc:.6f}",
            f"{train_mAP:.6f}",
            f"{val_loss:.6f}",
            f"{val_kd_loss:.6f}",
            f"{val_kl_loss:.6f}",
            f"{val_feat_loss:.6f}",
            *[f"{value:.6f}" for value in val_stage_losses],
            f"{float(val_statistics['accuracy']):.6f}",
            f"{float(np.mean(val_ap)):.6f}",
            f"{val_auc[0]:.6f}",
            f"{val_auc[1]:.6f}",
            f"{val_auc[2]:.6f}",
            f"{val_auc[3]:.6f}",
            f"{val_ap[0]:.6f}",
            f"{val_ap[1]:.6f}",
            f"{val_ap[2]:.6f}",
            f"{val_ap[3]:.6f}",
            f"{teacher_metrics['train_loss']:.6f}",
            f"{teacher_metrics['train_accuracy']:.6f}",
            f"{teacher_metrics['val_loss']:.6f}",
            f"{teacher_metrics['val_accuracy']:.6f}",
        ]
        with open(self.history_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def save_summary(self, training_time: float, inference_time_ms: float, val_statistics: dict, test_statistics: dict) -> None:
        summary_csv_path = os.path.join(self.log_dir, "summary.csv")
        file_exists = os.path.exists(summary_csv_path)
        headers = [
            "Training Time (s)",
            "Inference Time (ms/sample)",
            "Precision Val (Weighted)",
            "Recall Val (Weighted)",
            "F1-score Val (Weighted)",
            "Accuracy Val",
            "mAP Val",
            "Precision Test (Weighted)",
            "Recall Test (Weighted)",
            "F1-score Test (Weighted)",
            "Accuracy Test",
            "mAP Test",
        ]
        row = [
            f"{training_time:.2f}",
            f"{inference_time_ms:.3f}",
            f"{val_statistics['prec_weighted']:.6f}",
            f"{val_statistics['rec_weighted']:.6f}",
            f"{val_statistics['f1_weighted']:.6f}",
            f"{val_statistics['accuracy']:.6f}",
            f"{np.mean(val_statistics['average_precision']):.6f}",
            f"{test_statistics['prec_weighted']:.6f}",
            f"{test_statistics['rec_weighted']:.6f}",
            f"{test_statistics['f1_weighted']:.6f}",
            f"{test_statistics['accuracy']:.6f}",
            f"{np.mean(test_statistics['average_precision']):.6f}",
        ]
        with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
        logger.info(f"Saved summary to: '{summary_csv_path}'")

    def save_confusion_matrices(self, split: str, statistics: dict) -> None:
        """Save the multimodal confusion matrix for the selected evaluation split."""
        for head, head_statistics in statistics["heads"].items():
            path = os.path.join(self.log_dir, f"confusion_matrix_{split}_{head}.csv")
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["actual\\predicted", "none", "strong", "medium", "weak"])
                for label, row in zip(("none", "strong", "medium", "weak"), head_statistics["confu_matrix"]):
                    writer.writerow([label] + [int(value) for value in row])
            logger.info(f"Saved {split} confusion matrix for {head} head to: '{path}'")

    def save_classification_reports(self, split: str, statistics: dict) -> None:
        """Log and save the multimodal classification report for an evaluation split."""
        for head, head_statistics in statistics["heads"].items():
            path = os.path.join(self.log_dir, f"classification_report_{split}_{head}.txt")
            report = head_statistics["message"]
            with open(path, "w", encoding="utf-8") as file:
                file.write(report)
                if not report.endswith("\n"):
                    file.write("\n")
            logger.info("%s classification report (%s head):\n%s", split.upper(), head, report)
            logger.info("Saved %s classification report for %s head to: '%s'", split, head, path)

    def plot_history(self) -> None:
        return
