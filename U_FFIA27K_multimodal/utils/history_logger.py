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
                ]
            )

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, train_mAP: float, val_loss: float, val_statistics: dict, is_best: bool = False) -> None:
        val_auc = val_statistics["auc"]
        val_ap = val_statistics["average_precision"]
        row = [
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{train_mAP:.6f}",
            f"{val_loss:.6f}",
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
        """Save one confusion matrix per head for the selected evaluation split."""
        for head, head_statistics in statistics["heads"].items():
            path = os.path.join(self.log_dir, f"confusion_matrix_{split}_{head}.csv")
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["actual\\predicted", "none", "strong", "medium", "weak"])
                for label, row in zip(("none", "strong", "medium", "weak"), head_statistics["confu_matrix"]):
                    writer.writerow([label] + [int(value) for value in row])
            logger.info(f"Saved {split} confusion matrix for {head} head to: '{path}'")

    def plot_history(self) -> None:
        return
