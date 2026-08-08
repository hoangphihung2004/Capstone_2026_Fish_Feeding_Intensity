from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


CLASS_NAMES = ["none", "strong", "medium", "weak"]


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(true, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(true, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
    }


def save_metrics_csv(metrics: Dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def save_history_csv(history: List[Dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_history_plot(history: List[Dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
        if key in history[0]:
            plt.plot(epochs, [row[key] for row in history], label=key)
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_confusion_outputs(y_true: Iterable[int], y_pred: Iterable[int], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(list(y_true), list(y_pred), labels=list(range(len(CLASS_NAMES))))
    np.savetxt(output_path / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=200)
    plt.close()
