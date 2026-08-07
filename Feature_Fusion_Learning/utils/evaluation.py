from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def get_metrics(y_true, y_pred):
    return {
        "Precision (Weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (Weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score (Weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, output_dir, model_name, label_names=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(y_true) | set(y_pred))
    display_labels = [label_names.get(int(label), str(label)) if label_names else str(label) for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    csv_path = output_dir / f"{model_name}_confusion_matrix.csv"
    png_path = output_dir / f"{model_name}_confusion_matrix.png"

    pd.DataFrame(matrix, index=display_labels, columns=display_labels).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels).plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )
    ax.set_title(f"{model_name} Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return csv_path, png_path
