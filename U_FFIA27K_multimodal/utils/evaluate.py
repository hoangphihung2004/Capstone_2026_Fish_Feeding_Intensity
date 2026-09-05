from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from models.multimodal_model import HEAD_KEYS
from .losses import ClipCELoss


def classification_statistics(target, probabilities):
    labels = target.argmax(1)
    predictions = probabilities.argmax(1)
    classes = np.arange(target.shape[1])
    ap, auc = [], []
    for index in classes:
        truth = target[:, index]
        ap.append(float(metrics.average_precision_score(truth, probabilities[:, index]))
                  if truth.any() else float("nan"))
        auc.append(float(metrics.roc_auc_score(truth, probabilities[:, index]))
                   if np.unique(truth).size == 2 else float("nan"))
    result = {
        "average_precision": np.array(ap), "auc": np.array(auc),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confu_matrix": confusion_matrix(labels, predictions, labels=classes),
        "message": classification_report(labels, predictions, labels=classes, digits=4, zero_division=0),
    }
    for average in ("weighted", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, labels=classes, average=average, zero_division=0)
        result.update({f"prec_{average}": precision, f"rec_{average}": recall, f"f1_{average}": f1})
    return result


class MultimodalEvaluator:
    def __init__(self, model: nn.Module, loss_fn: ClipCELoss | None = None) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_fn = loss_fn or ClipCELoss()

    def _forward(self, data_loader: Any) -> Dict[str, np.ndarray]:
        output_dict: Dict[str, list] = {}
        loss_sums = {head: 0.0 for head in HEAD_KEYS}
        count = 0
        self.model.eval()
        for batch in tqdm(data_loader, desc="Running multimodal evaluation..."):
            waveform = batch["waveform"].to(self.device)
            video_form = batch["video_form"].to(self.device)
            with torch.no_grad():
                batch_output = self.model(waveform=waveform, video_form=video_form)
                losses = self.loss_fn.components(batch_output, {"target": batch["target"].to(self.device)})
            count += waveform.shape[0]
            for head, key in HEAD_KEYS.items():
                loss_sums[head] += losses[head].item() * waveform.shape[0]
                output_dict.setdefault(key, []).append(batch_output[key].softmax(1).cpu().numpy())
            target = batch["target"].cpu().numpy()
            if target.ndim == 1:
                target = np.eye(4)[target.astype(int)]
            output_dict.setdefault("target", []).append(target)
            output_dict.setdefault("audio_name", []).append(np.array(batch["audio_name"], dtype=object))
            output_dict.setdefault("video_name", []).append(np.array(batch["video_name"], dtype=object))
        if not count:
            raise ValueError("Evaluation loader is empty.")
        self.last_losses = {head: value / count for head, value in loss_sums.items()}
        return {key: np.concatenate(value, axis=0) for key, value in output_dict.items()}

    def evaluate(self, data_loader: Any, prediction_path=None) -> Dict[str, Any]:
        output_dict = self._forward(data_loader)
        target = output_dict["target"]
        heads = {head: classification_statistics(target, output_dict[key]) for head, key in HEAD_KEYS.items()}
        if prediction_path is not None:
            np.savez_compressed(prediction_path, **{
                key: value.astype(str) if value.dtype == object else value
                for key, value in output_dict.items()})
        return {**heads["multimodal"], "heads": heads, "head_losses": self.last_losses,
                "loss": sum(self.last_losses.values())}
