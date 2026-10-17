import torch
import torch.nn as nn
import torch.nn.functional as F


def _class_indices(target: torch.Tensor) -> torch.Tensor:
    if target.dim() > 1:
        target = target.argmax(dim=1)
    return target.long()


class ClipCELoss(nn.Module):
    def __init__(self, head: str = "multimodal") -> None:
        super().__init__()
        self.head = head

    def components(self, output_dict: dict, target_dict: dict) -> dict:
        return {
            self.head: F.cross_entropy(output_dict["clipwise_output"], _class_indices(target_dict["target"])),
        }

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return sum(self.components(output_dict, target_dict).values())


class FeatureMatchLoss(nn.Module):
    """MSE between the student's pooled 320-D feature and the teacher's pooled 320-D feature(s).

    target "audio": MSE(student, teacher_audio).
    target "video": MSE(student, teacher_video).
    target "both":  MSE(student, teacher_audio) + MSE(student, teacher_video).
    target "none":  the term is zero.
    No projectors: student and teacher features are both 320-D, so they are compared directly.
    """

    def __init__(self, target: str = "audio") -> None:
        super().__init__()
        if target not in ("none", "audio", "video", "both"):
            raise ValueError(f"Unknown feature target: {target}")
        self.target = target

    def forward(self, student_feature: torch.Tensor, teacher_features: dict) -> torch.Tensor:
        if self.target == "none":
            return student_feature.new_zeros(())
        names = ("audio", "video") if self.target == "both" else (self.target,)
        return sum(F.mse_loss(student_feature, teacher_features[name]) for name in names)


class HintonKDLoss(nn.Module):
    """loss = CE(student, labels) + alpha * T^2 * KL(teacher_T || student_T)."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def components(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, target: torch.Tensor) -> dict:
        """Unweighted terms: ce = CE(student, labels), kl = T^2 * KL(teacher_T || student_T)."""
        temperature = self.temperature
        ce = F.cross_entropy(student_logits, _class_indices(target))
        kl = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.log_softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
            log_target=True,
        ) * (temperature ** 2)
        return {"ce": ce, "kl": kl}

    def combine(self, ce: torch.Tensor, kl: torch.Tensor) -> torch.Tensor:
        return ce + self.alpha * kl

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        terms = self.components(student_logits, teacher_logits, target)
        return self.combine(terms["ce"], terms["kl"])
