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


STAGE_CHANNELS = (16, 24, 40, 112, 320)


class FeatureMatchLoss(nn.Module):
    """MSE between the student's pooled stage features and the teacher's pooled stage features.

    Both sides are [B, 512] = the five stage vectors (16+24+40+112+320) concatenated. Per stage k:
        target "audio": MSE_k = MSE(student_k, teacher_audio_k)
        target "video": MSE_k = MSE(student_k, teacher_video_k)
        target "both":  MSE_k = MSE(student_k, teacher_audio_k) + MSE(student_k, teacher_video_k)
        target "none":  MSE_k = 0
    L_feat = sum_k w_k * MSE_k / sum_k w_k over the selected stages (w_k = 1 when weights is None).
    Stage 5 is the final pooled 320-D feature. No projectors: shapes already match.
    forward returns (L_feat, per-stage MSE of all five stages) so unselected stages can still be logged.
    With source "delta" / "update" the teacher vector exists only for stages 2-5 (stage 1 has no fusion), so the
    stage-1 MSE is reported as 0.
    """

    def __init__(self, target: str = "audio", stages=(5,), weights=None, source: str = "fused") -> None:
        super().__init__()
        if target not in ("none", "audio", "video", "both"):
            raise ValueError(f"Unknown feature target: {target}")
        self.target = target
        self.stage_mask = [1.0] * len(STAGE_CHANNELS) if source == "fused" else [0.0] + [1.0] * (len(STAGE_CHANNELS) - 1)
        weights = [1.0] * len(stages) if weights is None else list(weights)
        self.stage_weights = [0.0] * len(STAGE_CHANNELS)
        for stage, weight in zip(stages, weights):
            self.stage_weights[stage - 1] = float(weight)

    def forward(self, student_feature: torch.Tensor, teacher_features: dict):
        names = {"none": (), "audio": ("audio",), "video": ("video",), "both": ("audio", "video")}[self.target]
        student_stages = torch.split(student_feature, STAGE_CHANNELS, dim=1)
        per_stage = student_feature.new_zeros(len(STAGE_CHANNELS))
        for name in names:
            teacher_stages = torch.split(teacher_features[name], STAGE_CHANNELS, dim=1)
            per_stage = per_stage + torch.stack([F.mse_loss(s, t) for s, t in zip(student_stages, teacher_stages)])
        per_stage = per_stage * per_stage.new_tensor(self.stage_mask)
        weights = per_stage.new_tensor(self.stage_weights)
        total = (weights * per_stage).sum() / weights.sum() if names else per_stage.sum()
        return total, per_stage


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
