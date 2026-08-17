from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> torch.Tensor:
    """
    Computes SOTA Decoupled Knowledge Distillation (DKD - CVPR 2022 / mdistiller):
    Decouples classical KD loss into:
    1. TCKD (Target Class KD): Transfers target class confidence.
    2. NTCKD (Non-Target Class KD): Transfers dark knowledge / confusion probabilities between non-target classes.
    Solves fine-grained boundary confusion for intermediate classes (Class 2 medium).
    """
    temperature = float(temperature)
    batch_size = student_logits.shape[0]
    num_classes = student_logits.shape[1]

    # Create binary mask for target and non-target classes
    gt_mask = torch.zeros_like(student_logits, dtype=torch.bool)
    gt_mask.scatter_(1, targets.unsqueeze(1), True)
    non_gt_mask = ~gt_mask

    # Calculate probabilities with Temperature
    s_probs = F.softmax(student_logits / temperature, dim=1)
    t_probs = F.softmax(teacher_logits / temperature, dim=1)

    # 1. Target Class Probabilities
    s_p_target = s_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    t_p_target = t_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # 2. Binary Probabilities for Target vs Non-Target
    s_binary = torch.stack([s_p_target, 1.0 - s_p_target], dim=1)
    t_binary = torch.stack([t_p_target, 1.0 - t_p_target], dim=1)

    log_s_binary = torch.log(s_binary + 1e-7)
    tckd_loss = F.kl_div(log_s_binary, t_binary, reduction="batchmean") * (temperature * temperature)

    # 3. Non-Target Class Probability Distributions
    s_non_target = s_probs[non_gt_mask].view(batch_size, num_classes - 1)
    t_non_target = t_probs[non_gt_mask].view(batch_size, num_classes - 1)

    s_non_target_normalized = s_non_target / (s_non_target.sum(dim=1, keepdim=True) + 1e-7)
    t_non_target_normalized = t_non_target / (t_non_target.sum(dim=1, keepdim=True) + 1e-7)

    log_s_non_target = torch.log(s_non_target_normalized + 1e-7)
    ntckd_loss = F.kl_div(log_s_non_target, t_non_target_normalized, reduction="batchmean") * (temperature * temperature)

    return alpha * tckd_loss + beta * ntckd_loss


def compute_feature_alignment_loss(
    student_proj_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
) -> torch.Tensor:
    """Computes Feature Alignment Loss using Cosine Distance with dynamic dimension matching."""
    if student_proj_feat.dim() == 2 and teacher_feat.dim() == 2:
        if student_proj_feat.shape[1] != teacher_feat.shape[1]:
            student_proj_feat = F.interpolate(
                student_proj_feat.unsqueeze(1),
                size=teacher_feat.shape[1],
                mode="linear",
                align_corners=False,
            ).squeeze(1)
    return (1.0 - F.cosine_similarity(student_proj_feat, teacher_feat, dim=-1)).mean()


class FocalLoss(nn.Module):
    """
    Formal Multi-class Focal Loss with optional class weights (alpha), focusing parameter (gamma),
    and label smoothing support.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean", label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)

        if self.label_smoothing > 0.0:
            with torch.no_grad():
                smooth_targets = torch.full_like(log_pt, self.label_smoothing / num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + (self.label_smoothing / num_classes))
            focal_weight = (1.0 - pt) ** self.gamma
            loss = -(smooth_targets * focal_weight * log_pt).sum(dim=1)
            if self.alpha is not None:
                alpha_t = self.alpha.to(inputs.device).gather(0, targets)
                loss = alpha_t * loss
        else:
            log_pt_target = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt_target = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - pt_target) ** self.gamma
            if self.alpha is not None:
                alpha_t = self.alpha.to(inputs.device).gather(0, targets)
                loss = -alpha_t * focal_weight * log_pt_target
            else:
                loss = -focal_weight * log_pt_target

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_multimodal_distillation_loss(
    student_outputs: dict[str, torch.Tensor],
    teacher_audio_outputs: dict[str, torch.Tensor],
    teacher_video_outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha_logit: float = 1.0,
    beta_feature: float = 2.0,
    lambda_aux: float = 0.3,
    class_weights: torch.Tensor | None = None,
    task_loss_fn: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes SOTA Multimodal Decoupled Knowledge Distillation (M-DKD from mdistiller):
    1. Task Supervised Loss: Cross-Entropy / Weighted CE / Focal Loss for 3 heads (Fused + Aux Audio & Video)
    2. Decoupled Logit KD (DKD): TCKD + NTCKD for Audio Head vs Audio Teacher and Video Head vs Video Teacher
    3. Feature KD Loss: Cosine Distance for Projected Audio/Video Features vs Teacher Features
    """
    if task_loss_fn is not None:
        ce_loss_fn = task_loss_fn
    else:
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 1. Supervised Task Losses
    l_ce_fused = ce_loss_fn(student_outputs["logits_fused"], targets)
    l_ce_audio = ce_loss_fn(student_outputs["logits_audio"], targets)
    l_ce_video = ce_loss_fn(student_outputs["logits_video"], targets)
    loss_ce = l_ce_fused + lambda_aux * (l_ce_audio + l_ce_video)

    # 2. Logit-Level Decoupled KD Losses (DKD - CVPR 2022)
    t_audio_logits = teacher_audio_outputs.get("logits", teacher_audio_outputs.get("clipwise_output"))
    t_video_logits = teacher_video_outputs.get("logits", teacher_video_outputs.get("clipwise_output"))

    # Audio teacher: loss_weight = 0.3, Video teacher: loss_weight = 0.7
    w_audio = 0.3
    w_video = 0.7

    l_dkd_audio = compute_dkd_loss(student_outputs["logits_audio"], t_audio_logits, targets, temperature=temperature, alpha=1.0, beta=2.0)
    l_dkd_video = compute_dkd_loss(student_outputs["logits_video"], t_video_logits, targets, temperature=temperature, alpha=1.0, beta=2.0)

    loss_kd_logit = w_audio * l_dkd_audio + w_video * l_dkd_video

    # 3. Feature-Level KD Losses
    t_audio_feat = teacher_audio_outputs.get("feature", t_audio_logits)
    t_video_feat = teacher_video_outputs.get("feature", t_video_logits)

    l_kd_feat_audio = compute_feature_alignment_loss(student_outputs["proj_f_audio"], t_audio_feat)
    l_kd_feat_video = compute_feature_alignment_loss(student_outputs["proj_f_video"], t_video_feat)
    loss_kd_feat = w_audio * l_kd_feat_audio + w_video * l_kd_feat_video

    # Total Multimodal DKD Loss
    total_loss = loss_ce + alpha_logit * loss_kd_logit + beta_feature * loss_kd_feat

    loss_stats = {
        "ce_loss": float(loss_ce.item()),
        "audio_dkd_loss": float(l_dkd_audio.item()),
        "video_dkd_loss": float(l_dkd_video.item()),
        "audio_kd_feat": float(l_kd_feat_audio.item()),
        "video_kd_feat": float(l_kd_feat_video.item()),
    }

    return total_loss, loss_stats
