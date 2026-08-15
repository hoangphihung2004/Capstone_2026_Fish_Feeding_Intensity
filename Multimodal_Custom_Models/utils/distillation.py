from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Computes Knowledge Distillation Loss using KL Divergence over soft logits."""
    temperature = float(temperature)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature * temperature)


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



def compute_multimodal_distillation_loss(
    student_outputs: dict[str, torch.Tensor],
    teacher_audio_outputs: dict[str, torch.Tensor],
    teacher_video_outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha_logit: float = 1.0,
    beta_feature: float = 2.0,
    lambda_aux: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes formal multi-head multi-level distillation loss for multimodal student models:
    1. Task Supervised Loss: Cross-Entropy for 3 heads (Primary Fused Head + Aux Audio & Video Heads)
    2. Logit KD Loss: KL Divergence for Audio Head vs Audio Teacher and Video Head vs Video Teacher
    3. Feature KD Loss: Cosine Distance for Projected Audio/Video Features vs Teacher Features
    """
    ce_loss_fn = nn.CrossEntropyLoss()

    # 1. Supervised Task Losses
    l_ce_fused = ce_loss_fn(student_outputs["logits_fused"], targets)
    l_ce_audio = ce_loss_fn(student_outputs["logits_audio"], targets)
    l_ce_video = ce_loss_fn(student_outputs["logits_video"], targets)
    loss_ce = l_ce_fused + lambda_aux * (l_ce_audio + l_ce_video)

    # 2. Logit-Level KD Losses
    t_audio_logits = teacher_audio_outputs.get("logits", teacher_audio_outputs.get("clipwise_output"))
    t_video_logits = teacher_video_outputs.get("logits", teacher_video_outputs.get("clipwise_output"))

    l_kd_logit_audio = compute_kl_divergence_loss(student_outputs["logits_audio"], t_audio_logits, temperature)
    l_kd_logit_video = compute_kl_divergence_loss(student_outputs["logits_video"], t_video_logits, temperature)
    loss_kd_logit = l_kd_logit_audio + l_kd_logit_video

    # 3. Feature-Level KD Losses
    t_audio_feat = teacher_audio_outputs.get("feature", t_audio_logits)
    t_video_feat = teacher_video_outputs.get("feature", t_video_logits)

    l_kd_feat_audio = compute_feature_alignment_loss(student_outputs["proj_f_audio"], t_audio_feat)
    l_kd_feat_video = compute_feature_alignment_loss(student_outputs["proj_f_video"], t_video_feat)
    loss_kd_feat = l_kd_feat_audio + l_kd_feat_video

    # Total Multimodal Distillation Loss
    total_loss = loss_ce + alpha_logit * loss_kd_logit + beta_feature * loss_kd_feat

    loss_stats = {
        "ce_loss": float(loss_ce.item()),
        "audio_logit_kd": float(l_kd_logit_audio.item()),
        "video_logit_kd": float(l_kd_logit_video.item()),
        "audio_feat_kd": float(l_kd_feat_audio.item()),
        "video_feat_kd": float(l_kd_feat_video.item()),
        "total_loss": float(total_loss.item()),
    }

    return total_loss, loss_stats
