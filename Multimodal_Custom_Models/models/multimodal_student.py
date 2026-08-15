from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
except ImportError:
    from torchvision.models import efficientnet_b0
    EfficientNet_B0_Weights = None

from config import AudioFeaturesConfig, VideoFeaturesConfig
from features.audio_frontend import AudioFrontend
from models.base import BaseMultimodalModel
from models.blocks.attention_blocks import BottleneckTokenAttentionBlock, ECABlock
from models.blocks.cross_modal_blocks import LightweightCrossAttentionUnit
from models.blocks.mobilevit_block import MobileViTv2Block


class FeatureProjectionAdapter(nn.Module):
    """Linear Feature Adapter projecting student features to match teacher feature dimension."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MultimodalStudentModel(BaseMultimodalModel):
    """
    Upgraded MobileViT-Multimodal Student Model featuring:
    - MobileViTv2 Linear Attention Blocks at Stage 4 & Stage 5 for Global-Local Feature Learning
    - Lightweight Multi-Scale Cross-Attention Units (Q-K-V) for Inter-Modality Interaction
    - Hierarchical Multi-Stage Feature Fusion (Stage 4 Fine-grained + Stage 5 High-level Features)
    - Tri-Head Classification Architecture (Audio Head, Video Head, Fused Head)
    """

    def __init__(
        self,
        num_classes: int = 4,
        audio_features_config: AudioFeaturesConfig | None = None,
        video_features_config: VideoFeaturesConfig | None = None,
        pretrained: bool = True,
        audio_teacher_dim: int = 2048,
        video_teacher_dim: int = 1280,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.model_name = "multimodal_student"
        self.embed_dim = embed_dim

        # 1. Standard Audio Frontend
        self.audio_frontend = AudioFrontend(audio_features_config)

        # 2. Audio Backbone Stages (24, 40 -> MobileViTv2 80, MobileViTv2 embed_dim)
        self.audio_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.audio_stage2 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            ECABlock(24),
        )
        self.audio_stage3 = nn.Sequential(
            nn.Conv2d(24, 40, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            ECABlock(40),
        )
        self.audio_stage4 = nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1),
            MobileViTv2Block(80, 80, attn_dim=80),
        )
        self.audio_stage5 = nn.Sequential(
            nn.Conv2d(80, self.embed_dim, kernel_size=3, stride=2, padding=1),
            MobileViTv2Block(self.embed_dim, self.embed_dim, attn_dim=128),
        )

        # 3. Video Backbone Stages (EfficientNet-B0 + MobileViTv2)
        efficientnet = self._build_efficientnet(pretrained=pretrained)
        v_layers = list(efficientnet.features.children())
        self.video_stem = v_layers[0]  # out: 32
        self.video_stage2 = nn.Sequential(v_layers[1], v_layers[2])  # out: 24
        self.video_stage3 = v_layers[3]  # out: 40
        self.video_stage4 = nn.Sequential(v_layers[4], MobileViTv2Block(80, 80, attn_dim=80))  # out: 80
        self.video_stage5 = nn.Sequential(
            v_layers[5],
            v_layers[6],
            nn.Conv2d(192, self.embed_dim, kernel_size=1),
            MobileViTv2Block(self.embed_dim, self.embed_dim, attn_dim=128),
        )

        # 4. Lightweight Cross-Attention Inter-Modality Units (Q-K-V)
        self.interaction_stage2 = LightweightCrossAttentionUnit(24)
        self.interaction_stage3 = LightweightCrossAttentionUnit(40)
        self.interaction_stage4 = LightweightCrossAttentionUnit(80)
        self.interaction_stage5 = LightweightCrossAttentionUnit(self.embed_dim)

        # 5. Hierarchical Feature Fusion & Token Attention
        self.stage4_proj = nn.Conv2d(80, self.embed_dim, kernel_size=1)
        self.fusion_module = BottleneckTokenAttentionBlock(embed_dim=self.embed_dim, num_heads=4)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 6. Tri-Head Classifiers
        self.audio_head = nn.Linear(self.embed_dim, num_classes)
        self.video_head = nn.Linear(self.embed_dim, num_classes)
        self.fused_head = nn.Linear(self.embed_dim, num_classes)

        # 7. Teacher Feature Projection Adapters
        self.audio_adapter = FeatureProjectionAdapter(self.embed_dim, audio_teacher_dim)
        self.video_adapter = FeatureProjectionAdapter(self.embed_dim, video_teacher_dim)

    @staticmethod
    def _build_efficientnet(pretrained: bool) -> nn.Module:
        if not pretrained or EfficientNet_B0_Weights is None:
            return efficientnet_b0(weights=None)
        try:
            return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            return efficientnet_b0(weights=None)

    def forward(self, waveform: torch.Tensor, video_form: torch.Tensor) -> dict[str, torch.Tensor]:
        # Step A: Preprocess waveform & video stems
        a = self.audio_frontend(waveform)
        a = self.audio_stem(a)
        v = self.video_stem(video_form)

        # Step B: Multi-Stage Feature Extraction & Cross-Attention Interaction
        a = self.audio_stage2(a)
        v = self.video_stage2(v)
        a, v = self.interaction_stage2(a, v)

        a = self.audio_stage3(a)
        v = self.video_stage3(v)
        a, v = self.interaction_stage3(a, v)

        a = self.audio_stage4(a)
        v = self.video_stage4(v)
        a, v = self.interaction_stage4(a, v)
        a4, v4 = a, v  # Save Stage 4 features for Hierarchical Fusion

        a = self.audio_stage5(a)
        v = self.video_stage5(v)
        a, v = self.interaction_stage5(a, v)

        # Step C: Global Pooling for Single-Modality Features
        f_audio = self.global_pool(a).flatten(1)
        f_video = self.global_pool(v).flatten(1)

        # Step D: Single-Modality Classification Heads
        logits_audio = self.audio_head(f_audio)
        logits_video = self.video_head(f_video)

        # Step E: Hierarchical Token Attention Fusion (Stage 4 + Stage 5 Tokens)
        a4_proj = self.stage4_proj(a4)
        v4_proj = self.stage4_proj(v4)

        a_tokens = torch.cat([a.flatten(2).transpose(1, 2), a4_proj.flatten(2).transpose(1, 2)], dim=1)
        v_tokens = torch.cat([v.flatten(2).transpose(1, 2), v4_proj.flatten(2).transpose(1, 2)], dim=1)
        tokens = torch.cat([a_tokens, v_tokens], dim=1)

        attended_tokens = self.fusion_module(tokens)
        f_fused = attended_tokens.mean(dim=1)

        # Step F: Fused Multimodal Head
        logits_fused = self.fused_head(f_fused)

        # Step G: Feature Projection for Distillation
        proj_f_audio = self.audio_adapter(f_audio)
        proj_f_video = self.video_adapter(f_video)

        return {
            "clipwise_output": logits_fused,
            "logits_audio": logits_audio,
            "logits_video": logits_video,
            "logits_fused": logits_fused,
            "f_audio": f_audio,
            "f_video": f_video,
            "f_fused": f_fused,
            "proj_f_audio": proj_f_audio,
            "proj_f_video": proj_f_video,
        }

    def get_name(self) -> str:
        return self.model_name
