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
from models.blocks.adaptive_gate_block import AdaptiveConfidenceGate
from models.blocks.attention_blocks import BottleneckTokenAttentionBlock, ECABlock
from models.blocks.depthwise_conv_block import DepthwiseAudioStage, DepthwiseSeparableConv2d
from models.blocks.mbt_block import MBTCrossAttentionBlock
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
    SOTA FF-MBT (Frequency-Spatial Feature Modulation & Multimodal Bottleneck Transformer Network):
    - Modular 5-Stage Backbone with Depthwise Separable Convolutions (~3.85M trainable params)
    - NeurIPS MBT Bottleneck Cross-Attention Blocks (K=4 tokens) at Stage 4 & Stage 5
    - IEEE Uncertainty-Aware Adaptive Confidence Gated Fusion (g_a, g_v)
    - MobileViTv2 Linear Self-Attention Blocks at Stage 4 & Stage 5
    - Hierarchical Stage 4 + Stage 5 Token Pyramid Attention Fusion
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

        # 2. SOTA Audio Backbone Stages (Depthwise Separable Conv + ECA + MobileViTv2)
        self.audio_stem = nn.Sequential(
            DepthwiseSeparableConv2d(1, 32, stride=2),
        )
        self.audio_stage2 = DepthwiseAudioStage(32, 24, stride=2)
        self.audio_stage3 = DepthwiseAudioStage(24, 40, stride=2)
        self.audio_stage4 = nn.Sequential(
            DepthwiseSeparableConv2d(40, 80, stride=2),
            MobileViTv2Block(80, 80, attn_dim=80),
        )
        self.audio_stage5 = nn.Sequential(
            DepthwiseSeparableConv2d(80, self.embed_dim, stride=2),
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

        # 4. SOTA NeurIPS MBT Bottleneck Cross-Attention Blocks (K=4 Bottleneck Tokens)
        self.mbt_stage4 = MBTCrossAttentionBlock(80, num_bottlenecks=4)
        self.mbt_stage5 = MBTCrossAttentionBlock(self.embed_dim, num_bottlenecks=4)

        # 5. SOTA IEEE Uncertainty-Aware Adaptive Gated Fusion Module
        self.adaptive_gate = AdaptiveConfidenceGate(self.embed_dim)

        # 6. Hierarchical Stage 4 + Stage 5 Token Pyramid Attention Fusion
        self.stage4_proj = nn.Conv2d(80, self.embed_dim, kernel_size=1)
        self.fusion_module = BottleneckTokenAttentionBlock(embed_dim=self.embed_dim, num_heads=4)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 7. Tri-Head Classifiers
        self.audio_head = nn.Linear(self.embed_dim, num_classes)
        self.video_head = nn.Linear(self.embed_dim, num_classes)
        self.fused_head = nn.Linear(self.embed_dim, num_classes)

        # 8. Teacher Feature Projection Adapters
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

        # Step B: Modality-Independent Stage 2 & 3
        a = self.audio_stage2(a)
        v = self.video_stage2(v)

        a = self.audio_stage3(a)
        v = self.video_stage3(v)

        # Step C: Stage 4 Extraction & MBT Bottleneck Cross-Attention Interaction
        a = self.audio_stage4(a)
        v = self.video_stage4(v)
        a, v = self.mbt_stage4(a, v)
        a4, v4 = a, v  # Save Stage 4 features for Hierarchical Token Pyramid Fusion

        # Step D: Stage 5 Extraction & MBT Bottleneck Cross-Attention Interaction
        a = self.audio_stage5(a)
        v = self.video_stage5(v)
        a, v = self.mbt_stage5(a, v)

        # Step E: Global Pooling for Single-Modality Features
        f_audio = self.global_pool(a).flatten(1)
        f_video = self.global_pool(v).flatten(1)

        # Step F: Single-Modality Heads
        logits_audio = self.audio_head(f_audio)
        logits_video = self.video_head(f_video)

        # Step G: Uncertainty-Aware Adaptive Gated Fusion
        f_adaptive, g_audio, g_video = self.adaptive_gate(f_audio, f_video)

        # Step H: Hierarchical Token Attention Fusion (Stage 4 + Stage 5 Tokens)
        a4_proj = self.stage4_proj(a4)
        v4_proj = self.stage4_proj(v4)

        a_tokens = torch.cat([a.flatten(2).transpose(1, 2), a4_proj.flatten(2).transpose(1, 2)], dim=1)
        v_tokens = torch.cat([v.flatten(2).transpose(1, 2), v4_proj.flatten(2).transpose(1, 2)], dim=1)
        tokens = torch.cat([a_tokens, v_tokens], dim=1)

        attended_tokens = self.fusion_module(tokens)
        f_token_fused = attended_tokens.mean(dim=1)

        f_fused = f_adaptive + f_token_fused

        # Step I: Fused Multimodal Head
        logits_fused = self.fused_head(f_fused)

        # Step J: Feature Projection for Distillation
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
            "gate_audio": g_audio,
            "gate_video": g_video,
        }

    def get_name(self) -> str:
        return self.model_name
