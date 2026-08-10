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
from models.blocks import (
    BottleneckTokenAttentionBlock,
    CBAMBlock,
    ConvBNReLU,
    DepthwiseSeparableProjection,
    DynamicInteractionUnit,
    ECABlock,
    MobileV2Block,
    SEBlock,
)


class AVMobileDIFMEfficientNet(BaseMultimodalModel):
    """
    Audio-video model with CNN14-MobileNetV2-style audio extraction,
    partial EfficientNet-B0 visual extraction, and four dynamic interaction units.

    Audio preprocessing remains the original log-mel frontend used by the
    pre-adaptive audio branch.
    """

    def __init__(
        self,
        num_classes: int = 4,
        audio_features_config: AudioFeaturesConfig | None = None,
        video_features_config: VideoFeaturesConfig | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = "av_mobile_difm_efficientnet"
        self.embed_dim = 512

        self.audio_frontend = AudioFrontend(audio_features_config)
        self.audio_stage1 = MobileV2Block(1, 16)
        self.audio_stage2 = MobileV2Block(16, 32)
        self.audio_stage3 = MobileV2Block(32, 64)
        self.audio_stage4 = MobileV2Block(64, 128)
        self.audio_stage5 = MobileV2Block(128, 256)
        self.audio_attn2 = ECABlock(32)
        self.audio_attn3 = ECABlock(64)
        self.audio_attn4 = SEBlock(128)
        self.audio_attn5 = SEBlock(256)

        efficientnet = self._build_efficientnet(pretrained=pretrained)
        self.video_layers = nn.ModuleList(list(efficientnet.features.children())[:6])
        self.video_proj2 = ConvBNReLU(24, 32, kernel_size=1)
        self.video_proj3 = ConvBNReLU(40, 64, kernel_size=1)
        self.video_proj4 = ConvBNReLU(80, 128, kernel_size=1)
        self.video_proj5 = ConvBNReLU(112, 256, kernel_size=1)
        self.video_attn2 = ECABlock(32)
        self.video_attn3 = ECABlock(64)
        self.video_attn4 = CBAMBlock(128)
        self.video_attn5 = CBAMBlock(256)

        self.interaction2 = DynamicInteractionUnit(32)
        self.interaction3 = DynamicInteractionUnit(64)
        self.interaction4 = DynamicInteractionUnit(128)
        self.interaction5 = DynamicInteractionUnit(256)

        self.audio_projection = DepthwiseSeparableProjection(256, self.embed_dim)
        self.video_projection = DepthwiseSeparableProjection(256, self.embed_dim)
        self.token_attention = BottleneckTokenAttentionBlock(
            embed_dim=self.embed_dim,
            attn_dim=128,
            num_heads=4,
            dropout=0.1,
        )
        self.final_gate = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, self.embed_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.context_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    @staticmethod
    def _build_efficientnet(pretrained: bool) -> nn.Module:
        if EfficientNet_B0_Weights is None:
            return efficientnet_b0(pretrained=pretrained)
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return efficientnet_b0(weights=weights)

    @staticmethod
    def _tokens_from_map(x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, output_size).flatten(2).transpose(1, 2)

    def _forward_audio(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.audio_frontend(waveform)
        x = F.dropout(self.audio_stage1(x), p=0.2, training=self.training)
        x2 = self.audio_attn2(F.dropout(self.audio_stage2(x), p=0.2, training=self.training))
        x3 = self.audio_attn3(F.dropout(self.audio_stage3(x2), p=0.2, training=self.training))
        x4 = self.audio_attn4(F.dropout(self.audio_stage4(x3), p=0.2, training=self.training))
        x5 = self.audio_attn5(F.dropout(self.audio_stage5(x4), p=0.2, training=self.training))
        return x2, x3, x4, x5

    def _forward_video(self, video_form: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.video_layers[0](video_form)
        x = self.video_layers[1](x)
        x = self.video_layers[2](x)
        x2 = self.video_attn2(self.video_proj2(x))
        x = self.video_layers[3](x)
        x3 = self.video_attn3(self.video_proj3(x))
        x = self.video_layers[4](x)
        x4 = self.video_attn4(self.video_proj4(x))
        x = self.video_layers[5](x)
        x5 = self.video_attn5(self.video_proj5(x))
        return x2, x3, x4, x5

    def forward(self, waveform: torch.Tensor, video_form: torch.Tensor) -> dict[str, torch.Tensor]:
        audio2, audio3, audio4, audio5 = self._forward_audio(waveform)
        video2, video3, video4, video5 = self._forward_video(video_form)

        audio2, video2 = self.interaction2(audio2, video2)
        audio3, video3 = self.interaction3(audio3, video3)
        audio4, video4 = self.interaction4(audio4, video4)
        audio5, video5 = self.interaction5(audio5, video5)

        audio_high = self.audio_projection(audio5)
        video_high = self.video_projection(video5)

        audio_tokens = self._tokens_from_map(audio_high, output_size=(4, 2))
        video_tokens = self._tokens_from_map(video_high, output_size=(2, 2))
        tokens = self.token_attention(torch.cat([audio_tokens, video_tokens], dim=1))
        context = tokens.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        context = self.context_proj(context)

        target_hw = audio_high.shape[-2:]
        video_high = F.adaptive_avg_pool2d(video_high, target_hw)
        gate = self.final_gate(torch.cat([audio_high, video_high], dim=1))
        fused = gate * audio_high + (1.0 - gate) * video_high + context

        embedding = torch.flatten(self.avgpool(fused), 1)
        logits = self.classifier(embedding)
        return {
            "clipwise_output": logits,
            "embedding": embedding,
        }

    def get_name(self) -> str:
        return self.model_name
