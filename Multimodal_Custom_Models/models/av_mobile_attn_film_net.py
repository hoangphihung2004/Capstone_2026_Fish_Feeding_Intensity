from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AudioFeaturesConfig, VideoFeaturesConfig
from features.audio_frontend import AudioFrontend
from models.base import BaseMultimodalModel
from models.blocks import CrossFiLMBlock, DepthwiseSeparableProjection, MobileV2Block, TokenSelfAttentionBlock


class AVMobileAttnFiLMNet(BaseMultimodalModel):
    """
    Lightweight audio-image model with mid-level cross-modal FiLM interaction
    and compact token self-attention before a MobileNet-style classifier.
    """
    def __init__(
        self,
        num_classes: int = 4,
        audio_features_config: AudioFeaturesConfig | None = None,
        video_features_config: VideoFeaturesConfig | None = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = "av_mobile_attn_film_net"
        self.embed_dim = 512
        self.audio_frontend = AudioFrontend(audio_features_config)

        self.audio_stage1 = MobileV2Block(1, 16)
        self.audio_stage2 = MobileV2Block(16, 32)
        self.audio_stage3 = MobileV2Block(32, 64)
        self.audio_stage4 = MobileV2Block(64, 128)
        self.audio_stage5 = MobileV2Block(128, 256)

        self.video_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.video_stage1 = MobileV2Block(16, 16)
        self.video_stage2 = MobileV2Block(16, 32)
        self.video_stage3 = MobileV2Block(32, 64)
        self.video_stage4 = MobileV2Block(64, 128)
        self.video_stage5 = MobileV2Block(128, 256)

        self.cross2 = CrossFiLMBlock(32)
        self.cross3 = CrossFiLMBlock(64)
        self.cross4 = CrossFiLMBlock(128)
        self.cross5 = CrossFiLMBlock(256)

        self.audio_projection = DepthwiseSeparableProjection(256, self.embed_dim)
        self.video_projection = DepthwiseSeparableProjection(256, self.embed_dim)
        self.attention = TokenSelfAttentionBlock(embed_dim=self.embed_dim, num_heads=4, dropout=0.1)
        self.joint_gate = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.embed_dim, num_classes),
        )

    def _tokens_from_map(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        tokens = F.adaptive_avg_pool2d(x, output_size).flatten(2).transpose(1, 2)
        return tokens

    def forward(self, waveform: torch.Tensor, video_form: torch.Tensor) -> dict[str, torch.Tensor]:
        audio = self.audio_frontend(waveform)
        video = self.video_stem(video_form)

        audio = self.audio_stage1(audio)
        video = self.video_stage1(video)

        audio = self.audio_stage2(audio)
        video = self.video_stage2(video)
        audio, video = self.cross2(audio, video)

        audio = self.audio_stage3(audio)
        video = self.video_stage3(video)
        audio, video = self.cross3(audio, video)

        audio = self.audio_stage4(audio)
        video = self.video_stage4(video)
        audio, video = self.cross4(audio, video)

        audio = self.audio_stage5(audio)
        video = self.video_stage5(video)
        audio, video = self.cross5(audio, video)

        audio_high = self.audio_projection(audio)
        video_high = self.video_projection(video)
        audio_tokens = self._tokens_from_map(audio_high, output_size=(4, 2))
        video_tokens = self._tokens_from_map(video_high, output_size=(2, 2))
        tokens = self.attention(torch.cat([audio_tokens, video_tokens], dim=1))
        context = tokens.mean(dim=1).unsqueeze(-1).unsqueeze(-1)

        target_hw = audio_high.shape[-2:]
        video_high = F.adaptive_avg_pool2d(video_high, target_hw)
        gate = self.joint_gate(torch.cat([audio_high, video_high], dim=1))
        joint = gate * audio_high + (1.0 - gate) * video_high + context

        pooled = torch.flatten(self.avgpool(joint), 1)
        logits = self.classifier(pooled)
        return {
            "clipwise_output": logits,
            "embedding": pooled,
        }

    def get_name(self) -> str:
        return self.model_name
