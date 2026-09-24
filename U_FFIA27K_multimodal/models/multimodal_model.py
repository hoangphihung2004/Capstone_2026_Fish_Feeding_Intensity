"""One jointly trained architecture, with a separate baseline audio frontend."""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from features.audio_frontend import AudioFrontend
from .audio_backbone import LightweightAudioEncoder
from .fusion import HierarchicalMessengerFusion
from .surgery import adapt_first_conv_to_multi_channels


HEAD_KEYS = {"multimodal": "clipwise_output"}
ABLATION_FUSION_NAMES = {
    "AF0": (),
    "AF1": ("F4",),
    "AF2": ("F3", "F4"),
    "AF3": ("F2", "F3", "F4"),
    "AF4": ("F1", "F2", "F3", "F4"),
    "AF5": ("F5", "F1", "F2", "F3", "F4"),
}
FUSION_CHANNELS = {"F5": 16, "F1": 24, "F2": 40, "F3": 112, "F4": 320}
FUSION_NAME_BY_BLOCK_INDEX = {0: "F5", 1: "F1", 2: "F2", 3: "F3", 4: "F4"}


class MultimodalArchitecture(nn.Module):
    """Log-mel + six-channel video with hierarchical messenger fusion."""

    def __init__(self, pretrained_video=True, classes_num=4, ablation_mode="AF4", dropout_rate=0.2):
        super().__init__()
        if ablation_mode not in ABLATION_FUSION_NAMES:
            raise ValueError(f"Unknown ablation_mode='{ablation_mode}'. Expected one of {tuple(ABLATION_FUSION_NAMES)}.")
        self.ablation_mode = ablation_mode
        self.enabled_fusion_names = ABLATION_FUSION_NAMES[ablation_mode]
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained_video else None
        self.video = efficientnet_b0(weights=weights).features[:8]
        adapt_first_conv_to_multi_channels(self.video, target_channels=6)
        self.audio = LightweightAudioEncoder()
        self.fusion = nn.ModuleDict(
            {
                name: HierarchicalMessengerFusion(FUSION_CHANNELS[name], FUSION_CHANNELS[name])
                for name in self.enabled_fusion_names
            }
        )
        self.classifier_dropout = nn.Dropout(p=dropout_rate)
        self.multimodal_head = nn.Linear(320 + 320 + 32 * len(self.enabled_fusion_names), classes_num)

    def forward(self, audio_features, video_form):
        if audio_features.ndim != 4 or audio_features.shape[1] != 1 or min(audio_features.shape[2:]) < 32:
            raise ValueError("Audio features must have shape [B, 1, T>=32, F>=32].")
        if video_form.ndim != 4 or video_form.shape[1] != 6:
            raise ValueError("Video must have shape [B, 6, H, W] in first_last order.")
        if audio_features.shape[0] != video_form.shape[0]:
            raise ValueError("Audio and video batch sizes must match.")
        audio = audio_features
        video = self.video[0](video_form)
        fusion_states = []
        for index, (block, video_stages) in enumerate(zip(
                self.audio.blocks, ((1,), (2,), (3,), (4, 5), (6, 7)))):
            audio = block(audio)
            for stage in video_stages:
                video = self.video[stage](video)
            fusion_name = FUSION_NAME_BY_BLOCK_INDEX[index]
            if fusion_name in self.enabled_fusion_names:
                audio, video, fusion_state = self.fusion[fusion_name](audio, video)
                fusion_states.append(fusion_state)
        audio = self.audio.pool(audio)
        video = video.mean(dim=(2, 3))
        features = (audio, video, *fusion_states)
        return {
            "clipwise_output": self.multimodal_head(self.classifier_dropout(torch.cat(features, dim=1))),
        }


class MultimodalModel(nn.Module):
    def __init__(self, audio_config=None, pretrained_video=True, ablation_mode="AF4", dropout_rate=0.2):
        super().__init__()
        self.frontend = AudioFrontend(audio_config)
        self.architecture = MultimodalArchitecture(
            pretrained_video=pretrained_video,
            ablation_mode=ablation_mode,
            dropout_rate=dropout_rate,
        )

    def forward(self, waveform, video_form):
        return self.architecture(self.frontend(waveform), video_form)

    def architecture_num_params(self):
        model = self.architecture
        return sum(p.numel() for p in model.parameters())
