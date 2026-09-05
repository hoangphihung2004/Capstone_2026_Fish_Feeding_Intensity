"""One jointly trained architecture, with a separate baseline audio frontend."""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from features.audio_frontend import AudioFrontend
from .audio_backbone import LightweightAudioEncoder
from .fusion import HierarchicalMessengerFusion
from .surgery import adapt_first_conv_to_multi_channels


HEAD_KEYS = {"audio": "audio_output", "video": "video_output", "multimodal": "clipwise_output"}


class MultimodalArchitecture(nn.Module):
    """Log-mel + six-channel video with hierarchical messenger fusion."""

    def __init__(self, pretrained_video=True, classes_num=4):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained_video else None
        self.video = efficientnet_b0(weights=weights).features[:8]
        adapt_first_conv_to_multi_channels(self.video, target_channels=6)
        self.audio = LightweightAudioEncoder()
        self.fusion = nn.ModuleList(HierarchicalMessengerFusion(c, c) for c in (40, 112, 320))
        self.fusion_aggregate = nn.Sequential(nn.Linear(3 * 64, 64), nn.LayerNorm(64))
        self.audio_head = nn.Linear(320, classes_num)
        self.video_head = nn.Linear(320, classes_num)
        self.multimodal_head = nn.Sequential(
            nn.Linear(704, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, classes_num))

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
            if index >= 2:
                audio, video, fusion_state = self.fusion[index - 2](audio, video)
                fusion_states.append(fusion_state)
        audio = self.audio.pool(audio)
        video = video.mean(dim=(2, 3))
        fusion_state = self.fusion_aggregate(torch.cat(fusion_states, dim=1))
        return {
            "audio_output": self.audio_head(audio),
            "video_output": self.video_head(video),
            "clipwise_output": self.multimodal_head(torch.cat((audio, video, fusion_state), dim=1)),
        }


class MultimodalModel(nn.Module):
    def __init__(self, audio_config=None, pretrained_video=True):
        super().__init__()
        self.frontend = AudioFrontend(audio_config)
        self.architecture = MultimodalArchitecture(pretrained_video=pretrained_video)

    def forward(self, waveform, video_form):
        return self.architecture(self.frontend(waveform), video_form)

    def architecture_num_params(self):
        model = self.architecture
        return sum(p.numel() for p in model.parameters())
