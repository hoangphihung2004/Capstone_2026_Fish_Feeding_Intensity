"""One jointly trained architecture, with a separate baseline audio frontend."""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from features.audio_frontend import AudioFrontend
from .audio_backbone import LightweightAudioEncoder
from .fusion import HierarchicalMessengerFusion
from .surgery import adapt_first_conv_to_multi_channels


HEAD_KEYS = {"multimodal": "clipwise_output"}
ABLATION_FUSION_INDICES = {
    "AF0": (),
    "AF1": (3,),
    "AF2": (2, 3),
    "AF3": (1, 2, 3),
    "AF4": (0, 1, 2, 3),
}


class MultimodalArchitecture(nn.Module):
    """Log-mel + six-channel video with hierarchical messenger fusion."""

    def __init__(self, pretrained_video=True, classes_num=4, ablation_mode="AF4"):
        super().__init__()
        if ablation_mode not in ABLATION_FUSION_INDICES:
            raise ValueError(f"Unknown ablation_mode='{ablation_mode}'. Expected one of {tuple(ABLATION_FUSION_INDICES)}.")
        self.ablation_mode = ablation_mode
        self.enabled_fusion_indices = ABLATION_FUSION_INDICES[ablation_mode]
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained_video else None
        self.video = efficientnet_b0(weights=weights).features[:8]
        adapt_first_conv_to_multi_channels(self.video, target_channels=6)
        self.audio = LightweightAudioEncoder()
        fusion_channels = (24, 40, 112, 320)
        self.fusion = nn.ModuleDict(
            {
                str(index): HierarchicalMessengerFusion(fusion_channels[index], fusion_channels[index])
                for index in self.enabled_fusion_indices
            }
        )
        self.multimodal_head = nn.Linear(320 + 320 + 32 * len(self.enabled_fusion_indices), classes_num)

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
            fusion_index = index - 1
            if fusion_index in self.enabled_fusion_indices:
                audio, video, fusion_state = self.fusion[str(fusion_index)](audio, video)
                fusion_states.append(fusion_state)
        audio = self.audio.pool(audio)
        video = video.mean(dim=(2, 3))
        features = (audio, video, *fusion_states)
        return {
            "clipwise_output": self.multimodal_head(torch.cat(features, dim=1)),
        }


class MultimodalModel(nn.Module):
    def __init__(self, audio_config=None, pretrained_video=True, ablation_mode="AF4"):
        super().__init__()
        self.frontend = AudioFrontend(audio_config)
        self.architecture = MultimodalArchitecture(
            pretrained_video=pretrained_video,
            ablation_mode=ablation_mode,
        )

    def forward(self, waveform, video_form):
        return self.architecture(self.frontend(waveform), video_form)

    def architecture_num_params(self):
        model = self.architecture
        return sum(p.numel() for p in model.parameters())
