"""Video-only student distilled from the frozen multimodal teacher."""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .surgery import adapt_first_conv_to_multi_channels


STUDENT_HEAD_KEYS = {"student": "clipwise_output"}
VIDEO_FEATURE_CHANNELS = 320
VIDEO_STAGE_ENDS = (1, 2, 3, 5, 7)


class VideoStudentArchitecture(nn.Module):
    """The teacher's EfficientNet-B0 video branch (features[:8], 6-channel first_last input) + a linear classifier."""

    def __init__(self, pretrained_video=True, classes_num=4):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained_video else None
        self.video = efficientnet_b0(weights=weights).features[:8]
        adapt_first_conv_to_multi_channels(self.video, target_channels=6)
        self.head = nn.Linear(VIDEO_FEATURE_CHANNELS, classes_num)

    def forward(self, video_form):
        if video_form.ndim != 4 or video_form.shape[1] != 6:
            raise ValueError("Video must have shape [B, 6, H, W] in first_last order.")
        # Stages 1-5 end at features[1], [2], [3], [5], [7]: the same points the teacher's video branch is fused at.
        stages = []
        x = self.video[0](video_form)
        for index in range(1, len(self.video)):
            x = self.video[index](x)
            if index in VIDEO_STAGE_ENDS:
                stages.append(x.mean(dim=(2, 3)))
        # "feature_stages" concatenates the five pooled stage features (16+24+40+112+320); the last one feeds the head.
        return {"clipwise_output": self.head(stages[-1]), "feature_stages": torch.cat(stages, dim=1)}


class VideoStudentModel(nn.Module):
    def __init__(self, pretrained_video=True):
        super().__init__()
        self.architecture = VideoStudentArchitecture(pretrained_video=pretrained_video)

    def forward(self, video_form):
        return self.architecture(video_form)

    def architecture_num_params(self):
        return sum(p.numel() for p in self.architecture.parameters())
