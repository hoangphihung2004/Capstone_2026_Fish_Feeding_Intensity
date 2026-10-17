"""Video-only student distilled from the frozen multimodal teacher."""

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .surgery import adapt_first_conv_to_multi_channels


STUDENT_HEAD_KEYS = {"student": "clipwise_output"}
VIDEO_FEATURE_CHANNELS = 320


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
        video = self.video(video_form).mean(dim=(2, 3))
        return {"clipwise_output": self.head(video)}


class VideoStudentModel(nn.Module):
    def __init__(self, pretrained_video=True):
        super().__init__()
        self.architecture = VideoStudentArchitecture(pretrained_video=pretrained_video)

    def forward(self, video_form):
        return self.architecture(video_form)

    def architecture_num_params(self):
        return sum(p.numel() for p in self.architecture.parameters())
