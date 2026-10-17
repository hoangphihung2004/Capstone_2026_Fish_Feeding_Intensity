"""Audio-only student distilled from the frozen multimodal teacher."""

import torch
from torch import nn

from features.audio_frontend import AudioFrontend
from .audio_backbone import LightweightAudioEncoder


STUDENT_HEAD_KEYS = {"student": "clipwise_output"}


class AudioStudentArchitecture(nn.Module):
    """LightweightAudioEncoder + the teacher's audio pooling + a linear classifier."""

    def __init__(self, classes_num=4):
        super().__init__()
        self.audio = LightweightAudioEncoder()
        self.head = nn.Linear(LightweightAudioEncoder.channels[-1], classes_num)

    def forward(self, audio_features):
        if audio_features.ndim != 4 or audio_features.shape[1] != 1 or min(audio_features.shape[2:]) < 32:
            raise ValueError("Audio features must have shape [B, 1, T>=32, F>=32].")
        stages = []
        x = audio_features
        for block in self.audio.blocks:
            x = block(x)
            stages.append(self.audio.pool(x))
        # "feature_stages" concatenates the five pooled stage features (16+24+40+112+320); the last one feeds the head.
        return {"clipwise_output": self.head(stages[-1]), "feature_stages": torch.cat(stages, dim=1)}


class AudioStudentModel(nn.Module):
    def __init__(self, audio_config=None):
        super().__init__()
        self.frontend = AudioFrontend(audio_config)
        self.architecture = AudioStudentArchitecture()

    def forward(self, waveform):
        return self.architecture(self.frontend(waveform))

    def architecture_num_params(self):
        return sum(p.numel() for p in self.architecture.parameters())
