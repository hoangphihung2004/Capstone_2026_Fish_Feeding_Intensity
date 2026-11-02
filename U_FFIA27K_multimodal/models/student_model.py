"""Audio-only student distilled from the frozen multimodal teacher."""

import torch
from torch import nn

from features.audio_frontend import AudioFrontend
from features.stft_tkeo_frontend import TkeoStftFrontend
from .audio_backbone import LightweightAudioEncoder
from .audio_mlp import AudioMLPBackbone


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
    input_keys = ("waveform",)  # batch entries fed to forward (64 kHz waveform)

    def __init__(self, audio_config=None):
        super().__init__()
        self.frontend = AudioFrontend(audio_config)
        self.architecture = AudioStudentArchitecture()

    def forward(self, waveform):
        return self.architecture(self.frontend(waveform))

    def architecture_num_params(self):
        return sum(p.numel() for p in self.architecture.parameters())


class MLPStudentArchitecture(nn.Module):
    """AudioMLPBackbone + a linear classifier. Input is a single feature vector per clip, [B, in_features]."""

    def __init__(self, in_features, mlp_config, classes_num=4):
        super().__init__()
        self.backbone = AudioMLPBackbone(in_features, tuple(mlp_config.hidden_dims), mlp_config.embed_dim, mlp_config.dropout)
        self.head = nn.Linear(mlp_config.embed_dim, classes_num)

    def forward(self, features):
        return {"clipwise_output": self.head(self.backbone(features))}


class StftMlpStudentModel(nn.Module):
    """TKEO-STFT vector (256 kHz audio, 2049 bins) -> spectral aug + LayerNorm -> MLP -> classifier.

    The dataloader caches the 2049-D vector per clip (batch["stft_vector"]); forward can also take the raw 256 kHz
    waveform and extract it on the fly, which is what the latency measurement uses (end to end).
    """

    input_keys = ("stft_vector",)

    def __init__(self, stft_config, mlp_config):
        super().__init__()
        self.frontend = TkeoStftFrontend(stft_config)
        self.architecture = MLPStudentArchitecture(stft_config.stft_bins, mlp_config)

    def forward(self, stft_vector=None, waveform=None):
        return self.architecture(self.frontend(stft_vector=stft_vector, waveform=waveform))

    def architecture_num_params(self):
        return sum(p.numel() for p in self.architecture.parameters())
