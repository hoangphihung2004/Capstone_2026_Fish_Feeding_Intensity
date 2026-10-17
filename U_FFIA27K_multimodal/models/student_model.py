"""Audio-only student distilled from the frozen multimodal teacher."""

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
        audio = self.audio.pool(self.audio(audio_features))
        # "feature" is the 320-D pooled (max + mean) vector, the same point the teacher's audio feature is taken at.
        return {"clipwise_output": self.head(audio), "feature": audio}


class AudioStudentModel(nn.Module):
    def __init__(self, audio_config=None):
        super().__init__()
        self.frontend = AudioFrontend(audio_config)
        self.architecture = AudioStudentArchitecture()

    def forward(self, waveform):
        return self.architecture(self.frontend(waveform))

    def architecture_num_params(self):
        return sum(p.numel() for p in self.architecture.parameters())
