"""Low-cost hierarchical bidirectional messenger fusion for audio and video."""

import torch
from torch import nn
from torch.nn import functional as F


class HierarchicalMessengerFusion(nn.Module):
    """Exchange cross-modal information through four compact messenger tokens."""

    def __init__(self, audio_channels, video_channels, latent_dim=32, tokens=4, output_dim=64):
        super().__init__()
        self.tokens = tokens
        self.audio_projection = nn.Sequential(nn.Linear(audio_channels, latent_dim), nn.LayerNorm(latent_dim))
        self.video_projection = nn.Sequential(nn.Linear(video_channels, latent_dim), nn.LayerNorm(latent_dim))
        self.messenger = nn.Parameter(torch.zeros(1, tokens, latent_dim))
        nn.init.normal_(self.messenger, std=0.02)
        self.audio_to_messenger = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        self.video_to_messenger = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        self.audio_from_messenger = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        self.video_from_messenger = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        self.audio_reliability = nn.Linear(latent_dim, 1)
        self.video_reliability = nn.Linear(latent_dim, 1)
        self.messenger_norm = nn.LayerNorm(latent_dim)
        self.audio_norm = nn.LayerNorm(latent_dim)
        self.video_norm = nn.LayerNorm(latent_dim)
        self.audio_update = nn.Linear(latent_dim, audio_channels)
        self.video_update = nn.Linear(latent_dim, video_channels)
        self.audio_scale = nn.Parameter(torch.tensor(0.0))
        self.video_scale = nn.Parameter(torch.tensor(0.0))
        self.summary = nn.Sequential(nn.Linear(latent_dim, output_dim), nn.LayerNorm(output_dim))

    @staticmethod
    def _tokens(feature, projection, messenger):
        pooled = F.adaptive_avg_pool2d(feature, (2, 2)).flatten(2).transpose(1, 2)
        return projection(pooled), messenger.expand(feature.shape[0], -1, -1)

    def forward(self, audio, video):
        audio_tokens, messenger = self._tokens(audio, self.audio_projection, self.messenger)
        video_tokens, _ = self._tokens(video, self.video_projection, self.messenger)

        audio_message, _ = self.audio_to_messenger(messenger, audio_tokens, audio_tokens)
        video_message, _ = self.video_to_messenger(messenger, video_tokens, video_tokens)
        reliability = torch.cat(
            [self.audio_reliability(audio_message.mean(1)), self.video_reliability(video_message.mean(1))], dim=1
        )
        reliability = F.softmax(reliability, dim=1)
        joint_message = self.messenger_norm(
            messenger
            + reliability[:, 0:1, None] * audio_message
            + reliability[:, 1:2, None] * video_message
        )

        audio_context, _ = self.audio_from_messenger(audio_tokens, joint_message, joint_message)
        video_context, _ = self.video_from_messenger(video_tokens, joint_message, joint_message)
        audio_context = self.audio_norm(audio_tokens + audio_context).mean(1)
        video_context = self.video_norm(video_tokens + video_context).mean(1)

        audio = audio + self.audio_scale.tanh() * self.audio_update(audio_context)[:, :, None, None]
        video = video + self.video_scale.tanh() * self.video_update(video_context)[:, :, None, None]
        return audio, video, self.summary(joint_message.mean(1))
