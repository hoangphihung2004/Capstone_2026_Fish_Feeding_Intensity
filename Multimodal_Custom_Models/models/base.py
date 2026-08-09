from __future__ import annotations

import torch
import torch.nn as nn


class BaseMultimodalModel(nn.Module):
    def forward(self, waveform: torch.Tensor, video_form: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError
