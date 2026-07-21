import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseVideoAug(nn.Module, ABC):
    """
    Abstract Base Class defining the PyTorch native interface for Video Augmentations.
    """
    @abstractmethod
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        pass
