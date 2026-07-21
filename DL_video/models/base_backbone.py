from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """
    Abstract Base Class (ABC / Interface) standardizing Video Backbone models.
    Enforces mandatory inheritance and implementation of abstract methods.

    Interface Contract:
      - Input:  5D Video tensor [Batch, Channels (3), Frames (T), Height (H), Width (W)]
      - Output: Classification logits tensor [Batch, Num_Classes]
    """
    def __init__(self) -> None:
        super(BaseBackbone, self).__init__()
        self.model_name = "base_backbone"

    def get_name(self) -> str:
        """
        Retrieve the name of the backbone model architecture.
        """
        return self.model_name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass. Subclasses MUST implement this abstract method.
        """
        pass
