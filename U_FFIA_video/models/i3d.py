import torch
import torch.nn as nn


class I3D(nn.Module):
    """
    I3D ResNet-50 video classifier loaded from PyTorchVideo via torch.hub.
    """

    input_type = "clip"
    minimum_frames = 8

    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        # Load model from PyTorchVideo via torch.hub
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=pretrained)
        
        # Replace projection layer to match classes_num
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, classes_num)
        
        self.model_name = "i3d_r50"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
