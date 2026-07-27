import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t

from models.base_backbone import BaseBackbone


def init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


def conv3_to_conv1(conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        if conv.weight.shape[1] == 3:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


def resize_spectrogram(x: torch.Tensor, input_size: int) -> torch.Tensor:
    if input_size <= 0:
        return x

    target_size = (input_size, input_size)
    if x.shape[-2:] == target_size:
        return x

    return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class SwinTiny(BaseBackbone):
    """
    Swin-Tiny backbone adapted for mono Mel-spectrogram classification.
    Input:  [B, 1, T, M]
    Output: [B, classes_num]
    """
    def __init__(
        self,
        classes_num: int = 4,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        input_size: int = 224,
    ) -> None:
        super(SwinTiny, self).__init__()
        self.model_name = "swin_tiny"
        self.input_size = input_size

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.model = swin_t(weights=weights)

        self.model.features[0][0] = conv3_to_conv1(self.model.features[0][0])

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, classes_num)
        init_linear(self.model.head)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model.head.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_spectrogram(x, self.input_size)
        return self.model(x)
