import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_backbone import BaseBackbone


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C * S)

    def forward(self, x):
        N, C, T, F_bins = x.size()
        x = x.reshape(N, C * self.S, T, F_bins // self.S)
        x = self.bn(x)
        return x.reshape(N, C, T, F_bins)


class BroadcastedBlock(nn.Module):
    def __init__(self, planes: int, dilation=1, stride=1, temp_pad=(0, 1)):
        super(BroadcastedBlock, self).__init__()
        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 4)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        identity = x
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        auxilary = out
        out = out.mean(2, keepdim=True)
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        out = out + identity + auxilary
        out = self.relu(out)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, dilation=1, stride=1, temp_pad=(0, 1)):
        super(TransitionBlock, self).__init__()
        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride, dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 4)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        auxilary = out
        out = out.mean(2, keepdim=True)
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        out = auxilary + out
        out = self.relu(out)
        return out


class BC_ResNet(BaseBackbone):
    """
    Broadcasted Residual Network (BC-ResNet) backbone model for Mel-spectrogram processing.
    Inherits from BaseBackbone and takes [Batch, 1, H, W] Mel-spectrogram input.
    """
    def __init__(self, classes_num: int = 4, norm: bool = False) -> None:
        super(BC_ResNet, self).__init__()
        self.model_name = "bc_resnet"
        c = 40
        self.conv1 = nn.Conv2d(1, 2 * c, 5, stride=(2, 2), padding=(2, 2))
        self.block1_1 = TransitionBlock(2 * c, c)
        self.block1_2 = BroadcastedBlock(c)

        self.block2_1 = nn.MaxPool2d(2)

        self.block3_1 = TransitionBlock(c, int(1.5 * c))
        self.block3_2 = BroadcastedBlock(int(1.5 * c))

        self.block4_1 = nn.MaxPool2d(2)

        self.block5_1 = TransitionBlock(int(1.5 * c), int(2 * c))
        self.block5_2 = BroadcastedBlock(int(2 * c))

        self.block6_1 = TransitionBlock(int(2 * c), int(2.5 * c))
        self.block6_2 = BroadcastedBlock(int(2.5 * c))
        self.block6_3 = BroadcastedBlock(int(2.5 * c))

        self.block7_1 = nn.Conv2d(int(2.5 * c), classes_num, 1)
        self.block8_1 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.norm = norm
        self.fc_audioset = nn.Linear(1, classes_num, bias=True)
        if norm:
            self.one = nn.InstanceNorm2d(1)
            self.two = nn.InstanceNorm2d(1)
            self.three = nn.InstanceNorm2d(1)
            self.four = nn.InstanceNorm2d(1)
            self.five = nn.InstanceNorm2d(1)
            self.lamb = nn.Parameter(torch.tensor(0.5))  # Fix for missing self.lamb bug in original U-FFIA

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.norm:
            out = self.lamb * out + self.one(out)
        out = self.conv1(out)
        out = self.block1_1(out)
        out = self.block1_2(out)
        if self.norm:
            out = self.lamb * out + self.two(out)

        out = self.block2_1(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        if self.norm:
            out = self.lamb * out + self.three(out)

        out = self.block4_1(out)
        out = self.block5_1(out)
        out = self.block5_2(out)
        if self.norm:
            out = self.lamb * out + self.four(out)

        out = self.block6_1(out)
        out = self.block6_2(out)
        out = self.block6_3(out)
        if self.norm:
            out = self.lamb * out + self.five(out)

        out = self.block7_1(out)
        out = self.block8_1(out)
        
        clipwise_output = torch.squeeze(torch.squeeze(out, dim=2), dim=2)
        return clipwise_output
