from .convnext_tiny import ConvNeXtTiny
from .densenet121 import DenseNet121
from .efficientnet_b0 import EfficientNetB0
from .mobilenet_v2 import MobileNetV2
from .resnet18 import ResNet18
from .resnet50 import ResNet50
from .swin_tiny import SwinTiny
from .vit_base_16 import ViTBase16


VIDEO_BACKBONES = {
    "EfficientNetB0": EfficientNetB0,
    "DenseNet121": DenseNet121,
    "MobileNetV2": MobileNetV2,
    "SwinTiny": SwinTiny,
    "ResNet18": ResNet18,
    "ResNet50": ResNet50,
    "ConvNeXtTiny": ConvNeXtTiny,
    "ViTBase16": ViTBase16,
}


def build_video_backbone(name: str, classes_num: int, pretrained: bool = True):
    if name not in VIDEO_BACKBONES:
        raise ValueError(f"Unsupported video backbone '{name}'. Expected one of {sorted(VIDEO_BACKBONES)}.")
    return VIDEO_BACKBONES[name](classes_num=classes_num, pretrained=pretrained)


__all__ = ["VIDEO_BACKBONES", "build_video_backbone"]
