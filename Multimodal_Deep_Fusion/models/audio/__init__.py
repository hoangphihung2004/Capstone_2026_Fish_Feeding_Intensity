from .cnn14_mobilev2 import Cnn14MobileV2
from .cnn14_mobilev2_1p9m import Cnn14MobileV2_1P9M
from .efficientnet_b0 import EfficientNetB0
from .mobilenet_v1 import MobileNetV1
from .mobilenet_v2 import MobileNetV2
from .panns_cnn6 import PANNS_Cnn6
from .panns_cnn10 import PANNS_Cnn10
from .panns_cnn14 import PANNS_Cnn14
from .resnet22 import ResNet22


AUDIO_BACKBONES = {
    "PANNS_Cnn6": PANNS_Cnn6,
    "PANNS_Cnn10": PANNS_Cnn10,
    "PANNS_Cnn14": PANNS_Cnn14,
    "Cnn14MobileV2": Cnn14MobileV2,
    "Cnn14MobileV2_1P9M": Cnn14MobileV2_1P9M,
    "MobileNetV1": MobileNetV1,
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0,
    "ResNet22": ResNet22,
}


def build_audio_backbone(name: str, classes_num: int, pretrained: bool = False):
    if name not in AUDIO_BACKBONES:
        raise ValueError(f"Unsupported audio backbone '{name}'. Expected one of {sorted(AUDIO_BACKBONES)}.")
    backbone_cls = AUDIO_BACKBONES[name]
    if name == "EfficientNetB0":
        return backbone_cls(classes_num=classes_num, pretrained=pretrained)
    return backbone_cls(classes_num=classes_num)


__all__ = ["AUDIO_BACKBONES", "build_audio_backbone"]
