import torch.nn as nn

try:
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
except ImportError:
    from torchvision.models import mobilenet_v2
    MobileNet_V2_Weights = None


class MobileNetV2(nn.Module):
    """
    MobileNetV2 feature extractor (classification head removed) with optional ImageNet pretrained weights.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        if MobileNet_V2_Weights is None:
            self.model = mobilenet_v2(pretrained=pretrained)
        else:
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.model = mobilenet_v2(weights=weights)

        self.embed_dim = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Identity()
        self.model_name = "mobilenet_v2"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
