import torch.nn as nn

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:
    from torchvision.models import resnet18
    ResNet18_Weights = None


class ResNet18(nn.Module):
    """
    ResNet18 feature extractor (classification head removed) with optional ImageNet pretrained weights.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        if ResNet18_Weights is None:
            self.model = resnet18(pretrained=pretrained)
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)

        self.embed_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model_name = "resnet18"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
