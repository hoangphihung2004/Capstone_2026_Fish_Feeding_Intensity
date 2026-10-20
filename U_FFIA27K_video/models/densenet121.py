import torch.nn as nn

try:
    from torchvision.models import DenseNet121_Weights, densenet121
except ImportError:
    from torchvision.models import densenet121
    DenseNet121_Weights = None


class DenseNet121(nn.Module):
    """
    DenseNet121 feature extractor (classification head removed) with optional ImageNet pretrained weights.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        if DenseNet121_Weights is None:
            self.model = densenet121(pretrained=pretrained)
        else:
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            self.model = densenet121(weights=weights)

        self.embed_dim = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model_name = "densenet121"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
