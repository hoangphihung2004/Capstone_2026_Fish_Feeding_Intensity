import torch.nn as nn

try:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
except ImportError:
    from torchvision.models import efficientnet_b0
    EfficientNet_B0_Weights = None


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 image classifier initialized with ImageNet pretrained weights.
    """
    def __init__(self, classes_num: int = 4) -> None:
        super().__init__()

        if EfficientNet_B0_Weights is None:
            self.model = efficientnet_b0(pretrained=True)
        else:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, classes_num)
        self.model_name = "efficientnet_b0"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
