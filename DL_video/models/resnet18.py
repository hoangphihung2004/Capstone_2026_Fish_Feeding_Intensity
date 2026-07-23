import torch.nn as nn

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:
    from torchvision.models import resnet18
    ResNet18_Weights = None


class ResNet18(nn.Module):
    """
    ResNet18 image classifier initialized with ImageNet pretrained weights.
    """
    def __init__(self, classes_num: int = 4) -> None:
        super().__init__()

        if ResNet18_Weights is None:
            self.model = resnet18(pretrained=True)
        else:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model.fc = nn.Linear(self.model.fc.in_features, classes_num)
        self.model_name = "resnet18"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
