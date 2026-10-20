import torch.nn as nn
import timm

class MobileViTv2_075(nn.Module):
    """
    MobileViTv2-0.75 feature extractor (classification head removed) using timm.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        
        self.model = timm.create_model("mobilevitv2_075", pretrained=pretrained, num_classes=0)
        self.embed_dim = self.model.num_features
        self.model_name = "mobilevitv2_075"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
