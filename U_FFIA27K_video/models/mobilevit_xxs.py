import torch.nn as nn
import timm

class MobileViTXXS(nn.Module):
    """
    MobileViT-XXS feature extractor (classification head removed) using timm.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        
        self.model = timm.create_model("mobilevit_xxs", pretrained=pretrained, num_classes=0)
        self.embed_dim = self.model.num_features
        self.model_name = "mobilevit_xxs"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
