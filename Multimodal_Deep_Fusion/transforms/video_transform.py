import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class VideoTransform:
    """Multi-frame early-fusion transform from U_FFIA27K_video_Multi_channels."""

    def __init__(self, image_size: int = 224, split: str = "train") -> None:
        self.image_size = image_size
        self.split = split

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[-1] in (3, 6, 9, 12):
            image = image.transpose(2, 0, 1)

        channels = image.shape[0]
        if channels not in (3, 6, 9, 12):
            raise ValueError(f"Expected 3, 6, 9, or 12 image channels, got shape {tuple(image.shape)}")
        mean = torch.tensor([0.485, 0.456, 0.406] * (channels // 3)).view(channels, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225] * (channels // 3)).view(channels, 1, 1)

        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = TF.resize(image_tensor, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        if self.split == "train":
            if torch.rand(1) < 0.5:
                image_tensor = TF.hflip(image_tensor)
            if torch.rand(1) < 0.5:
                image_tensor = TF.rotate(image_tensor, float(torch.empty(1).uniform_(-20, 20).item()))
            if torch.rand(1) < 0.5:
                translate = [
                    int(torch.empty(1).uniform_(-0.2, 0.2).item() * self.image_size),
                    int(torch.empty(1).uniform_(-0.2, 0.2).item() * self.image_size),
                ]
                image_tensor = TF.affine(image_tensor, angle=0.0, translate=translate, scale=1.0, shear=0.0)
        return (image_tensor - mean) / std

    @staticmethod
    def get_transforms(image_size: int = 224):
        return {
            "train": VideoTransform(image_size=image_size, split="train"),
            "val": VideoTransform(image_size=image_size, split="val"),
            "test": VideoTransform(image_size=image_size, split="test"),
        }
