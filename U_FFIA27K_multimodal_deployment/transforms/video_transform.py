import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class VideoTransform:
    """
    Transform for first-last RGB frame concatenation.

    Input is expected as uint8 CHW or HWC with 6 channels. Geometric
    augmentations are applied once to the whole 6-channel tensor so both frames
    receive exactly the same spatial transform.
    """

    def __init__(self, image_size: int = 224, split: str = "train") -> None:
        self.image_size = image_size
        self.split = split

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape={image.shape}")
        if image.shape[-1] == 6:
            image = image.transpose(2, 0, 1)
        if image.shape[0] != 6:
            raise ValueError(f"Expected first-last 6-channel video tensor, got shape={image.shape}")

        img = torch.from_numpy(image).float() / 255.0
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

        if self.split == "train":
            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
            if torch.rand(1) < 0.5:
                angle = float(torch.empty(1).uniform_(-20, 20).item())
                img = TF.rotate(img, angle)
            if torch.rand(1) < 0.5:
                tx = int(torch.empty(1).uniform_(-0.2, 0.2).item() * self.image_size)
                ty = int(torch.empty(1).uniform_(-0.2, 0.2).item() * self.image_size)
                img = TF.affine(img, angle=0.0, translate=[tx, ty], scale=1.0, shear=0.0)

        mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(6, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(6, 1, 1)
        return (img - mean) / std

    @staticmethod
    def get_transforms(image_size: int = 224) -> dict:
        return {
            "train": VideoTransform(image_size=image_size, split="train"),
            "val": VideoTransform(image_size=image_size, split="val"),
            "test": VideoTransform(image_size=image_size, split="test"),
        }
