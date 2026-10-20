import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class FramesToTensor:
    """Convert uint8 frames [T, C, H, W] (np.ndarray or tensor) to float32 in [0, 1]."""
    def __call__(self, frames):
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(np.ascontiguousarray(frames))

        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Expected frames with shape [T, 3, H, W], got {tuple(frames.shape)}")

        return frames.float().div(255.0)


class VideoTransform:
    """
    First+last frame transform pipeline.

    Every random augmentation samples its parameters once per call and applies them to
    all frames of the clip, so the first and last frames get identical flips/rotations/shifts.

    Input:  np.ndarray [T, C, H, W] uint8
    Output: torch.Tensor [T, C, H, W] float32 normalized with ImageNet statistics.
    """
    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        return self.transform(frames)

    @staticmethod
    def get_transforms(image_size: int = 224):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = [
            FramesToTensor(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ColorJitter(brightness=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.Normalize(mean, std),
        ]

        eval_transform = [
            FramesToTensor(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.Normalize(mean, std),
        ]

        return {
            "train": VideoTransform(transforms.Compose(train_transform)),
            "val": VideoTransform(transforms.Compose(eval_transform)),
            "test": VideoTransform(transforms.Compose(eval_transform)),
        }
