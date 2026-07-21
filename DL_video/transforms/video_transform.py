import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ClipToTensor:
    """
    Convert a video clip to float tensor [F, C, H, W].
    Accepts np.ndarray or torch.Tensor in [F, C, H, W] or [F, H, W, C].
    """
    def __call__(self, video: np.ndarray) -> torch.Tensor:
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        if video.ndim != 4:
            raise ValueError(f"Expected video with 4 dimensions, got shape {tuple(video.shape)}")

        if video.shape[1] == 3:  # [F, C, H, W]
            clip = video
        elif video.shape[-1] == 3:  # [F, H, W, C]
            clip = video.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected channel dimension with size 3, got shape {tuple(video.shape)}")

        if not torch.is_floating_point(clip):
            return clip.float().div(255.0)

        return clip.float()


class ClipResizeIfNeeded:
    """Resize the whole clip only when it is not already at the target size."""
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.shape[-2:] == (self.image_size, self.image_size):
            return clip

        return TF.resize(
            clip,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )


class ClipNormalize:
    """Normalize a float video clip [F, C, H, W]."""
    def __init__(self, mean, std) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=clip.device, dtype=clip.dtype)
        std = self.std.to(device=clip.device, dtype=clip.dtype)
        return (clip - mean) / std


class VideoTransform:
    """
    Tensor-based video transform pipeline applied once to the whole clip.

    Input:  np.ndarray [F, H, W, C] or [F, C, H, W] uint8, or torch.Tensor
    Output: torch.Tensor [F, C, H, W] float32
    """
    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform

    def __call__(self, video: np.ndarray) -> torch.Tensor:
        return self.transform(video)

    @staticmethod
    def get_transforms(image_size: int = 224):
        """
        Return transform dictionary for train / val / test splits.
        Additional video augmentations can be added here.
        """
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        data_transform = {
            "train": VideoTransform(transforms.Compose([
                ClipToTensor(),
                ClipResizeIfNeeded(image_size),
                # Add clip-level augmentations here, before ClipNormalize.
                # transforms.ColorJitter(brightness=0.15),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(20),
                # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                ClipNormalize(mean, std),
            ])),
            "val": VideoTransform(transforms.Compose([
                ClipToTensor(),
                ClipResizeIfNeeded(image_size),
                ClipNormalize(mean, std),
            ])),
            "test": VideoTransform(transforms.Compose([
                ClipToTensor(),
                ClipResizeIfNeeded(image_size),
                ClipNormalize(mean, std),
            ])),
        }

        return data_transform
