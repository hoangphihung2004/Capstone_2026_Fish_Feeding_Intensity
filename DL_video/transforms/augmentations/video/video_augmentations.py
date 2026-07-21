import logging
import numpy as np
import torch
import torch.nn as nn
from typing import List, Union, Tuple
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVF
from .base_augmentation import BaseVideoAug

logger = logging.getLogger(__name__)


class ToTensorVideo(BaseVideoAug):
    """
    Convert raw video array/tensor (uint8 [0, 255]) to float32 PyTorch Tensor in range [0.0, 1.0]
    using official torchvision.transforms.ToTensor.
    """
    def __init__(self) -> None:
        super().__init__()
        self.tv_to_tensor = TVT.ToTensor()
        logger.info("Initialized ToTensorVideo utilizing official torchvision.transforms.ToTensor.")

    def forward(self, video: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(video, np.ndarray):
            if video.ndim == 4 and video.shape[1] == 3:  # [F, C, H, W] uint8
                frames_t = [self.tv_to_tensor(video[i].transpose(1, 2, 0)) for i in range(video.shape[0])]
                return torch.stack(frames_t)
            elif video.ndim == 4 and video.shape[-1] == 3:  # [F, H, W, C] uint8
                frames_t = [self.tv_to_tensor(video[i]) for i in range(video.shape[0])]
                return torch.stack(frames_t)
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        if video_tensor.dtype == torch.uint8 or video_tensor.max() > 1.0:
            return video_tensor.to(torch.float32) / 255.0
        return video_tensor.to(torch.float32)


class NormalizeVideo(BaseVideoAug):
    """
    Video Normalization utilizing official torchvision.transforms.Normalize.
    Default mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5] maps [0.0, 1.0] to [-1.0, 1.0].
    """
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> None:
        super().__init__()
        self.mean = list(mean)
        self.std = list(std)
        self.tv_normalize = TVT.Normalize(mean=self.mean, std=self.std)
        logger.info(f"Initialized NormalizeVideo utilizing official torchvision.transforms.Normalize (mean={mean}, std={std}).")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() == 4 and video.shape[1] == 3:  # [F, C, H, W]
            return torch.stack([self.tv_normalize(frame) for frame in video])
        return self.tv_normalize(video)


class RandomFlipVideo(BaseVideoAug):
    """
    Random Horizontal Flip utilizing official torchvision.transforms.RandomHorizontalFlip & TVF.hflip.
    """
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.tv_flip = TVT.RandomHorizontalFlip(p=self.p)
        logger.info(f"Initialized RandomFlipVideo utilizing official torchvision.transforms.RandomHorizontalFlip (p={p}).")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return video

        if torch.rand(1).item() < self.p:
            if video.dim() == 4 and video.shape[1] == 3:  # [F, C, H, W]
                return torch.stack([TVF.hflip(frame) for frame in video])
            return TVF.hflip(video)
        return video


class CenterCropVideo(BaseVideoAug):
    """
    Center Crop utilizing official torchvision.transforms.CenterCrop.
    """
    def __init__(self, crop_size: Tuple[int, int] = (196, 196)) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.tv_crop = TVT.CenterCrop(size=self.crop_size)
        logger.info(f"Initialized CenterCropVideo utilizing official torchvision.transforms.CenterCrop (size={crop_size}).")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() == 4 and video.shape[1] == 3:  # [F, C, H, W]
            return torch.stack([self.tv_crop(frame) for frame in video])
        return self.tv_crop(video)


class ComposeVideo(BaseVideoAug):
    """
    Sequential composition container utilizing official torchvision.transforms.Compose.
    """
    def __init__(self, augmentations: List[BaseVideoAug]) -> None:
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)
        self.tv_compose = TVT.Compose([aug for aug in self.augmentations])
        aug_names = [aug.__class__.__name__ for aug in self.augmentations]
        logger.info(f"Initialized ComposeVideo utilizing official torchvision.transforms.Compose ({aug_names}).")

    def forward(self, video: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        out = video
        for aug in self.augmentations:
            out = aug(out)
        return out
