import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


class VideoTransform:
    """
    Wrapper duy nhất: nhận standard torchvision.transforms.Compose
    và áp dụng cho từng frame của video.

    Input:  np.ndarray [F, H, W, C] hoặc [F, C, H, W] uint8
    Output: torch.Tensor [F, C, H, W] float32
    """
    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform

    def __call__(self, video: np.ndarray) -> torch.Tensor:
        if isinstance(video, np.ndarray):
            if video.ndim == 4 and video.shape[1] == 3:  # [F, C, H, W]
                frames = [Image.fromarray(video[i].transpose(1, 2, 0)) for i in range(video.shape[0])]
            else:  # [F, H, W, C]
                frames = [Image.fromarray(video[i]) for i in range(video.shape[0])]
        else:
            frames = [Image.fromarray(video[i].permute(1, 2, 0).numpy()) for i in range(video.shape[0])]

        transformed = [self.transform(frame) for frame in frames]
        return torch.stack(transformed)

    @staticmethod
    def get_transforms(image_size: int = 224):
        """
        Trả về dictionary transform cho train / val / test.
        Muốn thêm augmentation chỉ cần sửa ở đây.
        """
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        img_size = (image_size, image_size)

        data_transform = {
            "train": VideoTransform(transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])),
            "val": VideoTransform(transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])),
            "test": VideoTransform(transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])),
        }

        return data_transform
