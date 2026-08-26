import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class VideoTransform:
    def __init__(self, image_size=224, split='train'):
        self.image_size = image_size
        self.split = split

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[-1] == 3 or image.shape[-1] == 6:
            image = image.transpose(2, 0, 1)
            
        C = image.shape[0]
        mean_3 = [0.485, 0.456, 0.406]
        std_3 = [0.229, 0.224, 0.225]
        
        if C == 6:
            mean = torch.tensor(mean_3 + mean_3).view(6, 1, 1)
            std = torch.tensor(std_3 + std_3).view(6, 1, 1)
        else:
            mean = torch.tensor(mean_3).view(3, 1, 1)
            std = torch.tensor(std_3).view(3, 1, 1)

        img = torch.from_numpy(image).float() / 255.0
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        
        if self.split == 'train':
            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
            if torch.rand(1) < 0.5:
                angle = float(torch.empty(1).uniform_(-20, 20).item())
                img = TF.rotate(img, angle)
            if torch.rand(1) < 0.5:
                tx = int(torch.empty(1).uniform_(-0.2, 0.2).item() * self.image_size)
                ty = int(torch.empty(1).uniform_(-0.2, 0.2).item() * self.image_size)
                img = TF.affine(img, angle=0.0, translate=[tx, ty], scale=1.0, shear=0.0)

        img = (img - mean) / std
        return img

    @staticmethod
    def get_transforms(image_size: int = 224):
        return {
            \'train\': VideoTransform(image_size=image_size, split=\'train\'),
            \'val\': VideoTransform(image_size=image_size, split=\'val\'),
            \'test\': VideoTransform(image_size=image_size, split=\'test\'),
        }
