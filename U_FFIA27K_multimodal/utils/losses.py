import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipCELoss(nn.Module):
    def __init__(self, audio_weight=1.0, video_weight=1.0, multimodal_weight=1.0):
        super().__init__()
        self.weights = {
            "audio": float(audio_weight),
            "video": float(video_weight),
            "multimodal": float(multimodal_weight),
        }

    def components(self, output_dict: dict, target_dict: dict) -> dict:
        target = target_dict["target"]
        if target.dim() > 1:
            target = target.argmax(dim=1)
        return {
            name: self.weights[name] * F.cross_entropy(output_dict[key], target.long())
            for name, key in (("audio", "audio_output"), ("video", "video_output"),
                              ("multimodal", "clipwise_output"))
        }

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return sum(self.components(output_dict, target_dict).values())
