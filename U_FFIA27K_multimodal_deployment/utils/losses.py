import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipCELoss(nn.Module):
    def components(self, output_dict: dict, target_dict: dict) -> dict:
        target = target_dict["target"]
        if target.dim() > 1:
            target = target.argmax(dim=1)
        return {
            "multimodal": F.cross_entropy(output_dict["clipwise_output"], target.long()),
        }

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return sum(self.components(output_dict, target_dict).values())
