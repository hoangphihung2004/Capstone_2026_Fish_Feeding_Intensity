import time
from typing import Tuple

import torch
import torch.nn as nn


class InferenceTimer:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def measure_latency_per_sample(self, waveform_shape: Tuple[int, ...], video_shape: Tuple[int, ...], warm_up_steps: int = 10, num_steps: int = 50) -> float:
        waveform = torch.randn(*waveform_shape).to(self.device)
        video_form = torch.randn(*video_shape).to(self.device)
        self.model.eval()
        is_cuda = self.device.type == "cuda"
        with torch.no_grad():
            for _ in range(warm_up_steps):
                _ = self.model(waveform=waveform, video_form=video_form)
                if is_cuda:
                    torch.cuda.synchronize()
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_steps):
                _ = self.model(waveform=waveform, video_form=video_form)
                if is_cuda:
                    torch.cuda.synchronize()
        return ((time.perf_counter() - start) / num_steps) * 1000.0
