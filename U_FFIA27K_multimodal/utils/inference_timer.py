import time
from typing import Optional, Tuple

import torch
import torch.nn as nn


class InferenceTimer:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def measure_latency_per_sample(self, waveform_shape: Optional[Tuple[int, ...]] = None, video_shape: Optional[Tuple[int, ...]] = None, warm_up_steps: int = 10, num_steps: int = 50) -> float:
        inputs = {}
        if waveform_shape is not None:
            inputs["waveform"] = torch.randn(*waveform_shape).to(self.device)
        if video_shape is not None:
            inputs["video_form"] = torch.randn(*video_shape).to(self.device)
        self.model.eval()
        is_cuda = self.device.type == "cuda"
        with torch.no_grad():
            for _ in range(warm_up_steps):
                _ = self.model(**inputs)
                if is_cuda:
                    torch.cuda.synchronize()
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_steps):
                _ = self.model(**inputs)
                if is_cuda:
                    torch.cuda.synchronize()
        return ((time.perf_counter() - start) / num_steps) * 1000.0
