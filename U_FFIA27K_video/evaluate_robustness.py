"""White-box robustness of the video models: first-last 6-channel video under PGD.

`--model` picks the attacked (and evaluated) video model; default efficientnet_b0:
    python evaluate_robustness.py --model efficientnet_b0
    python evaluate_robustness.py --model convnext_tiny
    python evaluate_robustness.py --model swin_tiny
Each model reads its own checkpoint paths (config/robustness_paths*.json, overridable with --paths) and writes
to <output_dir>/<model>/fold_XX. Every other option is the same as before (see robustness/core.py run_cli).
"""

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import TrainConfig
from dataset import FishVideoDataLoader
from dataset.data_split import _sample_identity_key
from models import ConvNeXtTiny, EfficientNetB0, SwinTiny, VideoModel
from robustness.core import Batch, BaseAdapter, fold_splitter_config, prepare_model, run_cli

FRAME_POLICY = "1_49_channels"
TARGET_CHANNELS = 6


@dataclass(frozen=True)
class VideoBackbone:
    build: Callable[..., torch.nn.Module]
    paths: str


BACKBONES: Dict[str, VideoBackbone] = {
    "efficientnet_b0": VideoBackbone(EfficientNetB0, "config/robustness_paths.json"),
    "convnext_tiny": VideoBackbone(ConvNeXtTiny, "config/robustness_paths_convnext_tiny.json"),
    "swin_tiny": VideoBackbone(SwinTiny, "config/robustness_paths_swin_tiny.json"),
}


class VideoAdapter(BaseAdapter):
    supports = ("video_pgd",)

    def __init__(self, fold, config, paths, checkpoint_path, device, backbone: str = "efficientnet_b0") -> None:
        self.name = backbone
        self.device = device
        train_config = TrainConfig.from_json(paths.train_config_path)
        if train_config.video_features.frame_policy != FRAME_POLICY:
            raise ValueError(
                f"Robustness evaluates {FRAME_POLICY} video, but {paths.train_config_path} says "
                f"{train_config.video_features.frame_policy}."
            )
        manager = FishVideoDataLoader(
            batch_size=config.batch_size,
            dataloader_workers=train_config.dataloader_workers,
            prefetch_factor=train_config.prefetch_factor,
            cache_mode=train_config.cache_mode,
            image_size=train_config.video_features.image_size,
            frame_policy=FRAME_POLICY,
            splitter_config=fold_splitter_config(train_config, fold, paths),
        )
        self.entries = manager.test_dict
        self.loader = manager.get_dataloader("test", shuffle=False, drop_last=False)
        model = VideoModel(backbone=BACKBONES[backbone].build(classes_num=4, pretrained=False), target_channels=TARGET_CHANNELS)
        self.model = prepare_model(model, checkpoint_path, device)

    def batches(self, max_batches: Optional[int] = None) -> Iterator[Batch]:
        for index, batch in enumerate(self.loader):
            if max_batches is not None and index >= max_batches:
                break
            yield Batch(
                keys=[_sample_identity_key(path) for path in batch["video_name"]],
                labels=batch["target"].argmax(dim=1).to(self.device),
                waveform=None,
                video=batch["video_form"].to(self.device),
            )

    def forward(self, waveform: Optional[torch.Tensor], video: Optional[torch.Tensor]) -> torch.Tensor:
        return self.model(video)["clipwise_output"]

    def split_records(self) -> set:
        return {(_sample_identity_key(entry[1]), int(entry[-1])) for entry in self.entries}


def main() -> None:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--model", choices=sorted(BACKBONES), default="efficientnet_b0",
                          help="Video model to attack (white-box) and evaluate.")
    args, rest = selector.parse_known_args()
    if not any(arg == "--paths" or arg.startswith("--paths=") for arg in rest):
        rest += ["--paths", str(PROJECT_ROOT / BACKBONES[args.model].paths)]
    sys.argv = [sys.argv[0], *rest]
    run_cli(partial(VideoAdapter, backbone=args.model), PROJECT_ROOT)


if __name__ == "__main__":
    main()
