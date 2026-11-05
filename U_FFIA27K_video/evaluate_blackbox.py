"""Black-box transfer attack on EfficientNet-B0 (first-last 6-channel video).

The adversarial video is generated with PGD on the source model chosen in config/blackbox_config.json
(ConvNeXt-Tiny, Swin-Tiny or ResNet18); EfficientNet-B0 only classifies it.
"""

import sys
from pathlib import Path
from typing import Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import TrainConfig
from dataset import FishVideoDataLoader
from dataset.data_split import _sample_identity_key
from models import EfficientNetB0, VideoModel
from robustness.blackbox import TargetAdapter, run_cli
from robustness.core import Batch, fold_splitter_config, prepare_model

FRAME_POLICY = "1_49_channels"
TARGET_CHANNELS = 6


def _key(entry) -> str:
    return _sample_identity_key(entry[1])


class VideoTarget(TargetAdapter):
    name = "efficientnet_b0"

    def __init__(self, fold, config, paths, checkpoint_path, device) -> None:
        self.device = device
        train_config = TrainConfig.from_json(paths.train_config_path)
        if train_config.model.backbone != "EfficientNetB0" or train_config.video_features.frame_policy != FRAME_POLICY:
            raise ValueError(
                f"The target is EfficientNetB0 with {FRAME_POLICY}, but {paths.train_config_path} says "
                f"{train_config.model.backbone} with {train_config.video_features.frame_policy}."
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
        # Sample-key order on both black-box branches, so every batch holds the same samples.
        manager.val_dict = sorted(manager.val_dict, key=_key)
        manager.test_dict = sorted(manager.test_dict, key=_key)
        self.entries = {"val": manager.val_dict, "test": manager.test_dict}
        self.loaders = {split: manager.get_dataloader(split, shuffle=False, drop_last=False) for split in config.splits}
        model = VideoModel(backbone=EfficientNetB0(classes_num=4, pretrained=False), target_channels=TARGET_CHANNELS)
        self.model = prepare_model(model, checkpoint_path, device)

    def batches(self, split: str, max_batches: Optional[int] = None) -> Iterator[Batch]:
        for index, batch in enumerate(self.loaders[split]):
            if max_batches is not None and index >= max_batches:
                break
            yield Batch(
                keys=[_sample_identity_key(path) for path in batch["video_name"]],
                labels=batch["target"].argmax(dim=1).to(self.device),
                waveform=None,
                video=batch["video_form"].to(self.device),
            )

    def forward(self, waveform: Optional[torch.Tensor], video: torch.Tensor) -> torch.Tensor:
        return self.model(video)["clipwise_output"]

    def split_records(self, split: str) -> set:
        return {(_key(entry), int(entry[-1])) for entry in self.entries[split]}


if __name__ == "__main__":
    run_cli(VideoTarget, PROJECT_ROOT)
