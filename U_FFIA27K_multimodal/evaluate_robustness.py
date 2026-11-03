"""Multimodal robustness: audio noise with clean video, and video PGD with clean audio."""

import sys
from pathlib import Path
from typing import Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import TrainConfig
from dataset import FishMultimodalDataLoader
from dataset.data_split import _sample_identity_key
from models import MultimodalModel
from robustness.core import Batch, BaseAdapter, fold_splitter_config, prepare_model, run_cli


class MultimodalAdapter(BaseAdapter):
    name = "multimodal"
    supports = ("audio_noise", "video_pgd")

    def __init__(self, fold, config, paths, checkpoint_path, device) -> None:
        self.device = device
        train_config = TrainConfig.from_json(paths.train_config_path)
        manager = FishMultimodalDataLoader(
            sample_rate=train_config.audio_features.sample_rate,
            batch_size=config.batch_size,
            dataloader_workers=train_config.dataloader_workers,
            prefetch_factor=train_config.prefetch_factor,
            cache_audio=train_config.cache_audio,
            cache_video_mode=train_config.cache_video_mode,
            image_size=train_config.video_features.image_size,
            splitter_config=fold_splitter_config(train_config, fold, paths),
        )
        self.entries = manager.test_dict
        missing = sum(1 for entry in self.entries if not entry[1])
        if missing:
            raise ValueError(f"{missing} test samples have no paired video; refusing to evaluate on blank frames.")
        self.loader = manager.get_dataloader("test", shuffle=False, drop_last=False)
        # Same constructor arguments as main.build_model, so the weights load into the trained architecture.
        model = MultimodalModel(
            train_config.audio_features,
            pretrained_video=False,
            ablation_mode=train_config.ablation.mode,
            dropout_rate=train_config.model.dropout_rate,
        )
        self.model = prepare_model(model, checkpoint_path, device)

    def batches(self, max_batches: Optional[int] = None) -> Iterator[Batch]:
        for index, batch in enumerate(self.loader):
            if max_batches is not None and index >= max_batches:
                break
            yield Batch(
                keys=[_sample_identity_key(path) for path in batch["audio_name"]],
                labels=batch["target"].argmax(dim=1).to(self.device),
                waveform=batch["waveform"].to(self.device),
                video=batch["video_form"].to(self.device),
            )

    def forward(self, waveform: Optional[torch.Tensor], video: Optional[torch.Tensor]) -> torch.Tensor:
        return self.model(waveform=waveform, video_form=video)["clipwise_output"]

    def split_records(self) -> set:
        return {(_sample_identity_key(entry[0]), int(entry[-1])) for entry in self.entries}


if __name__ == "__main__":
    run_cli(MultimodalAdapter, PROJECT_ROOT)
