"""AMARU robustness: Gaussian noise at fixed SNR levels. PGD is not applied to audio."""

import sys
from pathlib import Path
from typing import Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import TrainConfig
from dataset.data_split import _sample_identity_key
from dataset.dataloader_melspectrogram import FishVoiceDataLoader
from features import AudioFrontend
from models import AMARU, AudioModel
from robustness.core import Batch, BaseAdapter, fold_splitter_config, prepare_model, run_cli


class AudioAdapter(BaseAdapter):
    name = "amaru"
    supports = ("audio_noise",)

    def __init__(self, fold, config, paths, checkpoint_path, device) -> None:
        self.device = device
        train_config = TrainConfig.from_json(paths.train_config_path)
        manager = FishVoiceDataLoader(
            sample_rate=train_config.audio_features.sample_rate,
            batch_size=config.batch_size,
            num_workers=-1,
            cache_audio=train_config.cache_audio,
            splitter_config=fold_splitter_config(train_config, fold, paths),
        )
        self.entries = manager.test_dict
        self.loader = manager.get_dataloader("test", shuffle=False, drop_last=False)
        model = AudioModel(frontend=AudioFrontend(config=train_config.audio_features), backbone=AMARU(classes_num=4))
        self.model = prepare_model(model, checkpoint_path, device)

    def batches(self, max_batches: Optional[int] = None) -> Iterator[Batch]:
        for index, batch in enumerate(self.loader):
            if max_batches is not None and index >= max_batches:
                break
            yield Batch(
                keys=[_sample_identity_key(path) for path in batch["audio_name"]],
                labels=batch["target"].argmax(dim=1).to(self.device),
                waveform=batch["waveform"].to(self.device),
                video=None,
            )

    def forward(self, waveform: Optional[torch.Tensor], video: Optional[torch.Tensor]) -> torch.Tensor:
        return self.model(waveform)["clipwise_output"]

    def split_records(self) -> set:
        return {(_sample_identity_key(entry[0]), int(entry[-1])) for entry in self.entries}


if __name__ == "__main__":
    run_cli(AudioAdapter, PROJECT_ROOT)
