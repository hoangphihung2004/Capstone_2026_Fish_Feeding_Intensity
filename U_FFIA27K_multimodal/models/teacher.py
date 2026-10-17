"""Load the frozen multimodal teacher and check it matches the active split."""

import csv
import json
import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from dataset.data_split import _sample_identity_key
from .audio_backbone import LightweightAudioEncoder
from .multimodal_model import MultimodalModel

logger = logging.getLogger(__name__)

SPLIT_FIELDS = ("seed", "test_sample_per_class", "split_strategy", "evaluation_mode", "num_folds", "fold_index", "cv_val_ratio")
AUDIO_FIELDS = ("sample_rate", "window_size", "hop_size", "mel_bins", "fmin", "fmax")
THOP_BUFFER_SUFFIXES = ("total_ops", "total_params")
STAGES = (1, 2, 3, 4, 5)


class FeatureTeacher(nn.Module):
    """Frozen teacher that also returns the pooled feature of every stage, for the audio and the video branch.

    Stage 1 is the output of audio block 0 / video stage features[1] (the teacher fuses nothing there).
    Stages 2-5 are the audio / video outputs of fusion[0..3], i.e. after fusion. Each map is pooled right in the
    hook, exactly like the teacher pools its final features (audio: mean over frequency, then max + mean over time;
    video: spatial mean), so stage 5 equals the audio_320 / video_320 vectors feeding multimodal_head.
    The teacher architecture and checkpoint are untouched.
    """

    def __init__(self, teacher: MultimodalModel) -> None:
        super().__init__()
        self.teacher = teacher
        architecture = teacher.architecture
        self._audio = {}
        self._video = {}
        architecture.audio.blocks[0].register_forward_hook(lambda module, inputs, output: self._store_audio(1, output))
        architecture.video[1].register_forward_hook(lambda module, inputs, output: self._store_video(1, output))
        for index, fusion in enumerate(architecture.fusion):
            fusion.register_forward_hook(lambda module, inputs, output, stage=index + 2: self._store_fused(stage, output))

    def _store_audio(self, stage, audio) -> None:
        self._audio[stage] = LightweightAudioEncoder.pool(audio.detach())

    def _store_video(self, stage, video) -> None:
        self._video[stage] = video.detach().mean(dim=(2, 3))

    def _store_fused(self, stage, output) -> None:
        self._store_audio(stage, output[0])
        self._store_video(stage, output[1])

    def forward(self, waveform, video_form) -> dict:
        output = self.teacher(waveform=waveform, video_form=video_form)
        return {
            **output,
            "feature_stages_audio": torch.cat([self._audio[stage] for stage in STAGES], dim=1),
            "feature_stages_video": torch.cat([self._video[stage] for stage in STAGES], dim=1),
        }


def resolve_teacher_checkpoint(path: str) -> Path:
    if not path or not str(path).strip():
        raise ValueError("kd.teacher_checkpoint is empty. Set it to the teacher multimodal_best.pt of the selected fold.")
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: '{checkpoint}'")
    return checkpoint


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _teacher_train_config(checkpoint: Path) -> dict | None:
    for name in ("fold_train_config.json", "train_config.json"):
        path = checkpoint.parent / name
        if path.is_file():
            return _read_json(path)
    return None


def _mismatches(expected: dict, actual: dict, fields) -> List[str]:
    return [
        f"{field}: teacher={expected.get(field)!r}, current={actual.get(field)!r}"
        for field in fields
        if field in expected and expected.get(field) != actual.get(field)
    ]


def verify_teacher_config(checkpoint: Path, config) -> None:
    """Fail when the teacher was trained on another fold/split or audio frontend config."""
    teacher_config = _teacher_train_config(checkpoint)
    splitter_path = checkpoint.parent / "splitter_config.json"
    if splitter_path.is_file():
        teacher_splitter = _read_json(splitter_path)
    elif teacher_config is not None:
        teacher_splitter = teacher_config.get("dataset_splitter", {})
    else:
        teacher_splitter = None

    if teacher_splitter is None:
        logger.warning(
            "No splitter_config.json/train_config.json next to '%s'; cannot verify the teacher fold. "
            "Make sure dataset_splitter.fold_index matches the teacher checkpoint.",
            checkpoint,
        )
    else:
        errors = _mismatches(teacher_splitter, config.dataset_splitter.model_dump(), SPLIT_FIELDS)
        if errors:
            raise ValueError("Teacher checkpoint split does not match dataset_splitter:\n  " + "\n  ".join(errors))
        logger.info("Teacher split config matches dataset_splitter (fold_index=%s).", config.dataset_splitter.fold_index)

    if teacher_config is not None and "audio_features" in teacher_config:
        errors = _mismatches(teacher_config["audio_features"], config.audio_features.model_dump(), AUDIO_FIELDS)
        if errors:
            raise ValueError("Teacher audio_features do not match the current config:\n  " + "\n  ".join(errors))


def _read_split_keys(path: Path) -> set:
    with path.open("r", encoding="utf-8") as file:
        return {_sample_identity_key(row["audio_path"]) for row in csv.DictReader(file)}


def verify_teacher_splits(checkpoint: Path, splits: dict) -> None:
    """Compare the teacher's saved split files with the split generated for this run."""
    split_dir = checkpoint.parent / "splits"
    if not all((split_dir / f"{name}.csv").is_file() for name in splits):
        logger.warning("No saved splits next to the teacher checkpoint; skipping sample-level split verification.")
        return
    for name, entries in splits.items():
        teacher_keys = _read_split_keys(split_dir / f"{name}.csv")
        current_keys = {_sample_identity_key(entry[0]) for entry in entries}
        if teacher_keys != current_keys:
            raise ValueError(
                f"Teacher '{name}' split differs from the current split "
                f"(teacher={len(teacher_keys)}, current={len(current_keys)}, "
                f"only_teacher={len(teacher_keys - current_keys)}, only_current={len(current_keys - teacher_keys)})."
            )
    logger.info("Teacher saved splits match the current train/val/test splits sample by sample.")


def load_teacher(checkpoint: Path, config, device: torch.device) -> FeatureTeacher:
    """Build the multimodal teacher, load its weights, freeze it in eval mode and expose its 320-D features."""
    teacher = MultimodalModel(config.audio_features, pretrained_video=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state.get("model_state_dict", state)
    state_dict = {key: value for key, value in state_dict.items() if not key.endswith(THOP_BUFFER_SUFFIXES)}
    teacher.load_state_dict(state_dict, strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    logger.info(
        "Loaded frozen teacher from '%s' (epoch=%s, best metric=%s).",
        checkpoint,
        state.get("epoch"),
        state.get("ave_precision"),
    )
    return FeatureTeacher(teacher).to(device).eval()
