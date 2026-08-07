from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


SPLIT_ORDER = ("train", "val", "test")


@dataclass
class SplitFeatures:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_name: str
    label_names: Dict[int, str]


def feature_run_dir(feature_root: str, mode: str, fold: int | None = None) -> Path:
    root = Path(feature_root)
    if mode == "holdout":
        return root / "holdout"
    if mode == "cross_validation":
        if fold is None:
            raise ValueError("fold must be provided for cross_validation feature loading.")
        return root / "cross_validation" / f"fold_{fold}"
    raise ValueError(f"Unsupported mode: {mode}")


def read_feature_table(run_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    features_path = run_dir / "features.npy"
    metadata_path = run_dir / "metadata.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.npy: {features_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv: {metadata_path}")

    features = np.load(features_path)
    metadata = pd.read_csv(metadata_path)
    required = {"feature_index", "split", "sample_key"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns {sorted(missing)}: {metadata_path}")

    metadata["feature_index"] = metadata["feature_index"].astype(int)
    if metadata["feature_index"].min() < 0 or metadata["feature_index"].max() >= len(features):
        raise ValueError(f"feature_index is outside features.npy bounds: {metadata_path}")
    return features, metadata


def label_column(metadata: pd.DataFrame) -> str:
    if "label_id" in metadata.columns:
        return "label_id"
    if "label" in metadata.columns:
        return "label"
    raise ValueError("Metadata must contain label_id or label.")


def label_names_from_metadata(metadata: pd.DataFrame) -> Dict[int, str]:
    if "label_id" not in metadata.columns or "label" not in metadata.columns:
        return {}
    pairs = metadata[["label_id", "label"]].drop_duplicates()
    names = {}
    for _, row in pairs.iterrows():
        try:
            names[int(row["label_id"])] = str(row["label"])
        except (TypeError, ValueError):
            pass
    return names


def split_single_modality(features: np.ndarray, metadata: pd.DataFrame, feature_name: str) -> SplitFeatures:
    y_col = label_column(metadata)
    label_names = label_names_from_metadata(metadata)
    outputs = {}
    for split in SPLIT_ORDER:
        split_meta = metadata[metadata["split"] == split].copy()
        split_meta = split_meta.sort_values(["sample_key", "feature_index"])
        if split_meta.empty:
            raise ValueError(f"Split '{split}' has no samples in metadata.")
        outputs[f"x_{split}"] = features[split_meta["feature_index"].to_numpy()]
        outputs[f"y_{split}"] = split_meta[y_col].to_numpy()

    return SplitFeatures(
        x_train=outputs["x_train"],
        y_train=outputs["y_train"],
        x_val=outputs["x_val"],
        y_val=outputs["y_val"],
        x_test=outputs["x_test"],
        y_test=outputs["y_test"],
        feature_name=feature_name,
        label_names=label_names,
    )


def merge_modalities(
    audio_features: np.ndarray,
    audio_metadata: pd.DataFrame,
    video_features: np.ndarray,
    video_metadata: pd.DataFrame,
) -> SplitFeatures:
    audio_y_col = label_column(audio_metadata)
    video_y_col = label_column(video_metadata)
    label_names = label_names_from_metadata(audio_metadata) or label_names_from_metadata(video_metadata)
    outputs = {}

    for split in SPLIT_ORDER:
        a = audio_metadata[audio_metadata["split"] == split].copy()
        v = video_metadata[video_metadata["split"] == split].copy()
        if a.empty or v.empty:
            raise ValueError(f"Split '{split}' is empty in one modality.")

        a["_join_label"] = a[audio_y_col].astype(str)
        v["_join_label"] = v[video_y_col].astype(str)
        a["_target_label"] = a[audio_y_col]
        v["_target_label"] = v[video_y_col]
        key_cols = ["sample_key", "split", "_join_label"]
        merged = a.merge(v, on=key_cols, suffixes=("_audio", "_video"), how="inner")
        if len(merged) != len(a) or len(merged) != len(v):
            missing_audio = set(zip(a["sample_key"], a["split"], a["_join_label"])) - set(
                zip(merged["sample_key"], merged["split"], merged["_join_label"])
            )
            missing_video = set(zip(v["sample_key"], v["split"], v["_join_label"])) - set(
                zip(merged["sample_key"], merged["split"], merged["_join_label"])
            )
            raise ValueError(
                f"Audio/video feature mismatch for split '{split}'. "
                f"audio_rows={len(a)}, video_rows={len(v)}, matched={len(merged)}, "
                f"missing_audio_examples={list(missing_audio)[:3]}, "
                f"missing_video_examples={list(missing_video)[:3]}"
            )

        merged = merged.sort_values(["sample_key", "feature_index_audio", "feature_index_video"])
        x_audio = audio_features[merged["feature_index_audio"].astype(int).to_numpy()]
        x_video = video_features[merged["feature_index_video"].astype(int).to_numpy()]
        outputs[f"x_{split}"] = np.concatenate([x_audio, x_video], axis=1)
        outputs[f"y_{split}"] = merged["_target_label_audio"].to_numpy()

    return SplitFeatures(
        x_train=outputs["x_train"],
        y_train=outputs["y_train"],
        x_val=outputs["x_val"],
        y_val=outputs["y_val"],
        x_test=outputs["x_test"],
        y_test=outputs["y_test"],
        feature_name="audio_video",
        label_names=label_names,
    )


def load_features_for_run(config: dict, mode: str, fold: int | None = None) -> SplitFeatures:
    feature_mode = config["feature_mode"]
    if feature_mode not in {"audio", "video", "audio_video"}:
        raise ValueError("feature_mode must be one of: audio, video, audio_video")

    if feature_mode == "audio":
        run_dir = feature_run_dir(config["audio_feature_root"], mode=mode, fold=fold)
        features, metadata = read_feature_table(run_dir)
        return split_single_modality(features, metadata, "audio")

    if feature_mode == "video":
        run_dir = feature_run_dir(config["video_feature_root"], mode=mode, fold=fold)
        features, metadata = read_feature_table(run_dir)
        return split_single_modality(features, metadata, "video")

    audio_run_dir = feature_run_dir(config["audio_feature_root"], mode=mode, fold=fold)
    video_run_dir = feature_run_dir(config["video_feature_root"], mode=mode, fold=fold)
    audio_features, audio_metadata = read_feature_table(audio_run_dir)
    video_features, video_metadata = read_feature_table(video_run_dir)
    return merge_modalities(audio_features, audio_metadata, video_features, video_metadata)
