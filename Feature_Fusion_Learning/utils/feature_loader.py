from dataclasses import dataclass
from pathlib import Path
import re
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


def concat_split_features(feature_sets, feature_name: str) -> SplitFeatures:
    if not feature_sets:
        raise ValueError("feature_sets must not be empty.")

    reference = feature_sets[0]
    for feature_set in feature_sets[1:]:
        for split in SPLIT_ORDER:
            y_reference = getattr(reference, f"y_{split}")
            y_current = getattr(feature_set, f"y_{split}")
            if not np.array_equal(y_reference, y_current):
                raise ValueError(f"Cannot concatenate feature sets because y_{split} does not match.")

    return SplitFeatures(
        x_train=np.concatenate([feature_set.x_train for feature_set in feature_sets], axis=1),
        y_train=reference.y_train,
        x_val=np.concatenate([feature_set.x_val for feature_set in feature_sets], axis=1),
        y_val=reference.y_val,
        x_test=np.concatenate([feature_set.x_test for feature_set in feature_sets], axis=1),
        y_test=reference.y_test,
        feature_name=feature_name,
        label_names=reference.label_names,
    )


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


def fusion_sample_key(value) -> str:
    key = str(value).strip().replace("\\", "/")
    return re.sub(r"(?i)(^|[/_-])(audio|video)(?=[/_-]|$)", r"\1modality", key)


def validate_unique_fusion_keys(metadata: pd.DataFrame, split: str, modality: str) -> None:
    duplicate_mask = metadata.duplicated(["_fusion_key", "split", "_join_label"], keep=False)
    if not duplicate_mask.any():
        return

    examples = metadata.loc[
        duplicate_mask,
        ["sample_key", "_fusion_key", "split", "_join_label"],
    ].head(5)
    raise ValueError(
        f"Duplicate normalized sample keys in {modality} metadata for split '{split}'. "
        f"Examples: {examples.to_dict(orient='records')}"
    )


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


def split_aligned_modalities(
    audio_features: np.ndarray,
    audio_metadata: pd.DataFrame,
    video_features: np.ndarray,
    video_metadata: pd.DataFrame,
):
    audio_y_col = label_column(audio_metadata)
    video_y_col = label_column(video_metadata)
    label_names = label_names_from_metadata(audio_metadata) or label_names_from_metadata(video_metadata)
    audio_outputs = {}
    video_outputs = {}
    label_outputs = {}

    for split in SPLIT_ORDER:
        a = audio_metadata[audio_metadata["split"] == split].copy()
        v = video_metadata[video_metadata["split"] == split].copy()
        if a.empty or v.empty:
            raise ValueError(f"Split '{split}' is empty in one modality.")

        a["_join_label"] = a[audio_y_col].astype(str)
        v["_join_label"] = v[video_y_col].astype(str)
        a["_fusion_key"] = a["sample_key"].apply(fusion_sample_key)
        v["_fusion_key"] = v["sample_key"].apply(fusion_sample_key)
        a["_target_label"] = a[audio_y_col]
        v["_target_label"] = v[video_y_col]
        validate_unique_fusion_keys(a, split, "audio")
        validate_unique_fusion_keys(v, split, "video")

        key_cols = ["_fusion_key", "split", "_join_label"]
        merged = a.merge(v, on=key_cols, suffixes=("_audio", "_video"), how="inner")
        if len(merged) != len(a) or len(merged) != len(v):
            missing_audio = set(zip(a["_fusion_key"], a["split"], a["_join_label"])) - set(
                zip(merged["_fusion_key"], merged["split"], merged["_join_label"])
            )
            missing_video = set(zip(v["_fusion_key"], v["split"], v["_join_label"])) - set(
                zip(merged["_fusion_key"], merged["split"], merged["_join_label"])
            )
            raise ValueError(
                f"Audio/video feature mismatch for split '{split}'. "
                f"audio_rows={len(a)}, video_rows={len(v)}, matched={len(merged)}, "
                f"missing_audio_examples={list(missing_audio)[:3]}, "
                f"missing_video_examples={list(missing_video)[:3]}"
            )

        merged = merged.sort_values(["_fusion_key", "feature_index_audio", "feature_index_video"])
        audio_outputs[f"x_{split}"] = audio_features[merged["feature_index_audio"].astype(int).to_numpy()]
        video_outputs[f"x_{split}"] = video_features[merged["feature_index_video"].astype(int).to_numpy()]
        label_outputs[f"y_{split}"] = merged["_target_label_audio"].to_numpy()

    audio_set = SplitFeatures(
        x_train=audio_outputs["x_train"],
        y_train=label_outputs["y_train"],
        x_val=audio_outputs["x_val"],
        y_val=label_outputs["y_val"],
        x_test=audio_outputs["x_test"],
        y_test=label_outputs["y_test"],
        feature_name="audio",
        label_names=label_names,
    )
    video_set = SplitFeatures(
        x_train=video_outputs["x_train"],
        y_train=label_outputs["y_train"],
        x_val=video_outputs["x_val"],
        y_val=label_outputs["y_val"],
        x_test=video_outputs["x_test"],
        y_test=label_outputs["y_test"],
        feature_name="video",
        label_names=label_names,
    )
    return audio_set, video_set


def merge_modalities(
    audio_features: np.ndarray,
    audio_metadata: pd.DataFrame,
    video_features: np.ndarray,
    video_metadata: pd.DataFrame,
) -> SplitFeatures:
    audio_set, video_set = split_aligned_modalities(audio_features, audio_metadata, video_features, video_metadata)
    return concat_split_features([audio_set, video_set], "audio_video")


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


def load_modality_features_for_run(config: dict, mode: str, fold: int | None = None):
    feature_mode = config["feature_mode"]
    if feature_mode == "audio":
        run_dir = feature_run_dir(config["audio_feature_root"], mode=mode, fold=fold)
        features, metadata = read_feature_table(run_dir)
        return {"audio": split_single_modality(features, metadata, "audio")}

    if feature_mode == "video":
        run_dir = feature_run_dir(config["video_feature_root"], mode=mode, fold=fold)
        features, metadata = read_feature_table(run_dir)
        return {"video": split_single_modality(features, metadata, "video")}

    if feature_mode != "audio_video":
        raise ValueError("feature_mode must be one of: audio, video, audio_video")

    audio_run_dir = feature_run_dir(config["audio_feature_root"], mode=mode, fold=fold)
    video_run_dir = feature_run_dir(config["video_feature_root"], mode=mode, fold=fold)
    audio_features, audio_metadata = read_feature_table(audio_run_dir)
    video_features, video_metadata = read_feature_table(video_run_dir)
    audio_set, video_set = split_aligned_modalities(audio_features, audio_metadata, video_features, video_metadata)
    return {"audio": audio_set, "video": video_set}
