"""
Dataset preparation script for Jetson Orin Nano Edge Deployment.

This script selects a balanced subset of test samples (50 samples per class x 4 classes = 200 samples)
from the official test split (Fold 00) for cross-validation evaluation and web demonstration.
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
import pandas as pd

# Root directory of the repository (U_FFIA27K_multimodal_deployment)
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PrepareDataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare balanced test samples for Jetson evaluation")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.normpath(os.path.join(project_root, "..", "Dataset", "U_FFIA")),
        help="Root directory of the raw U_FFIA dataset (containing audio/ and video/)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=os.path.join(project_root, "checkpoint", "multimodal_model", "fold_00", "splits", "test.csv"),
        help="Path to the test.csv split file from Fold 00",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "samples"),
        help="Directory to save extracted sample files",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=50,
        help="Number of samples to pick per class (default: 50, total 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection",
    )
    return parser.parse_args()


def prepare_samples():
    args = parse_args()
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test split CSV not found: {args.test_csv}")
    if not os.path.exists(args.dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    logger.info(f"Reading test split: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    logger.info(f"Total samples in test split: {len(df)}")

    classes = ["none", "strong", "medium", "weak"]
    selected_subsets = []
    for cls in classes:
        cls_df = df[df["class_name"] == cls].sample(n=args.samples_per_class, random_state=args.seed)
        selected_subsets.append(cls_df)

    sample_df = pd.concat(selected_subsets, ignore_index=True)
    logger.info(f"Selected {len(sample_df)} balanced samples: {sample_df['class_name'].value_counts().to_dict()}")

    os.makedirs(args.output_dir, exist_ok=True)
    audio_out_dir = os.path.join(args.output_dir, "audio")
    video_out_dir = os.path.join(args.output_dir, "video")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(video_out_dir, exist_ok=True)

    records = []
    zip_path = os.path.join(args.output_dir, "test_samples_200.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, row in sample_df.iterrows():
            rel_video = row["video_path"].replace("/marimo/Fish_Feeding_Intensity_Dataset/", "").replace("/", os.sep)
            rel_audio = row["audio_path"].replace("/marimo/Fish_Feeding_Intensity_Dataset/", "").replace("/", os.sep)

            src_video = os.path.normpath(os.path.join(args.dataset_root, rel_video))
            src_audio = os.path.normpath(os.path.join(args.dataset_root, rel_audio))

            if not os.path.exists(src_video):
                logger.error(f"Missing video file: {src_video}")
                continue
            if not os.path.exists(src_audio):
                logger.error(f"Missing audio file: {src_audio}")
                continue

            # Prefix with class and index to ensure unique filenames without collision
            v_name = f"{row['class_name']}_{idx:03d}_{os.path.basename(src_video)}"
            a_name = f"{row['class_name']}_{idx:03d}_{os.path.basename(src_audio)}"

            dst_video = os.path.join(video_out_dir, v_name)
            dst_audio = os.path.join(audio_out_dir, a_name)

            shutil.copy2(src_video, dst_video)
            shutil.copy2(src_audio, dst_audio)

            archive_video = f"video/{v_name}"
            archive_audio = f"audio/{a_name}"
            zipf.write(dst_video, archive_video)
            zipf.write(dst_audio, archive_audio)

            records.append({
                "sample_id": idx,
                "video_path": archive_video,
                "audio_path": archive_audio,
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "date": row["date"],
                "session": row["session"],
                "orig_sample_id": row["sample_id"]
            })

        out_df = pd.DataFrame(records)
        csv_path = os.path.join(args.output_dir, "test_samples.csv")
        out_df.to_csv(csv_path, index=False)
        zipf.write(csv_path, "test_samples.csv")

    logger.info(f"Successfully prepared {len(records)} samples in: {args.output_dir}")
    logger.info(f"Manifest saved to: {csv_path}")
    logger.info(f"Zip package created: {zip_path} ({os.path.getsize(zip_path) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    prepare_samples()
