"""
Dataset preparation script for Jetson Orin Nano Edge Deployment.

Module: data_preparation
Responsibilities:
- Extract full test set (5,415 samples) from Fold 00 test split (or optional balanced subset)
- Safely COPY (shutil.copy2) media from raw dataset to samples/ preserving original dataset intact
- Generate clean manifest CSV for evaluation and web serving
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
import pandas as pd

# Root directory of the repository (U_FFIA27K_multimodal_deployment)
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataPreparation")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare test samples for Jetson Orin Nano evaluation")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.normpath(os.path.join(project_root, "..", "Dataset", "U_FFIA")),
        help="Root directory of raw U_FFIA dataset (containing audio/ and video/)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=os.path.join(project_root, "checkpoint", "multimodal_model", "fold_00", "splits", "test.csv"),
        help="Path to test.csv split file from Fold 00",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "samples"),
        help="Directory to save extracted sample files",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Clean output directory before copying",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on number of samples (default: None, full 5,415 samples)",
    )
    return parser.parse_args()


def prepare_samples():
    args = parse_args()
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test split CSV not found: {args.test_csv}")
    if not os.path.exists(args.dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    logger.info(f"Reading Fold 00 test split: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    total_test = len(df)
    logger.info(f"Total samples in Fold 00 test split: {total_test}")
    logger.info(f"Class breakdown in test split: {df['class_name'].value_counts().to_dict()}")

    if args.max_samples and args.max_samples < total_test:
        logger.info(f"Limiting to first {args.max_samples} samples as requested...")
        df = df.iloc[:args.max_samples]

    if args.clean and os.path.exists(args.output_dir):
        logger.info(f"Cleaning existing contents in: {args.output_dir}...")
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                logger.warning(f"Could not remove {item_path}: {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    audio_out_dir = os.path.join(args.output_dir, "audio")
    video_out_dir = os.path.join(args.output_dir, "video")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(video_out_dir, exist_ok=True)

    # 1. Pre-collect and pre-create all unique subdirectories
    logger.info("Pre-creating output subdirectories...")
    all_tasks = []
    unique_dirs = set()

    for idx, row in df.iterrows():
        rel_video = row["video_path"].replace("/marimo/Fish_Feeding_Intensity_Dataset/", "").replace("/", os.sep)
        rel_audio = row["audio_path"].replace("/marimo/Fish_Feeding_Intensity_Dataset/", "").replace("/", os.sep)

        src_video = os.path.normpath(os.path.join(args.dataset_root, rel_video))
        src_audio = os.path.normpath(os.path.join(args.dataset_root, rel_audio))

        v_sub = os.path.join(video_out_dir, row["date"], row["session"], row["class_name"])
        a_sub = os.path.join(audio_out_dir, row["date"], row["session"], row["class_name"])
        unique_dirs.add(v_sub)
        unique_dirs.add(a_sub)

        dst_video = os.path.join(v_sub, os.path.basename(src_video))
        dst_audio = os.path.join(a_sub, os.path.basename(src_audio))

        rel_dst_video = os.path.relpath(dst_video, args.output_dir).replace(os.sep, "/")
        rel_dst_audio = os.path.relpath(dst_audio, args.output_dir).replace(os.sep, "/")

        all_tasks.append((idx, row, src_video, src_audio, dst_video, dst_audio, rel_dst_video, rel_dst_audio))

    for d in unique_dirs:
        os.makedirs(d, exist_ok=True)

    logger.info(f"Created {len(unique_dirs)} subdirectories. Starting parallel safe COPY (shutil.copy2)...")

    # 2. Worker function for fast safe copy (pure copy, never move/delete)
    import ctypes
    is_windows = os.name == "nt"
    CopyFileW = ctypes.windll.kernel32.CopyFileW if is_windows else None

    def fast_copy(src, dst):
        if is_windows:
            success = CopyFileW(str(src), str(dst), False)
            if not success:
                raise OSError(f"CopyFileW failed from {src} to {dst}")
        else:
            shutil.copyfile(src, dst)

    def process_sample(task):
        idx, row, src_video, src_audio, dst_video, dst_audio, rel_dst_video, rel_dst_audio = task
        if not os.path.exists(src_video) or not os.path.exists(src_audio):
            return None

        # Check if already copied with identical size (fast resume support)
        v_exists = os.path.exists(dst_video) and os.path.getsize(dst_video) == os.path.getsize(src_video)
        a_exists = os.path.exists(dst_audio) and os.path.getsize(dst_audio) == os.path.getsize(src_audio)

        if not v_exists:
            fast_copy(src_video, dst_video)
        if not a_exists:
            fast_copy(src_audio, dst_audio)

        size = os.path.getsize(dst_video) + os.path.getsize(dst_audio)
        return {
            "sample_index": idx,
            "video_path": rel_dst_video,
            "audio_path": rel_dst_audio,
            "label": int(row["label"]),
            "class_name": row["class_name"],
            "date": row["date"],
            "session": row["session"],
            "orig_sample_id": row["sample_id"],
            "size": size,
        }

    records = []
    total_bytes = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_sample, t): t for t in all_tasks}
        count = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                total_bytes += res.pop("size")
                records.append(res)
                count += 1
                if count % 500 == 0 or count == len(all_tasks):
                    logger.info(f"Progress: Copied {count}/{len(all_tasks)} samples ({total_bytes / (1024**3):.2f} GB)...")

    # Sort records back by original sample_index
    records.sort(key=lambda r: r["sample_index"])

    # 3. Save manifest CSV
    manifest_df = pd.DataFrame(records)
    manifest_csv = os.path.join(args.output_dir, "manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)

    logger.info("==========================================================")
    logger.info("Fold 00 Full Test Set Preparation Completed Successfully!")
    logger.info(f"Total samples processed:  {len(records)} / {len(df)}")
    logger.info(f"Total size on disk:       {total_bytes / (1024**3):.2f} GB")
    logger.info(f"Samples directory:        {args.output_dir}")
    logger.info(f"Manifest saved to:        {manifest_csv}")
    logger.info(f"Class distribution:       {manifest_df['class_name'].value_counts().to_dict()}")
    logger.info("Original dataset integrity: 100% UNTOUCHED (safe read-only copy)")
    logger.info("==========================================================")


if __name__ == "__main__":
    prepare_samples()



