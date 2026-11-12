"""Build per-fold test manifests that point into the raw dataset (no copying).

Reads <dataset>/splits/cv/fold_XX/test.csv (absolute paths) and writes
<output>/manifests/fold_XX_test.csv with paths relative to the dataset root,
in the same format as the original samples/manifest.csv:

    sample_index,video_path,audio_path,label,class_name,date,session,orig_sample_id

<output>/manifest.csv is a copy of the default fold (used by the dashboard).
At runtime the dataset's video/ and audio/ folders are mounted under samples/.

Usage:
    python3 data_preparation/build_manifests.py \
        --dataset_dir ~/Desktop/Fish_Feeding_Intensity_Dataset \
        --output_dir samples --default_fold 04
"""

import argparse
import csv
import os
import shutil

FIELDS = ["sample_index", "video_path", "audio_path", "label", "class_name", "date", "session", "orig_sample_id"]


def to_relative(path: str, dataset_dir: str) -> str:
    path = path.replace("\\", "/")
    root = dataset_dir.rstrip("/") + "/"
    if path.startswith(root):
        return path[len(root):]
    for marker in ("video/", "audio/"):
        idx = path.find("/" + marker)
        if idx >= 0:
            return path[idx + 1:]
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default="samples")
    parser.add_argument("--default_fold", default="04")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    cv_dir = os.path.join(dataset_dir, "splits", "cv")
    out_dir = os.path.join(args.output_dir, "manifests")
    os.makedirs(out_dir, exist_ok=True)

    for fold in sorted(os.listdir(cv_dir)):
        src = os.path.join(cv_dir, fold, "test.csv")
        if not os.path.isfile(src):
            continue
        with open(src, newline="") as f:
            rows = list(csv.DictReader(f))

        dst = os.path.join(out_dir, f"{fold}_test.csv")
        missing = 0
        with open(dst, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for i, row in enumerate(rows):
                video = to_relative(row["video_path"], dataset_dir)
                audio = to_relative(row["audio_path"], dataset_dir)
                if not (os.path.isfile(os.path.join(dataset_dir, video)) and os.path.isfile(os.path.join(dataset_dir, audio))):
                    missing += 1
                writer.writerow({
                    "sample_index": i,
                    "video_path": video,
                    "audio_path": audio,
                    "label": row["label"],
                    "class_name": row["class_name"],
                    "date": row["date"],
                    "session": row["session"],
                    "orig_sample_id": row["sample_id"],
                })
        print(f"{dst}: {len(rows)} samples, {missing} missing files")

    default = os.path.join(out_dir, f"fold_{args.default_fold}_test.csv")
    shutil.copyfile(default, os.path.join(args.output_dir, "manifest.csv"))
    print(f"manifest.csv -> fold_{args.default_fold}")


if __name__ == "__main__":
    main()
