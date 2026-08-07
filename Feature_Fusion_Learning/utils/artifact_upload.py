import os
import re
import zipfile
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi


def safe_filename_part(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def build_artifact_name(fine_config, timestamp=None):
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = safe_filename_part(fine_config.get("dataset_tag", "dataset"))
    feature_mode = safe_filename_part(fine_config["feature_mode"])

    if fine_config["feature_mode"] == "audio_video":
        parts = [
            "FeatureFineTuning",
            dataset_tag,
            safe_filename_part(fine_config.get("audio_model_name", "audio")),
            safe_filename_part(fine_config.get("video_model_name", "video")),
            feature_mode,
            timestamp,
        ]
    elif fine_config["feature_mode"] == "audio":
        parts = [
            "FeatureFineTuning",
            dataset_tag,
            safe_filename_part(fine_config.get("audio_model_name", "audio")),
            feature_mode,
            timestamp,
        ]
    else:
        parts = [
            "FeatureFineTuning",
            dataset_tag,
            safe_filename_part(fine_config.get("video_model_name", "video")),
            feature_mode,
            timestamp,
        ]

    return "_".join(parts) + ".zip"


def zip_directory(source_dir, zip_path):
    source_dir = Path(source_dir).resolve()
    zip_path = Path(zip_path).resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.resolve() == zip_path:
                continue
            zip_file.write(path, path.relative_to(source_dir.parent))

    return zip_path


def upload_artifact_if_enabled(upload_config, fine_config, default_source_dir=None):
    if not upload_config.get("enabled", False):
        print("Artifact upload disabled.")
        return None

    source_dir = Path(upload_config.get("source_dir") or default_source_dir or ".").resolve()
    artifact_name = build_artifact_name(fine_config)
    zip_path = Path(upload_config.get("zip_path") or source_dir.parent / artifact_name)

    zip_path = zip_directory(source_dir, zip_path)
    repo_id = upload_config.get("repo_id", "").strip()
    if not repo_id:
        print(f"Saved artifact locally: {zip_path}")
        print("Skipped Hugging Face upload because repo_id is empty.")
        return zip_path

    repo_type = upload_config.get("repo_type", "dataset")
    path_in_repo = upload_config.get("path_in_repo") or artifact_name
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    if upload_config.get("create_repo", True):
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded artifact: {repo_id}/{path_in_repo}")
    return zip_path
