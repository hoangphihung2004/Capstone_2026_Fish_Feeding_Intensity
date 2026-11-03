"""Upload robustness results to Hugging Face. This file is byte-identical on all three branches.

Each upload is a zip whose name says which branch, model, fold and run it holds, for example
`audio_amaru/audio_amaru_fold00_full_20261103-101500.zip`. A new timestamp is used every time,
so earlier uploads are never overwritten.
"""

import csv
import json
import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODALITY = {"amaru": "audio", "efficientnet_b0": "video", "multimodal": "multimodal"}
UPLOAD_LOG_FIELDS = ["uploaded_at", "repo_id", "path_in_repo", "local_zip"]


class UploadConfig(BaseModel):
    enabled: bool = False
    repo_id: str = ""
    repo_type: str = "dataset"
    create_repo: bool = True
    private: bool = False


def load_upload_config(path: Path) -> UploadConfig:
    if not path.is_file():
        logger.info("No upload config at %s; results stay local.", path)
        return UploadConfig()
    with path.open("r", encoding="utf-8") as file:
        return UploadConfig(**json.load(file))


def _token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    from huggingface_hub import get_token
    return get_token()


def check_ready(config: UploadConfig) -> None:
    """Fail before any evaluation starts if the upload cannot work."""
    if not config.enabled:
        return
    if not config.repo_id:
        raise ValueError("repo_id must be set in the upload config when enabled is true.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for upload. Install it with requirements.txt.") from exc
    token = _token()
    if not token:
        raise EnvironmentError(
            "Set HF_TOKEN or run `huggingface-cli login` before uploading, or pass --no-upload."
        )
    api = HfApi(token=token)
    user = api.whoami()["name"]
    if config.create_repo:
        api.create_repo(config.repo_id, repo_type=config.repo_type, private=config.private, exist_ok=True)
    logger.info("Results will be uploaded to %s (%s) as %s.", config.repo_id, config.repo_type, user)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _zip(sources: list, arc_root: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
                if path.is_file():
                    archive.write(path, path.relative_to(arc_root).as_posix())
    return zip_path


def _upload(config: UploadConfig, zip_path: Path, path_in_repo: str, log_dir: Path) -> str:
    from huggingface_hub import HfApi

    logger.info("Uploading %s to %s/%s", zip_path.name, config.repo_id, path_in_repo)
    HfApi(token=_token()).upload_file(
        path_or_fileobj=str(zip_path), path_in_repo=path_in_repo, repo_id=config.repo_id,
        repo_type=config.repo_type, commit_message=f"Add {path_in_repo}",
    )
    log_path = log_dir / "uploads.csv"
    new_file = not log_path.is_file()
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=UPLOAD_LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "uploaded_at": datetime.now().isoformat(timespec="seconds"), "repo_id": config.repo_id,
            "path_in_repo": path_in_repo, "local_zip": str(zip_path),
        })
    logger.info("Uploaded %s", path_in_repo)
    return path_in_repo


def upload_fold(config: UploadConfig, output_dir: Path, model: str, fold: int, kind: str) -> Optional[str]:
    """Zip `<output_dir>/<model>/fold_XX` (archive included) and upload it.

    The zip keeps the `<model>/fold_XX/...` layout, so extracting every zip of one branch into a
    single folder gives a directory that `aggregate.py` reads directly.
    """
    if not config.enabled:
        return None
    fold_dir = output_dir / model / f"fold_{fold:02d}"
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"No results to upload: {fold_dir}")
    prefix = f"{MODALITY.get(model, model)}_{model}"
    name = f"{prefix}_fold{fold:02d}_{kind}_{_stamp()}.zip"
    zip_path = _zip([fold_dir], output_dir, Path(f"{output_dir}_uploads") / name)
    return _upload(config, zip_path, f"{prefix}/{name}", output_dir)


def upload_summary(config: UploadConfig, summary_dir: Path) -> Optional[str]:
    if not config.enabled:
        return None
    name = f"summary_audio_video_multimodal_{_stamp()}.zip"
    files = [path for path in summary_dir.iterdir() if path.is_file() and path.name != "uploads.csv"]
    zip_path = _zip(files, summary_dir, Path(f"{summary_dir}_uploads") / name)
    return _upload(config, zip_path, f"summary/{name}", summary_dir)
