from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, get_token

from config import ArtifactUploadConfig, TrainConfig

logger = logging.getLogger(__name__)


def build_artifact_name(cfg: TrainConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"MultimodalDL_{cfg.audio.backbone}_{cfg.video.backbone}_"
        f"{cfg.fusion.type}_{cfg.evaluation_mode}_{timestamp}.zip"
    )


def zip_source_tree(source_dir: str | Path, output_zip: str | Path) -> Path:
    source_path = Path(source_dir).resolve()
    output_zip = Path(output_zip).resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    base_name = str(output_zip.with_suffix(""))
    zip_path = shutil.make_archive(base_name, "zip", root_dir=source_path)
    return Path(zip_path)


def upload_artifact_if_enabled(
    upload_cfg: ArtifactUploadConfig,
    train_cfg: TrainConfig,
    source_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    if not upload_cfg.enabled:
        logger.info("Artifact upload is disabled.")
        return None

    token = get_token()
    if not token:
        raise EnvironmentError(
            "Hugging Face authentication token was not found. Run 'hf auth login' in this environment first."
        )

    artifact_name = upload_cfg.path_in_repo or build_artifact_name(train_cfg)
    source_path = Path(source_dir or upload_cfg.source_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / artifact_name
        zip_source_tree(source_path, zip_path)
        if upload_cfg.create_repo:
            create_repo(
                repo_id=upload_cfg.repo_id,
                repo_type=upload_cfg.repo_type,
                token=token,
                exist_ok=True,
            )
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo=artifact_name,
            repo_id=upload_cfg.repo_id,
            repo_type=upload_cfg.repo_type,
        )
        logger.info("Uploaded artifact to Hugging Face: %s/%s", upload_cfg.repo_id, artifact_name)
        return zip_path
