from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ArtifactUploadConfig(BaseModel):
    enabled: bool = True
    source_dir: str = "/marimo/Capstone_2026_Fish_Feeding_Intensity"
    repo_id: str = "hoangphihung442004/Results_Multimodal_Custom_Models"
    repo_type: str = "dataset"
    path_in_repo: str = ""
    create_repo: bool = True
    include_source_code: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ArtifactUploadConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


def load_artifact_upload_config(path: str | Path = "config/artifact_upload_config.json") -> ArtifactUploadConfig:
    return ArtifactUploadConfig.from_json(path)
