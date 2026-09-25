import json
from pathlib import Path

from pydantic import BaseModel, Field


class ArtifactUploadConfig(BaseModel):
    enabled: bool = Field(default=False)
    repo_id: str = Field(default="")
    repo_type: str = Field(default="dataset")
    create_repo: bool = Field(default=True)
    source_dir: str = Field(default="/marimo/Capstone_2026_Fish_Feeding_Intensity")
    zip_path: str = Field(default="/marimo/Capstone_2026_Fish_Feeding_Intensity.zip")
    path_in_repo: str = Field(default="Capstone_2026_Fish_Feeding_Intensity.zip")

    @classmethod
    def from_json(cls, path: str = "config/artifact_upload_config.json") -> "ArtifactUploadConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(**json.load(f))
