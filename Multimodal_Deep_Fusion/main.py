from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

from config import load_artifact_upload_config, load_train_config
from dataset import create_dataloaders, load_splits
from tasks import MultimodalTrainer, save_cv_summary
from utils import set_seed
from utils.artifact_upload import upload_artifact_if_enabled


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _log_run_configuration(cfg, root: Path) -> None:
    logger.info("==================================================")
    logger.info("Multimodal Deep Fusion configuration")
    logger.info("  - Evaluation mode:          %s", cfg.evaluation_mode)
    logger.info("  - Audio backbone:           %s", cfg.audio.backbone)
    logger.info("  - Video backbone:           %s", cfg.video.backbone)
    logger.info("  - Fusion head:              %s", cfg.fusion.type)
    logger.info("  - Epochs:                   %s", cfg.epochs)
    logger.info("  - Batch size:               %s", cfg.batch_size)
    logger.info("  - Optimizer:                %s", cfg.optimizer)
    logger.info("  - Learning rate:            %s", cfg.learning_rate)
    logger.info("  - Weight decay:             %s", cfg.weight_decay)
    logger.info("  - Early stopping:           %s", cfg.early_stopping)
    logger.info("  - Monitor metric:           %s", cfg.monitor)
    logger.info("  - Dataset path:             %s", cfg.dataset.dataset_path)
    logger.info("  - Split strategy:           %s", cfg.dataset.split_strategy)
    logger.info("  - Seed:                     %s", cfg.seed)
    logger.info("  - Output directory:         %s", root)
    logger.info("==================================================")


def _experiment_root(cfg, project_dir: Path) -> Path:
    name = f"{cfg.audio.backbone}_{cfg.video.backbone}_{cfg.fusion.type}"
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    return output_dir / name


def _run_holdout(cfg, root: Path) -> Dict[str, float]:
    logger.info("Starting holdout multimodal training.")
    splits = load_splits(cfg.dataset, mode="holdout")
    loaders = create_dataloaders(
        splits=splits,
        dataset_cfg=cfg.dataset,
        sample_rate=cfg.audio_features.sample_rate,
        image_size=cfg.video_features.image_size,
        batch_size=cfg.batch_size,
    )
    trainer = MultimodalTrainer(
        cfg=cfg,
        loaders=loaders,
        output_dir=root / "holdout",
        splits=splits,
        run_name="holdout",
    )
    return trainer.fit()


def _run_cross_validation(cfg, root: Path) -> List[Dict[str, float]]:
    logger.info("Starting cross-validation multimodal training with %d folds.", cfg.dataset.num_folds)
    fold_results: List[Dict[str, float]] = []
    for fold_index in range(cfg.dataset.num_folds):
        logger.info("Starting cross-validation fold_%02d.", fold_index)
        splits = load_splits(cfg.dataset, mode="cross_validation", fold_index=fold_index)
        loaders = create_dataloaders(
            splits=splits,
            dataset_cfg=cfg.dataset,
            sample_rate=cfg.audio_features.sample_rate,
            image_size=cfg.video_features.image_size,
            batch_size=cfg.batch_size,
        )
        trainer = MultimodalTrainer(
            cfg=cfg,
            loaders=loaders,
            output_dir=root / "cross_validation" / f"fold_{fold_index:02d}",
            splits=splits,
            run_name=f"cross_validation_fold_{fold_index:02d}",
        )
        fold_results.append(trainer.fit())
    save_cv_summary(fold_results, root / "cross_validation")
    return fold_results


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    cfg = load_train_config(project_dir / "config" / "train_config.json")
    set_seed(cfg.seed)
    root = _experiment_root(cfg, project_dir)
    root.mkdir(parents=True, exist_ok=True)
    _log_run_configuration(cfg, root)

    if cfg.evaluation_mode == "holdout":
        _run_holdout(cfg, root)
    elif cfg.evaluation_mode == "cross_validation":
        _run_cross_validation(cfg, root)

    upload_cfg = load_artifact_upload_config(project_dir / "config" / "artifact_upload_config.json")
    upload_artifact_if_enabled(upload_cfg, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Multimodal deep fusion run failed.")
        sys.exit(1)
