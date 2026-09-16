from __future__ import annotations

import copy
import csv
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
    logger.info("  - Early stopping:           %s", cfg.early_stopping)
    logger.info("  - Monitor metric:           %s", cfg.monitor)
    logger.info("  - Dataset path:             %s", cfg.dataset.dataset_path)
    logger.info("  - Split strategy:           %s", cfg.dataset.split_strategy)
    if cfg.evaluation_mode == "cross_validation":
        if cfg.dataset.fold_index is None:
            logger.info("  - CV fold selection:        all folds")
        else:
            logger.info("  - CV fold selection:        fold_%02d only", cfg.dataset.fold_index)
    logger.info("  - Seed:                     %s", cfg.seed)
    logger.info("  - Output directory:         %s", root)
    logger.info("==================================================")


def _experiment_root(cfg, project_dir: Path) -> Path:
    name = f"{cfg.audio.backbone}_{cfg.video.backbone}_{cfg.fusion.type}_{cfg.video_features.frame_policy}"
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
        frame_policy=cfg.video_features.frame_policy,
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


def selected_cv_fold_indices(cfg) -> List[int]:
    """Return all folds, or one validated fold selected in the JSON config."""
    requested_fold = cfg.dataset.fold_index
    num_folds = int(cfg.dataset.num_folds)
    if requested_fold is None:
        return list(range(num_folds))

    fold_index = int(requested_fold)
    if fold_index != requested_fold:
        raise ValueError(f"dataset.fold_index must be an integer fold id, got {requested_fold!r}.")
    if not 0 <= fold_index < num_folds:
        raise ValueError(f"dataset.fold_index={fold_index} is outside [0, {num_folds}).")
    return [fold_index]


def _load_fold_result(result_path: Path) -> Dict[str, float]:
    with result_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Fold result is empty: {result_path}")
    return {key: float(value) for key, value in rows[-1].items()}


def _cv_results_ready(root: Path, num_folds: int) -> bool:
    return all((root / f"fold_{fold_index:02d}" / "result.csv").is_file() for fold_index in range(num_folds))


def _aggregate_cv_results_if_ready(root: Path, num_folds: int) -> None:
    if not _cv_results_ready(root, num_folds):
        logger.info(
            "Cross-validation summary aggregation skipped because not all fold results exist yet. "
            "Run the remaining folds with the same seed, split_strategy, num_folds, and cv_val_ratio."
        )
        return
    fold_results = [_load_fold_result(root / f"fold_{fold_index:02d}" / "result.csv") for fold_index in range(num_folds)]
    save_cv_summary(fold_results, root)


def _run_cross_validation(cfg, root: Path) -> List[Dict[str, float]]:
    fold_indices = selected_cv_fold_indices(cfg)
    logger.info("Starting cross-validation for fold(s): %s.", ", ".join(f"{index:02d}" for index in fold_indices))
    fold_results: List[Dict[str, float]] = []
    for fold_index in fold_indices:
        # Match U_FFIA27K_audio: each fold receives a self-contained config.
        # FishDataSplitter is deterministic for this exact tuple of settings,
        # so running fold N alone produces the same train/val/test split as a
        # single process that runs all folds.
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.dataset.fold_index = fold_index
        logger.info("Starting cross-validation fold_%02d.", fold_index)
        splits = load_splits(fold_cfg.dataset, mode="cross_validation")
        loaders = create_dataloaders(
            splits=splits,
            dataset_cfg=fold_cfg.dataset,
            sample_rate=fold_cfg.audio_features.sample_rate,
            image_size=fold_cfg.video_features.image_size,
            frame_policy=fold_cfg.video_features.frame_policy,
            batch_size=fold_cfg.batch_size,
        )
        trainer = MultimodalTrainer(
            cfg=fold_cfg,
            loaders=loaders,
            output_dir=root / "cross_validation" / f"fold_{fold_index:02d}",
            splits=splits,
            run_name=f"cross_validation_fold_{fold_index:02d}",
        )
        fold_results.append(trainer.fit())
    cv_root = root / "cross_validation"
    if len(fold_indices) == cfg.dataset.num_folds:
        save_cv_summary(fold_results, cv_root)
    else:
        _aggregate_cv_results_if_ready(cv_root, cfg.dataset.num_folds)
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
