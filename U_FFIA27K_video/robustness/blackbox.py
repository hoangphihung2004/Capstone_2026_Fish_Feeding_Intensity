"""Black-box (transfer) PGD on video. Byte-identical on the video and multimodal black-box branches.

The perturbation is generated with PGD on a separately trained *source* video model (ConvNeXt-Tiny,
Swin-Tiny or ResNet18, chosen in config/blackbox_config.json) and then fed to the *target* model,
which never takes part in generating it:
- video branch:      target = EfficientNet-B0 (video only);
- multimodal branch: target = multimodal AF4 with the clean audio of the same sample.
Both branches build the same source model, read the same frames and seed the random start per
sample, so the two targets receive the same adversarial video. A digest of the clean input
and a per-level digest of the perturbation are saved, so this can be checked afterwards.

Scoring follows the clean-first rule: a sample the target misclassifies on the clean input keeps its
clean input (it counts as wrong); only clean-correct samples receive the adversarial video, so the
accuracy under attack can never exceed the clean accuracy.
"""

import argparse
import csv
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator

from .core import (
    METRIC_FIELDS, NUM_CLASSES, Batch, VideoPGDConfig, _archive_previous, _normalization, _read_csv,
    _uniform_delta, compute_metrics, load_json_model, prepare_model, random_linf_noise,
)
from .hub import MODALITY, _stamp, _upload, _zip, check_ready, load_upload_config
from .source_models import SOURCE_BUILDERS, SourceVideoModel

logger = logging.getLogger(__name__)

SourceName = Literal["ConvNeXtTiny", "SwinTiny", "ResNet18"]
SplitName = Literal["val", "test"]
SUMMARY_COLUMN = {"val": "Accuracy Val", "test": "Accuracy Test"}
SCENARIO = "video_pgd_transfer"
RESULT_FIELDS = [
    "model", "source_model", "fold", "split", "scenario", "level_unit", "level", "n_samples",
    *METRIC_FIELDS, "per_class", "confusion_matrix",
    "target_clean_correct", "source_accuracy", "source_loss_first", "source_loss_last",
    "input_digest", "perturbation_linf_max", "perturbation_mean_abs", "perturbation_digest",
    "partial", "seconds", "config_hash", "checkpoint", "source_checkpoint",
]
ARCHIVE_ITEMS = ("robustness.csv", "predictions", "run_info.json", "run.log")


class BlackBoxConfig(BaseModel):
    folds: List[int]
    splits: List[SplitName]
    source: SourceName
    batch_size: int = Field(gt=0)
    seed: int
    video_pgd: VideoPGDConfig

    @field_validator("splits")
    @classmethod
    def unique_splits(cls, value: List[str]) -> List[str]:
        if not value or len(set(value)) != len(value):
            raise ValueError("splits must be a non-empty list without repeats, e.g. ['val', 'test'].")
        return value


class BlackBoxPaths(BaseModel):
    """`targets` maps fold -> target checkpoint; `sources` maps source name -> template with {fold}."""

    train_config_path: str = "config/train_config.json"
    dataset_path: str = ""
    targets: Dict[str, str]
    sources: Dict[str, str]
    output_dir: str = "outputs/blackbox"

    def target_for(self, fold: int) -> Path:
        if str(fold) not in self.targets:
            raise ValueError(f"No target checkpoint configured for fold {fold}.")
        return _existing(Path(self.targets[str(fold)]), f"Target checkpoint for fold {fold}")

    def source_for(self, name: str, fold: int) -> Path:
        if name not in self.sources:
            raise ValueError(f"No checkpoint template for source '{name}'. Configured: {sorted(self.sources)}")
        return _existing(Path(self.sources[name].format(fold=fold)), f"{name} checkpoint for fold {fold}")


def _existing(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def config_hash(config: BlackBoxConfig) -> str:
    canonical = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def run_name(target: str, source: str) -> str:
    return f"{target}_from_{source}"


def transfer_pgd(
    source_forward: Callable[[torch.Tensor], torch.Tensor],
    video: torch.Tensor,
    labels: torch.Tensor,
    keys: Sequence[str],
    epsilon: float,
    alpha_ratio: float,
    steps: int,
    random_start: bool,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Untargeted L-infinity PGD (Madry et al.) on the source model only, in [0, 1] pixel space.

    Every sample is attacked and the final iterate is kept: for a transfer attack the useful signal
    is a perturbation that raises the source loss as far as the budget allows; stopping at the first
    source misclassification would leave a weak perturbation. Takes and returns normalized video,
    plus the pixel-space perturbation and the mean source loss before each step.
    """
    alpha = epsilon * alpha_ratio
    mean, std = _normalization(video)
    pixels = (video.detach() * std + mean).clamp(0.0, 1.0)
    adversarial = pixels.clone()
    if random_start:
        adversarial = (pixels + _uniform_delta(video, keys, epsilon, "transfer", seed)).clamp(0.0, 1.0)
    losses: List[float] = []
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        loss = F.cross_entropy(source_forward((adversarial - mean) / std), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        losses.append(loss.item())
        adversarial = adversarial.detach() + alpha * gradient.sign()
        adversarial = (pixels + (adversarial - pixels).clamp(-epsilon, epsilon)).clamp(0.0, 1.0)
    adversarial = adversarial.detach()
    return (adversarial - mean) / std, adversarial - pixels, losses


class TargetAdapter:
    """Implemented once per branch: the target model and its val/test loaders in sample-key order."""

    name: str
    device: torch.device

    def batches(self, split: str, max_batches: Optional[int] = None) -> Iterator[Batch]:
        raise NotImplementedError

    def forward(self, waveform: Optional[torch.Tensor], video: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def split_records(self, split: str) -> set:
        """Return {(sample_key, label)} for the loaded split."""
        raise NotImplementedError


def load_source(name: str, checkpoint_path: Path, device: torch.device) -> SourceVideoModel:
    return prepare_model(SourceVideoModel(name), checkpoint_path, device)


def verify_split(records: set, checkpoint_path: Path, split: str) -> None:
    """Fail if the regenerated split differs from the split saved with the weights."""
    from dataset.data_split import _sample_identity_key

    reference = checkpoint_path.parent / "splits" / f"{split}.csv"
    if not reference.is_file():
        raise FileNotFoundError(f"No saved {split} split next to the checkpoint: {reference}")
    with reference.open("r", encoding="utf-8", newline="") as file:
        expected = {
            (_sample_identity_key(row.get("video_path") or row.get("audio_path")), int(row["label"]))
            for row in csv.DictReader(file)
        }
    if expected != records:
        raise ValueError(
            f"{split} split differs from {reference}: {len(expected - records)} missing, "
            f"{len(records - expected)} unexpected."
        )
    logger.info("%s split matches %s (%d samples).", split, reference, len(records))


def _summary_accuracy(checkpoint_path: Path, split: str) -> Optional[float]:
    path = checkpoint_path.parent / "summary.csv"
    if not path.is_file():
        return None
    rows = _read_csv(path)
    value = rows[-1].get(SUMMARY_COLUMN[split]) if rows else None
    return float(value) if value not in (None, "") else None


def _digest(keys: Sequence[str], mean_abs: np.ndarray) -> str:
    order = np.argsort(np.asarray(keys))
    rounded = np.round(mean_abs[order].astype(np.float64), 8)
    payload = "\n".join(f"{keys[i]}:{value:.8f}" for i, value in zip(order, rounded))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clean_pass(
    target: TargetAdapter, source: torch.nn.Module, split: str, max_batches: Optional[int]
) -> Dict[str, Any]:
    keys: List[str] = []
    labels, target_probs, source_probs, input_means = [], [], [], []
    with torch.no_grad():
        for batch in target.batches(split, max_batches):
            input_means.append(batch.video.flatten(1).double().mean(dim=1).cpu().numpy())
            target_probs.append(target.forward(batch.waveform, batch.video).softmax(dim=1).cpu().numpy())
            source_probs.append(source(batch.video).softmax(dim=1).cpu().numpy())
            labels.append(batch.labels.cpu().numpy())
            keys.extend(batch.keys)
    if not keys:
        raise ValueError(f"The {split} loader produced no samples.")
    result = {
        "keys": keys, "labels": np.concatenate(labels),
        "target_probs": np.concatenate(target_probs), "source_probs": np.concatenate(source_probs),
    }
    result["input_digest"] = _digest(keys, np.concatenate(input_means))
    result["correct"] = dict(zip(keys, (result["target_probs"].argmax(axis=1) == result["labels"]).tolist()))
    return result


def _attack_pass(
    target: TargetAdapter,
    source: torch.nn.Module,
    split: str,
    epsilon: float,
    config: BlackBoxConfig,
    clean_correct: Dict[str, bool],
    max_batches: Optional[int],
    steps: Optional[int] = None,
) -> Dict[str, Any]:
    pgd = config.video_pgd
    keys: List[str] = []
    labels, target_probs, source_probs, linf, mean_abs = [], [], [], [], []
    loss_first, loss_last, weight = 0.0, 0.0, 0
    for batch in target.batches(split, max_batches):
        adversarial, delta, losses = transfer_pgd(
            source, batch.video, batch.labels, batch.keys, epsilon,
            pgd.alpha_ratio, steps or pgd.steps, pgd.random_start, config.seed,
        )
        attacked = torch.tensor([clean_correct[key] for key in batch.keys], device=batch.video.device)
        target_video = torch.where(attacked.view(-1, 1, 1, 1), adversarial, batch.video)
        with torch.no_grad():
            target_probs.append(target.forward(batch.waveform, target_video).softmax(dim=1).cpu().numpy())
            source_probs.append(source(adversarial).softmax(dim=1).cpu().numpy())
        flat = delta.flatten(1).double()
        linf.append(flat.abs().amax(dim=1).cpu().numpy())
        mean_abs.append(flat.abs().mean(dim=1).cpu().numpy())
        labels.append(batch.labels.cpu().numpy())
        keys.extend(batch.keys)
        size = len(batch.keys)
        loss_first += losses[0] * size
        loss_last += losses[-1] * size
        weight += size
    return {
        "keys": keys, "labels": np.concatenate(labels),
        "target_probs": np.concatenate(target_probs), "source_probs": np.concatenate(source_probs),
        "linf": np.concatenate(linf), "mean_abs": np.concatenate(mean_abs),
        "loss_first": loss_first / weight, "loss_last": loss_last / weight,
    }


def evaluate_fold(
    target: TargetAdapter,
    source: torch.nn.Module,
    config: BlackBoxConfig,
    fold: int,
    target_checkpoint: Path,
    source_checkpoint: Path,
    output_dir: Path,
    max_batches: Optional[int] = None,
    resume: bool = False,
) -> Path:
    """Evaluate clean + every epsilon on every configured split; save after each level."""
    fold_dir = output_dir / run_name(target.name, config.source) / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    partial = int(max_batches is not None)
    current_hash = config_hash(config)
    results_path = fold_dir / "robustness.csv"

    kept_rows: List[Dict[str, str]] = []
    if resume:
        for row in _read_csv(results_path):
            finished = (fold_dir / "predictions" / f"{row['split']}_{row['scenario']}_{row['level']}.npz").is_file()
            complete = all(row.get(name) not in (None, "") for name in RESULT_FIELDS)
            if finished and complete and row["config_hash"] == current_hash and row["partial"] == str(partial):
                kept_rows.append(row)
    else:
        _archive_previous(fold_dir, ARCHIVE_ITEMS)
    (fold_dir / "predictions").mkdir(exist_ok=True)

    log_handler = logging.FileHandler(fold_dir / "run.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(log_handler)
    run_info = {
        "model": target.name, "source_model": config.source, "fold": fold, "splits": config.splits,
        "checkpoint": str(target_checkpoint), "source_checkpoint": str(source_checkpoint),
        "config_hash": current_hash, "config": config.model_dump(), "partial": bool(partial), "resumed": resume,
        "started_at": datetime.now().isoformat(timespec="seconds"), "device": str(target.device),
        "torch_version": torch.__version__,
    }
    (fold_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    def row_for(split: str, scenario: str, unit: str, level: Any, result: Dict[str, Any], seconds: float) -> Dict[str, Any]:
        metrics = compute_metrics(result["labels"], result["target_probs"])
        attacked = scenario == SCENARIO
        return {
            "model": target.name, "source_model": config.source, "fold": fold, "split": split,
            "scenario": scenario, "level_unit": unit, "level": level, "n_samples": len(result["keys"]),
            **{name: metrics[name] for name in METRIC_FIELDS},
            "per_class": json.dumps(metrics["per_class"]), "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            "target_clean_correct": int(sum(result["clean_correct"])),
            "source_accuracy": float((result["source_probs"].argmax(axis=1) == result["labels"]).mean()),
            "source_loss_first": result["loss_first"] if attacked else 0.0,
            "source_loss_last": result["loss_last"] if attacked else 0.0,
            "input_digest": input_digest[split],
            "perturbation_linf_max": float(result["linf"].max()) if attacked else 0.0,
            "perturbation_mean_abs": float(result["mean_abs"].mean()) if attacked else 0.0,
            "perturbation_digest": _digest(result["keys"], result["mean_abs"]) if attacked else "none",
            "partial": partial, "seconds": round(seconds, 3), "config_hash": current_hash,
            "checkpoint": str(target_checkpoint), "source_checkpoint": str(source_checkpoint),
        }

    def save_predictions(split: str, scenario: str, level: Any, result: Dict[str, Any]) -> None:
        extra = {"linf": result["linf"], "mean_abs": result["mean_abs"]} if scenario == SCENARIO else {}
        np.savez_compressed(
            fold_dir / "predictions" / f"{split}_{scenario}_{level}.npz",
            keys=np.array(result["keys"]), labels=result["labels"],
            target_probabilities=result["target_probs"].astype(np.float32),
            source_probabilities=result["source_probs"].astype(np.float32), **extra,
        )

    input_digest: Dict[str, str] = {}

    try:
        done = {(row["split"], row["scenario"], row["level"]) for row in kept_rows}
        with results_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(kept_rows)
            file.flush()
            for split in config.splits:
                # The clean pass is always recomputed: it decides which samples are attacked.
                started = time.perf_counter()
                clean = _clean_pass(target, source, split, max_batches)
                clean["clean_correct"] = [clean["correct"][key] for key in clean["keys"]]
                input_digest[split] = clean["input_digest"]
                if (split, "clean", "clean") not in done:
                    save_predictions(split, "clean", "clean", clean)
                    row = row_for(split, "clean", "none", "clean", clean, time.perf_counter() - started)
                    writer.writerow(row)
                    file.flush()
                    logger.info(
                        "[%s fold %d %s] clean: target accuracy=%.4f | source %s accuracy=%.4f | input digest %s",
                        run_name(target.name, config.source), fold, split, row["accuracy"], config.source,
                        row["source_accuracy"], row["input_digest"],
                    )
                for epsilon in config.video_pgd.epsilon_levels():
                    if (split, SCENARIO, str(epsilon)) in done:
                        logger.info("[fold %d %s] epsilon %s already saved; skipping.", fold, split, epsilon)
                        continue
                    started = time.perf_counter()
                    result = _attack_pass(target, source, split, epsilon, config, clean["correct"], max_batches)
                    result["clean_correct"] = [clean["correct"][key] for key in result["keys"]]
                    save_predictions(split, SCENARIO, epsilon, result)
                    row = row_for(split, SCENARIO, "epsilon", epsilon, result, time.perf_counter() - started)
                    writer.writerow(row)
                    file.flush()
                    logger.info(
                        "[%s fold %d %s] epsilon %s: target accuracy=%.4f | source accuracy=%.4f | "
                        "source loss %.3f -> %.3f | linf %.3g (%.1fs)",
                        run_name(target.name, config.source), fold, split, epsilon, row["accuracy"],
                        row["source_accuracy"], row["source_loss_first"], row["source_loss_last"],
                        row["perturbation_linf_max"], row["seconds"],
                    )
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()

    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (fold_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return results_path


def run_sanity(
    target: TargetAdapter,
    source: torch.nn.Module,
    config: BlackBoxConfig,
    fold: int,
    target_checkpoint: Path,
    source_checkpoint: Path,
    output_dir: Path,
    max_batches: Optional[int] = None,
) -> bool:
    """Reliability checks for the transfer attack; writes sanity.json."""
    checks: List[Dict[str, Any]] = []

    def record(name: str, passed: Optional[bool], detail: Any) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})
        logger.info("[sanity] %s: %s (%s)", name, {True: "PASS", False: "FAIL", None: "INFO"}[passed], detail)

    pgd = config.video_pgd
    epsilon = pgd.sanity_epsilon
    cleans: Dict[str, Dict[str, Any]] = {}
    for split in config.splits:
        clean = _clean_pass(target, source, split, max_batches)
        cleans[split] = clean
        for role, checkpoint, probs in (
            ("target", target_checkpoint, clean["target_probs"]),
            ("source", source_checkpoint, clean["source_probs"]),
        ):
            accuracy = float((probs.argmax(axis=1) == clean["labels"]).mean())
            expected = _summary_accuracy(checkpoint, split) if max_batches is None else None
            if expected is None:
                record(f"{role}_clean_{split}_matches_summary", None, {"clean": accuracy, "note": "not compared"})
            else:
                record(f"{role}_clean_{split}_matches_summary", abs(accuracy - expected) <= 1e-3,
                       {"clean": accuracy, "summary": expected})

    split = "test" if "test" in config.splits else config.splits[0]
    batch = next(target.batches(split, 1))
    first, first_delta, losses = transfer_pgd(
        source, batch.video, batch.labels, batch.keys, epsilon, pgd.alpha_ratio, pgd.steps, pgd.random_start, config.seed,
    )
    second, _, _ = transfer_pgd(
        source, batch.video, batch.labels, batch.keys, epsilon, pgd.alpha_ratio, pgd.steps, pgd.random_start, config.seed,
    )
    record("perturbation_reproducible", bool(torch.equal(first, second)), {"epsilon": epsilon})
    linf = first_delta.abs().max().item()
    record("perturbation_within_budget", linf <= epsilon + 1e-6, {"epsilon": epsilon, "linf": linf})
    record("source_loss_increases", losses[-1] > losses[0], {"first": losses[0], "last": losses[-1]})

    result = _attack_pass(target, source, split, epsilon, config, cleans[split]["correct"], max_batches)
    source_accuracy = float((result["source_probs"].argmax(axis=1) == result["labels"]).mean())
    record("source_fooled_by_its_own_attack", source_accuracy <= 0.10,
           {"split": split, "epsilon": epsilon, "source_accuracy": source_accuracy})
    target_accuracy = float((result["target_probs"].argmax(axis=1) == result["labels"]).mean())
    target_clean = float((cleans[split]["target_probs"].argmax(axis=1) == cleans[split]["labels"]).mean())
    record("target_accuracy_not_above_clean", target_accuracy <= target_clean + 1e-12,
           {"clean": target_clean, "transfer": target_accuracy})

    random_probs, random_labels = [], []
    with torch.no_grad():
        for b in target.batches(split, max_batches):
            noisy = random_linf_noise(b.video, b.keys, epsilon, config.seed)
            attacked = torch.tensor([cleans[split]["correct"][key] for key in b.keys], device=b.video.device)
            noisy = torch.where(attacked.view(-1, 1, 1, 1), noisy, b.video)
            random_probs.append(target.forward(b.waveform, noisy).softmax(dim=1).cpu().numpy())
            random_labels.append(b.labels.cpu().numpy())
    random_accuracy = float((np.concatenate(random_probs).argmax(axis=1) == np.concatenate(random_labels)).mean())
    record("transfer_vs_random_noise", None,
           {"epsilon": epsilon, "target_transfer": target_accuracy, "target_random": random_accuracy})

    fold_dir = output_dir / run_name(target.name, config.source) / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous(fold_dir, ("sanity.json",))
    with (fold_dir / "sanity.json").open("w", encoding="utf-8") as file:
        json.dump({"partial": max_batches is not None, "source_model": config.source, "checks": checks}, file, indent=2)
    return all(check["passed"] is not False for check in checks)


def upload_fold(upload_config: Any, output_dir: Path, target: str, source: str, fold: int, splits: Sequence[str], kind: str) -> Optional[str]:
    """Upload `<output_dir>/<target>_from_<source>/fold_XX` as a self-describing zip, e.g.
    `blackbox_from_ConvNeXtTiny/video_efficientnet_b0_blackbox_from_ConvNeXtTiny_fold00_val-test_full_20261106-101500.zip`.
    """
    if not upload_config.enabled:
        return None
    name = run_name(target, source)
    fold_dir = output_dir / name / f"fold_{fold:02d}"
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"No results to upload: {fold_dir}")
    split_tag = "-".join(splits)
    file_name = f"{MODALITY.get(target, target)}_{target}_blackbox_from_{source}_fold{fold:02d}_{split_tag}_{kind}_{_stamp()}.zip"
    zip_path = _zip([fold_dir], output_dir, Path(f"{output_dir}_uploads") / file_name)
    return _upload(upload_config, zip_path, f"blackbox_from_{source}/{file_name}", output_dir)


def run_cli(build_target: Callable[[int, BlackBoxConfig, BlackBoxPaths, Path, torch.device], TargetAdapter], project_root: Path) -> None:
    parser = argparse.ArgumentParser(description="Black-box transfer PGD: perturbation from a source video model, scored on the target.")
    parser.add_argument("--config", default=str(project_root / "config/blackbox_config.json"))
    parser.add_argument("--paths", default=str(project_root / "config/blackbox_paths.json"))
    parser.add_argument("--source", choices=sorted(SOURCE_BUILDERS), help="Override the source model in the config.")
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], help="Override the splits in the config.")
    parser.add_argument("--folds", type=int, nargs="*", help="Subset of the configured folds.")
    parser.add_argument("--max-batches", type=int, help="Smoke test only; results are marked partial.")
    parser.add_argument("--sanity", action="store_true", help="Run reliability checks instead of the full evaluation.")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run and keep its finished levels.")
    parser.add_argument("--upload-config", default=str(project_root / "config/robustness_upload.json"))
    parser.add_argument("--no-upload", action="store_true", help="Keep results local even if upload is enabled.")
    parser.add_argument("--upload-only", action="store_true", help="Upload existing fold results without evaluating.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = load_json_model(BlackBoxConfig, Path(args.config))
    if args.source:
        config.source = args.source
    if args.splits:
        config.splits = list(dict.fromkeys(args.splits))
    config = BlackBoxConfig(**config.model_dump())
    paths = load_json_model(BlackBoxPaths, Path(args.paths))
    upload_config = load_upload_config(Path(args.upload_config))
    if args.no_upload:
        upload_config.enabled = False
    if args.upload_only and not upload_config.enabled:
        raise SystemExit("--upload-only needs upload enabled in the upload config.")
    check_ready(upload_config)

    folds = args.folds if args.folds else config.folds
    unknown = sorted(set(folds) - set(config.folds))
    if unknown:
        raise ValueError(f"Folds {unknown} are not in the config: {config.folds}")
    output_dir = Path(paths.output_dir)
    upload_failed: List[int] = []

    def upload(target_name: str, fold: int, kind: str) -> None:
        try:
            upload_fold(upload_config, output_dir, target_name, config.source, fold, config.splits, kind)
        except Exception:
            logger.exception("Upload failed for fold %d; results are still in %s", fold, output_dir)
            upload_failed.append(fold)

    if args.upload_only:
        targets = sorted(
            path.name[: -len(f"_from_{config.source}")] for path in output_dir.glob(f"*_from_{config.source}") if path.is_dir()
        ) if output_dir.is_dir() else []
        if not targets:
            raise SystemExit(f"No results for source {config.source} under {output_dir}")
        for target_name in targets:
            for fold in folds:
                upload(target_name, fold, "results")
    else:
        # Same convolution algorithms on every run, so both branches produce the same perturbation.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Black-box config hash: %s | source: %s | splits: %s | device: %s",
            config_hash(config), config.source, config.splits, device,
        )
        failed = []
        kind = "sanity" if args.sanity else "full"
        if args.max_batches is not None:
            kind += "-partial"
        for fold in folds:
            target_checkpoint = paths.target_for(fold)
            source_checkpoint = paths.source_for(config.source, fold)
            target = build_target(fold, config, paths, target_checkpoint, device)
            source = load_source(config.source, source_checkpoint, device)
            for split in config.splits:
                records = target.split_records(split)
                verify_split(records, target_checkpoint, split)
                verify_split(records, source_checkpoint, split)
            if args.sanity:
                if not run_sanity(target, source, config, fold, target_checkpoint, source_checkpoint, output_dir, args.max_batches):
                    failed.append(fold)
            else:
                evaluate_fold(target, source, config, fold, target_checkpoint, source_checkpoint, output_dir,
                              args.max_batches, args.resume)
            upload(target.name, fold, kind)
            del target, source
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if failed:
            raise SystemExit(f"Sanity checks failed on folds: {failed}")
    if upload_failed:
        folds_text = " ".join(str(fold) for fold in sorted(set(upload_failed)))
        raise SystemExit(f"Upload failed on folds {folds_text}. Retry with: --upload-only --folds {folds_text}")
