"""Shared robustness evaluation. This file is byte-identical on all three branches."""

import argparse
import copy
import csv
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support

from .hub import MODALITY, check_ready, load_upload_config, upload_fold

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 4
CLASS_NAMES = ("none", "strong", "medium", "weak")
METRIC_FIELDS = [
    "accuracy",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "mAP",
]
RESULT_FIELDS = [
    "model", "fold", "scenario", "level_unit", "level", "n_samples", *METRIC_FIELDS, "per_class",
    "confusion_matrix", "partial", "seconds", "config_hash", "checkpoint",
]
TRACE_FIELDS = ["epsilon", "step", "mean_loss", "max_perturbation"]
RESULT_ARCHIVE_ITEMS = ("robustness.csv", "pgd_traces.csv", "predictions", "run_info.json", "run.log")


class AudioNoiseConfig(BaseModel):
    type: Literal["gaussian_white"] = "gaussian_white"
    snr_db: List[float]
    domain: Literal["waveform"] = "waveform"


class EpsilonSweepConfig(BaseModel):
    start: float = Field(gt=0.0, le=1.0)
    end: float = Field(gt=0.0, le=1.0)
    num: int = Field(ge=2)
    spacing: Literal["log", "linear"] = "log"

    @model_validator(mode="after")
    def check_range(self) -> "EpsilonSweepConfig":
        if self.start >= self.end:
            raise ValueError("sweep.start must be smaller than sweep.end.")
        return self


class VideoPGDConfig(BaseModel):
    """Epsilon is in [0, 1] pixel units. Mode 'single' uses `epsilon`; 'sweep' uses `sweep`."""

    mode: Literal["single", "sweep"]
    epsilon: float = Field(gt=0.0, le=1.0)
    sweep: EpsilonSweepConfig
    include_epsilons: List[float] = Field(default_factory=list)
    sanity_epsilon: float = Field(gt=0.0, le=1.0)
    norm: Literal["linf"] = "linf"
    alpha_ratio: float = Field(gt=0.0)
    steps: int = Field(gt=0)
    random_start: bool = True
    loss: Literal["cross_entropy"] = "cross_entropy"

    def epsilon_levels(self) -> List[float]:
        if self.mode == "single":
            values = [self.epsilon]
        else:
            make = np.geomspace if self.sweep.spacing == "log" else np.linspace
            values = [*make(self.sweep.start, self.sweep.end, self.sweep.num).tolist(), *self.include_epsilons]
        return sorted({float(f"{value:.10g}") for value in values})


class RobustnessConfig(BaseModel):
    folds: List[int]
    split: Literal["test"] = "test"
    batch_size: int = Field(gt=0)
    seed: int
    audio_noise: AudioNoiseConfig
    video_pgd: VideoPGDConfig
    metrics: List[str]


class PathsConfig(BaseModel):
    """Per-branch paths: use `checkpoints` for explicit paths, otherwise `checkpoint_template`."""

    train_config_path: str = "config/train_config.json"
    dataset_path: str = ""
    checkpoint_template: str = ""
    checkpoints: Dict[str, str] = Field(default_factory=dict)
    output_dir: str = "outputs/robustness"

    def checkpoint_for(self, fold: int) -> Path:
        if str(fold) in self.checkpoints:
            path = Path(self.checkpoints[str(fold)])
        elif self.checkpoint_template:
            path = Path(self.checkpoint_template.format(fold=fold))
        else:
            raise ValueError(f"No checkpoint path configured for fold {fold}.")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint for fold {fold} not found: {path}")
        return path


def load_json_model(model_cls: Any, path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return model_cls(**json.load(file))


def config_hash(config: RobustnessConfig) -> str:
    canonical = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _sample_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def add_awgn(waveforms: torch.Tensor, keys: Sequence[str], snr_db: float, seed: int) -> torch.Tensor:
    """Add per-clip white noise with the exact SNR; noise depends only on (key, SNR, seed)."""
    noisy = waveforms.detach().cpu().clone()
    for index, key in enumerate(keys):
        signal = noisy[index]
        signal_power = signal.double().pow(2).mean()
        if signal_power == 0:
            logger.warning("Silent clip has undefined SNR; left unchanged: %s", key)
            continue
        generator = torch.Generator().manual_seed(_sample_seed("awgn", key, snr_db, seed))
        noise = torch.randn(signal.shape, generator=generator, dtype=torch.float64)
        target_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = noise * torch.sqrt(target_power / noise.pow(2).mean())
        noisy[index] = (signal.double() + noise).float()
    return noisy


def measured_snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
    clean, noisy = clean.double(), noisy.double()
    return 10.0 * torch.log10(clean.pow(2).mean(dim=1) / (noisy - clean).pow(2).mean(dim=1))


def _normalization(video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    frames = video.shape[1] // 3
    mean = torch.tensor(IMAGENET_MEAN * frames, dtype=video.dtype, device=video.device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD * frames, dtype=video.dtype, device=video.device).view(1, -1, 1, 1)
    return mean, std


def _uniform_delta(video: torch.Tensor, keys: Sequence[str], epsilon: float, tag: str, seed: int) -> torch.Tensor:
    deltas = []
    for key in keys:
        generator = torch.Generator().manual_seed(_sample_seed(tag, key, epsilon, seed))
        deltas.append(torch.rand(video.shape[1:], generator=generator, dtype=video.dtype) * 2.0 * epsilon - epsilon)
    return torch.stack(deltas).to(video.device)


def random_linf_noise(video: torch.Tensor, keys: Sequence[str], epsilon: float, seed: int) -> torch.Tensor:
    """Random baseline with the same L-infinity budget as PGD."""
    mean, std = _normalization(video)
    pixels = (video * std + mean).clamp(0.0, 1.0)
    adversarial = (pixels + _uniform_delta(video, keys, epsilon, "random", seed)).clamp(0.0, 1.0)
    return (adversarial - mean) / std


def pgd_attack(
    forward: Callable[[torch.Tensor], torch.Tensor],
    video: torch.Tensor,
    labels: torch.Tensor,
    keys: Sequence[str],
    epsilon: float,
    alpha_ratio: float,
    steps: int,
    random_start: bool,
    seed: int,
) -> Tuple[torch.Tensor, List[float]]:
    """Untargeted L-infinity PGD in [0, 1] pixel space. Takes and returns normalized video."""
    alpha = epsilon * alpha_ratio
    mean, std = _normalization(video)
    pixels = (video.detach() * std + mean).clamp(0.0, 1.0)
    adversarial = pixels.clone()
    if random_start:
        adversarial = (pixels + _uniform_delta(video, keys, epsilon, "pgd", seed)).clamp(0.0, 1.0)

    losses: List[float] = []
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        loss = F.cross_entropy(forward((adversarial - mean) / std), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        losses.append(loss.item())
        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - pixels).clamp(-epsilon, epsilon)
        adversarial = (pixels + delta).clamp(0.0, 1.0)
    return ((adversarial - mean) / std).detach(), losses


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    """Compute metrics on softmax outputs on every branch."""
    predictions = probabilities.argmax(axis=1)
    classes = list(range(NUM_CLASSES))
    average_precisions = {
        cls: float(average_precision_score(labels == cls, probabilities[:, cls]))
        for cls in classes
        if (labels == cls).any()
    }
    metrics: Dict[str, Any] = {"accuracy": float(accuracy_score(labels, predictions))}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, labels=classes, average=average, zero_division=0
        )
        metrics.update({f"precision_{average}": float(precision), f"recall_{average}": float(recall), f"f1_{average}": float(f1)})
    metrics["mAP"] = float(np.mean(list(average_precisions.values())))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=classes, average=None, zero_division=0
    )
    metrics["per_class"] = {
        CLASS_NAMES[cls]: {
            "precision": float(precision[cls]), "recall": float(recall[cls]), "f1": float(f1[cls]),
            "AP": average_precisions.get(cls), "support": int(support[cls]),
        }
        for cls in classes
    }
    metrics["confusion_matrix"] = confusion_matrix(labels, predictions, labels=classes).tolist()
    return metrics


@dataclass
class Batch:
    keys: List[str]
    labels: torch.Tensor
    waveform: Optional[torch.Tensor]
    video: Optional[torch.Tensor]


class BaseAdapter:
    """Implemented once per branch: wraps the existing model and test DataLoader."""

    name: str
    supports: Tuple[str, ...]
    device: torch.device

    def batches(self, max_batches: Optional[int] = None) -> Iterator[Batch]:
        raise NotImplementedError

    def forward(self, waveform: Optional[torch.Tensor], video: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def split_records(self) -> set:
        """Return {(sample_key, label)} for the loaded test split."""
        raise NotImplementedError


def prepare_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # thop's FLOP profiler registers total_ops/total_params counters on every module during training;
    # they are not weights. Everything else must match exactly.
    profiler_keys = [key for key in state_dict if key.rsplit(".", 1)[-1] in ("total_ops", "total_params")]
    if profiler_keys:
        state_dict = {key: value for key, value in state_dict.items() if key not in set(profiler_keys)}
        logger.info("Ignored %d thop profiler counters in %s", len(profiler_keys), checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def fold_splitter_config(train_config: Any, fold: int, paths: PathsConfig) -> Any:
    splitter = copy.deepcopy(train_config.dataset_splitter)
    splitter.evaluation_mode = "cross_validation"
    splitter.fold_index = fold
    splitter.save_results = False
    if paths.dataset_path:
        splitter.dataset_path = paths.dataset_path
    return splitter


def verify_test_split(adapter: BaseAdapter, checkpoint_path: Path) -> None:
    """Fail if the regenerated test split differs from the split saved with the weights."""
    from dataset.data_split import _sample_identity_key

    reference = checkpoint_path.parent / "splits" / "test.csv"
    if not reference.is_file():
        logger.warning("No training split next to the checkpoint; could not verify: %s", reference)
        return
    with reference.open("r", encoding="utf-8", newline="") as file:
        expected = set()
        for row in csv.DictReader(file):
            path = row.get("audio_path") or row.get("video_path")
            expected.add((_sample_identity_key(path), int(row["label"])))
    actual = adapter.split_records()
    if expected != actual:
        raise ValueError(
            f"Test split differs from {reference}: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected."
        )
    logger.info("Test split matches %s (%d samples).", reference, len(actual))


def _predict(
    adapter: BaseAdapter,
    transform: Callable[[Batch], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    max_batches: Optional[int],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    keys: List[str] = []
    labels: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    for batch in adapter.batches(max_batches):
        waveform, video = transform(batch)
        with torch.no_grad():
            probabilities.append(adapter.forward(waveform, video).softmax(dim=1).cpu().numpy())
        keys.extend(batch.keys)
        labels.append(batch.labels.cpu().numpy())
    if not keys:
        raise ValueError("Test loader produced no samples.")
    return keys, np.concatenate(labels), np.concatenate(probabilities)


def _clean_transform(batch: Batch) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    return batch.waveform, batch.video


def _noise_transform(adapter: BaseAdapter, snr_db: float, seed: int) -> Callable:
    def transform(batch: Batch) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return add_awgn(batch.waveform, batch.keys, snr_db, seed).to(adapter.device), batch.video

    return transform


def _pgd_transform(
    adapter: BaseAdapter,
    pgd: VideoPGDConfig,
    epsilon: float,
    seed: int,
    steps: int,
    tracker: Optional[Dict[str, Any]] = None,
) -> Callable:
    def transform(batch: Batch) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        video, losses = pgd_attack(
            lambda attacked: adapter.forward(batch.waveform, attacked),
            batch.video, batch.labels, batch.keys, epsilon,
            pgd.alpha_ratio, steps, pgd.random_start, seed,
        )
        if tracker is not None:
            _, std = _normalization(video)
            size = batch.labels.shape[0]
            tracker["loss_sum"] += np.asarray(losses) * size
            tracker["samples"] += size
            perturbation = ((video - batch.video) * std).abs().max().item()
            tracker["max_perturbation"] = max(tracker["max_perturbation"], perturbation)
        return batch.waveform, video

    return transform


def _scenario_plan(adapter: BaseAdapter, config: RobustnessConfig) -> List[Tuple[str, str, Any, Callable, Any]]:
    plan: List[Tuple[str, str, Any, Callable, Any]] = [("clean", "none", "clean", _clean_transform, None)]
    if "audio_noise" in adapter.supports:
        for snr_db in config.audio_noise.snr_db:
            plan.append(("audio_noise", "snr_db", snr_db, _noise_transform(adapter, snr_db, config.seed), None))
    if "video_pgd" in adapter.supports:
        pgd = config.video_pgd
        for epsilon in pgd.epsilon_levels():
            tracker = {"loss_sum": np.zeros(pgd.steps), "samples": 0, "max_perturbation": 0.0}
            transform = _pgd_transform(adapter, pgd, epsilon, config.seed, pgd.steps, tracker)
            plan.append(("video_pgd", "epsilon", epsilon, transform, tracker))
    return plan


def _archive_previous(fold_dir: Path, names: Sequence[str]) -> None:
    """Keep earlier results instead of overwriting them."""
    existing = [fold_dir / name for name in names if (fold_dir / name).exists()]
    if not existing:
        return
    target = fold_dir / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    target.mkdir(parents=True, exist_ok=True)
    for item in existing:
        shutil.move(str(item), str(target / item.name))
    logger.info("Archived previous results to %s", target)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def evaluate_fold(
    adapter: BaseAdapter,
    config: RobustnessConfig,
    fold: int,
    checkpoint_path: Path,
    output_dir: Path,
    max_batches: Optional[int] = None,
    resume: bool = False,
) -> Path:
    """Evaluate every planned level and save results after each level."""
    fold_dir = output_dir / adapter.name / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    partial = int(max_batches is not None)
    current_hash = config_hash(config)
    results_path = fold_dir / "robustness.csv"
    traces_path = fold_dir / "pgd_traces.csv"

    kept_rows: List[Dict[str, str]] = []
    kept_traces: List[Dict[str, str]] = []
    if resume:
        for row in _read_csv(results_path):
            finished = (fold_dir / "predictions" / f"{row['scenario']}_{row['level']}.npz").is_file()
            complete = all(row.get(name) not in (None, "") for name in RESULT_FIELDS)
            if finished and complete and row["config_hash"] == current_hash and row["partial"] == str(partial):
                kept_rows.append(row)
        finished_levels = {row["level"] for row in kept_rows}
        kept_traces = [row for row in _read_csv(traces_path) if row["epsilon"] in finished_levels]
    else:
        _archive_previous(fold_dir, RESULT_ARCHIVE_ITEMS)
    (fold_dir / "predictions").mkdir(exist_ok=True)

    log_handler = logging.FileHandler(fold_dir / "run.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(log_handler)
    run_info = {
        "model": adapter.name, "fold": fold, "checkpoint": str(checkpoint_path), "config_hash": current_hash,
        "config": config.model_dump(), "partial": bool(partial), "resumed": resume,
        "started_at": datetime.now().isoformat(timespec="seconds"), "device": str(adapter.device),
        "torch_version": torch.__version__,
    }
    (fold_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    try:
        done = {(row["scenario"], row["level"]) for row in kept_rows}
        with results_path.open("w", newline="", encoding="utf-8") as file, \
                traces_path.open("w", newline="", encoding="utf-8") as traces_file:
            writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(kept_rows)
            trace_writer = csv.DictWriter(traces_file, fieldnames=TRACE_FIELDS)
            trace_writer.writeheader()
            trace_writer.writerows(kept_traces)
            file.flush()
            traces_file.flush()
            for scenario, level_unit, level, transform, tracker in _scenario_plan(adapter, config):
                if (scenario, str(level)) in done:
                    logger.info("[%s fold %d] %s %s already saved; skipping.", adapter.name, fold, scenario, level)
                    continue
                started = time.perf_counter()
                keys, labels, probabilities = _predict(adapter, transform, max_batches)
                metrics = compute_metrics(labels, probabilities)
                np.savez_compressed(
                    fold_dir / "predictions" / f"{scenario}_{level}.npz",
                    keys=np.array(keys), labels=labels, probabilities=probabilities.astype(np.float32),
                )
                if tracker is not None:
                    for step, loss_sum in enumerate(tracker["loss_sum"]):
                        trace_writer.writerow({
                            "epsilon": level, "step": step, "mean_loss": loss_sum / tracker["samples"],
                            "max_perturbation": tracker["max_perturbation"],
                        })
                    traces_file.flush()
                writer.writerow({
                    "model": adapter.name, "fold": fold, "scenario": scenario, "level_unit": level_unit,
                    "level": level, "n_samples": len(keys),
                    **{name: metrics[name] for name in METRIC_FIELDS},
                    "per_class": json.dumps(metrics["per_class"]),
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]), "partial": partial,
                    "seconds": round(time.perf_counter() - started, 3), "config_hash": current_hash,
                    "checkpoint": str(checkpoint_path),
                })
                file.flush()
                logger.info(
                    "[%s fold %d] %s %s: accuracy=%.4f (%.1fs)", adapter.name, fold, scenario, level,
                    metrics["accuracy"], time.perf_counter() - started,
                )
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()

    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (fold_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return results_path


def _accuracy(adapter: BaseAdapter, transform: Callable, max_batches: Optional[int]) -> float:
    _, labels, probabilities = _predict(adapter, transform, max_batches)
    return float((probabilities.argmax(axis=1) == labels).mean())


def run_sanity(
    adapter: BaseAdapter,
    config: RobustnessConfig,
    fold: int,
    checkpoint_path: Path,
    output_dir: Path,
    max_batches: Optional[int] = None,
) -> bool:
    """Run the pre-registered reliability checks and write sanity.json."""
    checks: List[Dict[str, Any]] = []

    def record(name: str, passed: Optional[bool], detail: Any) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})
        logger.info("[sanity] %s: %s (%s)", name, {True: "PASS", False: "FAIL", None: "INFO"}[passed], detail)

    clean_accuracy = _accuracy(adapter, _clean_transform, max_batches)
    summary_path = checkpoint_path.parent / "summary.csv"
    if max_batches is None and summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", newline="") as file:
            expected = float(list(csv.DictReader(file))[-1]["Accuracy Test"])
        record("clean_matches_summary", abs(clean_accuracy - expected) <= 1e-3, {"clean": clean_accuracy, "summary": expected})
    else:
        record("clean_matches_summary", None, {"clean": clean_accuracy, "note": "not compared"})

    if "audio_noise" in adapter.supports:
        snr_db = min(config.audio_noise.snr_db)
        batch = next(adapter.batches(1))
        first = add_awgn(batch.waveform, batch.keys, snr_db, config.seed)
        second = add_awgn(batch.waveform, batch.keys, snr_db, config.seed)
        record("noise_reproducible", bool(torch.equal(first, second)), {"snr_db": snr_db})
        error = (measured_snr_db(batch.waveform.cpu(), first) - snr_db).abs().max().item()
        record("noise_snr_exact", error < 0.01, {"max_error_db": error})

    if "video_pgd" in adapter.supports:
        pgd = config.video_pgd
        epsilon = pgd.sanity_epsilon
        batch = next(adapter.batches(1))
        _, losses = pgd_attack(
            lambda attacked: adapter.forward(batch.waveform, attacked), batch.video, batch.labels, batch.keys,
            epsilon, pgd.alpha_ratio, pgd.steps, pgd.random_start, config.seed,
        )
        record("pgd_loss_increases", losses[-1] > losses[0], {"first": losses[0], "last": losses[-1]})
        accuracy = _accuracy(adapter, _pgd_transform(adapter, pgd, epsilon, config.seed, pgd.steps), max_batches)

        # Checks follow Carlini et al., "On Evaluating Adversarial Robustness" (arXiv:1902.06705).
        doubled = _accuracy(adapter, _pgd_transform(adapter, pgd, epsilon, config.seed, 2 * pgd.steps), max_batches)
        record("pgd_converged_when_steps_doubled", accuracy - doubled <= 0.01, {"steps": pgd.steps, "accuracy": accuracy, "accuracy_doubled": doubled})

        def fgsm(b: Batch) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            video, _ = pgd_attack(
                lambda attacked: adapter.forward(b.waveform, attacked), b.video, b.labels, b.keys,
                epsilon, 1.0, 1, False, config.seed,
            )
            return b.waveform, video

        single_step = _accuracy(adapter, fgsm, max_batches)
        record("pgd_at_least_as_strong_as_single_step", accuracy <= single_step, {"pgd": accuracy, "single_step": single_step})

        random_accuracy = _accuracy(
            adapter, lambda b: (b.waveform, random_linf_noise(b.video, b.keys, epsilon, config.seed)), max_batches
        )
        record("pgd_stronger_than_random", accuracy < random_accuracy, {"pgd": accuracy, "random": random_accuracy})

        budgets = [epsilon * fraction for fraction in (0.125, 0.25, 0.5, 1.0)]
        curve = [
            _accuracy(adapter, _pgd_transform(adapter, pgd, value, config.seed, pgd.steps), max_batches)
            for value in budgets
        ]
        monotonic = all(later <= earlier + 0.01 for earlier, later in zip(curve, curve[1:]))
        record("accuracy_falls_up_to_sanity_epsilon", monotonic, {"epsilon": budgets, "accuracy": curve})

        correct_any_restart = None
        for restart in range(3):
            _, labels, probabilities = _predict(
                adapter, _pgd_transform(adapter, pgd, epsilon, config.seed + restart, pgd.steps), max_batches
            )
            correct = probabilities.argmax(axis=1) == labels
            correct_any_restart = correct if correct_any_restart is None else correct_any_restart & correct
        worst_case = float(correct_any_restart.mean())
        record("restarts_add_little", accuracy - worst_case <= 0.02, {"one_restart": accuracy, "worst_of_three": worst_case})

        # At a very large epsilon the frames carry little information: a video-only model is expected near 1 / classes.
        largest = max(pgd.epsilon_levels())
        _, labels, probabilities = _predict(
            adapter, _pgd_transform(adapter, pgd, largest, config.seed, pgd.steps), max_batches
        )
        predictions = probabilities.argmax(axis=1)
        detail = {
            "epsilon": largest,
            "accuracy": float((predictions == labels).mean()),
            "chance": 1.0 / NUM_CLASSES,
            "most_predicted_class_share": float(np.bincount(predictions, minlength=NUM_CLASSES).max() / predictions.size),
        }
        if "audio_noise" in adapter.supports:
            detail["note"] = "clean audio is present; compare with clean AMARU instead of chance"
            record("largest_epsilon_behaviour", None, detail)
        elif largest < 0.5:
            detail["note"] = "largest configured epsilon is too small to expect chance level"
            record("largest_epsilon_near_chance", None, detail)
        else:
            record("largest_epsilon_near_chance", detail["accuracy"] <= 0.35, detail)

    fold_dir = output_dir / adapter.name / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous(fold_dir, ("sanity.json",))
    with (fold_dir / "sanity.json").open("w", encoding="utf-8") as file:
        json.dump({"partial": max_batches is not None, "checks": checks}, file, indent=2)
    return all(check["passed"] is not False for check in checks)


def run_cli(build_adapter: Callable[[int, RobustnessConfig, PathsConfig, Path, torch.device], BaseAdapter], project_root: Path) -> None:
    parser = argparse.ArgumentParser(description="Evaluate model robustness on the test split of each fold.")
    parser.add_argument("--config", default=str(project_root / "config/robustness_config.json"))
    parser.add_argument("--paths", default=str(project_root / "config/robustness_paths.json"))
    parser.add_argument("--folds", type=int, nargs="*", help="Subset of the configured folds.")
    parser.add_argument("--max-batches", type=int, help="Smoke test only; results are marked partial.")
    parser.add_argument("--sanity", action="store_true", help="Run reliability checks instead of the full evaluation.")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run and keep its finished levels.")
    parser.add_argument("--upload-config", default=str(project_root / "config/robustness_upload.json"))
    parser.add_argument("--no-upload", action="store_true", help="Keep results local even if upload is enabled.")
    parser.add_argument("--upload-only", action="store_true", help="Upload existing fold results without evaluating.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = load_json_model(RobustnessConfig, Path(args.config))
    paths = load_json_model(PathsConfig, Path(args.paths))
    upload_config = load_upload_config(Path(args.upload_config))
    if args.no_upload:
        upload_config.enabled = False
    if args.upload_only and not upload_config.enabled:
        raise SystemExit("--upload-only needs upload enabled in the upload config.")
    check_ready(upload_config)

    folds = args.folds if args.folds else config.folds
    unknown = sorted(set(folds) - set(config.folds))
    if unknown:
        raise ValueError(f"Folds {unknown} are not in the shared config: {config.folds}")
    output_dir = Path(paths.output_dir)

    upload_failed = []

    def upload(model: str, fold: int, kind: str) -> None:
        # A failed upload must not stop the evaluation; the results stay on disk.
        try:
            upload_fold(upload_config, output_dir, model, fold, kind)
        except Exception:
            logger.exception("Upload failed for %s fold %d; results are still in %s", model, fold, output_dir)
            upload_failed.append(fold)

    if args.upload_only:
        models = [path.name for path in output_dir.iterdir() if path.name in MODALITY] if output_dir.is_dir() else []
        if not models:
            raise SystemExit(f"No model results under {output_dir}")
        for model in models:
            for fold in folds:
                upload(model, fold, "results")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Robustness config hash: %s | device: %s", config_hash(config), device)
        failed = []
        kind = "sanity" if args.sanity else "full"
        if args.max_batches is not None:
            kind += "-partial"
        for fold in folds:
            checkpoint_path = paths.checkpoint_for(fold)
            adapter = build_adapter(fold, config, paths, checkpoint_path, device)
            verify_test_split(adapter, checkpoint_path)
            if args.sanity:
                if not run_sanity(adapter, config, fold, checkpoint_path, output_dir, args.max_batches):
                    failed.append(fold)
            else:
                evaluate_fold(adapter, config, fold, checkpoint_path, output_dir, args.max_batches, args.resume)
            upload(adapter.name, fold, kind)
        if failed:
            raise SystemExit(f"Sanity checks failed on folds: {failed}")
    if upload_failed:
        folds_text = " ".join(str(fold) for fold in sorted(set(upload_failed)))
        raise SystemExit(f"Upload failed on folds {folds_text}. Retry with: --upload-only --folds {folds_text}")
