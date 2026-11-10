"""Visualize imperceptible PGD perturbations on the video stream (paper figure, one representative fold).

For one fold and one epsilon this script:
1. loads the test split (first + last RGB frame, 6 channels) and five trained models: EfficientNet-B0 (video),
   HMMF-Net (multimodal, clean audio) and the three black-box source models (ConvNeXt-T, Swin-T, ResNet18);
2. checks that every checkpoint matches the regenerated test split and that each clean test accuracy
   matches its summary.csv;
3. attacks the test samples that all five models classify correctly on the clean input:
   - white-box PGD on EfficientNet-B0 and on HMMF-Net (same `pgd_attack` as the robustness runs);
   - black-box transfer PGD from each source (same `transfer_pgd` as the black-box runs), the same
     adversarial video is then fed to both EfficientNet-B0 and HMMF-Net;
4. ranks the samples (fooled by both white-box attacks first, then by the number of fooled black-box
   transfers, then by clean confidence), keeps the best `per_class` samples of every class (or the
   `sample_keys` given in the config), and saves for each of them:
   - every frame as a PNG (clean, amplified perturbation, adversarial) for both frames and all five attacks;
   - the float tensors (npz), predictions, L-inf, PSNR and SSIM (info.json);
   - the composite figure (figure.pdf / figure.png);
5. zips the fold folder and uploads it to Hugging Face (config/robustness_upload.json) unless --no-upload.

The perturbation is displayed as delta / (2 * epsilon) + 0.5 (amplified for visualization). Predictions are
always computed on the float tensors; the adversarial PNGs are 8-bit, so at small epsilon they round back to
the clean frame, which is exactly the point of the figure.
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torchvision.models import efficientnet_b0

from dataset.data_split import _sample_identity_key
from evaluate_blackbox import MultimodalTarget
from robustness.blackbox import _summary_accuracy, load_source, transfer_pgd, verify_split
from robustness.core import CLASS_NAMES, _normalization, pgd_attack, prepare_model
from robustness.hub import _stamp, _upload, _zip, check_ready, load_upload_config
from robustness.source_models import INPUT_CHANNELS, NUM_CLASSES, _Backbone, _widen_first_conv

logger = logging.getLogger("perturbation_viz")

SOURCE_LABELS = {"ConvNeXtTiny": "ConvNeXt-T", "SwinTiny": "Swin-T", "ResNet18": "ResNet18"}
TARGET_LABELS = {"effb0": "EfficientNet-B0", "hmmf": "HMMF-Net"}
FRAMES = (("first", "First frame", slice(0, 3)), ("last", "Last frame", slice(3, 6)))
ACCURACY_TOLERANCE = 0.005  # clean test accuracy must match summary.csv within 0.5 percentage points


class PGDSettings(BaseModel):
    alpha_ratio: float = Field(0.25, gt=0.0)
    steps: int = Field(20, gt=0)
    random_start: bool = True


class FeaturedSample(BaseModel):
    """A sample to render in addition to the ranked ones, on the fold whose test split holds it."""
    key: str
    fold: int


class VizConfig(BaseModel):
    fold: int = 2
    epsilon: float = Field(1e-4, gt=0.0, le=1.0)
    seed: int = 42
    batch_size: int = Field(32, gt=0)
    pgd: PGDSettings = PGDSettings()
    sources: List[str] = ["ConvNeXtTiny", "SwinTiny", "ResNet18"]
    per_class: int = Field(5, ge=1)
    sample_keys: List[str] = []
    display_original_size: bool = True
    featured_samples: List[FeaturedSample] = []


class VizPaths(BaseModel):
    train_config_path: str = "config/train_config.json"
    dataset_path: str = ""
    hmmf_net: Dict[str, str]
    efficientnet_b0: str
    sources: Dict[str, str]
    output_dir: str = "outputs/perturbation_viz"

    def checkpoint(self, path: str, label: str) -> Path:
        result = Path(path)
        if not result.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {result}")
        return result


class EfficientNetB0Video(nn.Module):
    """Same module layout as VideoModel(EfficientNetB0) of the video branch: backbone.model.<torchvision layers>."""

    def __init__(self) -> None:
        super().__init__()
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        _widen_first_conv(model, INPUT_CHANNELS)
        self.backbone = _Backbone(model)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.backbone(video)


@dataclass
class Models:
    target: MultimodalTarget  # HMMF-Net and the test loader
    effb0: nn.Module
    sources: Dict[str, nn.Module]
    checkpoints: Dict[str, Path]
    video_paths: Dict[str, str] = field(default_factory=dict)

    def hmmf(self, waveform: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda video: self.target.forward(waveform, video)


# ----------------------------------------------------------------------------- image metrics
def _pixels(video: torch.Tensor) -> torch.Tensor:
    mean, std = _normalization(video)
    return (video * std + mean).clamp(0.0, 1.0)


def psnr(clean: np.ndarray, adversarial: np.ndarray) -> float:
    mse = float(np.mean((clean.astype(np.float64) - adversarial.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(1.0 / mse)


def ssim(clean: np.ndarray, adversarial: np.ndarray) -> float:
    """Mean SSIM over RGB channels (Gaussian window 11, sigma 1.5, data range 1), images as HWC in [0, 1]."""
    coords = torch.arange(11, dtype=torch.float64) - 5
    gauss = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] * gauss[None, :]).view(1, 1, 11, 11)
    x = torch.from_numpy(clean.astype(np.float64)).permute(2, 0, 1).unsqueeze(1)
    y = torch.from_numpy(adversarial.astype(np.float64)).permute(2, 0, 1).unsqueeze(1)
    blur = lambda t: F.conv2d(t, window)
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(value.mean())


# ----------------------------------------------------------------------------- loading and checks
def load_models(config: VizConfig, paths: VizPaths, device: torch.device, fold: int) -> Models:
    hmmf_path = paths.checkpoint(paths.hmmf_net[str(fold)], f"HMMF-Net fold {fold}")
    effb0_path = paths.checkpoint(paths.efficientnet_b0.format(fold=fold), f"EfficientNet-B0 fold {fold}")
    source_paths = {name: paths.checkpoint(paths.sources[name].format(fold=fold), f"{name} fold {fold}") for name in config.sources}

    loader_config = SimpleNamespace(batch_size=config.batch_size, splits=["test"])
    target = MultimodalTarget(fold, loader_config, paths, hmmf_path, device)
    effb0 = prepare_model(EfficientNetB0Video(), effb0_path, device)
    sources = {name: load_source(name, path, device) for name, path in source_paths.items()}

    checkpoints = {"hmmf": hmmf_path, "effb0": effb0_path, **source_paths}
    records = target.split_records("test")
    for path in checkpoints.values():
        verify_split(records, path, "test")
    video_paths = {_sample_identity_key(entry[0]): entry[1] for entry in target.entries["test"]}
    return Models(target, effb0, sources, checkpoints, video_paths)


def clean_pass(models: Models) -> Dict[str, Dict[str, np.ndarray]]:
    """Clean softmax probabilities of all five models for every test sample, keyed by sample key."""
    probs: Dict[str, Dict[str, np.ndarray]] = {}
    labels: Dict[str, int] = {}
    with torch.no_grad():
        for batch in models.target.batches("test"):
            outputs = {"hmmf": models.target.forward(batch.waveform, batch.video), "effb0": models.effb0(batch.video)}
            outputs.update({name: source(batch.video) for name, source in models.sources.items()})
            outputs = {name: logits.softmax(dim=1).cpu().numpy() for name, logits in outputs.items()}
            for index, key in enumerate(batch.keys):
                probs[key] = {name: value[index] for name, value in outputs.items()}
                labels[key] = int(batch.labels[index])
    return {"probs": probs, "labels": labels}


def check_clean_accuracy(models: Models, clean: Dict) -> Dict[str, Dict[str, float]]:
    keys = list(clean["labels"])
    labels = np.array([clean["labels"][key] for key in keys])
    report = {}
    for name, path in models.checkpoints.items():
        predicted = np.array([clean["probs"][key][name].argmax() for key in keys])
        accuracy = float((predicted == labels).mean())
        expected = _summary_accuracy(path, "test")
        report[name] = {"accuracy": accuracy, "summary_accuracy": expected}
        if expected is None:
            logger.warning("%s: no summary.csv next to %s; clean accuracy %.4f not checked.", name, path, accuracy)
        elif abs(accuracy - expected) > ACCURACY_TOLERANCE:
            raise ValueError(f"{name}: clean test accuracy {accuracy:.4f} differs from summary.csv {expected:.4f} ({path}).")
        else:
            logger.info("%s: clean test accuracy %.4f matches summary.csv %.4f.", name, accuracy, expected)
    return report


# ----------------------------------------------------------------------------- attacks
def attack_batch(models: Models, config: VizConfig, batch) -> Dict[str, Dict[str, torch.Tensor]]:
    """Run the five attacks on one batch. Returns, per attack, the normalized adversarial video and the
    softmax probabilities of the target(s) on it."""
    pgd, epsilon = config.pgd, config.epsilon
    results: Dict[str, Dict[str, torch.Tensor]] = {}
    hmmf_forward = models.hmmf(batch.waveform)

    for name, forward in (("wb_effb0", models.effb0), ("wb_hmmf", hmmf_forward)):
        adversarial, _ = pgd_attack(forward, batch.video, batch.labels, batch.keys, epsilon,
                                    pgd.alpha_ratio, pgd.steps, pgd.random_start, config.seed)
        with torch.no_grad():
            target = "effb0" if name == "wb_effb0" else "hmmf"
            results[name] = {"video": adversarial, target: forward(adversarial).softmax(dim=1)}

    for source_name, source in models.sources.items():
        adversarial, _, _ = transfer_pgd(source, batch.video, batch.labels, batch.keys, epsilon,
                                         pgd.alpha_ratio, pgd.steps, pgd.random_start, config.seed)
        with torch.no_grad():
            results[f"bb_{source_name}"] = {
                "video": adversarial,
                "effb0": models.effb0(adversarial).softmax(dim=1),
                "hmmf": hmmf_forward(adversarial).softmax(dim=1),
                "source": source(adversarial).softmax(dim=1),
            }
    return results


def attack_targets(attack: str) -> List[str]:
    if attack == "wb_effb0":
        return ["effb0"]
    if attack == "wb_hmmf":
        return ["hmmf"]
    return ["effb0", "hmmf"]


def search(models: Models, config: VizConfig, clean: Dict, max_batches: Optional[int]) -> List[Dict]:
    """Attack every test sample that all five models classify correctly; one row per sample."""
    model_names = ["hmmf", "effb0", *models.sources]
    rows = []
    attacks = ["wb_effb0", "wb_hmmf", *[f"bb_{name}" for name in models.sources]]
    for index, batch in enumerate(models.target.batches("test", max_batches)):
        results = attack_batch(models, config, batch)
        for i, key in enumerate(batch.keys):
            label = int(batch.labels[i])
            probs = clean["probs"][key]
            if not all(probs[name].argmax() == label for name in model_names):
                continue
            row = {"key": key, "label": CLASS_NAMES[label],
                   "clean_min_confidence": float(min(probs[name][label] for name in model_names))}
            fooled_bb = 0
            for attack in attacks:
                for target in attack_targets(attack):
                    predicted = int(results[attack][target][i].argmax())
                    row[f"{attack}_{target}_pred"] = CLASS_NAMES[predicted]
                    row[f"{attack}_{target}_conf"] = float(results[attack][target][i].max())
                    if attack.startswith("bb_") and predicted != label:
                        fooled_bb += 1
            row["wb_both_fooled"] = row["wb_effb0_effb0_pred"] != row["label"] and row["wb_hmmf_hmmf_pred"] != row["label"]
            row["bb_fooled_count"] = fooled_bb
            rows.append(row)
        if index % 10 == 0:
            logger.info("Search: %d batches done, %d clean-correct samples so far.", index + 1, len(rows))
    if not rows:
        raise ValueError("No test sample is classified correctly by all five models.")
    rows.sort(key=lambda row: (not row["wb_both_fooled"], -row["clean_min_confidence"], -row["bb_fooled_count"], row["key"]))
    return rows


def success_rates(rows: List[Dict], attacks: Sequence[str]) -> Dict[str, float]:
    rates = {}
    for attack in attacks:
        for target in attack_targets(attack):
            column = f"{attack}_{target}_pred"
            rates[f"{attack}->{target}"] = float(np.mean([row[column] != row["label"] for row in rows]))
    return rates


def select(rows: List[Dict], config: VizConfig) -> List[Dict]:
    if config.sample_keys:
        by_key = {row["key"]: row for row in rows}
        missing = [key for key in config.sample_keys if key not in by_key]
        if missing:
            raise ValueError(f"sample_keys not among the clean-correct test samples: {missing}")
        return [by_key[key] for key in config.sample_keys]
    chosen = []
    for label in CLASS_NAMES:
        chosen.extend([row for row in rows if row["label"] == label][: config.per_class])
    return chosen


# ----------------------------------------------------------------------------- saving
def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def _save_png(path: Path, image: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    plt.imsave(path, _to_uint8(image))


def _hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor.permute(1, 2, 0).cpu().numpy().astype(np.float64)


def original_frames(video_path: str) -> Optional[List[np.ndarray]]:
    """First and last frame of the clip at the original resolution, RGB HWC in [0, 1]. The model input is the
    same two frames resized by the decoder to 224x224 (aspect ratio not kept); this is only for display."""
    try:
        from decord import VideoReader, cpu
        reader = VideoReader(video_path, ctx=cpu(0))
        batch = reader.get_batch([0, len(reader) - 1]).asnumpy()
        return [batch[0].astype(np.float64) / 255.0, batch[1].astype(np.float64) / 255.0]
    except Exception:
        logger.exception("decord could not read %s; trying OpenCV.", video_path)
    try:
        import cv2
        capture = cv2.VideoCapture(video_path)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for index in (0, max(count - 1, 0)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                return None
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0)
        capture.release()
        return frames
    except Exception:
        logger.exception("Could not read the original frames of %s; showing the 224x224 model input.", video_path)
        return None


def _resize_delta(delta: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Bilinear resize of a [3, 224, 224] pixel-space perturbation to the original frame size, HWC."""
    resized = F.interpolate(delta.unsqueeze(0).double().cpu(), size=(height, width), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).numpy()


def _pred_text(probs: torch.Tensor) -> str:
    index = int(probs.argmax())
    return f"{CLASS_NAMES[index].capitalize()} ({float(probs[index]):.2f})"


def render_sample(models: Models, config: VizConfig, batch, out_dir: Path, rank: Optional[int],
                  search_row: Optional[Dict], fold: int) -> Dict:
    """Recompute the five attacks for one sample and save every image, the tensors and the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    epsilon = config.epsilon
    label = int(batch.labels[0])
    results = attack_batch(models, config, batch)
    with torch.no_grad():
        clean_probs = {"effb0": models.effb0(batch.video).softmax(dim=1)[0],
                       "hmmf": models.target.forward(batch.waveform, batch.video).softmax(dim=1)[0]}
    clean_pixels = _pixels(batch.video)[0]

    info = {"key": batch.keys[0], "label": CLASS_NAMES[label], "fold": fold, "epsilon": epsilon,
            "epsilon_in_gray_levels": epsilon * 255.0, "rank_in_search": rank,
            "clean": {target: _pred_text(probs) for target, probs in clean_probs.items()}, "attacks": {}}
    arrays = {"clean_pixels": clean_pixels.cpu().numpy()}

    # Display frames: the original resolution when available (the perturbation computed at the 224x224 model
    # input is resized to it), otherwise the model input itself. The *_224.png files are the model input.
    video_path = models.video_paths.get(batch.keys[0])
    originals = original_frames(video_path) if (config.display_original_size and video_path) else None
    info["video_path"] = video_path
    info["display"] = (f"original {originals[0].shape[1]}x{originals[0].shape[0]}, perturbation resized from 224x224"
                       if originals else "model input 224x224")
    for index, (frame, _, channels) in enumerate(FRAMES):
        _save_png(out_dir / f"clean_{frame}_224.png", _hwc(clean_pixels[channels]))
        _save_png(out_dir / f"clean_{frame}.png", originals[index] if originals else _hwc(clean_pixels[channels]))

    for attack, result in results.items():
        adv_pixels = _pixels(result["video"])[0]
        delta = adv_pixels - clean_pixels
        arrays[f"{attack}_adv_pixels"] = adv_pixels.cpu().numpy()
        arrays[f"{attack}_delta"] = delta.cpu().numpy()
        entry = {"linf": float(delta.abs().max()), "linf_gray_levels": float(delta.abs().max()) * 255.0,
                 "predictions": {}, "frames": {}}
        for target in attack_targets(attack):
            entry["predictions"][target] = _pred_text(result[target][0])
            search_pred = search_row[f"{attack}_{target}_pred"] if search_row else None
            if search_pred is not None and CLASS_NAMES[int(result[target][0].argmax())] != search_pred:
                logger.warning("%s %s on %s: prediction %s differs from the batched search (%s).",
                               attack, target, batch.keys[0], entry["predictions"][target], search_pred)
        if "source" in result:
            entry["predictions"]["source"] = _pred_text(result["source"][0])
        for index, (frame, _, channels) in enumerate(FRAMES):
            clean_img, adv_img = _hwc(clean_pixels[channels]), _hwc(adv_pixels[channels])
            amplified = np.clip(_hwc(delta[channels]) / (2.0 * epsilon) + 0.5, 0.0, 1.0)
            _save_png(out_dir / f"{attack}_{frame}_perturbation_224.png", amplified)
            _save_png(out_dir / f"{attack}_{frame}_adversarial_224.png", adv_img)
            if originals:
                height, width = originals[index].shape[:2]
                big_delta = _resize_delta(delta[channels], height, width)
                _save_png(out_dir / f"{attack}_{frame}_perturbation.png", np.clip(big_delta / (2.0 * epsilon) + 0.5, 0.0, 1.0))
                _save_png(out_dir / f"{attack}_{frame}_adversarial.png", np.clip(originals[index] + big_delta, 0.0, 1.0))
            else:
                _save_png(out_dir / f"{attack}_{frame}_perturbation.png", amplified)
                _save_png(out_dir / f"{attack}_{frame}_adversarial.png", adv_img)
            entry["frames"][frame] = {
                "psnr_db_float": psnr(clean_img, adv_img), "ssim_float": ssim(clean_img, adv_img),
                "png_identical_to_clean": bool(np.array_equal(_to_uint8(clean_img), _to_uint8(adv_img))),
            }
        info["attacks"][attack] = entry

    np.savez_compressed(out_dir / "tensors.npz", **arrays)
    with (out_dir / "info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)
    draw_figure(out_dir, info, list(results))
    return info


def draw_figure(out_dir: Path, info: Dict, attacks: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    exponent = np.log10(info["epsilon"])
    eps_text = rf"$\epsilon=10^{{{int(round(exponent))}}}$" if abs(exponent - round(exponent)) < 1e-9 else rf"$\epsilon={info['epsilon']:.2g}$"
    row_titles = {"wb_effb0": "White-box\nEfficientNet-B0", "wb_hmmf": "White-box\nHMMF-Net"}
    row_titles.update({f"bb_{name}": f"Black-box\nfrom {label}" for name, label in SOURCE_LABELS.items()})
    columns = [("first", "perturbation"), ("first", "adversarial"), ("last", "perturbation"), ("last", "adversarial")]
    column_titles = ["First frame: $\\delta$ (amplified)", "First frame: $x+\\delta$",
                     "Last frame: $\\delta$ (amplified)", "Last frame: $x+\\delta$"]

    rows = 1 + len(attacks)
    clean_first = imread(out_dir / "clean_first.png")
    ratio = clean_first.shape[0] / clean_first.shape[1]
    fig, axes = plt.subplots(rows, 5, figsize=(7.16, (1.3 * ratio + 0.3) * rows),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0, 1].imshow(imread(out_dir / "clean_first.png")); axes[0, 1].set_title("First frame: clean $x$", fontsize=7)
    axes[0, 3].imshow(imread(out_dir / "clean_last.png")); axes[0, 3].set_title("Last frame: clean $x$", fontsize=7)
    axes[0, 0].set_ylabel("Clean input", fontsize=7)
    axes[0, 4].text(0.0, 0.5, f"Label: {info['label'].capitalize()}\nEfficientNet-B0: {info['clean']['effb0']}\n"
                    f"HMMF-Net: {info['clean']['hmmf']}", fontsize=6.5, va="center", transform=axes[0, 4].transAxes)

    for r, attack in enumerate(attacks, start=1):
        for c, (frame, kind) in enumerate(columns):
            axes[r, c].imshow(imread(out_dir / f"{attack}_{frame}_{kind}.png"))
            if r == 1:
                axes[r, c].set_title(column_titles[c], fontsize=7)
        axes[r, 0].set_ylabel(row_titles.get(attack, attack), fontsize=7)
        predictions = info["attacks"][attack]["predictions"]
        lines = [f"{TARGET_LABELS[t]}: {info['clean'][t]} → {predictions[t]}" for t in ("effb0", "hmmf") if t in predictions]
        frame_info = info["attacks"][attack]["frames"]["first"]
        lines.append(f"PSNR {frame_info['psnr_db_float']:.1f} dB, SSIM {frame_info['ssim_float']:.4f}")
        axes[r, 4].text(0.0, 0.5, "\n".join(lines), fontsize=6.5, va="center", transform=axes[r, 4].transAxes)

    note = ("\n$\\delta$ computed at the 224$\\times$224 model input, resized to the original frame for display"
            if info.get("display", "").startswith("original") else "")
    fig.suptitle(f"PGD $L_\\infty$, {eps_text} ({info['epsilon_in_gray_levels']:.3f} gray levels), sample {info['key']}{note}",
                 fontsize=7, y=0.995)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    for extension in ("pdf", "png"):
        fig.savefig(out_dir / f"figure.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def single_batch(models: Models, key: str):
    for batch in models.target.batches("test"):
        if key in batch.keys:
            i = batch.keys.index(key)
            return SimpleNamespace(keys=[key], labels=batch.labels[i:i + 1], waveform=batch.waveform[i:i + 1],
                                   video=batch.video[i:i + 1])
    raise KeyError(key)


# ----------------------------------------------------------------------------- main
def run_fold(config: VizConfig, paths: VizPaths, device: torch.device, fold: int, main_fold: bool,
             featured: List[str], max_batches: Optional[int]) -> Dict:
    """Main fold: checks, search over the test split, `per_class` samples per class, plus featured samples.
    Other folds: checks and only the featured samples whose test split is this fold."""
    fold_dir = Path(paths.output_dir) / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    logger.info("===== fold %d | %s | featured samples: %s", fold, "search + featured" if main_fold else "featured only", featured)
    models = load_models(config, paths, device, fold)
    clean = clean_pass(models)
    accuracy = check_clean_accuracy(models, clean)
    summary: Dict = {"fold": fold, "epsilon": config.epsilon, "clean_accuracy": accuracy, "n_test": len(clean["labels"])}
    rendered = []

    rows: List[Dict] = []
    if main_fold:
        rows = search(models, config, clean, max_batches)
        attacks = ["wb_effb0", "wb_hmmf", *[f"bb_{name}" for name in config.sources]]
        rates = success_rates(rows, attacks)
        write_csv(fold_dir / "candidates.csv", rows)
        for name, rate in rates.items():
            logger.info("Attack success on the %d clean-correct samples: %-28s %.2f%%", len(rows), name, 100 * rate)
        summary.update({"n_clean_correct_all_models": len(rows), "partial_search": max_batches is not None,
                        "attack_success_rate": rates})
        rank = {row["key"]: index for index, row in enumerate(rows)}
        counters: Dict[str, int] = {}
        for row in select(rows, config):
            counters[row["label"]] = counters.get(row["label"], 0) + 1
            sample_dir = fold_dir / f"{row['label']}_{counters[row['label']]:02d}_{row['key'].replace('/', '_')}"
            info = render_sample(models, config, single_batch(models, row["key"]), sample_dir, rank[row["key"]], row, fold)
            rendered.append({"dir": sample_dir.name, "key": info["key"], "label": info["label"], "featured": False,
                             "wb_both_fooled": row["wb_both_fooled"], "bb_fooled_count": row["bb_fooled_count"]})
            logger.info("Saved %s (white-box both fooled: %s, black-box transfers fooled: %d)",
                        sample_dir, row["wb_both_fooled"], row["bb_fooled_count"])

    by_key = {row["key"]: row for row in rows}
    for key in featured:
        if key not in clean["labels"]:
            logger.error("Featured sample %s is not in the fold-%d test split; skipped.", key, fold)
            continue
        label = CLASS_NAMES[clean["labels"][key]]
        sample_dir = fold_dir / f"featured_{label}_{key.replace('/', '_')}"
        info = render_sample(models, config, single_batch(models, key), sample_dir, None, by_key.get(key), fold)
        predictions = {attack: entry["predictions"] for attack, entry in info["attacks"].items()}
        rendered.append({"dir": sample_dir.name, "key": key, "label": label, "featured": True,
                         "clean": info["clean"], "predictions": predictions})
        logger.info("Saved featured sample %s: clean %s, attacks %s", sample_dir, info["clean"], predictions)

    summary.update({"samples": rendered, "seconds": round(time.time() - started, 1),
                    "finished_at": datetime.now().isoformat(timespec="seconds")})
    with (fold_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Save PGD perturbation visualizations for the paper.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/perturbation_viz_config.json"))
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "config/perturbation_viz_paths.json"))
    parser.add_argument("--upload-config", default=str(PROJECT_ROOT / "config/robustness_upload.json"))
    parser.add_argument("--no-upload", action="store_true", help="Keep the results local.")
    parser.add_argument("--max-batches", type=int, help="Smoke test: search only the first N test batches.")
    args = parser.parse_args()

    config = VizConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    paths = VizPaths(**json.loads(Path(args.paths).read_text(encoding="utf-8")))
    unknown = sorted(set(config.sources) - set(SOURCE_LABELS))
    if unknown:
        raise ValueError(f"Unknown sources {unknown}; choose from {sorted(SOURCE_LABELS)}.")
    upload_config = load_upload_config(Path(args.upload_config))
    if args.no_upload:
        upload_config.enabled = False
    check_ready(upload_config)

    output_dir = Path(paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")])
    for noisy in ("fontTools", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    logger.info("Perturbation visualization | main fold %d | epsilon %g | sources %s | device %s | command: %s",
                config.fold, config.epsilon, config.sources, device, " ".join(sys.argv))

    featured: Dict[int, List[str]] = {}
    for sample in config.featured_samples:
        featured.setdefault(sample.fold, []).append(sample.key)
    folds = [config.fold] + sorted(set(featured) - {config.fold})
    summaries = [run_fold(config, paths, device, fold, fold == config.fold, featured.get(fold, []), args.max_batches)
                 for fold in folds]
    with (output_dir / "summary_all.json").open("w", encoding="utf-8") as file:
        json.dump({"config": config.model_dump(), "folds": summaries}, file, indent=2)
    logger.info("Done in %.1f s. Results in %s", time.time() - started, output_dir)

    if upload_config.enabled:
        kind = "partial" if args.max_batches is not None else "full"
        name = f"perturbation_viz_video_eps{config.epsilon:g}_{kind}_{_stamp()}.zip"
        zip_path = _zip([output_dir], output_dir.parent, Path(f"{output_dir}_uploads") / name)
        _upload(upload_config, zip_path, f"perturbation_viz/{name}", output_dir)


if __name__ == "__main__":
    main()
