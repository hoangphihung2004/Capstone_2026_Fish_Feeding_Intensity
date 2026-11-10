"""Visualize how white noise masks the audio (paper figure, one representative fold).

Stage 2 of the audio-noise visualization. Stage 1 (U_FFIA27K_audio/export_noise_predictions.py) saves the
AMARU test predictions under noise. This script:
1. loads the test split and HMMF-Net (clean video), checks the split and the clean accuracy against summary.csv;
2. predicts with HMMF-Net on the clean audio and at every SNR, with the same `add_awgn` and seed as the
   robustness runs (noise depends only on sample key, SNR and seed);
3. joins the AMARU predictions by sample key and ranks the samples that both models classify correctly on
   the clean input: AMARU correct at the mildest SNR and wrong at the strongest first, then by the number of
   noisy SNRs at which HMMF-Net stays correct, then by clean confidence;
4. keeps the best `per_class` samples of every class (or the `sample_keys` given in the config) and saves,
   per sample: waveform and log-mel PNGs per SNR, the noisy WAV files, info.json and the figure
   (figure.pdf / figure.png); per class an overview sheet (overview_<class>.png) to compare the samples;
5. zips the fold folder and uploads it to Hugging Face (config/robustness_upload.json) unless --no-upload.

The log-mel shown is the one the HMMF-Net frontend computes (before its BatchNorm), in dB, on one shared
colour scale per sample so that the noise floor rising with lower SNR is visible.
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from pydantic import BaseModel, Field

from evaluate_blackbox import MultimodalTarget
from robustness.blackbox import _summary_accuracy, verify_split
from robustness.core import CLASS_NAMES, add_awgn, measured_snr_db
from robustness.hub import _stamp, _upload, _zip, check_ready, load_upload_config

logger = logging.getLogger("audio_noise_viz")
ACCURACY_TOLERANCE = 0.005
MODELS = (("amaru", "AMARU"), ("hmmf", "HMMF-Net"))


class FeaturedSample(BaseModel):
    """A sample to render in addition to the ranked ones, on the fold whose test split holds it."""
    key: str
    fold: int


class AudioVizConfig(BaseModel):
    fold: int = 2
    seed: int = 42
    snr_db: List[float] = [20.0, 0.0, -20.0]
    batch_size: int = Field(32, gt=0)
    per_class: int = Field(5, ge=1)
    sample_keys: List[str] = []
    featured_samples: List[FeaturedSample] = []
    amaru_predictions: str = "../U_FFIA27K_audio/outputs/audio_noise_viz/amaru_fold{fold:02d}.json"
    figure_width_in: float = 7.16


class AudioVizPaths(BaseModel):
    train_config_path: str = "config/train_config.json"
    dataset_path: str = ""
    hmmf_net: Dict[str, str]
    output_dir: str = "outputs/audio_noise_viz"


def level_name(snr: Optional[float]) -> str:
    return "clean" if snr is None else f"{snr:g}"


def level_title(level: str) -> str:
    return "Clean" if level == "clean" else f"SNR {float(level):g} dB"


def figure_mark(text: str) -> str:
    """The serif figure font has no check/cross glyphs; use the math-text ones."""
    return text.replace("✓", r"$\checkmark$").replace("✗", r"$\times$")


def pred_text(probs: np.ndarray, label: int) -> str:
    index = int(np.argmax(probs))
    mark = "✓" if index == label else "✗"
    return f"{CLASS_NAMES[index].capitalize()} ({float(probs[index]):.2f}) {mark}"


# ----------------------------------------------------------------------------- predictions
def hmmf_predictions(target: MultimodalTarget, config: AudioVizConfig, levels: List[str]) -> Dict:
    probs: Dict[str, Dict[str, np.ndarray]] = {}
    labels: Dict[str, int] = {}
    with torch.no_grad():
        for batch in target.batches("test"):
            per_level = {"clean": target.forward(batch.waveform, batch.video)}
            for snr in config.snr_db:
                noisy = add_awgn(batch.waveform, batch.keys, snr, config.seed).to(batch.waveform.device)
                per_level[level_name(snr)] = target.forward(noisy, batch.video)
            per_level = {level: logits.softmax(dim=1).cpu().numpy() for level, logits in per_level.items()}
            for i, key in enumerate(batch.keys):
                probs[key] = {level: per_level[level][i] for level in levels}
                labels[key] = int(batch.labels[i])
    return {"probs": probs, "labels": labels}


def load_amaru(path: Path, config: AudioVizConfig, levels: List[str], fold: int) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"AMARU predictions not found: {path}. Run U_FFIA27K_audio/export_noise_predictions.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["fold"] != fold or data["seed"] != config.seed or data["levels"] != levels:
        raise ValueError(f"AMARU predictions in {path} are for fold {data['fold']}, seed {data['seed']}, levels "
                         f"{data['levels']}; this run needs fold {fold}, seed {config.seed}, levels {levels}.")
    return {
        "probs": {key: {level: np.array(value["probs"][level]) for level in levels} for key, value in data["samples"].items()},
        "labels": {key: int(value["label"]) for key, value in data["samples"].items()},
    }


def rank_samples(hmmf: Dict, amaru: Dict, levels: List[str]) -> List[Dict]:
    shared = sorted(set(hmmf["labels"]) & set(amaru["labels"]))
    mismatch = [key for key in shared if hmmf["labels"][key] != amaru["labels"][key]]
    if mismatch:
        raise ValueError(f"{len(mismatch)} samples have different labels in the audio and multimodal splits, e.g. {mismatch[:3]}")
    logger.info("Samples: %d HMMF-Net, %d AMARU, %d shared.", len(hmmf["labels"]), len(amaru["labels"]), len(shared))
    noisy = levels[1:]
    mildest, strongest = noisy[0], noisy[-1]
    rows = []
    for key in shared:
        label = hmmf["labels"][key]
        correct = {name: {level: int(np.argmax(source["probs"][key][level])) == label for level in levels}
                   for name, source in (("amaru", amaru), ("hmmf", hmmf))}
        if not (correct["amaru"]["clean"] and correct["hmmf"]["clean"]):
            continue
        row = {"key": key, "label": CLASS_NAMES[label]}
        for name, source in (("amaru", amaru), ("hmmf", hmmf)):
            for level in levels:
                row[f"{name}_{level}"] = CLASS_NAMES[int(np.argmax(source["probs"][key][level]))]
        row["amaru_story"] = correct["amaru"][mildest] and not correct["amaru"][strongest]
        row["hmmf_noisy_correct"] = sum(correct["hmmf"][level] for level in noisy)
        row["amaru_noisy_wrong"] = sum(not correct["amaru"][level] for level in noisy)
        row["clean_min_confidence"] = float(min(amaru["probs"][key]["clean"][label], hmmf["probs"][key]["clean"][label]))
        rows.append(row)
    rows.sort(key=lambda r: (not r["amaru_story"], -r["hmmf_noisy_correct"], -r["amaru_noisy_wrong"], -r["clean_min_confidence"], r["key"]))
    return rows


def accuracy_report(source: Dict, keys: List[str], levels: List[str]) -> Dict[str, Dict[str, float]]:
    labels = np.array([source["labels"][key] for key in keys])
    report = {}
    for level in levels:
        predicted = np.array([int(np.argmax(source["probs"][key][level])) for key in keys])
        report[level] = {
            "accuracy": float((predicted == labels).mean()),
            "recall": {CLASS_NAMES[c]: float((predicted[labels == c] == c).mean()) for c in range(len(CLASS_NAMES))},
            "predicted_share": {CLASS_NAMES[c]: float((predicted == c).mean()) for c in range(len(CLASS_NAMES))},
        }
    return report


def select(rows: List[Dict], config: AudioVizConfig) -> List[Dict]:
    if config.sample_keys:
        by_key = {row["key"]: row for row in rows}
        missing = [key for key in config.sample_keys if key not in by_key]
        if missing:
            raise ValueError(f"sample_keys not among the samples both models classify correctly when clean: {missing}")
        return [by_key[key] for key in config.sample_keys]
    chosen = []
    for label in CLASS_NAMES:
        chosen.extend([row for row in rows if row["label"] == label][: config.per_class])
    return chosen


# ----------------------------------------------------------------------------- rendering
def logmel_db(target: MultimodalTarget, waveform: torch.Tensor) -> np.ndarray:
    """[time, mel] log-mel in dB from the HMMF-Net frontend, before its BatchNorm."""
    frontend = target.model.frontend
    with torch.no_grad():
        spec = frontend.logmel_extractor(frontend.spectrogram_extractor(waveform.to(target.device)))
    return spec[0, 0].cpu().numpy()


def single_sample(target: MultimodalTarget, key: str):
    for batch in target.batches("test"):
        if key in batch.keys:
            i = batch.keys.index(key)
            return SimpleNamespace(keys=[key], labels=batch.labels[i:i + 1], waveform=batch.waveform[i:i + 1], video=batch.video[i:i + 1])
    raise KeyError(key)


def _style(plt) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})


def render_sample(target: MultimodalTarget, config: AudioVizConfig, levels: List[str], row: Dict, amaru: Dict,
                  hmmf: Dict, sample_rate: int, out_dir: Path, fold: int) -> Dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    _style(plt)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = single_sample(target, row["key"])
    label = int(sample.labels[0])
    clean = sample.waveform.detach().cpu()
    waves = {"clean": clean}
    for snr in config.snr_db:
        waves[level_name(snr)] = add_awgn(clean, sample.keys, snr, config.seed)

    with torch.no_grad():  # recompute HMMF-Net on this single sample to confirm the batched prediction
        for level, wave in waves.items():
            single = target.forward(wave.to(target.device), sample.video).softmax(dim=1)[0].cpu().numpy()
            if int(single.argmax()) != int(np.argmax(hmmf["probs"][row["key"]][level])):
                logger.warning("HMMF-Net %s on %s: single-sample prediction differs from the batched one.", level, row["key"])

    mels = {level: logmel_db(target, wave) for level, wave in waves.items()}
    vmax = max(float(np.percentile(m, 99.5)) for m in mels.values())
    vmin = vmax - 80.0
    amp = max(float(w.abs().max()) for w in waves.values())
    seconds = clean.shape[1] / sample_rate
    t = np.arange(clean.shape[1]) / sample_rate

    info = {"key": row["key"], "label": CLASS_NAMES[label], "fold": fold, "seed": config.seed,
            "sample_rate": sample_rate, "duration_s": seconds, "levels": {}}
    for level, wave in waves.items():
        snr = None if level == "clean" else float(measured_snr_db(clean, wave)[0])
        info["levels"][level] = {
            "measured_snr_db": snr,
            "amaru": pred_text(amaru["probs"][row["key"]][level], label),
            "hmmf": pred_text(hmmf["probs"][row["key"]][level], label),
            "amaru_probs": [round(float(p), 4) for p in amaru["probs"][row["key"]][level]],
            "hmmf_probs": [round(float(p), 4) for p in hmmf["probs"][row["key"]][level]],
        }
        sf.write(out_dir / f"audio_{level}.wav", wave[0].numpy(), sample_rate)
        plt.imsave(out_dir / f"logmel_{level}.png", mels[level].T[::-1], cmap="magma", vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(figsize=(2.0, 0.6))
        ax.plot(t, wave[0].numpy(), lw=0.3, color="#2a78d6")
        ax.set_ylim(-amp, amp); ax.set_xlim(0, seconds); ax.axis("off")
        fig.savefig(out_dir / f"waveform_{level}.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    (out_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    n = len(levels)
    fig, axes = plt.subplots(2, n, figsize=(config.figure_width_in, 2.35), gridspec_kw={"height_ratios": [0.55, 1.6]},
                             sharex="col")
    for c, level in enumerate(levels):
        ax_w, ax_m = axes[0, c], axes[1, c]
        ax_w.plot(t, waves[level][0].numpy(), lw=0.3, color="#2a78d6")
        ax_w.set_ylim(-amp, amp); ax_w.set_xlim(0, seconds)
        ax_w.set_title(level_title(level), fontsize=7.5)
        ax_w.set_yticks([]); ax_w.tick_params(labelbottom=False, length=0)
        image = ax_m.imshow(mels[level].T, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                            extent=(0, seconds, 0, mels[level].shape[1]))
        ax_m.set_xlabel("Time (s)", fontsize=6.5)
        ax_m.set_xticks(np.arange(0.0, seconds - 0.05, 0.5 if seconds > 1.2 else 0.25))
        ax_m.set_yticks([0, 64, 128] if mels[level].shape[1] >= 128 else [])
        if c == 0:
            ax_m.set_ylabel("Mel bin", fontsize=6.5)
            ax_w.set_ylabel("Wave", fontsize=6.5)
        else:
            ax_m.set_yticklabels([])
        ax_m.tick_params(labelsize=6, length=2)
        entry = info["levels"][level]
        ax_m.text(0.5, -0.42, f"AMARU: {figure_mark(entry['amaru'])}\nHMMF-Net: {figure_mark(entry['hmmf'])}",
                  transform=ax_m.transAxes,
                  ha="center", va="top", fontsize=6.5)
        for spine in (*ax_w.spines.values(),):
            spine.set_visible(False)
    fig.subplots_adjust(bottom=0.27, hspace=0.12, wspace=0.08, right=0.9)
    last = axes[1, -1].get_position()
    bar = fig.colorbar(image, cax=fig.add_axes((last.x1 + 0.012, last.y0, 0.01, last.height)))
    bar.set_label("dB", fontsize=6.5); bar.ax.tick_params(labelsize=6, length=2)
    fig.suptitle(f"Label: {CLASS_NAMES[label].capitalize()}  |  sample {row['key']}", fontsize=7, y=1.02)
    for extension in ("pdf", "png"):
        fig.savefig(out_dir / f"figure.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return info


def draw_overview(fold_dir: Path, rendered: List[Dict], levels: List[str], class_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    _style(plt)
    items = [item for item in rendered if item["label"] == class_name and not item.get("featured")]
    if not items:
        return
    fig, axes = plt.subplots(len(items), len(levels), figsize=(1.9 * len(levels), 1.55 * len(items)), squeeze=False)
    for r, item in enumerate(items):
        for c, level in enumerate(levels):
            ax = axes[r, c]
            ax.imshow(imread(fold_dir / item["dir"] / f"logmel_{level}.png"), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            entry = item["info"]["levels"][level]
            ax.set_title(f"{level_title(level)}\nA: {figure_mark(entry['amaru'])} | H: {figure_mark(entry['hmmf'])}", fontsize=5.5)
            if c == 0:
                ax.set_ylabel(f"#{r + 1}\n{item['dir'][-28:]}", fontsize=5)
    fig.suptitle(f"Class {class_name.capitalize()}: candidates (A = AMARU, H = HMMF-Net)", fontsize=7)
    fig.tight_layout()
    fig.savefig(fold_dir / f"overview_{class_name}.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run_fold(config: AudioVizConfig, paths: AudioVizPaths, device: torch.device, fold: int, main_fold: bool,
             featured: List[str], levels: List[str]) -> Dict:
    """Main fold: checks, ranking, `per_class` samples per class with overview sheets, plus featured samples.
    Other folds: checks and only the featured samples whose test split is this fold."""
    fold_dir = Path(paths.output_dir) / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    logger.info("===== fold %d | %s | featured samples: %s", fold, "ranking + featured" if main_fold else "featured only", featured)

    amaru = load_amaru(Path(config.amaru_predictions.format(fold=fold)), config, levels, fold)
    if str(fold) not in paths.hmmf_net:
        raise ValueError(f"No HMMF-Net checkpoint configured for fold {fold}.")
    checkpoint = Path(paths.hmmf_net[str(fold)])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HMMF-Net checkpoint not found: {checkpoint}")
    target = MultimodalTarget(fold, SimpleNamespace(batch_size=config.batch_size, splits=["test"]), paths, checkpoint, device)
    verify_split(target.split_records("test"), checkpoint, "test")
    hmmf = hmmf_predictions(target, config, levels)

    hmmf_report = accuracy_report(hmmf, sorted(hmmf["labels"]), levels)
    expected = _summary_accuracy(checkpoint, "test")
    clean_accuracy = hmmf_report["clean"]["accuracy"]
    if expected is not None and abs(clean_accuracy - expected) > ACCURACY_TOLERANCE:
        raise ValueError(f"HMMF-Net clean test accuracy {clean_accuracy:.4f} differs from summary.csv {expected:.4f}.")
    logger.info("HMMF-Net clean test accuracy %.4f (summary.csv %s).", clean_accuracy, expected)
    amaru_report = accuracy_report(amaru, sorted(amaru["labels"]), levels)
    for level in levels:
        logger.info("%-6s accuracy AMARU %.4f | HMMF-Net %.4f | share predicted Strong: AMARU %.3f, HMMF-Net %.3f", level,
                    amaru_report[level]["accuracy"], hmmf_report[level]["accuracy"],
                    amaru_report[level]["predicted_share"]["strong"], hmmf_report[level]["predicted_share"]["strong"])

    sample_rate = target.model.frontend.config.sample_rate
    rows = rank_samples(hmmf, amaru, levels)
    rendered = []
    summary: Dict = {"fold": fold, "levels": levels, "hmmf_net": hmmf_report, "amaru": amaru_report}
    if main_fold:
        with (fold_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        counters: Dict[str, int] = {}
        for row in select(rows, config):
            counters[row["label"]] = counters.get(row["label"], 0) + 1
            name = f"{row['label']}_{counters[row['label']]:02d}_{row['key'].replace('/', '_')}"
            info = render_sample(target, config, levels, row, amaru, hmmf, sample_rate, fold_dir / name, fold)
            rendered.append({"dir": name, "key": row["key"], "label": row["label"], "featured": False,
                             "amaru_story": row["amaru_story"], "hmmf_noisy_correct": row["hmmf_noisy_correct"], "info": info})
            logger.info("Saved %s", fold_dir / name)
        for class_name in CLASS_NAMES:
            draw_overview(fold_dir, rendered, levels, class_name)
        summary.update({"n_ranked": len(rows), "n_amaru_story": int(sum(r["amaru_story"] for r in rows))})

    shared = set(hmmf["labels"]) & set(amaru["labels"])
    for key in featured:
        if key not in shared:
            logger.error("Featured sample %s is not in the fold-%d test split of both models; skipped.", key, fold)
            continue
        label = CLASS_NAMES[hmmf["labels"][key]]
        row = {"key": key, "label": label}
        name = f"featured_{label}_{key.replace('/', '_')}"
        info = render_sample(target, config, levels, row, amaru, hmmf, sample_rate, fold_dir / name, fold)
        rendered.append({"dir": name, "key": key, "label": label, "featured": True, "info": info})
        logger.info("Saved featured sample %s: %s", fold_dir / name,
                    {level: (entry["amaru"], entry["hmmf"]) for level, entry in info["levels"].items()})

    summary.update({"samples": [{k: v for k, v in item.items() if k != "info"} for item in rendered],
                    "seconds": round(time.time() - started, 1), "finished_at": datetime.now().isoformat(timespec="seconds")})
    (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize white noise on the audio stream.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/audio_noise_viz_config.json"))
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "config/perturbation_viz_paths.json"))
    parser.add_argument("--upload-config", default=str(PROJECT_ROOT / "config/robustness_upload.json"))
    parser.add_argument("--no-upload", action="store_true", help="Keep the results local.")
    args = parser.parse_args()

    config = AudioVizConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    paths_raw = json.loads(Path(args.paths).read_text(encoding="utf-8"))
    paths = AudioVizPaths(**{k: v for k, v in paths_raw.items() if k in AudioVizPaths.model_fields and k != "output_dir"})
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
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    levels = ["clean"] + [level_name(snr) for snr in config.snr_db]
    logger.info("Audio-noise visualization | main fold %d | levels %s | seed %d | device %s", config.fold, levels, config.seed, device)

    featured: Dict[int, List[str]] = {}
    for sample in config.featured_samples:
        featured.setdefault(sample.fold, []).append(sample.key)
    folds = [config.fold] + sorted(set(featured) - {config.fold})
    summaries = [run_fold(config, paths, device, fold, fold == config.fold, featured.get(fold, []), levels) for fold in folds]
    (output_dir / "summary_all.json").write_text(json.dumps({"config": config.model_dump(), "folds": summaries}, indent=2),
                                                 encoding="utf-8")
    logger.info("Done in %.1f s. Results in %s", time.time() - started, output_dir)

    if upload_config.enabled:
        name = f"audio_noise_viz_{_stamp()}.zip"
        zip_path = _zip([output_dir], output_dir.parent, Path(f"{output_dir}_uploads") / name)
        _upload(upload_config, zip_path, f"perturbation_viz/{name}", output_dir)


if __name__ == "__main__":
    main()
