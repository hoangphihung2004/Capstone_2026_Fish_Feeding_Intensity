"""White-box PGD perturbation figures for the paper sample: EfficientNet-B0, ConvNeXt-T, Swin-T and HMMF-Net.

`visualize_perturbation.py` attacks EfficientNet-B0 and HMMF-Net white-box and uses ConvNeXt-T / Swin-T only as
black-box sources (`transfer_pgd`: final iterate, every sample attacked). This script renders the white-box
attack on ConvNeXt-T and Swin-T with exactly the same `pgd_attack` (and settings) as the white-box robustness
runs and as the EfficientNet-B0 / HMMF-Net rows, for the samples chosen for the paper only. EfficientNet-B0 and
HMMF-Net are attacked again in the same run, so all four rows come from one run and can be checked against the
figures made earlier (same seed, same key-seeded random start).

    python visualize_perturbation_whitebox.py                               # the paper sample (fold 1)
    python visualize_perturbation_whitebox.py --sample 2022_6_21/pm_70/medium/21_sample_7:1

Output: <output_dir>_whitebox/fold_XX/<label>_<key>/ with, per attack, <attack>_<first|last>_<perturbation|adversarial>.png
(attacks wb_effb0, wb_convnext, wb_swin, wb_hmmf), clean_<first|last>.png, info.json, tensors.npz and figure.pdf/png.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from robustness.core import CLASS_NAMES, pgd_attack
from robustness.hub import _stamp, _upload, _zip, check_ready, load_upload_config
from visualize_perturbation import (
    FRAMES, Models, VizConfig, VizPaths, _hwc, _pixels, _pred_text, _resize_delta, _save_png, _to_uint8,
    check_clean_accuracy, clean_pass, load_models, original_frames, psnr, single_batch, ssim,
)

logger = logging.getLogger("perturbation_viz_whitebox")

PAPER_SAMPLES = ["2022_6_21/pm_70/medium/21_sample_7:1"]  # selected_for_paper/video
SOURCES = ["ConvNeXtTiny", "SwinTiny"]
# attack id -> (model id in clean_pass / checkpoints, display label)
ATTACKS: Dict[str, Tuple[str, str]] = {
    "wb_effb0": ("effb0", "EfficientNet-B0"),
    "wb_convnext": ("ConvNeXtTiny", "ConvNeXt-T"),
    "wb_swin": ("SwinTiny", "Swin-T"),
    "wb_hmmf": ("hmmf", "HMMF-Net"),
}


def forwards(models: Models, waveform: torch.Tensor) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {"effb0": models.effb0, "ConvNeXtTiny": models.sources["ConvNeXtTiny"],
            "SwinTiny": models.sources["SwinTiny"], "hmmf": models.hmmf(waveform)}


def render(models: Models, config: VizConfig, key: str, fold: int, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = single_batch(models, key)
    label, epsilon, pgd = int(batch.labels[0]), config.epsilon, config.pgd
    nets = forwards(models, batch.waveform)
    clean_pixels = _pixels(batch.video)[0]
    with torch.no_grad():
        clean_probs = {name: net(batch.video).softmax(dim=1)[0] for name, net in nets.items()}
    for name, probs in clean_probs.items():
        if int(probs.argmax()) != label:
            logger.warning("%s misclassifies %s on the clean input; pgd_attack leaves it unattacked.", name, key)

    info = {"key": key, "label": CLASS_NAMES[label], "fold": fold, "epsilon": epsilon, "seed": config.seed,
            "pgd": pgd.model_dump(), "attack_function": "robustness.core.pgd_attack (same as the white-box runs)",
            "clean": {name: _pred_text(probs) for name, probs in clean_probs.items()}, "attacks": {}}
    video_path = models.video_paths.get(key)
    originals = original_frames(video_path) if (config.display_original_size and video_path) else None
    info["video_path"] = video_path
    info["display"] = (f"original {originals[0].shape[1]}x{originals[0].shape[0]}, perturbation resized from 224x224"
                       if originals else "model input 224x224")
    arrays = {"clean_pixels": clean_pixels.cpu().numpy()}
    for index, (frame, _, channels) in enumerate(FRAMES):
        _save_png(out_dir / f"clean_{frame}_224.png", _hwc(clean_pixels[channels]))
        _save_png(out_dir / f"clean_{frame}.png", originals[index] if originals else _hwc(clean_pixels[channels]))

    for attack, (name, _) in ATTACKS.items():
        net = nets[name]
        adversarial, losses = pgd_attack(net, batch.video, batch.labels, batch.keys, epsilon,
                                         pgd.alpha_ratio, pgd.steps, pgd.random_start, config.seed)
        with torch.no_grad():
            probs = net(adversarial).softmax(dim=1)[0]
        adv_pixels = _pixels(adversarial)[0]
        delta = adv_pixels - clean_pixels
        arrays[f"{attack}_adv_pixels"] = adv_pixels.cpu().numpy()
        arrays[f"{attack}_delta"] = delta.cpu().numpy()
        entry = {"model": name, "prediction": _pred_text(probs), "fooled": int(probs.argmax()) != label,
                 "linf": float(delta.abs().max()), "linf_gray_levels": float(delta.abs().max()) * 255.0,
                 "pgd_losses": [round(value, 6) for value in losses], "frames": {}}
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
        logger.info("%s on %s: clean %s -> %s (L-inf %.3g)", attack, key, info["clean"][name], entry["prediction"], entry["linf"])

    np.savez_compressed(out_dir / "tensors.npz", **arrays)
    with (out_dir / "info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, ensure_ascii=False)
    draw_figure(out_dir, info)
    return info


def draw_figure(out_dir: Path, info: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    columns = [("first", "perturbation"), ("first", "adversarial"), ("last", "perturbation"), ("last", "adversarial")]
    titles = ["First frame: $\\delta$ (amplified)", "First frame: $x+\\delta$", "Last frame: $\\delta$ (amplified)", "Last frame: $x+\\delta$"]
    clean_first = imread(out_dir / "clean_first.png")
    ratio = clean_first.shape[0] / clean_first.shape[1]
    rows = 1 + len(ATTACKS)
    fig, axes = plt.subplots(rows, 5, figsize=(7.16, (1.3 * ratio + 0.3) * rows), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0, 1].imshow(clean_first); axes[0, 1].set_title("First frame: clean $x$", fontsize=7)
    axes[0, 3].imshow(imread(out_dir / "clean_last.png")); axes[0, 3].set_title("Last frame: clean $x$", fontsize=7)
    axes[0, 0].set_ylabel("Clean input", fontsize=7)
    axes[0, 4].text(0.0, 0.5, f"Label: {info['label'].capitalize()}", fontsize=6.5, va="center", transform=axes[0, 4].transAxes)
    for r, (attack, (name, display)) in enumerate(ATTACKS.items(), start=1):
        for c, (frame, kind) in enumerate(columns):
            axes[r, c].imshow(imread(out_dir / f"{attack}_{frame}_{kind}.png"))
            if r == 1:
                axes[r, c].set_title(titles[c], fontsize=7)
        axes[r, 0].set_ylabel(f"White-box\n{display}", fontsize=7)
        entry = info["attacks"][attack]
        frame_info = entry["frames"]["first"]
        axes[r, 4].text(0.0, 0.5, f"{display}: {info['clean'][name]} → {entry['prediction']}\n"
                        f"PSNR {frame_info['psnr_db_float']:.1f} dB, SSIM {frame_info['ssim_float']:.4f}",
                        fontsize=6.5, va="center", transform=axes[r, 4].transAxes)
    fig.suptitle(f"White-box PGD $L_\\infty$, $\\epsilon$={info['epsilon']:g}, sample {info['key']}", fontsize=7, y=0.995)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    for extension in ("pdf", "png"):
        fig.savefig(out_dir / f"figure.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_sample(text: str) -> Tuple[str, int]:
    key, _, fold = text.rpartition(":")
    if not key or not fold.isdigit():
        raise argparse.ArgumentTypeError(f"Expected <sample key>:<fold>, got {text!r}")
    return key, int(fold)


def main() -> None:
    parser = argparse.ArgumentParser(description="White-box PGD figures (EfficientNet-B0, ConvNeXt-T, Swin-T, HMMF-Net) for chosen samples.")
    parser.add_argument("--sample", action="append", type=parse_sample,
                        help=f"<sample key>:<fold>, repeatable (default: the paper sample {PAPER_SAMPLES}).")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/perturbation_viz_config.json"))
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "config/perturbation_viz_paths.json"))
    parser.add_argument("--upload-config", default=str(PROJECT_ROOT / "config/robustness_upload.json"))
    parser.add_argument("--no-upload", action="store_true", help="Keep the results local.")
    args = parser.parse_args()

    samples = args.sample or [parse_sample(text) for text in PAPER_SAMPLES]
    config = VizConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    config.sources = SOURCES
    paths = VizPaths(**json.loads(Path(args.paths).read_text(encoding="utf-8")))
    upload_config = load_upload_config(Path(args.upload_config))
    if args.no_upload:
        upload_config.enabled = False
    check_ready(upload_config)

    output_dir = Path(f"{paths.output_dir}_whitebox")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")])
    for noisy in ("fontTools", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    logger.info("White-box viz | epsilon %g | pgd %s | samples %s | device %s", config.epsilon, config.pgd.model_dump(), samples, device)

    by_fold: Dict[int, List[str]] = {}
    for key, fold in samples:
        by_fold.setdefault(fold, []).append(key)
    results = []
    for fold, keys in sorted(by_fold.items()):
        models = load_models(config, paths, device, fold)
        check_clean_accuracy(models, clean_pass(models))
        for key in keys:
            label = CLASS_NAMES[int(single_batch(models, key).labels[0])]
            out_dir = output_dir / f"fold_{fold:02d}" / f"{label}_{key.replace('/', '_')}"
            info = render(models, config, key, fold, out_dir)
            results.append({"dir": str(out_dir), "key": key, "fold": fold, "clean": info["clean"],
                            "attacks": {attack: entry["prediction"] for attack, entry in info["attacks"].items()}})
            logger.info("Saved %s", out_dir)
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"config": config.model_dump(), "samples": results}, file, indent=2, ensure_ascii=False)
    logger.info("Done in %.1f s. Results in %s", time.time() - started, output_dir)

    if upload_config.enabled:
        name = f"perturbation_viz_video_whitebox_eps{config.epsilon:g}_{_stamp()}.zip"
        zip_path = _zip([output_dir], output_dir.parent, Path(f"{output_dir}_uploads") / name)
        _upload(upload_config, zip_path, f"perturbation_viz/{name}", output_dir)


if __name__ == "__main__":
    main()
