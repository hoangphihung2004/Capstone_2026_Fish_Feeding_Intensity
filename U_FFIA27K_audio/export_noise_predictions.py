"""Export AMARU test predictions under Gaussian white noise for the audio-noise figure (one fold).

Stage 1 of the audio-noise visualization (stage 2 is U_FFIA27K_multimodal/visualize_audio_noise.py, which
adds HMMF-Net, picks the samples and draws the figures). It uses exactly the robustness setup of this branch:
the same AudioAdapter (data loader, AudioModel, checkpoint) and the same `add_awgn` (noise depends only on
sample key, SNR and seed), so the predictions are the ones behind the audio-noise table.

Output: one JSON with, for every test sample, the label and the softmax probabilities on the clean input and
at each SNR. The clean test accuracy is checked against summary.csv next to the checkpoint.
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from evaluate_robustness import AudioAdapter
from robustness.core import PathsConfig, add_awgn, load_json_model, verify_test_split

logger = logging.getLogger("audio_noise_export")
ACCURACY_TOLERANCE = 0.005  # clean test accuracy must match summary.csv within 0.5 percentage points


def summary_accuracy(checkpoint_path: Path):
    path = checkpoint_path.parent / "summary.csv"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    value = rows[-1].get("Accuracy Test") if rows else None
    return float(value) if value not in (None, "") else None


def main() -> None:
    parser = argparse.ArgumentParser(description="AMARU test predictions under white noise (one fold).")
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "config/robustness_paths.json"))
    parser.add_argument("--folds", type=int, nargs="+", default=[2])
    parser.add_argument("--snr", type=float, nargs="+", default=[20.0, 0.0, -20.0])
    parser.add_argument("--seed", type=int, default=42, help="Same seed as config/robustness_config.json.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs/audio_noise_viz/amaru_fold{fold:02d}.json"),
                        help="Output file; {fold} is replaced by the fold number.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_json_model(PathsConfig, Path(args.paths))
    for fold in args.folds:
        export_fold(args, paths, fold, device)


def export_fold(args, paths, fold: int, device: torch.device) -> None:
    checkpoint = paths.checkpoint_for(fold)
    adapter = AudioAdapter(fold, SimpleNamespace(batch_size=args.batch_size), paths, checkpoint, device)
    verify_test_split(adapter, checkpoint)

    levels = ["clean"] + [f"{snr:g}" for snr in args.snr]
    keys, labels = [], []
    probs = {level: [] for level in levels}
    with torch.no_grad():
        for batch in adapter.batches():
            keys.extend(batch.keys)
            labels.extend(batch.labels.cpu().tolist())
            probs["clean"].append(adapter.forward(batch.waveform, None).softmax(dim=1).cpu().numpy())
            for snr in args.snr:
                noisy = add_awgn(batch.waveform, batch.keys, snr, args.seed).to(device)
                probs[f"{snr:g}"].append(adapter.forward(noisy, None).softmax(dim=1).cpu().numpy())
    probs = {level: np.concatenate(values) for level, values in probs.items()}
    labels_np = np.array(labels)

    accuracy = {level: float((value.argmax(axis=1) == labels_np).mean()) for level, value in probs.items()}
    for level, value in accuracy.items():
        logger.info("AMARU fold %d %-6s accuracy %.4f", fold, level, value)
    expected = summary_accuracy(checkpoint)
    if expected is None:
        logger.warning("No summary.csv next to %s; clean accuracy not checked.", checkpoint)
    elif abs(accuracy["clean"] - expected) > ACCURACY_TOLERANCE:
        raise ValueError(f"AMARU clean test accuracy {accuracy['clean']:.4f} differs from summary.csv {expected:.4f}.")
    else:
        logger.info("AMARU clean test accuracy %.4f matches summary.csv %.4f.", accuracy["clean"], expected)

    out = Path(args.out.format(fold=fold))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "amaru", "fold": fold, "seed": args.seed, "levels": levels, "checkpoint": str(checkpoint),
        "accuracy": accuracy, "created_at": datetime.now().isoformat(timespec="seconds"),
        "samples": {key: {"label": int(label), "probs": {level: probs[level][i].round(6).tolist() for level in levels}}
                    for i, (key, label) in enumerate(zip(keys, labels))},
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("Saved %d AMARU predictions to %s", len(keys), out)


if __name__ == "__main__":
    main()
