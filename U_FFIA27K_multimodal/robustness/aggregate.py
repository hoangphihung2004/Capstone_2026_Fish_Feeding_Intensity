"""Combine fold results from the three branches into comparison tables and figures."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
MULTIMODAL_COLOR = "#2a78d6"
SINGLE_COLOR = "#eb6834"
REFERENCE_COLOR = "#52514e"
REPORT_EPSILON = 8.0 / 255.0

COMPARISONS = {
    "audio_noise": {
        "single": "amaru", "single_label": "AMARU", "reference": "efficientnet_b0",
        "reference_label": "Clean EfficientNet-B0", "unit": "SNR (dB)", "descending": True,
        "title": "Audio noise with clean video",
    },
    "video_pgd": {
        "single": "efficientnet_b0", "single_label": "EfficientNet-B0", "reference": "amaru",
        "reference_label": "Clean AMARU", "unit": "PGD ε", "descending": False,
        "title": "Video PGD with clean audio",
    },
}
METRICS = ("accuracy", "f1_macro", "f1_weighted", "mAP")


def read_results(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(root.rglob("robustness.csv")):
        if "archive" in path.parts:
            continue
        with path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row["_dir"] = str(path.parent)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No robustness.csv under {root}")
    return rows


def validate(rows: List[Dict[str, str]]) -> List[int]:
    hashes = {row["config_hash"] for row in rows}
    if len(hashes) != 1:
        raise ValueError(f"Results come from different robustness configs: {sorted(hashes)}")
    partial = {row["_dir"] for row in rows if row["partial"] != "0"}
    if partial:
        raise ValueError(f"Partial smoke-test results cannot be aggregated: {sorted(partial)}")
    folds_by_model: Dict[str, set] = {}
    for row in rows:
        folds_by_model.setdefault(row["model"], set()).add(int(row["fold"]))
    fold_sets = {tuple(sorted(folds)) for folds in folds_by_model.values()}
    if len(fold_sets) != 1:
        raise ValueError(f"Models cover different folds: {folds_by_model}")
    if set(folds_by_model) != {"amaru", "efficientnet_b0", "multimodal"}:
        raise ValueError(f"Expected all three models, found: {sorted(folds_by_model)}")
    for scenario, spec in COMPARISONS.items():
        level_sets = {
            model: {row["level"] for row in rows if row["model"] == model and row["scenario"] == scenario}
            for model in ("multimodal", spec["single"])
        }
        if len({frozenset(levels) for levels in level_sets.values()}) != 1:
            raise ValueError(f"Incomplete results: {scenario} levels differ between models. Resume unfinished runs.")
    return list(fold_sets.pop())


def lookup(rows: List[Dict[str, str]], model: str, scenario: str, level: str, field: str) -> Dict[int, float]:
    return {
        int(row["fold"]): float(row[field])
        for row in rows
        if row["model"] == model and row["scenario"] == scenario and row["level"] == level
    }


def ordered_levels(rows: List[Dict[str, str]], scenario: str, descending: bool) -> List[str]:
    levels = {row["level"] for row in rows if row["scenario"] == scenario}
    return sorted(levels, key=float, reverse=descending)


def mean_std(values: List[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def correct_by_key(fold_dir: str, scenario: str, level: str) -> Dict[str, bool]:
    with np.load(Path(fold_dir) / "predictions" / f"{scenario}_{level}.npz") as data:
        correct = data["probabilities"].argmax(axis=1) == data["labels"]
        return dict(zip(data["keys"].tolist(), correct.tolist()))


def mcnemar(rows: List[Dict[str, str]], single: str, scenario: str, level: str) -> Tuple[int, int, float]:
    """Exact pooled test; test folds are disjoint, so each sample appears once."""
    only_multimodal = only_single = 0
    dirs: Dict[Tuple[str, int], str] = {
        (row["model"], int(row["fold"])): row["_dir"] for row in rows
    }
    for (model, fold), fold_dir in dirs.items():
        if model != "multimodal":
            continue
        multimodal = correct_by_key(fold_dir, scenario, level)
        other = correct_by_key(dirs[(single, fold)], scenario, level)
        if multimodal.keys() != other.keys():
            raise ValueError(f"Fold {fold} test samples differ between multimodal and {single}.")
        for key, is_correct in multimodal.items():
            only_multimodal += int(is_correct and not other[key])
            only_single += int(other[key] and not is_correct)
    discordant = only_multimodal + only_single
    p_value = float(stats.binomtest(only_multimodal, discordant, 0.5).pvalue) if discordant else 1.0
    return only_multimodal, only_single, p_value


def epsilon_at_half_accuracy(epsilons: np.ndarray, accuracy: np.ndarray, clean: float) -> Optional[float]:
    """Interpolate on log(epsilon) the first level where accuracy falls to half of clean."""
    target = clean / 2.0
    for index, value in enumerate(accuracy):
        if value > target:
            continue
        if index == 0:
            return float(epsilons[0])
        previous = accuracy[index - 1]
        position = (previous - target) / (previous - value)
        logs = np.log10(epsilons[index - 1 : index + 1])
        return float(10.0 ** (logs[0] + position * (logs[1] - logs[0])))
    return None


def curve_summary(scenario: str, levels: List[str], clean: Dict[int, float], accuracy_by_level: Dict[str, Dict[int, float]], folds: List[int]) -> Dict:
    areas: List[float] = []
    half_points: List[float] = []
    for fold in folds:
        values = np.array([accuracy_by_level[level][fold] for level in levels])
        if scenario == "video_pgd":
            epsilons = np.array([float(level) for level in levels])
            if epsilons.size > 1:
                logs = np.log10(epsilons)
                areas.append(float(((values[:-1] + values[1:]) / 2.0 * np.diff(logs)).sum() / (logs[-1] - logs[0])))
                half = epsilon_at_half_accuracy(epsilons, values, clean[fold])
                if half is not None:
                    half_points.append(half)
        else:
            path = np.concatenate(([clean[fold]], values))
            areas.append(float(((path[:-1] + path[1:]) / 2.0).mean()))
    summary = {"area_mean": "", "area_std": "", "epsilon50_geometric_mean": "", "folds_reaching_epsilon50": ""}
    if areas:
        summary["area_mean"], summary["area_std"] = mean_std(areas)
    if scenario == "video_pgd":
        summary["folds_reaching_epsilon50"] = len(half_points)
        if half_points:
            summary["epsilon50_geometric_mean"] = float(10.0 ** np.mean(np.log10(half_points)))
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows: List[Dict[str, str]], folds: List[int], output: Path) -> Dict[str, Dict]:
    level_rows: List[Dict] = []
    test_rows: List[Dict] = []
    area_rows: List[Dict] = []
    plot_data: Dict[str, Dict] = {}
    for scenario, spec in COMPARISONS.items():
        levels = ordered_levels(rows, scenario, spec["descending"])
        models = {"multimodal": "multimodal", "single": spec["single"]}
        clean = {key: lookup(rows, name, "clean", "clean", "accuracy") for key, name in models.items()}
        curves = {key: {"clean": mean_std(list(clean[key].values())), "points": []} for key in models}
        reference = mean_std(list(lookup(rows, spec["reference"], "clean", "clean", "accuracy").values()))[0]
        accuracy_by_level: Dict[str, Dict[str, Dict[int, float]]] = {key: {} for key in models}

        for level in levels:
            metric_values = {
                key: {metric: lookup(rows, name, scenario, level, metric) for metric in METRICS}
                for key, name in models.items()
            }
            drops = {
                key: {fold: clean[key][fold] - metric_values[key]["accuracy"][fold] for fold in folds}
                for key in models
            }
            for key, name in models.items():
                accuracy_by_level[key][level] = metric_values[key]["accuracy"]
                row = {"comparison": scenario, "model": name, "unit": spec["unit"], "level": level}
                for metric in METRICS:
                    mean, std = mean_std(list(metric_values[key][metric].values()))
                    row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
                row["drop_mean"], row["drop_std"] = mean_std(list(drops[key].values()))
                row["reference_clean_accuracy"] = reference
                row["below_reference"] = bool(row["accuracy_mean"] < reference) if key == "multimodal" else ""
                level_rows.append(row)
                curves[key]["points"].append((row["accuracy_mean"], row["accuracy_std"]))
            differences = np.array([drops["single"][fold] - drops["multimodal"][fold] for fold in folds])
            margin = float(stats.t.ppf(0.975, differences.size - 1) * stats.sem(differences)) if differences.size > 1 else 0.0
            only_multimodal, only_single, p_value = mcnemar(rows, spec["single"], scenario, level)
            test_rows.append({
                "comparison": scenario, "level": level, "drop_single_minus_multimodal_mean": float(differences.mean()),
                "ci95_low": float(differences.mean() - margin), "ci95_high": float(differences.mean() + margin),
                "only_multimodal_correct": only_multimodal, "only_single_correct": only_single,
                "mcnemar_p": p_value,
            })

        for key, name in models.items():
            area_rows.append({
                "comparison": scenario, "model": name, "clean_accuracy_mean": curves[key]["clean"][0],
                "clean_accuracy_std": curves[key]["clean"][1],
                **curve_summary(scenario, levels, clean[key], accuracy_by_level[key], folds),
            })
        plot_data[scenario] = {"levels": levels, "curves": curves, "reference": reference}

    merged = [{key: value for key, value in row.items() if key != "_dir"} for row in rows]
    write_csv(output / "all_results.csv", merged)
    write_csv(output / "summary_by_level.csv", level_rows)
    write_csv(output / "paired_tests.csv", test_rows)
    write_csv(output / "curve_summary.csv", area_rows)
    run_summary = {"folds": folds, "config_hash": rows[0]["config_hash"], "levels": {k: v["levels"] for k, v in plot_data.items()}}
    (output / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return plot_data


def _style_axes(ax, spec: Dict, xlabel: str) -> None:
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel("Test accuracy", color=TEXT_SECONDARY)
    ax.set_ylim(0, 1)
    ax.set_title(spec["title"], color=TEXT_PRIMARY, loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="lower left", labelcolor=TEXT_SECONDARY, fontsize=9)


def plot(plot_data: Dict[str, Dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for scenario, data in plot_data.items():
        spec = COMPARISONS[scenario]
        if len(data["levels"]) < 2:
            print(f"Skipped {scenario} figure: a curve needs at least two levels.")
            continue
        fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(data["reference"], color=REFERENCE_COLOR, linewidth=1.5, linestyle=(0, (4, 3)), label=spec["reference_label"])
        series = [
            ("multimodal", "Multimodal", MULTIMODAL_COLOR, "o"),
            ("single", spec["single_label"], SINGLE_COLOR, "s"),
        ]
        last_values = {key: data["curves"][key]["points"][-1][0] for key, *_ in series}
        upper_key = max(last_values, key=last_values.get)
        close = abs(last_values["multimodal"] - last_values["single"]) < 0.05
        for key, label, color, marker in series:
            points = data["curves"][key]["points"]
            if scenario == "video_pgd":
                x = np.array([float(level) for level in data["levels"]])
                means = np.array([point[0] for point in points])
                errors = np.array([point[1] for point in points])
                ax.plot(x, means, color=color, linewidth=2, marker=marker, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
                ax.fill_between(x, means - errors, means + errors, color=color, alpha=0.15, linewidth=0)
            else:
                x = np.arange(len(data["levels"]) + 1)
                means = np.array([data["curves"][key]["clean"][0]] + [point[0] for point in points])
                errors = np.array([data["curves"][key]["clean"][1]] + [point[1] for point in points])
                ax.errorbar(x, means, yerr=errors, color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=3, elinewidth=1, label=label)
            offset = 0 if not close else (7 if key == upper_key else -7)
            ax.annotate(label, (x[-1], means[-1]), xytext=(10, offset), textcoords="offset points", va="center", color=TEXT_PRIMARY, fontsize=9, annotation_clip=False)
        if scenario == "video_pgd":
            ax.set_xscale("log")
            if any(abs(float(level) - REPORT_EPSILON) < 1e-6 for level in data["levels"]):
                ax.axvline(REPORT_EPSILON, color=GRID, linewidth=1.2, zorder=0)
                ax.text(REPORT_EPSILON, 0.985, " 8/255", color=TEXT_SECONDARY, fontsize=8, va="top")
            xlabel = f"{spec['unit']} on [0, 1] pixels, log scale"
        else:
            ax.set_xticks(np.arange(len(data["levels"]) + 1), ["clean"] + [f"{float(level):g}" for level in data["levels"]])
            xlabel = f"{spec['unit']}, more severe to the right"
        _style_axes(ax, spec, xlabel)
        fig.tight_layout()
        fig.subplots_adjust(right=0.82)
        fig.savefig(output / f"comparison_{scenario}.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--multimodal-dir", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs/robustness_summary"), type=Path)
    args = parser.parse_args()

    rows = read_results(args.audio_dir) + read_results(args.video_dir) + read_results(args.multimodal_dir)
    folds = validate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(analyze(rows, folds, args.output), args.output)
    print(f"Saved tables and figures to {args.output}")


if __name__ == "__main__":
    main()
