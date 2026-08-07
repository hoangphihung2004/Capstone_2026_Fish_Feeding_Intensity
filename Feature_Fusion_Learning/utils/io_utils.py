import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "Precision Val (Weighted)",
    "Recall Val (Weighted)",
    "F1-score Val (Weighted)",
    "Accuracy Val",
    "Precision Val (Macro)",
    "Recall Val (Macro)",
    "F1-score Val (Macro)",
    "Precision Test (Weighted)",
    "Recall Test (Weighted)",
    "F1-score Test (Weighted)",
    "Accuracy Test",
    "Precision Test (Macro)",
    "Recall Test (Macro)",
    "F1-score Test (Macro)",
]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_result_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def write_cv_summary(fold_result_paths, output_path):
    frames = []
    for result_path in fold_result_paths:
        frame = pd.read_csv(result_path)
        frames.append(frame)

    if not frames:
        raise ValueError("No fold result files were provided for CV summary.")

    all_results = pd.concat(frames, ignore_index=True)
    group_cols = ["Dataset", "Feature Mode", "Feature", "Number Feature", "Model"]
    rows = []

    for keys, group in all_results.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["Num Folds"] = int(group["Fold"].nunique())
        for col in METRIC_COLUMNS:
            mean_value = group[col].mean()
            std_value = group[col].std(ddof=1) if len(group) > 1 else 0.0
            row[f"{col} Mean"] = mean_value
            row[f"{col} Std"] = std_value
            row[f"{col} Mean +- Std"] = f"{mean_value:.6f} +- {std_value:.6f}"
        rows.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary
