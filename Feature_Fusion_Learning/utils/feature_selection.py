import math
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from utils.feature_loader import SplitFeatures
from utils.io_utils import write_json
from utils.search_space import fine_tune_model
from utils.search_space import get_models


DEFAULT_SELECTION_INFO = {
    "enabled": False,
    "selector": "",
    "ratio": "",
    "n_trials": "",
    "trial_n_jobs": "",
    "original_num_features": "",
    "selected_num_features": "",
    "best_params": {},
    "tuning_time_seconds": 0.0,
    "include_default_params": False,
    "candidate_count": "",
    "artifact_dir": "",
}


def feature_selection_enabled(config):
    return bool(config.get("feature_selection", {}).get("enabled", False))


def validate_feature_selection_config(selection_config):
    selector = str(selection_config.get("selector", "RF")).upper()
    if selector not in {"RF", "LGBM"}:
        raise ValueError("feature_selection.selector must be one of: RF, LGBM")

    ratio = float(selection_config.get("ratio", 1.0))
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("feature_selection.ratio must be greater than 0.0 and less than or equal to 1.0")

    n_trials = int(selection_config.get("n_trials", 100))
    if n_trials < 1:
        raise ValueError("feature_selection.n_trials must be at least 1")

    trial_n_jobs = int(selection_config.get("trial_n_jobs", 1))
    if trial_n_jobs < 1:
        raise ValueError("feature_selection.trial_n_jobs must be at least 1")

    return {
        "enabled": True,
        "selector": selector,
        "ratio": ratio,
        "n_trials": n_trials,
        "trial_n_jobs": trial_n_jobs,
    }


def selection_dir_name(selector, ratio, n_trials):
    return f"{selector}_ratio_{ratio:g}_trials_{n_trials}"


def apply_selected_features(feature_set, selected_indices, feature_name_suffix):
    return SplitFeatures(
        x_train=feature_set.x_train[:, selected_indices],
        y_train=feature_set.y_train,
        x_val=feature_set.x_val[:, selected_indices],
        y_val=feature_set.y_val,
        x_test=feature_set.x_test[:, selected_indices],
        y_test=feature_set.y_test,
        feature_name=f"{feature_set.feature_name}_{feature_name_suffix}",
        label_names=feature_set.label_names,
    )


def write_feature_selection_outputs(output_dir, selection_config, importance_rows, selected_rows, info):
    relative_dir = Path("feature_selection") / selection_dir_name(
        selection_config["selector"],
        selection_config["ratio"],
        selection_config["n_trials"],
    )
    artifact_dir = Path(output_dir) / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(importance_rows).to_csv(artifact_dir / "feature_importances.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(artifact_dir / "selected_features.csv", index=False)

    info = dict(info)
    info["artifact_dir"] = str(relative_dir).replace("\\", "/")
    write_json(info, artifact_dir / "selection_info.json")
    return artifact_dir, info


def select_features(feature_set, config, output_dir):
    selection_config = validate_feature_selection_config(config.get("feature_selection", {}))
    selector_name = selection_config["selector"]
    selector_model, param_grid, _, _ = get_models([selector_name])[selector_name]

    best_param, tuning_time = fine_tune_model(
        selector_name,
        selector_model,
        param_grid,
        selection_config["n_trials"],
        feature_set.x_train,
        feature_set.y_train,
        feature_set.x_val,
        feature_set.y_val,
        trial_n_jobs=selection_config["trial_n_jobs"],
        include_default=True,
    )

    best_selector = clone(selector_model)
    best_selector.set_params(**best_param)
    best_selector.fit(feature_set.x_train, feature_set.y_train)

    if not hasattr(best_selector, "feature_importances_"):
        raise RuntimeError(f"{selector_name} does not expose feature_importances_.")

    importances = best_selector.feature_importances_
    original_num_features = int(feature_set.x_train.shape[1])
    selected_num_features = max(1, int(math.ceil(original_num_features * selection_config["ratio"])))
    ranked_indices = sorted(range(original_num_features), key=lambda index: importances[index], reverse=True)
    selected_indices = ranked_indices[:selected_num_features]
    selected_set = set(selected_indices)

    importance_rows = [
        {
            "rank": rank,
            "feature_index": int(index),
            "importance": float(importances[index]),
            "selected": bool(index in selected_set),
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]
    selected_rows = [
        {
            "rank": rank,
            "feature_index": int(index),
            "importance": float(importances[index]),
        }
        for rank, index in enumerate(selected_indices, start=1)
    ]

    info = {
        **selection_config,
        "original_num_features": original_num_features,
        "selected_num_features": selected_num_features,
        "best_params": best_param,
        "tuning_time_seconds": tuning_time,
        "include_default_params": True,
        "candidate_count": selection_config["n_trials"] + 1,
        "artifact_dir": "",
    }
    _, info = write_feature_selection_outputs(output_dir, selection_config, importance_rows, selected_rows, info)

    feature_name_suffix = f"fs_{selector_name}_ratio_{selection_config['ratio']:g}_trials_{selection_config['n_trials']}"
    return apply_selected_features(feature_set, selected_indices, feature_name_suffix), info
