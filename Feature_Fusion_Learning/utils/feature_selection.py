import math
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from utils.feature_loader import concat_split_features
from utils.feature_loader import SplitFeatures
from utils.io_utils import write_json
from utils.search_space import fine_tune_model
from utils.search_space import get_models


DEFAULT_SELECTION_INFO = {
    "enabled": False,
    "selector": "",
    "strategy": "",
    "ratio": "",
    "n_trials": "",
    "trial_n_jobs": "",
    "original_num_features": "",
    "selected_num_features": "",
    "best_params": {},
    "tuning_time_seconds": 0.0,
    "include_default_params": False,
    "candidate_count": "",
    "modality_details": {},
    "artifact_dir": "",
}


def feature_selection_enabled(config):
    return bool(config.get("feature_selection", {}).get("enabled", False))


def validate_feature_selection_config(selection_config):
    strategy = str(selection_config.get("strategy", "global")).lower()
    if strategy not in {"global", "modality"}:
        raise ValueError("feature_selection.strategy must be one of: global, modality")

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
        "strategy": strategy,
        "ratio": ratio,
        "n_trials": n_trials,
        "trial_n_jobs": trial_n_jobs,
    }


def selection_dir_name(strategy, selector, ratio, n_trials):
    return f"{strategy}_{selector}_ratio_{ratio:g}_trials_{n_trials}"


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


def write_feature_selection_outputs(output_dir, selection_config, importance_rows, selected_rows, info, subdir=None):
    relative_dir = Path("feature_selection") / selection_dir_name(
        selection_config["strategy"],
        selection_config["selector"],
        selection_config["ratio"],
        selection_config["n_trials"],
    )
    if subdir:
        relative_dir = relative_dir / subdir
    artifact_dir = Path(output_dir) / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(importance_rows).to_csv(artifact_dir / "feature_importances.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(artifact_dir / "selected_features.csv", index=False)

    info = dict(info)
    info["artifact_dir"] = str(relative_dir).replace("\\", "/")
    write_json(info, artifact_dir / "selection_info.json")
    return artifact_dir, info


def select_one_feature_set(feature_set, selection_config, output_dir, output_subdir=None, feature_name_suffix=None):
    selector_name = selection_config["selector"]
    selector_model, param_grid, _, _ = get_models([selector_name])[selector_name]
    label = output_subdir or "global"

    print(
        f"[feature-selection:{label}] selector={selector_name}, strategy={selection_config['strategy']}, "
        f"ratio={selection_config['ratio']:g}, candidates={selection_config['n_trials'] + 1} "
        f"(default + {selection_config['n_trials']} random)"
    )
    print(
        f"[feature-selection:{label}] input shapes: "
        f"train={feature_set.x_train.shape}, val={feature_set.x_val.shape}, test={feature_set.x_test.shape}"
    )

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
    print(f"[feature-selection:{label}] best selector params: {best_param}")

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
    print(
        f"[feature-selection:{label}] selected {selected_num_features}/{original_num_features} "
        f"features ({selection_config['ratio']:g})"
    )

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
        "modality_details": {},
        "artifact_dir": "",
    }
    _, info = write_feature_selection_outputs(
        output_dir,
        selection_config,
        importance_rows,
        selected_rows,
        info,
        subdir=output_subdir,
    )

    feature_name_suffix = feature_name_suffix or (
        f"fs_{selection_config['strategy']}_{selector_name}_ratio_"
        f"{selection_config['ratio']:g}_trials_{selection_config['n_trials']}"
    )
    return apply_selected_features(feature_set, selected_indices, feature_name_suffix), info


def select_features(feature_set, config, output_dir):
    selection_config = validate_feature_selection_config(config.get("feature_selection", {}))
    if selection_config["strategy"] != "global":
        raise ValueError("select_features only supports feature_selection.strategy='global'.")
    return select_one_feature_set(feature_set, selection_config, output_dir)


def select_features_by_modality(modality_feature_sets, config, output_dir):
    selection_config = validate_feature_selection_config(config.get("feature_selection", {}))
    if selection_config["strategy"] != "modality":
        raise ValueError("select_features_by_modality only supports feature_selection.strategy='modality'.")

    print("[feature-selection] running modality strategy")
    selected_sets = []
    modality_details = {}
    original_total = 0
    selected_total = 0

    for modality_name, modality_feature_set in modality_feature_sets.items():
        selected_set, modality_info = select_one_feature_set(
            modality_feature_set,
            selection_config,
            output_dir,
            output_subdir=modality_name,
            feature_name_suffix=f"{modality_name}_fs_{selection_config['selector']}_ratio_"
            f"{selection_config['ratio']:g}_trials_{selection_config['n_trials']}",
        )
        selected_sets.append(selected_set)
        modality_details[modality_name] = modality_info
        original_total += int(modality_info["original_num_features"])
        selected_total += int(modality_info["selected_num_features"])

    combined = concat_split_features(
        selected_sets,
        f"{'_'.join(modality_feature_sets.keys())}_fs_modality_"
        f"{selection_config['selector']}_ratio_{selection_config['ratio']:g}_trials_{selection_config['n_trials']}",
    )
    info = {
        **selection_config,
        "original_num_features": original_total,
        "selected_num_features": selected_total,
        "best_params": {},
        "tuning_time_seconds": sum(float(item["tuning_time_seconds"]) for item in modality_details.values()),
        "include_default_params": True,
        "candidate_count": selection_config["n_trials"] + 1,
        "modality_details": modality_details,
        "artifact_dir": str(
            Path("feature_selection")
            / selection_dir_name(
                selection_config["strategy"],
                selection_config["selector"],
                selection_config["ratio"],
                selection_config["n_trials"],
            )
        ).replace("\\", "/"),
    }
    write_json(info, Path(output_dir) / info["artifact_dir"] / "selection_info.json")
    print(
        f"[feature-selection] modality strategy selected {selected_total}/{original_total} "
        f"features after concat"
    )
    return combined, info
