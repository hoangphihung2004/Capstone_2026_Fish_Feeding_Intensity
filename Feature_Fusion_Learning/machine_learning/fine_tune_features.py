import sys
import time
from pathlib import Path

from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.artifact_upload import upload_artifact_if_enabled
from utils.evaluation import get_metrics
from utils.evaluation import save_confusion_matrix
from utils.feature_selection import DEFAULT_SELECTION_INFO
from utils.feature_selection import feature_selection_enabled
from utils.feature_selection import select_features
from utils.feature_selection import select_features_by_modality
from utils.feature_loader import load_features_for_run
from utils.feature_loader import load_modality_features_for_run
from utils.io_utils import load_json
from utils.io_utils import write_cv_summary
from utils.io_utils import write_json
from utils.io_utils import write_result_csv
from utils.search_space import fine_tune_model
from utils.search_space import get_models
from utils.search_space import normalize_data
from utils.search_space import set_random_seed


CONFIG_PATH = PROJECT_ROOT / "config" / "fine_tune_config.json"
UPLOAD_CONFIG_PATH = PROJECT_ROOT / "config" / "artifact_upload_config.json"


def mode_output_dir(config, mode, fold=None):
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs")
    if mode == "holdout":
        return output_root / "holdout"
    if mode == "cross_validation":
        if fold is None:
            raise ValueError("fold must be provided for cross_validation output.")
        return output_root / "cross_validation" / f"fold_{fold}"
    raise ValueError(f"Unsupported mode: {mode}")


def result_row(
    config,
    mode,
    fold,
    feature_set,
    model_name,
    tuning_time,
    training_time,
    val_metrics,
    test_metrics,
    best_param,
    selection_info,
):
    return {
        "Dataset": config.get("dataset_tag", ""),
        "Mode": mode,
        "Fold": fold if fold is not None else "",
        "Feature Mode": config["feature_mode"],
        "Model": model_name,
        "Feature": feature_set.feature_name,
        "Number Feature": feature_set.x_train.shape[1],
        "Feature Selection Enabled": selection_info["enabled"],
        "Feature Selector": selection_info["selector"],
        "Feature Selection Strategy": selection_info["strategy"],
        "Feature Selection Ratio": selection_info["ratio"],
        "Feature Selection Trials": selection_info["n_trials"],
        "Feature Selection Trial Jobs": selection_info["trial_n_jobs"],
        "Feature Selection Tuning Time (s)": selection_info["tuning_time_seconds"],
        "Original Number Feature": selection_info["original_num_features"],
        "Selected Number Feature": selection_info["selected_num_features"],
        "Tuning Time (s)": tuning_time,
        "Training Time (s)": training_time,
        "Precision Val (Weighted)": val_metrics["Precision (Weighted)"],
        "Recall Val (Weighted)": val_metrics["Recall (Weighted)"],
        "F1-score Val (Weighted)": val_metrics["F1-score (Weighted)"],
        "Accuracy Val": val_metrics["Accuracy"],
        "Precision Val (Macro)": val_metrics["Precision (Macro)"],
        "Recall Val (Macro)": val_metrics["Recall (Macro)"],
        "F1-score Val (Macro)": val_metrics["F1-score (Macro)"],
        "Precision Test (Weighted)": test_metrics["Precision (Weighted)"],
        "Recall Test (Weighted)": test_metrics["Recall (Weighted)"],
        "F1-score Test (Weighted)": test_metrics["F1-score (Weighted)"],
        "Accuracy Test": test_metrics["Accuracy"],
        "Precision Test (Macro)": test_metrics["Precision (Macro)"],
        "Recall Test (Macro)": test_metrics["Recall (Macro)"],
        "F1-score Test (Macro)": test_metrics["F1-score (Macro)"],
        "Best Param": best_param,
    }


def run_one_feature_set(config, mode, fold=None):
    output_dir = mode_output_dir(config, mode, fold=fold)
    confusion_dir = output_dir / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_info = dict(DEFAULT_SELECTION_INFO)
    run_label = mode if fold is None else f"{mode}/fold_{fold}"

    print(f"[run:{run_label}] loading features")

    if feature_selection_enabled(config):
        strategy = config.get("feature_selection", {}).get("strategy", "global")
        print(f"[run:{run_label}] feature selection enabled with strategy={strategy}")
        if strategy == "modality":
            modality_feature_sets = load_modality_features_for_run(config, mode=mode, fold=fold)
            for modality_name, modality_feature_set in modality_feature_sets.items():
                print(
                    f"[run:{run_label}] loaded {modality_name}: "
                    f"train={modality_feature_set.x_train.shape}, "
                    f"val={modality_feature_set.x_val.shape}, "
                    f"test={modality_feature_set.x_test.shape}"
                )
            feature_set, selection_info = select_features_by_modality(modality_feature_sets, config, output_dir)
        else:
            feature_set = load_features_for_run(config, mode=mode, fold=fold)
            print(
                f"[run:{run_label}] loaded fused features: "
                f"train={feature_set.x_train.shape}, val={feature_set.x_val.shape}, test={feature_set.x_test.shape}"
            )
            feature_set, selection_info = select_features(feature_set, config, output_dir)
    else:
        feature_set = load_features_for_run(config, mode=mode, fold=fold)
        print(
            f"[run:{run_label}] loaded features: "
            f"train={feature_set.x_train.shape}, val={feature_set.x_val.shape}, test={feature_set.x_test.shape}"
        )
        num_features = int(feature_set.x_train.shape[1])
        selection_info["original_num_features"] = num_features
        selection_info["selected_num_features"] = num_features

    models = get_models(config.get("models"))
    trial_n_jobs = int(config.get("trial_n_jobs", 1))
    rows = []
    print(
        f"[run:{run_label}] final feature matrix: "
        f"train={feature_set.x_train.shape}, val={feature_set.x_val.shape}, test={feature_set.x_test.shape}"
    )
    print(f"[run:{run_label}] training models: {', '.join(models.keys())}")

    for model_name, model_info in models.items():
        model, param_grid, n_iter, use_scaler = model_info
        print(f"[run:{run_label}] fine tuning model {model_name} ({n_iter} random trials)")

        if use_scaler:
            print(f"[run:{run_label}] normalizing data for {model_name}")
            x_train_use, x_val_use, x_test_use = normalize_data(
                feature_set.x_train,
                feature_set.x_val,
                feature_set.x_test,
            )
        else:
            x_train_use = feature_set.x_train
            x_val_use = feature_set.x_val
            x_test_use = feature_set.x_test

        best_param, tuning_time = fine_tune_model(
            model_name,
            model,
            param_grid,
            n_iter,
            x_train_use,
            feature_set.y_train,
            x_val_use,
            feature_set.y_val,
            trial_n_jobs=trial_n_jobs,
        )

        best_model = clone(model)
        best_model.set_params(**best_param)
        print(f"[run:{run_label}] best params for {model_name}: {best_param}")

        start_time = time.time()
        best_model.fit(x_train_use, feature_set.y_train)
        training_time = time.time() - start_time

        y_pred_val = best_model.predict(x_val_use)
        y_pred_test = best_model.predict(x_test_use)

        val_metrics = get_metrics(feature_set.y_val, y_pred_val)
        test_metrics = get_metrics(feature_set.y_test, y_pred_test)
        print(
            f"[run:{run_label}] {model_name} metrics: "
            f"val_acc={val_metrics['Accuracy']:.4f}, test_acc={test_metrics['Accuracy']:.4f}, "
            f"test_f1_macro={test_metrics['F1-score (Macro)']:.4f}"
        )
        save_confusion_matrix(
            feature_set.y_test,
            y_pred_test,
            confusion_dir,
            model_name,
            label_names=feature_set.label_names,
        )

        rows.append(
            result_row(
                config,
                mode,
                fold,
                feature_set,
                model_name,
                tuning_time,
                training_time,
                val_metrics,
                test_metrics,
                best_param,
                selection_info,
            )
        )

    result_path = output_dir / "result.csv"
    result = write_result_csv(rows, result_path)
    write_json(
        {
            "dataset": config.get("dataset_tag", ""),
            "mode": mode,
            "fold": fold,
            "feature_mode": config["feature_mode"],
            "feature_name": feature_set.feature_name,
            "num_train": int(len(feature_set.y_train)),
            "num_val": int(len(feature_set.y_val)),
            "num_test": int(len(feature_set.y_test)),
            "num_features": int(feature_set.x_train.shape[1]),
            "models": list(models.keys()),
            "trial_n_jobs": trial_n_jobs,
            "feature_selection": selection_info,
        },
        output_dir / "run_info.json",
    )
    print(f"Saved result: {result_path}")
    return result_path, result


def resolve_folds(config):
    folds = config.get("folds", "all")
    if folds == "all":
        return list(range(int(config.get("num_folds", 5))))
    if isinstance(folds, int):
        return [folds]
    return [int(fold) for fold in folds]


def run_holdout(config):
    print("Running holdout fine tuning...")
    return run_one_feature_set(config, mode="holdout", fold=None)


def run_cross_validation(config):
    print("Running cross validation fine tuning...")
    fold_paths = []
    for fold in resolve_folds(config):
        print(f"Running fold_{fold}...")
        result_path, _ = run_one_feature_set(config, mode="cross_validation", fold=fold)
        fold_paths.append(result_path)

    summary_path = PROJECT_ROOT / config.get("output_dir", "outputs") / "cross_validation" / "cv_summary.csv"
    summary = write_cv_summary(fold_paths, summary_path)
    print(f"Saved CV summary: {summary_path}")
    return summary_path, summary


def main():
    config = load_json(CONFIG_PATH)
    upload_config = load_json(UPLOAD_CONFIG_PATH)
    set_random_seed(int(config.get("seed", 42)))

    evaluation_mode = config.get("evaluation_mode", "both")
    if evaluation_mode not in {"holdout", "cross_validation", "both"}:
        raise ValueError("evaluation_mode must be one of: holdout, cross_validation, both")

    if evaluation_mode in {"holdout", "both"}:
        run_holdout(config)
    if evaluation_mode in {"cross_validation", "both"}:
        run_cross_validation(config)

    upload_artifact_if_enabled(upload_config, config, default_source_dir=PROJECT_ROOT)


if __name__ == "__main__":
    main()
