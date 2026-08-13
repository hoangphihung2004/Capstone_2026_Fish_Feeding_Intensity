import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from math import prod
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


RANDOM_STATE = 42
warnings.filterwarnings("ignore")


def set_random_seed(seed: int) -> None:
    global RANDOM_STATE
    RANDOM_STATE = seed
    random.seed(seed)
    np.random.seed(seed)


def normalize_data(x_train, x_val, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test


def get_models(selected_models: Iterable[str] | None = None):
    param_lr = {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1.0, 10, 100],
        "solver": ["liblinear", "saga"],
        "max_iter": np.random.randint(300, 500, size=100),
    }

    param_knn = {
        "n_neighbors": np.arange(3, 31, 2),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    }

    param_svm = {
        "kernel": ["linear", "rbf"],
        "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        "gamma": ["scale", "auto"],
    }

    param_rf = {
        "n_estimators": np.random.randint(10, 400, size=300),
        "max_depth": np.append([None], np.random.randint(5, 51, size=30)),
        "criterion": ["gini", "entropy"],
    }

    param_lgbm = {
        "boosting_type": ["gbdt", "dart"],
        "num_leaves": np.random.randint(15, 50, size=34),
        "max_depth": [-1] + list(range(5, 41)),
        "learning_rate": [0.001, 0.01, 0.1, 1, 3, 10],
        "n_estimators": np.random.randint(50, 501, size=300),
        "class_weight": ["balanced", None],
    }

    param_et = {
        "n_estimators": np.random.randint(10, 501, size=300),
        "max_depth": np.append([None], np.random.randint(5, 51, size=40)),
        "criterion": ["gini", "entropy"],
        "bootstrap": [True, False],
        "max_leaf_nodes": np.append([None], np.random.randint(2, 13, size=10)),
        "min_samples_split": [2, 3, 5, 7, 9, 11],
        "min_samples_leaf": [1, 3, 5, 8, 9, 11],
        "max_features": ["sqrt", 0.5, 0.6, 0.7],
    }

    models = {
        "LR": (LogisticRegression(random_state=RANDOM_STATE), param_lr, 20, True),
        "KNN": (KNeighborsClassifier(), param_knn, 100, True),
        "SVM": (SVC(), param_svm, 50, True),
        "RF": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), param_rf, 100, True),
        "ET": (ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1), param_et, 100, True),
    }

    if LGBMClassifier is not None:
        models["LGBM"] = (
            LGBMClassifier(
                objective="multiclass",
                random_state=RANDOM_STATE,
                force_col_wise=True,
                verbose=-1,
                n_jobs=-1,
            ),
            param_lgbm,
            100,
            True,
        )
    elif selected_models is None or "LGBM" in selected_models:
        raise ImportError("lightgbm is required because LGBM is enabled in config.")

    if selected_models is None:
        return models

    unknown = set(selected_models) - set(models)
    if unknown:
        raise ValueError(f"Unknown or unavailable models: {sorted(unknown)}")
    return {name: models[name] for name in selected_models}


def get_random_params(param_grid, n_iter, include_default=False):
    keys, values = zip(*param_grid.items())
    values = [list(value) for value in values]
    total_combinations = prod(len(value) for value in values)
    params_list = []
    seen = set()

    if include_default:
        params_list.append({})

    if n_iter >= total_combinations:
        all_params = list(product(*values))
    else:
        all_params = []
        while len(all_params) < n_iter:
            value = tuple(random.choice(options) for options in values)
            if value in seen:
                continue
            seen.add(value)
            all_params.append(value)

    for value in all_params:
        params_list.append(dict(zip(keys, value)))

    return params_list


def _set_inner_n_jobs(model_instance, outer_n_jobs: int):
    if outer_n_jobs > 1 and "n_jobs" in model_instance.get_params():
        model_instance.set_params(n_jobs=1)
    return model_instance


def _evaluate_params(index, model_name, model, params, x_train, y_train, x_val, y_val, trial_n_jobs):
    model_instance = clone(model)
    model_instance.set_params(**params)
    model_instance = _set_inner_n_jobs(model_instance, trial_n_jobs)
    model_instance.fit(x_train, y_train)
    y_pred_val = model_instance.predict(x_val)
    acc = accuracy_score(y_val, y_pred_val)
    return index, params, acc


def fine_tune_model(
    model_name,
    model,
    param_grid,
    n_iter,
    x_train,
    y_train,
    x_val,
    y_val,
    trial_n_jobs=1,
    include_default=False,
):
    best_param = None
    best_acc = -1
    best_index = None
    trial_n_jobs = max(1, int(trial_n_jobs))
    params_list = get_random_params(param_grid, n_iter, include_default=include_default)
    start_time = time.time()

    if trial_n_jobs == 1:
        progress = tqdm(list(enumerate(params_list)), desc=f"Fine tuning - {model_name}")
        for index, params in progress:
            try:
                _, candidate_params, acc = _evaluate_params(
                    index, model_name, model, params, x_train, y_train, x_val, y_val, trial_n_jobs
                )
                if acc > best_acc or (acc == best_acc and (best_index is None or index < best_index)):
                    best_acc = acc
                    best_param = candidate_params
                    best_index = index
                progress.set_postfix(best_val_acc=f"{best_acc:.4f}", n_jobs=trial_n_jobs)
            except Exception:
                pass
    else:
        progress = tqdm(total=len(params_list), desc=f"Fine tuning - {model_name}")
        with ThreadPoolExecutor(max_workers=trial_n_jobs) as executor:
            futures = [
                executor.submit(
                    _evaluate_params,
                    index,
                    model_name,
                    model,
                    params,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    trial_n_jobs,
                )
                for index, params in enumerate(params_list)
            ]
            for future in as_completed(futures):
                try:
                    index, candidate_params, acc = future.result()
                    if acc > best_acc or (acc == best_acc and (best_index is None or index < best_index)):
                        best_acc = acc
                        best_param = candidate_params
                        best_index = index
                except Exception:
                    pass
                progress.update(1)
                progress.set_postfix(best_val_acc=f"{best_acc:.4f}", n_jobs=trial_n_jobs)
        progress.close()

    tuning_time = time.time() - start_time
    if best_param is None:
        raise RuntimeError(f"No valid parameter set found for {model_name}.")
    return best_param, tuning_time
