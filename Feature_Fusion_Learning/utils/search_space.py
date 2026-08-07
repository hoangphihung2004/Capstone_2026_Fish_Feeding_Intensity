import random
import time
import warnings
from itertools import product
from typing import Iterable

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
        "max_depth": [None] + list(range(5, 41)),
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


def get_random_params(param_grid, n_iter):
    keys, values = zip(*param_grid.items())
    all_params = list(product(*values))

    if n_iter < len(all_params):
        all_params = random.sample(all_params, n_iter)

    params_list = []
    for value in all_params:
        params_list.append(dict(zip(keys, value)))

    return params_list


def fine_tune_model(model_name, model, param_grid, n_iter, x_train, y_train, x_val, y_val):
    best_param = None
    best_acc = -1
    params_list = get_random_params(param_grid, n_iter)
    start_time = time.time()

    progress = tqdm(params_list, desc=f"Fine tuning - {model_name}")
    for params in progress:
        try:
            model_instance = clone(model)
            model_instance.set_params(**params)
            model_instance.fit(x_train, y_train)
            y_pred_val = model_instance.predict(x_val)
            acc = accuracy_score(y_val, y_pred_val)
            if acc > best_acc:
                best_acc = acc
                best_param = params
            progress.set_postfix(best_val_acc=f"{best_acc:.4f}")
        except Exception:
            pass

    tuning_time = time.time() - start_time
    if best_param is None:
        raise RuntimeError(f"No valid parameter set found for {model_name}.")
    return best_param, tuning_time
