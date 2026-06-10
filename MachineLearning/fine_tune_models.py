import json
import random
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")


def load_feature(feature_path,
                 label="label"):

    data = pd.read_csv(feature_path)

    data["feature"] = data["feature"].apply(json.loads)

    train = data[data["type"] == "train"].copy()
    val = data[data["type"] == "val"].copy()
    test = data[data["type"] == "test"].copy()

    x_train = np.vstack(train["feature"].values)
    x_val = np.vstack(val["feature"].values)
    x_test = np.vstack(test["feature"].values)

    y_train = train[label].values
    y_val = val[label].values
    y_test = test[label].values

    return x_train, y_train, x_val, y_val, x_test, y_test


def normalize_data(x_train, x_val, x_test):

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test


def get_models():
    param_LR = {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1.0, 10, 100],
        "solver": ["liblinear", "saga"],
        "max_iter": np.random.randint(300, 1000, size=300)
    }

    param_KNN = {
        "n_neighbors": np.arange(3, 31, 2),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    }

    param_SVM = {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"]
    }

    param_RF = {
        "n_estimators": np.random.randint(10, 400, size=300),
        "max_depth": np.append([None], np.random.randint(5, 51, size=30)),
        "criterion": ["gini", "entropy"]
    }

    param_LGBM = {
        "boosting_type": ["gbdt", "dart"],
        "num_leaves": np.random.randint(15, 50, size=34),
        "max_depth": [None] + list(range(5, 41)),
        "learning_rate": [0.001, 0.01, 0.1, 1, 3, 10],
        "n_estimators": np.random.randint(50, 501, size=300),
        "class_weight": ["balanced", None]
    }

    param_ET = {
        "n_estimators": np.random.randint(10, 501, size=300),
        "max_depth": np.append([None], np.random.randint(5, 51, size=40)),
        "criterion": ["gini", "entropy"],
        "bootstrap": [True, False],
        "max_leaf_nodes": np.append([None], np.random.randint(2, 13, size=10)),
        "min_samples_split": [2, 3, 5, 7, 9, 11],
        "min_samples_leaf": [1, 3, 5, 8, 9, 11],
        "max_features": ["sqrt", 0.5, 0.6, 0.7]
    }

    models = {
        "LR": (LogisticRegression(random_state=RANDOM_STATE), param_LR, 3, True),
        "KNN": (KNeighborsClassifier(), param_KNN, 40, True),
        "SVM": (SVC(), param_SVM, 3, True),
        "RF": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), param_RF, 20, False),
        "ET": (ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1), param_ET, 50, False),
        "LGBM": (LGBMClassifier(objective="multiclass", random_state=RANDOM_STATE, force_col_wise=True, verbose=-1, n_jobs=-1), param_LGBM, 30, False)
    }

    return models


def get_random_params(param_grid, n_iter):

    keys, values = zip(*param_grid.items())
    all_params = list(product(*values))

    if n_iter < len(all_params):
        all_params = random.sample(all_params, n_iter)

    params_list = []

    for value in all_params:
        params = dict(zip(keys, value))
        params_list.append(params)

    return params_list


def get_metrics(y_true, y_pred):

    metrics = {
        "Precision (Weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (Weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score (Weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }

    return metrics


def fine_tune_model(model_name, model, param_grid, n_iter, x_train, y_train, x_val, y_val):

    best_param = None
    best_acc = -1

    params_list = get_random_params(param_grid, n_iter)

    start_time = time.time()

    for params in tqdm(params_list, desc=f"Fine tuning - {model_name}"):
        try:
            model_instance = clone(model)
            model_instance.set_params(**params)
            model_instance.fit(x_train, y_train)

            y_pred_val = model_instance.predict(x_val)

            acc = accuracy_score(y_val, y_pred_val)

            if acc > best_acc:
                best_acc = acc
                best_param = params

        except Exception:
            pass

    tuning_time = time.time() - start_time

    return best_param, tuning_time


def fine_tune_models(feature_path, label="label"):

    feature_path = Path(feature_path)
    feature_name = feature_path.parent.name
    output_dir = feature_path.parent

    x_train, y_train, x_val, y_val, x_test, y_test = load_feature(feature_path, label)

    models = get_models()
    results = []

    for model_name, model_info in tqdm(models.items(), desc="Models"):
        model = model_info[0]
        param_grid = model_info[1]
        n_iter = model_info[2]
        use_scaler = model_info[3]

        if use_scaler:
            x_train_use, x_val_use, x_test_use = normalize_data(x_train, x_val, x_test)
        else:
            x_train_use = x_train
            x_val_use = x_val
            x_test_use = x_test

        best_param, tuning_time = fine_tune_model(model_name, model, param_grid, n_iter, x_train_use, y_train, x_val_use, y_val)

        best_model = clone(model)
        best_model.set_params(**best_param)

        start_time = time.time()
        best_model.fit(x_train_use, y_train)
        training_time = time.time() - start_time

        y_pred_val = best_model.predict(x_val_use)
        y_pred_test = best_model.predict(x_test_use)

        val_metrics = get_metrics(y_val, y_pred_val)
        test_metrics = get_metrics(y_test, y_pred_test)

        results.append({
            "Model": model_name,
            "Feature": feature_name,
            "Number Feature": x_train.shape[1],
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

            "Best Param": best_param
        })

    result = pd.DataFrame(results)
    result.to_csv(output_dir / f"{feature_name}_result.csv", index=False)

    return result


def main():
    feature_path = r""

    result = fine_tune_models(feature_path)

    print(result)


if __name__ == "__main__":
    main()
