import json
import os
import random
import re
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Optional

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


def load_feature(feature_path, label="label"):
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
        "max_iter": np.random.randint(300, 500, size=100)
    }

    param_KNN = {
        "n_neighbors": np.arange(3, 31, 2),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    }

    param_SVM = {
        "kernel": ["linear", "rbf"],
        "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
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
        "LR": (LogisticRegression(random_state=RANDOM_STATE), param_LR, 10, True),
        "KNN": (KNeighborsClassifier(), param_KNN, 100, True),
        "SVM": (SVC(), param_SVM, 10, True),
        "RF": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), param_RF, 100, False),
        "ET": (ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1), param_ET, 100, False),
        "LGBM": (LGBMClassifier(objective="multiclass", random_state=RANDOM_STATE, force_col_wise=True, verbose=-1, n_jobs=-1), param_LGBM, 100, False)
    }

    return models


def get_random_params(param_grid, n_iter):
    # Tính tổng số tổ hợp khả dĩ
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)

    # Nếu tổng số tổ hợp nhỏ thì dùng tích Descartes an toàn
    if total_combinations <= n_iter:
        keys, values = zip(*param_grid.items())
        all_params = list(product(*values))
        return [dict(zip(keys, v)) for v in all_params]

    # Tránh tràn RAM bằng cách lấy mẫu ngẫu nhiên trực tiếp
    params_list = []
    seen = set()
    attempts = 0
    max_attempts = n_iter * 20

    while len(params_list) < n_iter and attempts < max_attempts:
        attempts += 1
        params = {k: random.choice(list(v)) for k, v in param_grid.items()}
        sig = tuple(sorted((k, str(v)) for k, v in params.items()))
        if sig not in seen:
            seen.add(sig)
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

    pbar = tqdm(params_list, desc=f"Fine tuning - {model_name}")
    for params in pbar:
        try:
            model_instance = clone(model)
            model_instance.set_params(**params)
            model_instance.fit(x_train, y_train)

            y_pred_val = model_instance.predict(x_val)

            acc = accuracy_score(y_val, y_pred_val)

            if acc > best_acc:
                best_acc = acc
                best_param = params
                tqdm.write(f"  [+] {model_name} -> New Best Val Acc: {best_acc:.4f}")

            pbar.set_postfix({"best_val_acc": f"{best_acc:.4f}" if best_acc >= 0 else "N/A"})

        except Exception:
            pass

    tuning_time = time.time() - start_time

    # Giữ nguyên lệnh in khi kết thúc tuning model
    print(f"[{model_name}] Final Best Val Acc: {best_acc:.4f} (Tuning time: {tuning_time:.1f}s)")

    return best_param, tuning_time, best_acc


def fine_tune_single_fold(feature_path, label="label"):
    feature_path = Path(feature_path).resolve()
    feature_name = feature_path.parent.parent.name
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

        best_param, tuning_time, best_val_acc = fine_tune_model(
            model_name, model, param_grid, n_iter, x_train_use, y_train, x_val_use, y_val
        )

        best_model = clone(model)
        if best_param:
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

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_dir / "result.csv", index=False)

    return result_df


def fine_tune_cv(parent_dir_path: str, label: str = "label") -> pd.DataFrame:
    # Làm sạch chuỗi đường dẫn nhập từ terminal (loại bỏ dấu ngoặc kép, khoảng trắng thừa)
    raw_path = parent_dir_path.strip().strip('"').strip("'")
    parent_path = Path(raw_path).expanduser().resolve()

    if not parent_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {parent_path}")

    # Tìm các thư mục fold (hỗ trợ fold_00, fold_0, FOLD_00, Fold_01, ...)
    def natural_sort_key(p):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', p.name)]

    fold_dirs = [
        d for d in parent_path.iterdir()
        if d.is_dir() and "fold" in d.name.lower()
    ]
    fold_dirs = sorted(fold_dirs, key=natural_sort_key)

    if not fold_dirs:
        # Nếu đường dẫn chỉ thẳng tới 1 fold đơn lẻ hoặc thư mục chứa file csv
        has_csv = any((parent_path / name).exists() for name in ["features.csv", "feature.csv"])
        if has_csv:
            fold_dirs = [parent_path]
        else:
            raise ValueError(f"Không tìm thấy thư mục fold nào trong: {parent_path}")

    print("=" * 80)
    print(f"BẮT ĐẦU FINE-TUNE CROSS-VALIDATION ({len(fold_dirs)} FOLDS)")
    print(f"Hệ điều hành          : {sys.platform.upper()}")
    print(f"Thư mục đặc trưng mẹ  : {parent_path.name}")
    print(f"Đường dẫn tuyệt đối   : {parent_path}")
    print("=" * 80)

    all_fold_dfs = []

    for idx, fdir in enumerate(fold_dirs, start=1):
        # Tìm file features.csv
        feature_csv = None
        for candidate in ["features.csv", "feature.csv"]:
            if (fdir / candidate).exists():
                feature_csv = fdir / candidate
                break

        if feature_csv is None:
            candidate_csvs = [c for c in fdir.glob("*.csv") if c.name not in ["result.csv", "cv_summary.csv"]]
            if len(candidate_csvs) == 1:
                feature_csv = candidate_csvs[0]

        if feature_csv is None or not feature_csv.exists():
            print(f"\n[CẢNH BÁO] Bỏ qua {fdir.name} vì không tìm thấy file features.csv!")
            continue

        print(f"\n[{idx}/{len(fold_dirs)}] >>> Đang chạy: {fdir.name}")
        fold_df = fine_tune_single_fold(feature_csv, label=label)
        fold_df["Fold"] = fdir.name
        all_fold_dfs.append(fold_df)
        print(f"  [+] Đã lưu kết quả fold: {fdir / 'result.csv'}")

    if not all_fold_dfs:
        raise RuntimeError("Không có fold nào được xử lý thành công!")

    # Gộp kết quả toàn bộ các fold
    combined_df = pd.concat(all_fold_dfs, ignore_index=True)

    # Danh sách các cột số để tính mean +- std
    numeric_cols = [
        "Tuning Time (s)",
        "Training Time (s)",
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

    summary_rows = []
    models_list = combined_df["Model"].unique()

    for model_name in models_list:
        sub_df = combined_df[combined_df["Model"] == model_name]
        row = {
            "Model": model_name,
            "Feature": sub_df["Feature"].iloc[0] if "Feature" in sub_df.columns else parent_path.name,
            "Number Feature": sub_df["Number Feature"].iloc[0] if "Number Feature" in sub_df.columns else None,
            "Folds Count": len(sub_df),
        }

        for col in numeric_cols:
            if col in sub_df.columns:
                mean_val = sub_df[col].mean()
                std_val = sub_df[col].std()
                row[f"{col} (Mean ± Std)"] = f"{mean_val:.4f} ± {std_val:.4f}"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = parent_path / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("HOÀN TẤT TẤT CẢ CÁC FOLD! TỔNG HỢP KẾT QUẢ CROSS-VALIDATION:")
    print(f"File tổng kết đã lưu tại: {summary_path}\n")

    # In bảng tóm tắt Accuracy và F1-score Test (Mean +- Std)
    display_cols = ["Model", "Accuracy Val (Mean ± Std)", "Accuracy Test (Mean ± Std)", "F1-score Test (Weighted) (Mean ± Std)"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)

    # Tự động tải kết quả lên Hugging Face (sử dụng token từ hf login)
    upload_results_to_hf(parent_path)

    return summary_df


def upload_results_to_hf(
    parent_path: Path,
    repo_id: str = "hoangphihung442004/Audio_Features_ML_Results",
    token: Optional[str] = None,
) -> bool:
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        print("\n[HUGGINGFACE] Chưa cài đặt thư viện 'huggingface_hub'. Bỏ qua bước upload.")
        return False

    active_token = token or os.environ.get("HF_TOKEN") or get_token()
    if not active_token:
        print("\n" + "=" * 80)
        print("[HUGGINGFACE] CHƯA ĐĂNG NHẬP TÀI KHOẢN HUGGING FACE!")
        print("Để kích hoạt tính năng tự động đẩy kết quả lên repo, bạn chỉ cần mở terminal và chạy:")
        print("    hf auth login")
        print("Toàn bộ kết quả huấn luyện và file cv_summary.csv đã được lưu an toàn tại máy của bạn.")
        print("=" * 80)
        return False

    # Xác định đường dẫn tương đối trên repo (giữ nguyên cấu trúc thư mục đối xứng 1:1)
    parts = list(parent_path.parts)
    rel_parts = []
    for anchor in ["Audio_Features", "features_256k", "stft_pre_features", "features"]:
        if anchor in parts:
            idx = parts.index(anchor)
            if anchor == "Audio_Features":
                rel_parts = parts[idx + 1 :]
            else:
                rel_parts = parts[idx:]
            break

    if not rel_parts:
        rel_parts = parts[-2:] if len(parts) >= 2 else parts[-1:]

    path_in_repo = "/".join(rel_parts)

    print("\n" + "=" * 80)
    print("BẮT ĐẦU TỰ ĐỘNG UPLOAD KẾT QUẢ LÊN HUGGING FACE...")
    print(f"- Repo đích     : {repo_id}")
    print(f"- Đường dẫn repo: {path_in_repo}/")

    api = HfApi(token=active_token)
    try:
        commit_info = api.upload_folder(
            folder_path=str(parent_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[
                "cv_summary.csv",
                "config.txt",
                "fold_*/result.csv",
                "fold_*/config.txt",
                "*/result.csv",
                "*/config.txt",
            ],
            ignore_patterns=["*features.csv", "*.wav", "*.py", "*.pyc"],
            commit_message=f"Add ML evaluation results for {path_in_repo}",
        )
        commit_url = getattr(commit_info, "commit_url", commit_info)
        print("TỰ ĐỘNG UPLOAD THÀNH CÔNG RỰC RỠ!")
        print(f"Commit URL: {commit_url}")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"\n[CẢNH BÁO] Không thể upload kết quả lên Hugging Face: {e}")
        print(f"Toàn bộ kết quả huấn luyện vẫn được lưu an toàn tại máy của bạn: {parent_path}")
        print("=" * 80)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune 6 mô hình ML trên tập đặc trưng Cross-Validation (5 Folds)")
    parser.add_argument("path", nargs="?", default=None, help="Đường dẫn thư mục đặc trưng chứa 5 fold (tùy chọn)")
    parser.add_argument("--path", dest="opt_path", default=None, help="Đường dẫn thư mục đặc trưng")
    args = parser.parse_args()

    feature_path = args.path or args.opt_path
    if not feature_path:
        feature_path = input("Path: ")

    result = fine_tune_cv(feature_path)


if __name__ == "__main__":
    main()
