"""Huấn luyện, đánh giá và xuất pipeline gợi ý OutfitAI (Bước 1)."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "DatasetDSS.xlsx"
MODEL_DIR = ROOT / "model"
CHART_DIR = MODEL_DIR / "charts"
DATA_DIR = ROOT / "data"

CATEGORICAL_FEATURES = ["Hoàn cảnh", "Phong cách"]
NUMERIC_FEATURES = [
    "Tmin (°C)",
    "Tmax (°C)",
    "Nhiệt độ ngày (°C)",
    "Giữ ấm",
    "Thoáng khí",
    "Vận động",
    "Chỉn chu",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Điểm phù hợp (0–10)"
GROUP_FEATURES = [
    "Phong cách",
    "Tmin (°C)",
    "Tmax (°C)",
    "Giữ ấm",
    "Thoáng khí",
    "Vận động",
    "Chỉn chu",
]
K_CANDIDATES = [1, 2, 3, 5, 7, 10, 15, 20]
RANDOM_STATE = 42
RF_TREES = 200
JOBLIB_COMPRESSION = 9
RF_SIZE_CANDIDATES = [
    {"name": "full", "min_samples_leaf": 1, "max_depth": None},
    {"name": "leaf_2", "min_samples_leaf": 2, "max_depth": None},
    {"name": "leaf_3", "min_samples_leaf": 3, "max_depth": None},
    {"name": "leaf_2_depth_20", "min_samples_leaf": 2, "max_depth": 20},
    {"name": "leaf_3_depth_20", "min_samples_leaf": 3, "max_depth": 20},
    {"name": "leaf_5_depth_14", "min_samples_leaf": 5, "max_depth": 14},
]
MAX_RF_MAE_REGRESSION = 0.02
CONFIDENCE_QUANTILES = (1 / 3, 2 / 3)
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
HYBRID_WEIGHT_CANDIDATES = np.linspace(0.05, 0.95, 19)


def make_pipeline(*, scale_numeric: bool, estimator: object) -> Pipeline:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def validate_training_data(data: pd.DataFrame, wardrobe: pd.DataFrame) -> None:
    required = [*FEATURES, TARGET, "ID_SP"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong DuLieu_ML: {missing}")
    wardrobe_required = [
        "ID_SP",
        "Tên trang phục",
        "Loại TP",
        "Giới tính",
        "Phong cách",
        "Màu sắc",
        *GROUP_FEATURES[1:],
    ]
    wardrobe_missing = [
        column for column in dict.fromkeys(wardrobe_required) if column not in wardrobe.columns
    ]
    if wardrobe_missing:
        raise ValueError(f"Thiếu cột bắt buộc trong MonDo: {wardrobe_missing}")

    null_columns = [
        column
        for column in required
        if data[column].isna().any()
    ]
    if null_columns:
        raise ValueError(f"Dữ liệu huấn luyện có giá trị thiếu ở cột: {null_columns}")
    wardrobe_nulls = [
        column
        for column in wardrobe_required
        if wardrobe[column].isna().any()
    ]
    if wardrobe_nulls:
        raise ValueError(f"Danh mục MonDo có giá trị thiếu ở cột: {wardrobe_nulls}")

    numeric = data[NUMERIC_FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Các đặc trưng số và điểm mục tiêu phải là số hữu hạn.")
    target = numeric[TARGET]
    if not target.between(0, 10).all():
        raise ValueError("Điểm mục tiêu 'Điểm phù hợp (0–10)' phải nằm trong khoảng 0–10.")
    for column in CATEGORICAL_FEATURES:
        if data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Cột phân loại '{column}' chứa giá trị rỗng.")


def confidence_calibration(
    uncertainties: np.ndarray, absolute_errors: np.ndarray
) -> tuple[dict[str, object], dict[str, dict[str, float | int]]]:
    low_threshold, high_threshold = np.quantile(uncertainties, CONFIDENCE_QUANTILES)
    thresholds = {
        "high_std_max": float(low_threshold),
        "medium_std_max": float(high_threshold),
        "source": "quantiles_validation_set",
        "quantiles": {
            "high_std_max": CONFIDENCE_QUANTILES[0],
            "medium_std_max": CONFIDENCE_QUANTILES[1],
        },
    }
    labels = np.where(
        uncertainties <= low_threshold,
        "Cao",
        np.where(uncertainties <= high_threshold, "Trung bình", "Thấp"),
    )
    calibration: dict[str, dict[str, float | int]] = {}
    for label in ("Cao", "Trung bình", "Thấp"):
        mask = labels == label
        calibration[label] = {
            "rows": int(mask.sum()),
            "coverage": float(mask.mean()),
            "mean_std": float(uncertainties[mask].mean()) if mask.any() else 0.0,
            "mae": float(absolute_errors[mask].mean()) if mask.any() else 0.0,
        }
    return thresholds, calibration


def verify_fresh_process(
    model_path: Path,
    metadata_path: Path,
    expected: dict[str, object],
) -> dict[str, float]:
    worker_code = """
import json
import sys
import joblib
import numpy as np
import pandas as pd

model_path, metadata_path, sample_json = sys.argv[1:]
artifact = joblib.load(model_path)
with open(metadata_path, encoding="utf-8") as stream:
    metadata = json.load(stream)
features = json.loads(sample_json)
frame = pd.DataFrame([features], columns=metadata["features"])
rf_pipe = artifact["rf_pipe"]
transformed = rf_pipe.named_steps["preprocess"].transform(frame)
rf_model = rf_pipe.named_steps["model"]
tree_scores = np.asarray(
    [tree.predict(transformed)[0] for tree in rf_model.estimators_], dtype=float
)
rf_mean = float(tree_scores.mean())
spread = float(tree_scores.std(ddof=0))
selected = artifact.get("selected_model")
if selected == "Lai":
    knn_score = float(artifact["knn_pipe"].predict(frame)[0])
    weight = float(artifact["w"])
    score = weight * rf_mean + (1 - weight) * knn_score
elif selected == "KNN":
    score = float(artifact["knn_pipe"].predict(frame)[0])
elif selected == "Mốc đoán trung bình":
    score = float(artifact["baseline_value"])
else:
    score = rf_mean
print(json.dumps({"score": float(np.clip(score, 0, 10)), "std": spread}))
"""
    sample_json = json.dumps(expected["features"], ensure_ascii=False)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            worker_code,
            str(model_path),
            str(metadata_path),
            sample_json,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    for key, metadata_key in (("score", "score"), ("std", "std_between_trees")):
        if not np.isclose(actual[key], float(expected[metadata_key]), rtol=0, atol=1e-12):
            raise RuntimeError(
                f"Dự đoán tiến trình mới không khớp: {key} "
                f"{actual[key]} != {expected[metadata_key]}"
            )
    print(
        "Tiến trình mới: "
        f"score={actual['score']:.12f}, std={actual['std']:.12f}; "
        "khớp tiến trình huấn luyện."
    )
    return {"score": float(actual["score"]), "std_between_trees": float(actual["std"])}


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    clipped = np.clip(y_pred, 0, 10)
    return {
        "MAE": float(mean_absolute_error(y_true, clipped)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, clipped))),
        "R2": float(r2_score(y_true, clipped)),
    }


def print_metrics(results: dict[str, dict[str, float]]) -> None:
    print("\nBẢNG SO SÁNH TRÊN TẬP TEST")
    print(f"{'Mô hình':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 52)
    for name, values in results.items():
        print(
            f"{name:<20} {values['MAE']:>10.4f} "
            f"{values['RMSE']:>10.4f} {values['R2']:>10.4f}"
        )


def error_analysis(
    data: pd.DataFrame,
    y_test: pd.Series,
    prediction: np.ndarray,
) -> dict[str, object]:
    errors = pd.DataFrame(
        {
            "Hoàn cảnh": data.loc[y_test.index, "Hoàn cảnh"],
            "actual": y_test,
            "predicted": np.clip(prediction, 0, 10),
        },
        index=y_test.index,
    )
    errors["absolute_error"] = (errors["actual"] - errors["predicted"]).abs()
    by_activity: dict[str, dict[str, float | int]] = {}
    print("\nPHÂN TÍCH LỖI THEO HOÀN CẢNH (mô hình có MAE Test thấp nhất)")
    for activity, subset in errors.groupby("Hoàn cảnh", sort=True):
        mae = float(subset["absolute_error"].mean())
        by_activity[str(activity)] = {
            "rows": int(len(subset)),
            "MAE": mae,
            "zero_score_rows": int((subset["actual"] == 0).sum()),
        }
        print(
            f"- {activity}: {len(subset)} dòng; MAE={mae:.4f}; "
            f"điểm 0={int((subset['actual'] == 0).sum())}"
        )

    zero_all = data[TARGET].astype(float).eq(0)
    zero_test = errors["actual"].eq(0)
    zero_test_mae = (
        float(errors.loc[zero_test, "absolute_error"].mean()) if zero_test.any() else None
    )
    nonzero_test_mae = (
        float(errors.loc[~zero_test, "absolute_error"].mean()) if (~zero_test).any() else None
    )
    print("\nPHÂN TÍCH NHÓM ĐIỂM 0")
    print(
        f"- Toàn bộ dữ liệu: {int(zero_all.sum())}/{len(data)} dòng "
        f"({zero_all.mean()*100:.2f}%) có điểm 0; quy tắc tạo nhãn cho 0 khi nhiệt độ ngoài [Tmin, Tmax]."
    )
    print(
        f"- Tập Test: {int(zero_test.sum())}/{len(errors)} dòng; "
        f"MAE nhóm điểm 0={zero_test_mae if zero_test_mae is not None else float('nan'):.4f}; "
        f"MAE nhóm điểm khác 0={nonzero_test_mae if nonzero_test_mae is not None else float('nan'):.4f}"
    )
    return {
        "by_activity": by_activity,
        "zero_score": {
            "all_rows": int(len(data)),
            "all_zero_rows": int(zero_all.sum()),
            "all_zero_percent": float(zero_all.mean() * 100),
            "test_rows": int(len(errors)),
            "test_zero_rows": int(zero_test.sum()),
            "test_zero_mae": zero_test_mae,
            "test_nonzero_mae": nonzero_test_mae,
        },
    }


def save_charts(
    results: dict[str, dict[str, float]],
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    importances: list[dict[str, float | str]],
    hybrid_weight: float,
    y_train: pd.Series,
    oof_rf: np.ndarray,
    oof_knn: np.ndarray,
    selected_name: str,
) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = list(results)
    ax.bar(labels, [results[name]["MAE"] for name in labels], color="#7c3aed")
    ax.set_title("So sánh MAE trên tập Test")
    ax.set_ylabel("MAE (điểm)")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "mae_comparison.png", dpi=160)
    plt.close(fig)

    selected_pred = np.clip(predictions[selected_name], 0, 10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, selected_pred, s=13, alpha=0.45, color="#7c3aed")
    ax.plot([0, 10], [0, 10], linestyle="--", color="#6b7280", linewidth=1)
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="Điểm thực tế", ylabel="Điểm dự đoán")
    ax.set_title(f"Thực tế và dự đoán — {selected_name}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    top = sorted(importances, key=lambda item: float(item["importance"]), reverse=True)[:9]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [str(item["feature"]) for item in reversed(top)],
        [float(item["importance"]) for item in reversed(top)],
        color="#8b5cf6",
    )
    ax.set_title("Độ quan trọng đặc trưng — Random Forest")
    ax.set_xlabel("Feature importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)

    # Biểu đồ 4: MAE theo trọng số lai trên dự đoán OOF của Train (không dùng Test để chọn w).
    weights = np.linspace(0.0, 1.0, 21)
    maes = [mean_absolute_error(y_train, np.clip(w * oof_rf + (1 - w) * oof_knn, 0, 10))
            for w in weights]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(weights, maes, marker="o", markersize=3, color="#ea580c")
    ax.axvline(hybrid_weight, linestyle="--", color="#15803d",
               label=f"w được chọn = {hybrid_weight:.2f}")
    ax.set_xlabel("Trọng số Random Forest (w); KNN = 1 − w")
    ax.set_ylabel("MAE OOF trên Train")
    ax.set_title("MAE của mô hình lai theo trọng số")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "mae_by_weight.png", dpi=160)
    plt.close(fig)


def main() -> None:
    if not WORKBOOK.is_file():
        raise FileNotFoundError(f"Không tìm thấy workbook: {WORKBOOK}")

    data = pd.read_excel(WORKBOOK, sheet_name="DuLieu_ML", engine="openpyxl")
    wardrobe = pd.read_excel(WORKBOOK, sheet_name="MonDo", engine="openpyxl")
    validate_training_data(data, wardrobe)
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype(str).str.strip()

    # Chia theo ID_SP để không có sản phẩm xuất hiện ở nhiều tập.
    all_groups = data["ID_SP"].astype(str)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    development_idx, test_idx = next(
        splitter_test.split(data, None, groups=all_groups)
    )
    development = data.iloc[development_idx].copy()
    test = data.iloc[test_idx].copy()
    validation_fraction = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    splitter_validation = GroupShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=RANDOM_STATE + 1
    )
    train_rel_idx, validation_rel_idx = next(
        splitter_validation.split(
            development, None,
            groups=development["ID_SP"].astype(str),
        )
    )
    train = development.iloc[train_rel_idx].copy()
    validation = development.iloc[validation_rel_idx].copy()
    X_train = train.loc[:, FEATURES].copy()
    X_test = test.loc[:, FEATURES].copy()
    X_validation = validation.loc[:, FEATURES].copy()
    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)
    y_validation = validation[TARGET].astype(float)
    groups = train["ID_SP"].astype(str).to_numpy()
    partitions = {
        "train": set(train["ID_SP"].astype(str)),
        "validation": set(validation["ID_SP"].astype(str)),
        "test": set(test["ID_SP"].astype(str)),
    }
    if (partitions["train"] & partitions["validation"] or
            partitions["train"] & partitions["test"] or
            partitions["validation"] & partitions["test"]):
        raise RuntimeError("Rò rỉ nhóm ID_SP giữa Train/Validation/Test.")
    overlap: set[str] = set()
    group_count = int(pd.Series(groups).nunique())
    if group_count < 5:
        raise ValueError("Cần ít nhất 5 giá trị ID_SP trong Train cho GroupKFold(5).")
    cv = GroupKFold(n_splits=5)

    print(f"Tổng số dòng DuLieu_ML: {len(data)}")
    print(f"Train theo ID_SP: {len(train)} dòng ({len(train)/len(data):.1%})")
    print(f"Validation theo ID_SP: {len(validation)} dòng ({len(validation)/len(data):.1%})")
    print(f"Test theo ID_SP: {len(test)} dòng ({len(test)/len(data):.1%})")
    print("Số ID_SP giao nhau giữa các tập: 0 (bắt buộc = 0)")
    print(f"Nhóm ID_SP khác nhau cho GroupKFold: {group_count}; số fold: 5")
    print(f"Đầu vào model ({len(FEATURES)} cột): {', '.join(FEATURES)}")

    print(
        f"\nThử cấu hình RF gọn hơn (n_jobs=-1, joblib compress={JOBLIB_COMPRESSION}):"
    )
    rf_size_trials: list[dict[str, object]] = []
    selected_rf_pipe: Pipeline | None = None
    selected_rf_trial: dict[str, object] | None = None
    allowed_rf_mae: float | None = None
    full_rf_mae: float | None = None
    for configuration in RF_SIZE_CANDIDATES:
        candidate_pipe = make_pipeline(
            scale_numeric=False,
            estimator=RandomForestRegressor(
                n_estimators=RF_TREES,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=int(configuration["min_samples_leaf"]),
                max_depth=configuration["max_depth"],
            ),
        )
        candidate_pipe.fit(X_train, y_train)
        candidate_prediction = np.clip(candidate_pipe.predict(X_validation), 0, 10)
        candidate_metrics = metrics(y_validation, candidate_prediction)
        with tempfile.TemporaryDirectory(prefix="outfitai-model-size-") as temp_dir:
            temporary_model = Path(temp_dir) / "candidate.joblib"
            joblib.dump(candidate_pipe, temporary_model, compress=JOBLIB_COMPRESSION)
            compressed_bytes = temporary_model.stat().st_size
        trial = {
            **configuration,
            "metrics_validation": candidate_metrics,
            "compressed_bytes": int(compressed_bytes),
        }
        rf_size_trials.append(trial)
        print(
            f"- {configuration['name']}: Validation MAE={candidate_metrics['MAE']:.4f}; "
            f"nén={compressed_bytes:,} bytes"
        )
        if configuration["name"] == "full":
            full_rf_mae = candidate_metrics["MAE"]
            allowed_rf_mae = full_rf_mae * (1 + MAX_RF_MAE_REGRESSION)
            selected_rf_pipe = candidate_pipe
            selected_rf_trial = trial
        elif (
            candidate_metrics["MAE"] <= float(allowed_rf_mae) + 1e-12
            and int(compressed_bytes) < int(selected_rf_trial["compressed_bytes"])
        ):
            selected_rf_pipe = candidate_pipe
            selected_rf_trial = trial

    if selected_rf_pipe is None or selected_rf_trial is None or allowed_rf_mae is None:
        raise RuntimeError("Không huấn luyện được cấu hình Random Forest.")
    selected_rf_name = str(selected_rf_trial["name"])
    rf_pipe = selected_rf_pipe
    selected_rf_configuration = {
        "name": selected_rf_name,
        "min_samples_leaf": int(selected_rf_trial["min_samples_leaf"]),
        "max_depth": selected_rf_trial["max_depth"],
        "full_rf_mae": full_rf_mae,
        "max_allowed_mae": allowed_rf_mae,
        "max_mae_regression_percent": MAX_RF_MAE_REGRESSION * 100,
    }
    print(
        f"RF đã chọn: {selected_rf_name}; Validation MAE={selected_rf_trial['metrics_validation']['MAE']:.4f} "
        f"(giới hạn {allowed_rf_mae:.4f}); file nén thử={selected_rf_trial['compressed_bytes']:,} bytes"
    )
    knn_cv_rows: list[tuple[int, float]] = []
    for k in K_CANDIDATES:
        candidate = make_pipeline(
            scale_numeric=True,
            estimator=KNeighborsRegressor(n_neighbors=k, n_jobs=-1),
        )
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            groups=groups,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        knn_cv_rows.append((k, float(-scores.mean())))
        print(f"CV KNN K={k:<2}: MAE={-scores.mean():.4f}")
    best_k, best_knn_cv_mae = min(knn_cv_rows, key=lambda item: item[1])
    knn_pipe = make_pipeline(
        scale_numeric=True,
        estimator=KNeighborsRegressor(n_neighbors=best_k, n_jobs=-1),
    )

    print("Đang tạo dự đoán out-of-fold cho Random Forest và KNN để chọn trọng số lai...")
    rf_oof = cross_val_predict(
        rf_pipe, X_train, y_train, groups=groups, cv=cv, n_jobs=1, method="predict"
    )
    knn_oof = cross_val_predict(
        knn_pipe, X_train, y_train, groups=groups, cv=cv, n_jobs=1, method="predict"
    )
    best_weight = 0.0
    best_oof_mae = float("inf")
    for weight in HYBRID_WEIGHT_CANDIDATES:
        hybrid_oof = np.clip(weight * rf_oof + (1 - weight) * knn_oof, 0, 10)
        score = mean_absolute_error(y_train, hybrid_oof)
        if score < best_oof_mae:
            best_oof_mae = float(score)
            best_weight = float(weight)

    # Chọn mô hình mặc định bằng Validation; mọi dự đoán lựa chọn đều trước khi mở Test.
    validation_rf = np.clip(rf_pipe.predict(X_validation), 0, 10)
    validation_knn_pipe = make_pipeline(scale_numeric=True,
        estimator=KNeighborsRegressor(n_neighbors=best_k, n_jobs=-1))
    validation_knn_pipe.fit(X_train, y_train)
    validation_knn = np.clip(validation_knn_pipe.predict(X_validation), 0, 10)
    baseline_train = float(np.clip(y_train.mean(), 0, 10))
    validation_predictions = {
        "Mốc đoán trung bình": np.full(len(y_validation), baseline_train),
        "KNN": validation_knn,
        "RF": validation_rf,
        "Lai": np.clip(best_weight * validation_rf + (1 - best_weight) * validation_knn, 0, 10),
    }
    validation_metrics = {
        name: metrics(y_validation, pred) for name, pred in validation_predictions.items()
    }
    validation_best_single_model = min(
        validation_metrics, key=lambda name: validation_metrics[name]["MAE"]
    )
    # The shipped prediction intentionally blends both models; the weight is OOF-selected.
    chosen_name = "Lai"
    validation_transformed = rf_pipe.named_steps["preprocess"].transform(X_validation)
    validation_tree_predictions = np.asarray(
        [tree.predict(validation_transformed) for tree in rf_pipe.named_steps["model"].estimators_],
        dtype=float,
    )
    validation_uncertainties = validation_tree_predictions.std(axis=0, ddof=0)
    validation_absolute_errors = np.abs(
        y_validation.to_numpy(dtype=float) - validation_predictions[chosen_name]
    )
    confidence_thresholds, confidence_calibration_validation = confidence_calibration(
        validation_uncertainties, validation_absolute_errors
    )

    # Chốt cấu hình xong mới fit mô hình phát hành bằng Train + Validation.
    X_development = development.loc[:, FEATURES].copy()
    y_development = development[TARGET].astype(float)
    rf_pipe.fit(X_development, y_development)
    knn_pipe.fit(X_development, y_development)
    pred_rf = np.clip(rf_pipe.predict(X_test), 0, 10)
    pred_knn = np.clip(knn_pipe.predict(X_test), 0, 10)
    baseline_value = float(np.clip(y_development.mean(), 0, 10))
    pred_mean = np.full(len(y_test), baseline_value)
    pred_hybrid = np.clip(best_weight * pred_rf + (1 - best_weight) * pred_knn, 0, 10)
    predictions = {
        "Mốc đoán trung bình": pred_mean,
        "KNN": pred_knn,
        "RF": pred_rf,
        "Lai": pred_hybrid,
    }
    results = {name: metrics(y_test, pred) for name, pred in predictions.items()}
    print_metrics(results)
    diagnostics = error_analysis(data, y_test, predictions[chosen_name])
    transformed_test = rf_pipe.named_steps["preprocess"].transform(X_test)
    rf_estimator = rf_pipe.named_steps["model"]
    tree_predictions_test = np.asarray(
        [tree.predict(transformed_test) for tree in rf_estimator.estimators_],
        dtype=float,
    )
    test_uncertainties = tree_predictions_test.std(axis=0, ddof=0)
    print(f"KNN tốt nhất theo GroupKFold: K={best_k} (CV MAE={best_knn_cv_mae:.4f})")
    print(f"Trọng số lai chọn từ OOF: w_RF={best_weight:.2f}; w_KNN={1-best_weight:.2f}")
    print(
        f"Mô hình mặc định: Lai (chọn trọng số OOF trên Train); "
        f"mô hình có MAE Validation thấp nhất: {validation_best_single_model}"
    )
    if best_weight >= 0.95:
        print("LƯU Ý: trọng số lai gần 1.0; mô hình lai chủ yếu dựa vào RF.")
    print("Đối chiếu HuongDan: RF tham khảo MAE≈0.30, R²≈0.97; KNN K=5 tham khảo MAE≈0.75, R²≈0.80.")
    print("Ghi chú: KNN trong bảng Test dùng K được chọn bằng GroupKFold, không cố định K=5.")

    feature_importances: list[dict[str, float | str]] = []
    transformed_names = rf_pipe.named_steps["preprocess"].get_feature_names_out()
    raw_importances = rf_pipe.named_steps["model"].feature_importances_
    grouped_importances = {feature: 0.0 for feature in FEATURES}
    for transformed_name, importance in zip(transformed_names, raw_importances):
        name = str(transformed_name)
        if name.startswith("num__"):
            source = name.removeprefix("num__")
        else:
            source = next(
                (feature for feature in CATEGORICAL_FEATURES if name.startswith(f"cat__{feature}_")),
                name,
            )
        grouped_importances[source] = grouped_importances.get(source, 0.0) + float(importance)
    feature_importances = [
        {"feature": feature, "importance": float(value)}
        for feature, value in grouped_importances.items()
    ]

    category_columns = {
        "Giới tính": "Giới tính",
        "Loại TP": "Loại TP",
        "Màu sắc": "Màu sắc",
    }
    categories = {
        feature: sorted(data[feature].dropna().astype(str).str.strip().unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }
    for public_name, source_name in category_columns.items():
        categories[public_name] = sorted(
            wardrobe[source_name].dropna().astype(str).str.strip().unique().tolist()
        )
    feature_ranges = {
        column: {"min": float(data[column].min()), "max": float(data[column].max())}
        for column in NUMERIC_FEATURES
    }
    feature_ranges["Nhiệt độ ngày (°C)"] = {"min": 5.0, "max": 40.0}

    limitations = [
        "Nhãn được sinh từ công thức quy tắc; chỉ số R² cao phản ánh phần lớn việc học lại công thức, không phải kiểm chứng sở thích thực tế.",
        "Train/Validation/Test được chia theo ID_SP bằng GroupShuffleSplit (mục tiêu 70/15/15), không dùng cột 'Tập'.",
        "Mô hình không sử dụng giới tính, lịch học, vị trí, mưa hoặc gió làm đặc trưng.",
        "Độ lệch chuẩn giữa các cây RF là chỉ báo phân tán; các nhãn tin cậy chia theo phân vị độ lệch chuẩn trên Validation, không phải khoảng tin cậy thống kê.",
        "Cấu hình RF và mô hình mặc định được chọn bằng Validation; Test chỉ dùng để đánh giá cuối cùng.",
    ]
    roundtrip_features = json.loads(X_test.iloc[[0]].to_json(orient="records"))[0]
    roundtrip_score = float(predictions[chosen_name][0])
    roundtrip_std = float(test_uncertainties[0])
    roundtrip_check = {
        "features": roundtrip_features,
        "score": roundtrip_score,
        "std_between_trees": roundtrip_std,
        "score_low": float(max(0, roundtrip_score - roundtrip_std)),
        "score_high": float(min(10, roundtrip_score + roundtrip_std)),
    }
    metadata = {
        "model_name": chosen_name,
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "dataset": "data/DatasetDSS.xlsx",
        "sheet": "DuLieu_ML",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "rows": {"total": int(len(data)), "train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "split": {
            "method": "GroupShuffleSplit", "group_column": "ID_SP",
            "target_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "group_counts": {name: len(values) for name, values in partitions.items()},
            "group_ids": {name: sorted(values) for name, values in partitions.items()},
            "groups_disjoint": True,
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "target": TARGET,
        "categories": categories,
        "feature_ranges": feature_ranges,
        "metrics_test": results,
        "metrics_validation": validation_metrics,
        "baseline_mae": results["Mốc đoán trung bình"]["MAE"],
        "error_analysis_test": diagnostics,
        "selected_model": chosen_name,
        "baseline_value": baseline_value,
        "hybrid_weight": {"rf": best_weight, "knn": 1 - best_weight},
        "hybrid_oof_mae": best_oof_mae,
        "hybrid_weight_selection": {
            "method": "GroupKFold(5) out-of-fold MAE on Train",
            "candidate_rf_weights": [float(value) for value in HYBRID_WEIGHT_CANDIDATES],
            "both_models_required": True,
            "minimum_knn_weight": 0.05,
        },
        "validation_best_model_by_mae": validation_best_single_model,
        "best_k": best_k,
        "group_kfold": {"n_splits": 5, "group_column": "ID_SP", "n_groups_train": group_count},
        "rf_size_trials": rf_size_trials,
        "selected_rf_configuration": selected_rf_configuration,
        "confidence_thresholds": confidence_thresholds,
        "confidence_calibration_validation": confidence_calibration_validation,
        "roundtrip_check": roundtrip_check,
        "feature_importances": feature_importances,
        "limitations": limitations,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "rf_pipe": rf_pipe,
        "knn_pipe": knn_pipe,
        "w": best_weight,
        "selected_model": chosen_name,
        "baseline_value": baseline_value,
        "version": metadata["version"],
        "features": FEATURES,
    }
    model_path = MODEL_DIR / "outfit_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"
    csv_path = DATA_DIR / "mon_do.csv"
    with tempfile.TemporaryDirectory(prefix="outfitai-model-baseline-") as temp_dir:
        baseline_model = Path(temp_dir) / "baseline_compress3.joblib"
        joblib.dump(model_artifact, baseline_model, compress=3)
        compression_baseline_bytes = baseline_model.stat().st_size
    joblib.dump(model_artifact, model_path, compress=JOBLIB_COMPRESSION)
    metadata["model_size_bytes"] = int(model_path.stat().st_size)
    metadata["model_size_baseline_bytes"] = int(compression_baseline_bytes)
    metadata["joblib_compression"] = JOBLIB_COMPRESSION
    metadata["model_size_reduction_percent"] = float(
        (compression_baseline_bytes - model_path.stat().st_size)
        / compression_baseline_bytes
        * 100
    )
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    wardrobe.to_csv(csv_path, index=False, encoding="utf-8-sig")

    importances_for_chart = feature_importances
    save_charts(results, y_test, predictions, importances_for_chart, best_weight, y_train, rf_oof, knn_oof, chosen_name)

    print("\nKiểm tra nạp joblib trong tiến trình Python mới:")
    verify_fresh_process(model_path, metadata_path, roundtrip_check)

    generated_files = [model_path, metadata_path, csv_path, *sorted(CHART_DIR.glob("*.png"))]
    print("\nFILE ĐÃ SINH:")
    for path in generated_files:
        print(f"- {path.relative_to(ROOT).as_posix()} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Lỗi huấn luyện/export: {exc}", file=sys.stderr)
        raise
