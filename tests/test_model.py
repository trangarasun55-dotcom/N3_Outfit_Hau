"""Regression checks for the model imported from Train.rar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "outfit_model.joblib"
METADATA_PATH = ROOT / "model" / "model_metadata.json"
WORKBOOK_PATH = ROOT / "data" / "DatasetDSS.xlsx"
CHARTS = (
    ROOT / "model" / "charts" / "mae_comparison.png",
    ROOT / "model" / "charts" / "actual_vs_predicted.png",
    ROOT / "model" / "charts" / "feature_importance.png",
)

def load_metadata() -> dict[str, object]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

def test_imported_model_matches_api_contract_and_metadata() -> None:
    artifact = joblib.load(MODEL_PATH)
    metadata = load_metadata()
    assert artifact["features"] == metadata["features"]
    assert artifact["selected_model"] == "Lai"
    assert 0.05 <= artifact["w"] <= 0.95
    assert artifact["w"] > 0 and 1 - artifact["w"] > 0
    assert len(metadata["features"]) == 9
    encoder = artifact["rf_pipe"].named_steps["preprocess"].named_transformers_["cat"]
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "ignore"
    assert isinstance(artifact["knn_pipe"].named_steps["model"], KNeighborsRegressor)
    assert artifact["knn_pipe"].named_steps["model"].n_neighbors == metadata["best_k"]
    assert artifact["rf_pipe"].named_steps["model"].n_estimators == 200

def test_workbook_partition_matches_metadata_and_has_no_id_overlap() -> None:
    data = pd.read_excel(WORKBOOK_PATH, sheet_name="DuLieu_ML", engine="openpyxl")
    metadata = load_metadata()
    assert len(data) == metadata["rows"]["total"]
    split = metadata["split"]
    assert metadata["n_train"] == metadata["rows"]["train"]
    assert metadata["n_validation"] == metadata["rows"]["validation"]
    assert metadata["n_test"] == metadata["rows"]["test"]
    assert split["groups_disjoint"] is True
    group_ids = split["group_ids"]
    assert set(group_ids["train"]).isdisjoint(group_ids["validation"])
    assert set(group_ids["train"]).isdisjoint(group_ids["test"])
    assert set(group_ids["validation"]).isdisjoint(group_ids["test"])
    assert set(data["ID_SP"].astype(str)) == set().union(*map(set, group_ids.values()))

def test_unknown_categories_and_api_scoring_path_are_finite() -> None:
    sys.path.insert(0, str(ROOT))
    from api.recommender import _model_scores

    artifact = joblib.load(MODEL_PATH)
    metadata = load_metadata()
    frame = pd.DataFrame([metadata["roundtrip_check"]["features"]], columns=metadata["features"])
    frame.loc[:, "Hoàn cảnh"] = "Giá trị phân loại chưa gặp"
    rf_pipe = artifact["rf_pipe"]
    transformed = rf_pipe.named_steps["preprocess"].transform(frame)
    tree_predictions = np.asarray([tree.predict(transformed)[0] for tree in rf_pipe.named_steps["model"].estimators_])
    rf_mean = float(tree_predictions.mean())
    selected = artifact["selected_model"]
    if selected == "Lai":
        score = artifact["w"] * rf_mean + (1 - artifact["w"]) * float(artifact["knn_pipe"].predict(frame)[0])
    elif selected == "KNN":
        score = float(artifact["knn_pipe"].predict(frame)[0])
    elif selected == "Mốc đoán trung bình":
        score = float(artifact["baseline_value"])
    else:
        score = rf_mean
    assert np.isfinite(score)
    assert 0 <= score <= 10
    assert np.isfinite(tree_predictions).all()
    for model_name in ("Mốc đoán trung bình", "KNN", "RF", "Lai"):
        variant = {**artifact, "selected_model": model_name}
        api_scores, api_std = _model_scores(variant, frame)
        assert np.isfinite(api_scores).all() and np.isfinite(api_std).all()
        assert ((api_scores >= 0) & (api_scores <= 10)).all()

def test_metadata_metrics_and_artifacts_are_present() -> None:
    metadata = load_metadata()
    assert set(metadata["metrics_test"]) == {"Mốc đoán trung bình", "KNN", "RF", "Lai"}
    assert metadata["selected_model"] == "Lai"
    assert metadata["validation_best_model_by_mae"] in metadata["metrics_validation"]
    assert metadata["hybrid_weight_selection"]["both_models_required"] is True
    assert metadata["confidence_thresholds"]["source"] == "quantiles_validation_set"
    assert "confidence_calibration_test" not in metadata
    assert MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0
    assert METADATA_PATH.is_file()
    assert (ROOT / "data" / "mon_do.csv").is_file()
    assert all(path.is_file() and path.stat().st_size > 0 for path in CHARTS)

def test_public_model_info_omits_test_set_diagnostics() -> None:
    sys.path.insert(0, str(ROOT))
    from api.main import model_info

    public_info = model_info()
    assert "metrics_test" not in public_info
    assert "metrics_validation" not in public_info
    assert "group_ids" not in public_info.get("split", {})
    assert "categories" in public_info and "features" in public_info
