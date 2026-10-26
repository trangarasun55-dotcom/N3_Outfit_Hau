"""Reproduce held-out model metrics and non-personalized CBF quality checks."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.config import MODEL_METADATA, OUTFIT_MIN_SCORE, OUTFIT_MMR_LAMBDA, TOP_K_PER_SLOT
from api.recommender import _model_scores, recommend
from api.schemas import RecommendRequest

WORKBOOK = ROOT / "data" / "DatasetDSS.xlsx"
MODEL_PATH = ROOT / "model" / "outfit_model.joblib"
OUTPUT = ROOT / "artifacts" / "evaluation"
SEED = 42
RELEVANCE_THRESHOLD = 6.0
K_VALUES = (1, 3, 5)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranking_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> dict[str, float]:
    relevant = y_true >= RELEVANCE_THRESHOLD
    order = np.argsort(-y_score, kind="stable")
    top = order[:k]
    hits = relevant[top]
    denominator = min(k, len(y_true))
    precision = float(hits.sum() / denominator) if denominator else 0.0
    relevant_count = int(relevant.sum())
    recall = float(hits.sum() / relevant_count) if relevant_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precisions = [float(relevant[order[:rank]].sum() / rank) for rank in range(1, k + 1)]
    ap = (
        sum(precision_at_rank for precision_at_rank, is_relevant
            in zip(precisions, relevant[order[:k]]) if is_relevant)
        / min(relevant_count, k)
        if relevant_count
        else 0.0
    )
    gains = relevant[top].astype(float)
    discounts = np.log2(np.arange(2, len(top) + 2))
    dcg = float(np.sum(gains / discounts))
    ideal_count = min(relevant_count, k)
    idcg = float(np.sum(np.ones(ideal_count) / np.log2(np.arange(2, ideal_count + 2))))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": float(f1),
        "map_at_k": float(ap),
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def main() -> None:
    started = time.perf_counter()
    metadata = MODEL_METADATA
    artifact = joblib.load(MODEL_PATH)
    frame = pd.read_excel(WORKBOOK, sheet_name=metadata["sheet"], engine="openpyxl")
    test_ids = set(metadata["split"]["group_ids"]["test"])
    test = frame[frame["ID_SP"].astype(str).isin(test_ids)].copy()
    if len(test) != metadata["n_test"]:
        raise RuntimeError("Số dòng Test không khớp metadata; dừng đánh giá.")
    features = metadata["features"]
    target = metadata["target"]
    X_test = test.loc[:, features]
    y_test = test[target].astype(float).to_numpy()
    predictions = {
        "Mốc đoán trung bình": np.full(len(test), float(artifact["baseline_value"])),
        "KNN": artifact["knn_pipe"].predict(X_test),
        "RF": artifact["rf_pipe"].predict(X_test),
    }
    alpha = float(artifact["w"])
    predictions["Lai"] = alpha * predictions["RF"] + (1.0 - alpha) * predictions["KNN"]

    model_metrics: dict[str, dict[str, float]] = {}
    for name, raw in predictions.items():
        clipped = np.clip(raw, 0, 10)
        model_metrics[name] = {
            "mae_points": float(mean_absolute_error(y_test, clipped)),
            "rmse_points": float(np.sqrt(mean_squared_error(y_test, clipped))),
            "r2": float(r2_score(y_test, clipped)),
        }
    for name, metrics in model_metrics.items():
        for metric, value in metrics.items():
            expected = metadata["metrics_test"][name][{"mae_points": "MAE", "rmse_points": "RMSE", "r2": "R2"}[metric]]
            if not np.isclose(value, float(expected), rtol=0, atol=1e-10):
                raise RuntimeError(f"Test metric không tái lập cho {name}.{metric}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame([
        {"model": name, **values} for name, values in model_metrics.items()
    ])
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False, encoding="utf-8")
    _json(OUTPUT / "model_metrics.json", {
        "status": "PASS",
        "test_rows": len(test),
        "test_product_groups": len(test_ids),
        "target": target,
        "target_semantics": "rule_generated_suitability_score; not observed user preference",
        "metrics": model_metrics,
        "selected_model": metadata["selected_model"],
        "hybrid_weight": {"rf": alpha, "knn": 1.0 - alpha},
    })

    grouped_rankings: dict[str, dict[str, list[float]]] = {
        name: {f"{metric}@{k}": [] for k in K_VALUES
               for metric in ("precision", "recall", "f1", "map", "ndcg")}
        for name in predictions
    }
    group_columns = ["Hoàn cảnh", "Nhiệt độ ngày (°C)"]
    for _, indices in test.groupby(group_columns, sort=True).groups.items():
        positions = test.index.get_indexer(indices)
        if len(positions) < 1 or not np.any(y_test[positions] >= RELEVANCE_THRESHOLD):
            continue
        group_truth = y_test[positions]
        for name, scores in predictions.items():
            for k in K_VALUES:
                values = _ranking_at_k(group_truth, np.clip(scores[positions], 0, 10), k)
                for metric, value in values.items():
                    key = metric.replace("_at_k", f"@{k}")
                    grouped_rankings[name][key].append(value)
    ranking_result = {
        "status": "MEASURED_PROXY",
        "ranking_scope": "held-out product groups within activity and exact temperature groups",
        "relevance_definition": f"workbook score >= {RELEVANCE_THRESHOLD:g}/10",
        "ground_truth_source": "independent held-out workbook labels generated by deterministic business rules",
        "limitation": "These ranking metrics measure agreement with the workbook rule labels, not observed user preference.",
        "groups_with_relevant_items": max((len(v) for model in grouped_rankings.values() for v in model.values()), default=0),
        "k_values": list(K_VALUES),
        "models": {
            name: {
                metric: (float(np.mean(values)) if values else None)
                for metric, values in metric_values.items()
            }
            for name, metric_values in grouped_rankings.items()
        },
    }
    _json(OUTPUT / "ranking_metrics.json", ranking_result)

    catalog = pd.read_csv(ROOT / "data" / "mon_do.csv", dtype={"ID_SP": "string"})
    cases = [
        (gender, activity, temperature, style)
        for gender in ("Nam", "Nữ")
        for activity in ("Đi học", "Xưởng/Công trình", "Vẽ ngoại cảnh", "Thuyết trình")
        for temperature, style in ((12.0, "Casual"), (28.0, "Casual"))
    ]
    cases.extend([
        ("Nam", "Đi học", 20.0, "Smart Casual"),
        ("Nữ", "Xưởng/Công trình", 34.0, "Streetwear"),
        ("Nam", "Vẽ ngoại cảnh", 34.0, "Minimalist"),
        ("Nữ", "Thuyết trình", 12.0, "Artsy"),
    ])
    recommended_ids: set[str] = set()
    scenario_sizes: list[int] = []
    scenario_latencies_ms: list[float] = []
    pairwise_diversity: list[float] = []
    for gender, activity, temperature, style in cases:
        request = RecommendRequest(
            gender=gender, activity=activity, temperature=temperature,
            styles=[style], top_n=3,
        )
        before = time.perf_counter()
        result = recommend(catalog, artifact, request)
        scenario_latencies_ms.append((time.perf_counter() - before) * 1000)
        outfits = result["outfits"]
        scenario_sizes.append(len(outfits))
        for outfit in outfits:
            recommended_ids.update(str(item["id"]) for item in outfit["items"])
        for index, left in enumerate(outfits):
            left_ids = {str(item["id"]) for item in left["items"]}
            for right in outfits[index + 1:]:
                right_ids = {str(item["id"]) for item in right["items"]}
                union = left_ids | right_ids
                pairwise_diversity.append(1.0 - len(left_ids & right_ids) / len(union) if union else 0.0)

    _json(OUTPUT / "recommendation_quality.json", {
        "status": "MEASURED_SYNTHETIC_SCENARIOS",
        "scenario_count": len(cases),
        "scenario_source": "deterministic test contexts from recommender regression suite; not real user requests",
        "real_user_relevance": {
            "status": "NOT_EVALUABLE",
            "reason": "The project snapshot has no observed outfit ratings/feedback suitable as user-item relevance ground truth.",
        },
        "diversity": {
            "definition": "mean pairwise 1 - Jaccard(item IDs) across returned outfits within each request",
            "measured_pairs": len(pairwise_diversity),
            "mean": float(np.mean(pairwise_diversity)) if pairwise_diversity else None,
        },
        "catalog_coverage": {
            "unique_items_recommended": len(recommended_ids),
            "catalog_items": int(catalog["ID_SP"].nunique()),
            "fraction": len(recommended_ids) / max(1, int(catalog["ID_SP"].nunique())),
            "scope": "synthetic scenarios only; not production traffic coverage",
        },
        "outfits_per_scenario": {
            "requested": 3,
            "minimum": min(scenario_sizes),
            "mean": float(np.mean(scenario_sizes)),
            "maximum": max(scenario_sizes),
            "scenarios_with_fewer_than_requested": sum(size < 3 for size in scenario_sizes),
        },
    })

    feature_frame = X_test.iloc[: min(900, len(X_test))].copy()
    inference_times = []
    for _ in range(3):
        before = time.perf_counter()
        _model_scores(artifact, feature_frame)
        inference_times.append((time.perf_counter() - before) * 1000)
    runtime = {
        "status": "MEASURED",
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "model_batch": {
            "rows": len(feature_frame), "repetitions": len(inference_times),
            "median_ms": float(statistics.median(inference_times)),
            "mean_ms": float(statistics.mean(inference_times)),
            "unit": "milliseconds per batch",
        },
        "recommendation_scenarios": {
            "count": len(scenario_latencies_ms),
            "median_ms": float(statistics.median(scenario_latencies_ms)),
            "mean_ms": float(statistics.mean(scenario_latencies_ms)),
            "unit": "milliseconds per in-process recommendation call",
        },
        "limitations": "No HTTP, cold-start, concurrent load, or production hardware benchmark was run.",
    }
    _json(OUTPUT / "runtime_metrics.json", runtime)

    _json(OUTPUT / "evaluation_config.json", {
        "seed": SEED,
        "split": metadata["split"],
        "test_rows": len(test),
        "target": target,
        "relevance_threshold_for_proxy_ranking": RELEVANCE_THRESHOLD,
        "ranking_groups": group_columns,
        "ranking_k_values": list(K_VALUES),
        "rf": metadata["selected_rf_configuration"],
        "knn": {"n_neighbors": metadata["best_k"], "selection": "GroupKFold(5) on Train"},
        "hybrid": metadata["hybrid_weight_selection"],
        "content_based_filtering": {
            "similarity": "cosine",
            "profile": ["explicit styles", "explicit color choices", "requested formality"],
            "candidate_limit_per_slot": TOP_K_PER_SLOT,
        },
        "dss": {
            "minimum_outfit_score": OUTFIT_MIN_SCORE,
            "mmr_lambda": OUTFIT_MMR_LAMBDA,
            "top_n_cases": 3,
        },
    })

    missing_ml = frame.isna().sum()
    wardrobe = pd.read_excel(WORKBOOK, sheet_name="MonDo", engine="openpyxl")
    _json(OUTPUT / "data_profile.json", {
        "workbook_sha256": _sha256(WORKBOOK),
        "ml_rows": len(frame),
        "ml_columns": list(frame.columns),
        "ml_missing_cells": int(missing_ml.sum()),
        "ml_missing_by_column": {str(key): int(value) for key, value in missing_ml.items() if value},
        "distinct_product_groups": int(frame["ID_SP"].astype(str).nunique()),
        "label_range": {
            "min": float(frame[target].min()), "max": float(frame[target].max()), "unit": "points on 0–10 rule-generated scale"
        },
        "wardrobe_rows": len(wardrobe),
        "wardrobe_distinct_products": int(wardrobe["ID_SP"].astype(str).nunique()),
        "wardrobe_missing_cells": int(wardrobe.isna().sum().sum()),
        "temporal_fields": [],
        "split": metadata["split"],
    })

    files_to_hash = [MODEL_PATH, ROOT / "model" / "model_metadata.json", WORKBOOK]
    _json(OUTPUT / "run_manifest.json", {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_revision": None,
        "revision_status": "No Git repository/commit was available in the supplied ZIP.",
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "status": "PASS",
        "duration_seconds": round(time.perf_counter() - started, 3),
        "files_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in files_to_hash},
        "outputs": [
            "model_comparison.csv", "model_metrics.json", "ranking_metrics.json",
            "recommendation_quality.json", "runtime_metrics.json", "evaluation_config.json",
            "data_profile.json",
        ],
    })
    print(f"Evaluation complete: {len(test)} held-out rows; outputs written to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
