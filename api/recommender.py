"""Ghép outfit trên API từ điểm pipeline; rule scores chỉ để diễn giải."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

try:
    from .config import (
        COLOR_MAP, CONFIDENCE_HIGH_STD, CONFIDENCE_MEDIUM_STD,
        JACKET_SCORE_THRESHOLD, MODEL_METADATA, MODEL_TEMP_RANGE, OUTFIT_MIN_SCORE,
        OUTFIT_MMR_LAMBDA, OUTFIT_SCORE_BAND, TOP_K_PER_SLOT,
    )
    from .rules import explain_item
except ImportError:  # chạy `uvicorn main:app` từ thư mục api
    from config import (
        COLOR_MAP, CONFIDENCE_HIGH_STD, CONFIDENCE_MEDIUM_STD,
        JACKET_SCORE_THRESHOLD, MODEL_METADATA, MODEL_TEMP_RANGE, OUTFIT_MIN_SCORE,
        OUTFIT_MMR_LAMBDA, OUTFIT_SCORE_BAND, TOP_K_PER_SLOT,
    )
    from rules import explain_item

FEATURES = MODEL_METADATA["features"]
CONTENT_WEIGHT = 0.35
MIN_SCORE = 5.0
MIN_ITEMS = 1


def _normalize_label(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold().strip()
    return "".join(char for char in text if not unicodedata.combining(char))


def confidence_label(std: float) -> str:
    if std <= CONFIDENCE_HIGH_STD:
        return "Cao"
    if std <= CONFIDENCE_MEDIUM_STD:
        return "Trung bình"
    return "Thấp"


def _model_scores(artifact: dict[str, Any], feature_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rf_pipe = artifact["rf_pipe"]
    transformed = rf_pipe.named_steps["preprocess"].transform(feature_frame.loc[:, FEATURES])
    rf_model = rf_pipe.named_steps["model"]
    tree_predictions = np.vstack([tree.predict(transformed) for tree in rf_model.estimators_])
    rf_mean = tree_predictions.mean(axis=0)
    rf_std = tree_predictions.std(axis=0)
    selected_model = artifact.get("selected_model", "RF")
    if selected_model == "Lai":
        knn_mean = artifact["knn_pipe"].predict(feature_frame.loc[:, FEATURES])
        weight = float(artifact["w"])
        score = weight * rf_mean + (1 - weight) * knn_mean
    elif selected_model == "KNN":
        score = artifact["knn_pipe"].predict(feature_frame.loc[:, FEATURES])
    elif selected_model == "Mốc đoán trung bình":
        score = np.full(len(feature_frame), float(artifact["baseline_value"]))
    else:
        score = rf_mean
    return np.clip(score, 0, 10), rf_std


def _dataset_color_match(item_color: str, selected_colors: list[str]) -> bool:
    return any(item_color in COLOR_MAP.get(color, ()) for color in selected_colors)


def _content_similarity(
    catalog: pd.DataFrame,
    *,
    styles: list[str],
    colors: list[str],
    formality: int,
) -> np.ndarray:
    """Cosine match between catalog content and the user's explicit session profile."""
    style_values = sorted(catalog["Phong cách"].astype(str).unique())
    color_values = sorted(catalog["Màu sắc"].astype(str).unique())
    style_index = {value: index for index, value in enumerate(style_values)}
    color_index = {value: index for index, value in enumerate(color_values)}
    item_vectors = np.zeros(
        (len(catalog), len(style_values) + len(color_values) + 2), dtype=float
    )
    style_column = catalog["Phong cách"].astype(str).to_numpy()
    color_column = catalog["Màu sắc"].astype(str).to_numpy()
    neatness_column = catalog["Chỉn chu"].astype(float).to_numpy()
    for row_index, (style, color, neatness_value) in enumerate(
        zip(style_column, color_column, neatness_column)
    ):
        if style in style_index:
            item_vectors[row_index, style_index[style]] = 1.0
        if color in color_index:
            item_vectors[row_index, len(style_values) + color_index[color]] = 1.0
        neatness = neatness_value / 5.0
        item_vectors[row_index, -2:] = (neatness, 1.0 - neatness)

    profile = np.zeros(item_vectors.shape[1], dtype=float)
    selected_styles = [value for value in styles if value in style_index]
    if selected_styles:
        profile[[style_index[value] for value in selected_styles]] = 1.0 / np.sqrt(len(selected_styles))
    selected_colors = sorted({
        actual for selected in colors for actual in COLOR_MAP.get(selected, ())
        if actual in color_index
    })
    if selected_colors:
        start = len(style_values)
        profile[[start + color_index[value] for value in selected_colors]] = 1.0 / np.sqrt(len(selected_colors))
    target_formality = float(np.clip(formality, 0, 100)) / 100.0
    profile[-2:] = (target_formality, 1.0 - target_formality)

    item_norms = np.linalg.norm(item_vectors, axis=1)
    profile_norm = float(np.linalg.norm(profile))
    if profile_norm == 0:
        return np.zeros(len(catalog), dtype=float)
    similarities = item_vectors @ profile / np.maximum(item_norms * profile_norm, 1e-12)
    return np.clip(similarities, 0.0, 1.0)


def _slot(item_type: str, gender: str) -> str | None:
    if item_type == "Áo":
        return "ao"
    if item_type in {"Quần", "Chân Váy"}:
        if item_type == "Chân Váy" and gender != "Nữ":
            return None
        return "quan_vay"
    if item_type == "Đầm":
        return "dam" if gender == "Nữ" else None
    if item_type == "Áo khoác":
        return "ao_khoac"
    if item_type in {"Giày", "Dép"}:
        return "giay_dep"
    return None


def _score_catalog(
    catalog: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    activity: str,
    temperature: float,
    gender: str,
    styles: list[str],
    colors: list[str],
    formality: int,
    allow_style_fallback: bool = False,
    filter_colors: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    warnings: list[str] = []
    eligible = catalog.drop_duplicates(subset=["ID_SP"], keep="first")
    eligible = eligible[eligible["Giới tính"].map(_normalize_label) == _normalize_label(gender)].copy()
    if colors and filter_colors:
        eligible = eligible[eligible["Màu sắc"].astype(str).map(
            lambda value: _dataset_color_match(value, colors)
        )].copy()
    eligible = eligible[
        (eligible["Tmin (°C)"].astype(float) <= temperature)
        & (eligible["Tmax (°C)"].astype(float) >= temperature)
    ].copy()
    eligible["slot"] = eligible["Loại TP"].astype(str).map(lambda value: _slot(value, gender))
    eligible = eligible[eligible["slot"].notna()].copy()
    if eligible.empty:
        if colors and catalog[catalog["Giới tính"] == gender].shape[0] and not catalog[
            (catalog["Giới tính"] == gender)
            & catalog["Màu sắc"].astype(str).map(lambda value: _dataset_color_match(value, colors))
        ].shape[0]:
            return {}, ["Không tìm thấy món đồ có màu sắc phù hợp với lựa chọn của bạn trong tủ đồ."]
        return {}, ["Danh mục hiện không có món phù hợp với giới tính đã chọn."]

    eligible["content_similarity"] = _content_similarity(
        eligible, styles=styles, colors=colors, formality=formality
    )

    numeric_values = {
        "Tmin (°C)": eligible["Tmin (°C)"].astype(float).to_numpy(),
        "Tmax (°C)": eligible["Tmax (°C)"].astype(float).to_numpy(),
        "Nhiệt độ ngày (°C)": np.full(len(eligible), temperature, dtype=float),
        "Giữ ấm": eligible["Giữ ấm"].astype(float).to_numpy(),
        "Thoáng khí": eligible["Thoáng khí"].astype(float).to_numpy(),
        "Vận động": eligible["Vận động"].astype(float).to_numpy(),
        "Chỉn chu": eligible["Chỉn chu"].astype(float).to_numpy(),
    }
    feature_frame = pd.DataFrame(
        {
            "Hoàn cảnh": activity,
            "Phong cách": eligible["Phong cách"].astype(str).to_numpy(),
            **numeric_values,
        },
        columns=FEATURES,
    )
    scores, stds = _model_scores(artifact, feature_frame)
    eligible["model_score"] = scores
    eligible["std"] = stds
    eligible["color_match"] = eligible["Màu sắc"].astype(str).map(
        lambda value: _dataset_color_match(value, colors)
    )
    # CBF selects candidates from actual style/color/formality features; the
    # fitted RF+KNN relevance score ranks that candidate set.
    eligible["rank_score"] = (
        CONTENT_WEIGHT * eligible["content_similarity"] * 10
        + (1 - CONTENT_WEIGHT) * eligible["model_score"].clip(0, 10)
    )

    buckets: dict[str, list[dict[str, Any]]] = {}
    for slot_name in ("ao", "quan_vay", "dam", "ao_khoac", "giay_dep"):
        slot_frame = eligible[eligible["slot"] == slot_name]
        if slot_frame.empty:
            continue
        if styles and not allow_style_fallback:
            preferred = slot_frame[slot_frame["Phong cách"].isin(styles)]
            if len(preferred) >= min(3, TOP_K_PER_SLOT):
                slot_frame = preferred
            else:
                warnings.append(
                    f"Không đủ món thuộc phong cách đã chọn cho vị trí {slot_name}; "
                    "đã nới lỏng bộ lọc phong cách."
                )
        slot_frame = slot_frame.sort_values(
            ["content_similarity", "rank_score", "model_score"],
            ascending=False,
            kind="stable",
        ).head(TOP_K_PER_SLOT)
        records = []
        for row in slot_frame.to_dict(orient="records"):
            item = dict(row)
            item["ID_SP"] = str(item["ID_SP"])
            item["score"] = float(item["model_score"])
            item["score_low"] = float(max(0, item["model_score"] - item["std"]))
            item["score_high"] = float(min(10, item["model_score"] + item["std"]))
            item["rank_score"] = float(item["rank_score"])
            item["content_similarity"] = float(item["content_similarity"])
            item["std"] = float(item["std"])
            item["color_match"] = bool(item["color_match"])
            records.append(item)
        buckets[slot_name] = records
    return buckets, warnings


def _make_combinations(buckets: dict[str, list[dict[str, Any]]], temperature: float) -> list[list[dict[str, Any]]]:
    shoes = buckets.get("giay_dep", [])
    if not shoes:
        return []
    jacket_candidates = [
        item for item in buckets.get("ao_khoac", [])
        if item["score"] >= JACKET_SCORE_THRESHOLD
    ]
    cold_needs_layer = temperature <= 18
    # Ở thời tiết lạnh, các bộ hợp lệ phải có lớp áo khoác nếu catalog có món đạt ngưỡng.
    jackets: list[dict[str, Any] | None] = jacket_candidates if cold_needs_layer and jacket_candidates else [None]
    if not cold_needs_layer and temperature <= 30:
        jackets += jacket_candidates
    combinations: list[list[dict[str, Any]]] = []
    for shirt, lower, shoe, jacket in product(
        buckets.get("ao", []), buckets.get("quan_vay", []), shoes, jackets
    ):
        combo = [shirt, lower, shoe]
        if jacket is not None:
            combo.append(jacket)
        if len({item["ID_SP"] for item in combo}) == len(combo):
            combinations.append(combo)
    for dress, shoe, jacket in product(buckets.get("dam", []), shoes, jackets):
        combo = [dress, shoe]
        if jacket is not None:
            combo.append(jacket)
        if len({item["ID_SP"] for item in combo}) == len(combo):
            combinations.append(combo)
    return combinations


def _combo_model_score(items: list[dict[str, Any]]) -> float:
    return float(np.mean([item["score"] for item in items]))


def _combo_rank(items: list[dict[str, Any]]) -> float:
    return float(np.mean([item["rank_score"] for item in items]))


def _main_item_ids(items: list[dict[str, Any]]) -> set[str]:
    return {str(item["ID_SP"]) for item in items}


def _combo_similarity(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    left_ids = {str(item["ID_SP"]) for item in left}
    right_ids = {str(item["ID_SP"]) for item in right}
    union = left_ids | right_ids
    item_similarity = len(left_ids & right_ids) / len(union) if union else 0.0
    left_styles = {str(item["Phong cách"]) for item in left}
    right_styles = {str(item["Phong cách"]) for item in right}
    style_similarity = len(left_styles & right_styles) / max(
        1, len(left_styles | right_styles)
    )
    left_colors = {str(item["Màu sắc"]) for item in left}
    right_colors = {str(item["Màu sắc"]) for item in right}
    color_similarity = len(left_colors & right_colors) / max(
        1, len(left_colors | right_colors)
    )
    return 0.7 * item_similarity + 0.15 * style_similarity + 0.15 * color_similarity


def _select_diverse(
    combos: list[list[dict[str, Any]]], count: int
) -> list[list[dict[str, Any]]]:
    if not combos or count <= 0:
        return []

    # Gom phép tính mean theo lô để tránh tạo hàng trăm nghìn mảng NumPy nhỏ.
    combos_by_identity = {id(combo): combo for combo in combos}

    def combo_mean_scores() -> tuple[dict[int, float], dict[int, float]]:
        grouped: dict[int, list[list[dict[str, Any]]]] = {}
        for combo in combos:
            grouped.setdefault(len(combo), []).append(combo)
        model_scores: dict[int, float] = {}
        rank_scores: dict[int, float] = {}
        for group in grouped.values():
            item_count = len(group[0])
            values = np.fromiter(
                (
                    float(item[field])
                    for combo in group
                    for item in combo
                    for field in ("score", "rank_score")
                ),
                dtype=float,
                count=len(group) * item_count * 2,
            ).reshape(len(group), item_count, 2)
            means = np.mean(values, axis=1)
            model_scores.update(
                (id(combo), float(score)) for combo, score in zip(group, means[:, 0])
            )
            rank_scores.update(
                (id(combo), float(score)) for combo, score in zip(group, means[:, 1])
            )
        return model_scores, rank_scores

    model_scores, rank_scores = combo_mean_scores()
    # Tuple giữ ID gọn hơn hàng trăm nghìn set; phép giao vẫn kiểm tra cùng các ID.
    combo_item_ids = {
        id(combo): tuple(str(item["ID_SP"]) for item in combo) for combo in combos
    }

    def model_score(combo_identity: int) -> float:
        return model_scores[combo_identity]

    def rank_score(combo_identity: int) -> float:
        return rank_scores[combo_identity]

    @lru_cache(maxsize=None)
    def similarity(left_identity: int, right_identity: int) -> float:
        return _combo_similarity(
            combos_by_identity[left_identity], combos_by_identity[right_identity]
        )

    def similarity_for(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
        left_identity, right_identity = id(left), id(right)
        if left_identity > right_identity:
            left_identity, right_identity = right_identity, left_identity
        return similarity(left_identity, right_identity)

    best_score = max(model_score(id(combo)) for combo in combos)
    def choose(candidates: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        candidates.sort(
            key=lambda combo: (
                -rank_score(id(combo)),
                tuple(str(item["ID_SP"]) for item in combo),
            )
        )
        selected: list[list[dict[str, Any]]] = []
        remaining = list(candidates)
        while remaining and len(selected) < count:
            available = [
                combo for combo in remaining
                if all(
                    not any(
                        item_id in combo_item_ids[id(prior)]
                        for item_id in combo_item_ids[id(combo)]
                    )
                    for prior in selected
                )
            ]
            if not available:
                break
            chosen = max(
                available,
                key=lambda combo: (
                    (1 - OUTFIT_MMR_LAMBDA) * (rank_score(id(combo)) / 10)
                    - OUTFIT_MMR_LAMBDA
                    * max(
                        (similarity_for(combo, prior) for prior in selected),
                        default=0.0,
                    ),
                    rank_score(id(combo)),
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    maximum_band = max(0.0, best_score - OUTFIT_MIN_SCORE)
    band = min(OUTFIT_SCORE_BAND, maximum_band)
    selected = []
    while True:
        candidates = [
            combo for combo in combos
            if model_score(id(combo))
            >= max(OUTFIT_MIN_SCORE, best_score - band)
        ]
        selected = choose(candidates)
        if len(selected) >= count or band >= maximum_band:
            break
        band = min(maximum_band, band + 0.25)
    return sorted(selected, key=lambda combo: model_score(id(combo)), reverse=True)


def _item_public(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["ID_SP"],
        "name": str(item["Tên trang phục"]),
        "category": str(item["Loại TP"]),
        "color": str(item["Màu sắc"]),
        "style": str(item["Phong cách"]),
        "material": str(item.get("Chất liệu", "")),
        "occasion": str(item.get("Hoàn cảnh hợp nhất", "")),
        "content_match": round(float(item.get("content_similarity", 0.0)) * 100, 2),
        "gender": str(item["Giới tính"]),
        "score": round(item["score"], 3),
        "score_low": round(item["score_low"], 3),
        "score_high": round(item["score_high"], 3),
        "temperature_range": {
            "min": float(item["Tmin (°C)"]),
            "max": float(item["Tmax (°C)"]),
        },
        "characteristics": {
            "warmth": float(item["Giữ ấm"]),
            "breathability": float(item["Thoáng khí"]),
            "mobility": float(item["Vận động"]),
            "formality": float(item["Chỉn chu"]),
        },
        "google_link": str(item.get("Link Google", "")),
        "pinterest_link": str(item.get("Link Pinterest", "")),
    }


def _explanation(
    items: list[dict[str, Any]],
    activity: str,
    temperature: float,
    schedule_summary: str | None = None,
    rain_probability: int | None = None,
) -> str:
    characteristics = ["Giữ ấm", "Thoáng khí", "Vận động", "Chỉn chu"]
    means = [
        float(np.mean([float(item[name]) for item in items])) for name in characteristics
    ]
    standout = characteristics[int(np.argmax(means))]
    mean_std = float(np.mean([item["std"] for item in items]))
    explanation = (
        f"Bộ này được xếp cho hoạt động {activity} ở {temperature:g}°C. "
        f"Đặc tính nổi bật là {standout.lower()}; mức tin cậy {confidence_label(mean_std).lower()}. "
        "Các điểm thời tiết, hoạt động và phong cách bên dưới là diễn giải theo công thức dữ liệu."
    )
    if schedule_summary:
        explanation += f" Lịch hôm nay: {schedule_summary}"
    if rain_probability is not None and rain_probability >= 40:
        explanation += " Có khả năng mưa, nên cân nhắc mang ô hoặc áo mưa."
    if not (MODEL_TEMP_RANGE["min"] <= temperature <= MODEL_TEMP_RANGE["max"]):
        explanation += " Nhiệt độ nằm ngoài khoảng huấn luyện nên mức tin cậy thấp."
    return explanation


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(char for char in text if not unicodedata.combining(char))


def _item_content_scores(
    item: dict[str, Any], request: Any, rule_part: dict[str, float]
) -> tuple[float | None, float]:
    """Return preference and selected-criteria fit scores on a 0–100 scale."""
    notes = _normalized_text(getattr(request, "notes", ""))
    style_values = list(getattr(request, "styles", []) or [])
    color_values = list(getattr(request, "colors", []) or [])
    attrs = [
        _normalized_text(item.get("Màu sắc")),
        _normalized_text(item.get("Phong cách")),
        _normalized_text(item.get("Loại TP")),
        _normalized_text(item.get("Chất liệu")),
        _normalized_text(item.get("Hoàn cảnh hợp nhất")),
    ]
    note_terms = [term for term in notes.split() if len(term) >= 3]
    preference_parts: list[float] = []
    if style_values:
        preference_parts.append(100.0 if item.get("Phong cách") in style_values else 0.0)
    if color_values:
        preference_parts.append(100.0 if _dataset_color_match(str(item.get("Màu sắc", "")), color_values) else 0.0)
    if note_terms:
        preference_parts.append(100.0 * sum(any(term in attr for attr in attrs) for term in note_terms) / len(note_terms))
    preference_score = float(np.mean(preference_parts)) if preference_parts else None

    temp = float(request.temperature)
    low, high = float(item["Tmin (°C)"]), float(item["Tmax (°C)"])
    deviation = max(0.0, low - temp, temp - high)
    criteria_parts = [
        (100.0, 0.10),  # Gender is a hard-filtered criterion in the candidate set.
        (float(np.clip(rule_part["activity_score"] * 10, 0, 100)), 0.20),
        (float(np.clip(100 - 20 * deviation, 0, 100)), 0.20),
        (100.0 if not style_values or item.get("Phong cách") in style_values else 0.0, 0.15),
        (100.0 if not color_values or _dataset_color_match(str(item.get("Màu sắc", "")), color_values) else 0.0, 0.15),
        (100.0 if not request.activity or item.get("Hoàn cảnh hợp nhất") == request.activity else float(np.clip(rule_part["activity_score"] * 10, 0, 100)), 0.10),
        (100.0 if not notes or any(term in " ".join(attrs) for term in note_terms) else 50.0, 0.10),
    ]
    total_weight = sum(weight for _, weight in criteria_parts)
    criteria_score = sum(score * weight for score, weight in criteria_parts) / total_weight
    return preference_score, float(np.clip(criteria_score, 0, 100))


def _detailed_explanation(
    items: list[dict[str, Any]], scores: dict[str, float | None],
    request: Any, confidence: str,
) -> str:
    labels = {
        "temp_score": "thời tiết", "activity_score": "hoàn cảnh",
        "style_score": "phong cách", "preference_score": "sở thích",
        "criteria_score": "tiêu chí đã chọn", "match_score": "xếp hạng cuối",
    }
    available = [(key, value) for key, value in scores.items() if value is not None]
    if not available or not items:
        return "Chưa đủ dữ liệu để phân tích chi tiết; độ tin cậy " + confidence.lower() + "."
    strongest = max(available, key=lambda entry: entry[1])
    weakest = min(available, key=lambda entry: entry[1])
    item = items[0]
    matched = []
    if request.colors and _dataset_color_match(str(item.get("Màu sắc", "")), request.colors):
        matched.append(f"màu {item.get('Màu sắc')}")
    if request.styles and item.get("Phong cách") in request.styles:
        matched.append(f"phong cách {item.get('Phong cách')}")
    if request.activity:
        matched.append(f"hoạt động {request.activity}")
    if item.get("Hoàn cảnh hợp nhất") == request.activity:
        matched.append(f"dịp {item.get('Hoàn cảnh hợp nhất')}")
    material = str(item.get("Chất liệu", "")).strip()
    note_text = _normalized_text(getattr(request, "notes", ""))
    if material and _normalized_text(material) in note_text:
        matched.append(f"chất liệu {material}")
    matched.append(f"giới tính {item.get('Giới tính')}")
    sentence = "Món tiêu biểu khớp " + ", ".join(matched) + "." if matched else "Món tiêu biểu được chọn theo nhiệt độ và điểm mô hình."
    combo_items = [
        f"{entry.get('Loại TP', 'Món đồ')} {entry.get('Tên trang phục', '')} ({entry.get('Màu sắc', '')})"
        for entry in items
    ]
    if combo_items:
        sentence += " Tổ hợp cụ thể: " + "; ".join(combo_items) + "."
    sentence += f" Điểm mạnh nhất là {labels[strongest[0]]} ({strongest[1]:.0f}%)."
    if weakest[0] != strongest[0]:
        sentence += f" Điểm cần cân nhắc là {labels[weakest[0]]} ({weakest[1]:.0f}%)."
    if scores.get("preference_score") is not None:
        preference_details = []
        if request.styles:
            preference_details.append("phong cách " + ", ".join(request.styles))
        if request.colors:
            preference_details.append("màu " + ", ".join(request.colors))
        if note_text:
            preference_details.append("ghi chú sở thích bạn nhập")
        if preference_details:
            sentence += " Sở thích được đối chiếu: " + ", ".join(preference_details) + "."
        sentence += f" Mức khớp sở thích đạt {scores['preference_score']:.0f}%."
    sentence += f" Độ tin cậy {confidence.lower()}."
    return sentence


def recommend(
    catalog: pd.DataFrame,
    artifact: dict[str, Any],
    request: Any,
) -> dict[str, Any]:
    temp = float(request.temperature)
    genders = ["Unisex", "Nam"] if request.gender == "Unisex" else [request.gender]
    bucket_sets: list[dict[str, list[dict[str, Any]]]] = []
    warnings: list[str] = []
    for candidate_gender in genders:
        candidate_buckets, candidate_warnings = _score_catalog(
            catalog, artifact, activity=request.activity, temperature=temp,
            gender=candidate_gender, styles=request.styles, colors=request.colors,
            formality=request.formality,
        )
        bucket_sets.append(candidate_buckets)
        warnings.extend(candidate_warnings)
    buckets = bucket_sets[0] if bucket_sets else {}
    if not (MODEL_TEMP_RANGE["min"] <= temp <= MODEL_TEMP_RANGE["max"]):
        warnings.append(
            f"Nhiệt độ {temp:g}°C nằm ngoài khoảng huấn luyện "
            f"{MODEL_TEMP_RANGE['min']:g}–{MODEL_TEMP_RANGE['max']:g}°C; kết quả ngoại suy, độ tin cậy thấp."
        )
    combos = _make_combinations(buckets, temp)
    if temp <= 18 and buckets.get("ao_khoac") and not any(
        item["score"] >= JACKET_SCORE_THRESHOLD for item in buckets["ao_khoac"]
    ):
        warnings.append(
            "Không có áo khoác đạt ngưỡng điểm model tối thiểu; các bộ lạnh có thể thiếu lớp áo khoác."
        )
    selected = _select_diverse(combos, request.top_n)
    # Unisex is ranked first; only fill remaining slots from men's items.
    if request.gender == "Unisex" and len(selected) < request.top_n and len(bucket_sets) > 1:
        male_combos = _make_combinations(bucket_sets[1], temp)
        male_selected = _select_diverse(male_combos, request.top_n)
        selected_ids = {str(item["ID_SP"]) for outfit in selected for item in outfit}
        for combo in male_selected:
            if len(selected) >= request.top_n:
                break
            if not any(str(item["ID_SP"]) in selected_ids for item in combo):
                selected.append(combo)
                selected_ids.update(str(item["ID_SP"]) for item in combo)
    if len(selected) < request.top_n and request.styles:
        relaxed_sets = [
            _score_catalog(
                catalog, artifact, activity=request.activity, temperature=temp,
                gender=candidate_gender, styles=request.styles, colors=request.colors,
                formality=request.formality, allow_style_fallback=True,
            )[0]
            for candidate_gender in genders
        ]
        relaxed_buckets = relaxed_sets[0] if relaxed_sets else {}
        combos.extend(_make_combinations(relaxed_buckets, temp))
        buckets = relaxed_buckets
        warnings.append(
            "Đã nới lỏng bộ lọc phong cách để tìm thêm outfit đủ điểm và không trùng món chính."
        )
        selected = _select_diverse(combos, request.top_n)
        if request.gender == "Unisex" and len(selected) < request.top_n and len(relaxed_sets) > 1:
            relaxed_male = _select_diverse(
                _make_combinations(relaxed_sets[1], temp), request.top_n
            )
            selected_ids = {str(item["ID_SP"]) for outfit in selected for item in outfit}
            for combo in relaxed_male:
                if len(selected) >= request.top_n:
                    break
                if not any(str(item["ID_SP"]) in selected_ids for item in combo):
                    selected.append(combo)
                    selected_ids.update(str(item["ID_SP"]) for item in combo)

    if len(selected) < request.top_n and request.colors:
        relaxed_color_sets = [
            _score_catalog(
                catalog,
                artifact,
                activity=request.activity,
                temperature=temp,
                gender=candidate_gender,
                styles=request.styles,
                colors=request.colors,
                formality=request.formality,
                allow_style_fallback=bool(request.styles),
                filter_colors=False,
            )[0]
            for candidate_gender in genders
        ]
        relaxed_color_buckets = relaxed_color_sets[0] if relaxed_color_sets else {}
        relaxed_color_combos = [
            combo
            for candidate_buckets in relaxed_color_sets
            for combo in _make_combinations(candidate_buckets, temp)
        ]
        if relaxed_color_combos:
            color_fill = _select_diverse(relaxed_color_combos, request.top_n * 2)
            occupied_main_ids = {
                item_id for combo in selected for item_id in _main_item_ids(combo)
            }
            initial_count = len(selected)
            for combo in color_fill:
                if len(selected) >= request.top_n:
                    break
                main_ids = _main_item_ids(combo)
                if main_ids & occupied_main_ids:
                    continue
                selected.append(combo)
                occupied_main_ids.update(main_ids)
            if len(selected) > initial_count:
                warnings.append(
                    "Không đủ combo hoàn chỉnh đúng các màu đã chọn; đã bổ sung một số món "
                    "màu khác theo thời tiết, phong cách và điểm DSS."
                )
                buckets = relaxed_color_buckets
                selected.sort(key=_combo_model_score, reverse=True)
    if len(selected) < request.top_n:
        warnings.append(f"Chỉ ghép được {len(selected)} bộ outfit đủ đa dạng từ danh mục hiện có.")
    if selected:
        warnings = [warning for warning in warnings if "Không tìm thấy món đồ có màu sắc phù hợp" not in warning]

    recommendations = []
    for rank, combo in enumerate(selected, start=1):
        base_score = _combo_model_score(combo)
        final_rank_score = _combo_rank(combo)
        mean_std = float(np.mean([item["std"] for item in combo]))
        rule_parts = [explain_item(item, request.activity, temp) for item in combo]
        temp_score = float(np.mean([part["temp_factor"] for part in rule_parts]) * 100)
        activity_score = float(np.mean([part["activity_score"] for part in rule_parts]) * 10)
        style_score = float(np.mean([part["style_score"] for part in rule_parts]) * 10)
        item_scores = [
            _item_content_scores(item, request, rule_part)
            for item, rule_part in zip(combo, rule_parts)
        ]
        available_preferences = [
            preference for preference, _ in item_scores if preference is not None
        ]
        preference_score = (
            float(np.mean(available_preferences)) if available_preferences else None
        )
        criteria_score = float(np.mean([criteria for _, criteria in item_scores]))
        named_style = next(
            (str(item["Phong cách"]) for item in combo if item["Phong cách"] in request.styles),
            str(combo[0]["Phong cách"]),
        )
        confidence = "Thấp" if warnings and any("ngoài khoảng" in w for w in warnings) else confidence_label(mean_std)
        analysis_scores = {
            "temp_score": float(np.clip(temp_score, 0, 100)),
            "activity_score": float(np.clip(activity_score, 0, 100)),
            "style_score": float(np.clip(style_score, 0, 100)),
            "preference_score": preference_score,
            "criteria_score": criteria_score,
            "match_score": float(np.clip(final_rank_score * 10, 0, 100)),
        }
        recommendation_text = _detailed_explanation(
            combo, analysis_scores, request, confidence
        )
        summary_labels = {
            "temp_score": "thời tiết",
            "activity_score": "hoạt động",
            "style_score": "phong cách",
            "preference_score": "sở thích",
            "criteria_score": "tiêu chí đã chọn",
            "match_score": "điểm phù hợp tổng thể",
        }
        strongest_label, strongest_value = max(
            ((summary_labels[key], value) for key, value in analysis_scores.items() if value is not None),
            key=lambda entry: entry[1],
        )
        summary_items = ", ".join(
            f"{item.get('Tên trang phục', 'Món đồ')} ({item.get('Màu sắc', 'chưa rõ màu')})"
            for item in combo
        )
        explanation_summary = (
            f"Được gợi ý cho {request.activity} ở {temp:g}°C; điểm nổi bật là "
            f"{strongest_label} ({strongest_value:.0f}%). Combo gồm {summary_items}."
        )
        recommendations.append(
            {
                "id": rank,
                "name": f"{named_style} – {request.activity} #{rank}",
                "score": round(base_score, 3),
                "score_low": round(max(0, base_score - mean_std), 3),
                "score_high": round(min(10, base_score + mean_std), 3),
                "std_between_trees": round(mean_std, 4),
                "confidence_level": confidence,
                "match_score": round(float(np.clip(final_rank_score, 0, 10) * 10), 2),
                "temp_score": round(float(np.clip(temp_score, 0, 100)), 2),
                "activity_score": round(float(np.clip(activity_score, 0, 100)), 2),
                "style_score": round(float(np.clip(style_score, 0, 100)), 2),
                "preference_score": round(preference_score, 2) if preference_score is not None else None,
                "criteria_score": round(criteria_score, 2),
                "items": [_item_public(item) for item in combo],
                "breakdown": [_item_public(item) for item in combo],
                "alternatives": {
                    slot: [_item_public(item) for item in items if item["ID_SP"] not in {x["ID_SP"] for x in combo}][:3]
                    for slot, items in buckets.items()
                },
                "recommendation": recommendation_text,
                "explanation": recommendation_text,
                "explanation_summary": explanation_summary,
            }
        )
    return {
        "outfits": recommendations,
        "warnings": list(dict.fromkeys(warnings)),
        "model_version": MODEL_METADATA["version"],
        "selected_model": artifact.get("selected_model", MODEL_METADATA["model_name"]),
    }
