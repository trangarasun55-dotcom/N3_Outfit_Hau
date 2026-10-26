"""Công thức quy tắc của workbook, chỉ dùng giải thích/đối chiếu (không xếp hạng)."""

from __future__ import annotations

from typing import Any

import numpy as np


CHARACTERISTICS = ("Giữ ấm", "Thoáng khí", "Vận động", "Chỉn chu")
ACTIVITY_WEIGHTS = {
    "Đi học": (0.1, 0.3, 0.3, 0.3),
    "Xưởng/Công trình": (0.2, 0.2, 0.5, 0.1),
    "Vẽ ngoại cảnh": (0.0, 0.3, 0.5, 0.2),
    "Thuyết trình": (0.0, 0.2, 0.2, 0.6),
}
STYLE_COMPATIBILITY = {
    "Casual": {"Đi học": 5, "Xưởng/Công trình": 4, "Vẽ ngoại cảnh": 5, "Thuyết trình": 3},
    "Smart Casual": {"Đi học": 4, "Xưởng/Công trình": 2, "Vẽ ngoại cảnh": 2, "Thuyết trình": 5},
    "Sporty": {"Đi học": 4, "Xưởng/Công trình": 5, "Vẽ ngoại cảnh": 4, "Thuyết trình": 1},
    "Streetwear": {"Đi học": 4, "Xưởng/Công trình": 3, "Vẽ ngoại cảnh": 4, "Thuyết trình": 2},
    "Minimalist": {"Đi học": 4, "Xưởng/Công trình": 3, "Vẽ ngoại cảnh": 3, "Thuyết trình": 4},
    "Artsy": {"Đi học": 4, "Xưởng/Công trình": 2, "Vẽ ngoại cảnh": 5, "Thuyết trình": 3},
}


def scale(value: Any) -> Any:
    """Đổi thang 1–5 thành 1–10 theo bảng CachTinhDiem."""
    return 1 + (np.asarray(value) - 1) * 9 / 4


def calculate_components(
    activity: Any,
    style: Any,
    tmin: Any,
    tmax: Any,
    temperature: Any,
    warmth: Any,
    breathability: Any,
    mobility: Any,
    neatness: Any,
) -> dict[str, Any]:
    """Tính Bước 1, Bước 2, hệ số nhiệt và điểm cuối cho scalar hoặc arrays."""
    activity_values = np.asarray(activity, dtype=object)
    style_values = np.asarray(style, dtype=object)
    raw_traits = np.stack(
        [np.asarray(value, dtype=float) for value in (warmth, breathability, mobility, neatness)],
        axis=-1,
    )
    step1_raw = np.empty(activity_values.shape, dtype=float)
    step2_raw = np.empty(style_values.shape, dtype=float)
    for value, weights in ACTIVITY_WEIGHTS.items():
        mask = activity_values == value
        if np.any(mask):
            step1_raw[mask] = raw_traits[mask] @ np.asarray(weights, dtype=float)
    for style_name, by_activity in STYLE_COMPATIBILITY.items():
        style_mask = style_values == style_name
        for activity_name, raw_score in by_activity.items():
            mask = style_mask & (activity_values == activity_name)
            if np.any(mask):
                step2_raw[mask] = raw_score

    if np.isnan(step1_raw).any() or np.isnan(step2_raw).any():
        raise ValueError("Hoàn cảnh hoặc phong cách không có trong bảng quy tắc.")
    # Khớp quy ước làm tròn 1 chữ số của workbook, tránh sai số nhị phân như 4.6000000001.
    step1 = np.round(scale(step1_raw), 1)
    step2 = np.round(scale(step2_raw), 1)
    temp = np.asarray(temperature, dtype=float)
    low = np.asarray(tmin, dtype=float)
    high = np.asarray(tmax, dtype=float)
    deviation = np.maximum.reduce([np.zeros_like(temp), low - temp, temp - high])
    temp_factor = np.maximum(0.0, 1.0 - 0.1 * deviation)
    raw_score = (0.7 * step1 + 0.3 * step2) * temp_factor
    # Nhãn workbook được làm tròn đến một chữ số; điểm luôn không âm.
    score = np.floor(np.clip(raw_score, 0, 10) * 10 + 0.5) / 10
    return {
        "step1": step1,
        "step2": step2,
        "temp_deviation": deviation,
        "temp_factor": temp_factor,
        "score": score,
    }


def explain_item(item: dict[str, Any], activity: str, temperature: float) -> dict[str, float]:
    parts = calculate_components(
        activity,
        str(item["Phong cách"]),
        float(item["Tmin (°C)"],),
        float(item["Tmax (°C)"],),
        temperature,
        float(item["Giữ ấm"]),
        float(item["Thoáng khí"]),
        float(item["Vận động"]),
        float(item["Chỉn chu"]),
    )
    return {
        "activity_score": float(parts["step1"]),
        "style_score": float(parts["step2"]),
        "temp_factor": float(parts["temp_factor"]),
    }

