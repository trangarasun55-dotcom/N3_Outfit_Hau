"""Tải metadata huấn luyện và các cấu hình dùng chung cho API."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "outfit_model.joblib"
METADATA_PATH = ROOT / "model" / "model_metadata.json"
CATALOG_PATH = ROOT / "data" / "mon_do.csv"

if not MODEL_PATH.is_file() or not METADATA_PATH.is_file() or not CATALOG_PATH.is_file():
    raise RuntimeError(
        "Thiếu artifact Bước 1; hãy chạy `python model/train_export.py` trước khi khởi động API."
    )

with METADATA_PATH.open("r", encoding="utf-8") as file:
    MODEL_METADATA = json.load(file)

ACTIVITIES = tuple(MODEL_METADATA["categories"]["Hoàn cảnh"])
STYLES = tuple(MODEL_METADATA["categories"]["Phong cách"])
GENDERS = tuple(dict.fromkeys((*MODEL_METADATA["categories"]["Giới tính"], "Unisex")))
DATASET_COLORS = tuple(MODEL_METADATA["categories"]["Màu sắc"])

# Giữ sáu nhóm màu cũ để tương thích; bổ sung các màu cụ thể có trong metadata.
COLOR_MAP = {
    "Đen": ("Đen", "Đen trắng"),
    "Trắng": ("Trắng", "Xanh trắng"),
    "Xám": ("Xám", "Xám tro"),
    "Be": ("Be", "Be sữa", "Kem", "Khaki"),
    "Nâu đất": ("Nâu", "Nâu caramel", "Nâu đậm"),
    "Navy": ("Xanh navy", "Xanh dương", "Xanh denim đậm"),
}
for _dataset_color in DATASET_COLORS:
    COLOR_MAP.setdefault(_dataset_color, (_dataset_color,))
COLOR_CHOICES = tuple(COLOR_MAP)
MODEL_TEMP_RANGE = MODEL_METADATA["feature_ranges"]["Nhiệt độ ngày (°C)"]
_CONFIDENCE_THRESHOLDS = MODEL_METADATA.get("confidence_thresholds", {})
CONFIDENCE_HIGH_STD = float(
    _CONFIDENCE_THRESHOLDS.get("high_std_max", 0.5)
)
CONFIDENCE_MEDIUM_STD = float(
    _CONFIDENCE_THRESHOLDS.get("medium_std_max", 1.0)
)
JACKET_SCORE_THRESHOLD = 5.0
TOP_K_PER_SLOT = 30
DEFAULT_FORMALITY = 35
OUTFIT_MIN_SCORE = 6.0
OUTFIT_SCORE_BAND = 1.0
OUTFIT_MMR_LAMBDA = 0.25
