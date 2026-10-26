"""Đối chiếu quy tắc API với các nhãn trung gian trong toàn bộ DuLieu_ML."""

from pathlib import Path

import numpy as np
import pandas as pd

from api.rules import calculate_components


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "DatasetDSS.xlsx"


def test_rule_reconstruction_matches_all_6000_rows() -> None:
    frame = pd.read_excel(WORKBOOK, sheet_name="DuLieu_ML", engine="openpyxl")
    result = calculate_components(
        frame["Hoàn cảnh"].to_numpy(),
        frame["Phong cách"].to_numpy(),
        frame["Tmin (°C)"].to_numpy(),
        frame["Tmax (°C)"].to_numpy(),
        frame["Nhiệt độ ngày (°C)"].to_numpy(),
        frame["Giữ ấm"].to_numpy(),
        frame["Thoáng khí"].to_numpy(),
        frame["Vận động"].to_numpy(),
        frame["Chỉn chu"].to_numpy(),
    )
    checks = {
        "Bước 1": ("step1", "Bước 1: Điểm hoàn cảnh"),
        "Bước 2": ("step2", "Bước 2: Điểm phong cách"),
        "Bước 3b": ("temp_factor", "Bước 3b: Hệ số nhiệt độ"),
        "Điểm mục tiêu": ("score", "Điểm phù hợp (0–10)"),
    }
    for label, (actual_key, source_column) in checks.items():
        delta = np.abs(np.asarray(result[actual_key], dtype=float) - frame[source_column].to_numpy(float))
        max_delta = float(delta.max())
        safe_label = label.encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"{safe_label}: max_abs_error={max_delta:.6f} ({len(frame)} rows)")
        assert max_delta <= 0.100001, f"{label} lệch tối đa {max_delta:.6f}, vượt 0.1"


def test_published_example() -> None:
    result = calculate_components(
        "Xưởng/Công trình", "Streetwear", 20, 38, 12, 2, 5, 2, 2
    )
    assert float(result["step1"]) == 4.6
    assert float(result["step2"]) == 5.5
    assert np.isclose(float(result["temp_factor"]), 0.2)
    assert float(result["score"]) == 1.0

