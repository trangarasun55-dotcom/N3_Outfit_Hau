"""Kiểm tra nhanh dataset, catalog và ảnh sản phẩm trước khi chạy web."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "DatasetDSS.xlsx"
CATALOG = ROOT / "data" / "mon_do.csv"
IMAGE_DIR = ROOT / "web" / "static" / "images" / "images"

def main() -> int:
    errors: list[str] = []
    if not WORKBOOK.is_file():
        errors.append(f"Thiếu workbook: {WORKBOOK}")
    if not CATALOG.is_file():
        errors.append(f"Thiếu catalog: {CATALOG}")
    if errors:
        print("\\n".join("ERROR: " + item for item in errors))
        return 1

    xls = pd.ExcelFile(WORKBOOK)
    required_sheets = {"MonDo", "DuLieu_ML"}
    if not required_sheets.issubset(xls.sheet_names):
        errors.append(f"Workbook thiếu sheet: {sorted(required_sheets - set(xls.sheet_names))}")
        print("\\n".join("ERROR: " + item for item in errors))
        return 1

    wardrobe = pd.read_excel(WORKBOOK, sheet_name="MonDo")
    ml = pd.read_excel(WORKBOOK, sheet_name="DuLieu_ML")
    catalog = pd.read_csv(CATALOG)
    for frame_name, frame in (("MonDo", wardrobe), ("DuLieu_ML", ml), ("mon_do.csv", catalog)):
        if "ID_SP" not in frame.columns:
            errors.append(f"{frame_name} thiếu cột ID_SP")
    if "ID_SP" in wardrobe.columns and wardrobe["ID_SP"].duplicated().any():
        errors.append("Sheet MonDo có ID_SP trùng lặp.")
    if "ID_SP" in catalog.columns and catalog["ID_SP"].duplicated().any():
        errors.append("data/mon_do.csv có ID_SP trùng lặp.")

    images = {p.stem.strip().upper() for p in IMAGE_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    ids = set(wardrobe["ID_SP"].astype(str).str.strip().str.upper()) if "ID_SP" in wardrobe.columns else set()
    matched = len(ids & images)
    missing = sorted(ids - images)
    print(f"Workbook: {len(wardrobe)} món; {len(ml)} dòng huấn luyện.")
    print(f"Catalog CSV: {len(catalog)} dòng.")
    print(f"Ảnh: {len(images)} file; khớp {matched}/{len(ids)} ID sản phẩm.")
    if missing:
        print(f"INFO: {len(missing)} sản phẩm chưa có ảnh (web sẽ dùng placeholder): {', '.join(missing[:12])}" + (" ..." if len(missing) > 12 else ""))
    if errors:
        print("\\n".join("ERROR: " + item for item in errors))
        return 1
    print("Kiểm tra dữ liệu cơ bản: OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
