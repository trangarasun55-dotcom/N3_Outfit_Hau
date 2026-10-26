"""Kiểm tra kết nối API, Open-Meteo và Gemini mà không đọc tệp bí mật.

Chạy khi FastAPI đang hoạt động: python scripts/check_connections.py
Chỉ dùng các biến môi trường của tiến trình hiện tại; không đăng nhập admin,
tránh cập nhật last_login trong cơ sở dữ liệu thật.
"""

from __future__ import annotations

import os
import sys

import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT = (3, 5)
WEATHER_TIMEOUT = (3, 8)
FAILED = False


def report(name: str, ok: bool, detail: str = "", *, required: bool = True) -> None:
    global FAILED
    mark = "OK  " if ok else ("LỖI" if required else "CẢNH BÁO")
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))
    if required and not ok:
        FAILED = True


def check_api() -> bool:
    """Health thành công nghĩa là API đã khởi động xong, gồm cả SQLite."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("health response is not an object")
        ok = data.get("status") == "ok"
        version = data.get("model_version")
        report("API FastAPI và khởi động SQLite", ok, f"model {version}" if version else "")
        return ok
    except (requests.RequestException, ValueError, TypeError) as exc:
        report("API FastAPI và khởi động SQLite", False, type(exc).__name__)
        return False


def check_weather() -> None:
    try:
        response = requests.get(
            f"{API_URL}/weather", params={"city": "Hà Nội"}, timeout=WEATHER_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
        ok = isinstance(payload, dict) and payload.get("source") == "open-meteo"
        report("Thời tiết Open-Meteo qua API", ok, "có dữ liệu" if ok else "phản hồi không đúng định dạng", required=False)
    except (requests.RequestException, ValueError, TypeError) as exc:
        report("Thời tiết Open-Meteo qua API", False, type(exc).__name__, required=False)


def check_gemini() -> None:
    """Gọi thử bằng cấu hình đã truyền vào môi trường, không đọc .env."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "").strip()
    if not api_key:
        report("Gemini", False, "bỏ qua: GEMINI_API_KEY không có trong môi trường kiểm tra", required=False)
        return
    if not model:
        report("Gemini", False, "bỏ qua: GEMINI_MODEL không có trong môi trường kiểm tra", required=False)
        return
    try:
        from google import genai
        from google.genai import types

        with genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=8_000),
        ) as client:
            reply = client.models.generate_content(
                model=model,
                contents="Trả lời đúng một từ: OK",
                config=types.GenerateContentConfig(max_output_tokens=4, temperature=0),
            )
        report("Gemini", bool(getattr(reply, "text", None)), f"model {model}", required=False)
    except Exception as exc:  # noqa: BLE001 - chẩn đoán nhưng không in nội dung exception/key
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        detail = type(exc).__name__ + (f" (HTTP {status})" if status else "")
        report("Gemini", False, detail, required=False)


def main() -> int:
    if check_api():
        check_weather()
    check_gemini()
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
