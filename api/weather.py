"""Open-Meteo weather and geocoding client used only by the FastAPI backend."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from dotenv import load_dotenv
from fastapi import HTTPException

try:
    from .config import ROOT
except ImportError:  # `uvicorn main:app` from api/
    from config import ROOT

load_dotenv(ROOT / ".env")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_TIMEOUT = 5
CACHE_TTL_SECONDS = 600
RAIN_ALERT_THRESHOLD = 40
DEFAULT_CITY = os.environ.get("DEFAULT_CITY", "Hà Nội")
DEFAULT_LAT = 21.0285
DEFAULT_LON = 105.8542

# WMO codes: (Vietnamese condition, icon key).
WEATHER_CODES: dict[int, tuple[str, str]] = {
    0: ("Trời quang", "sun"), 1: ("Ít mây", "partly"),
    2: ("Ít mây", "partly"), 3: ("Có mây", "cloud"),
    45: ("Sương mù", "fog"), 48: ("Sương mù", "fog"),
    51: ("Mưa phùn", "rain"), 53: ("Mưa phùn", "rain"), 55: ("Mưa phùn", "rain"),
    56: ("Mưa phùn", "rain"), 57: ("Mưa phùn", "rain"),
    61: ("Mưa", "rain"), 63: ("Mưa", "rain"), 65: ("Mưa", "rain"),
    66: ("Mưa", "rain"), 67: ("Mưa", "rain"),
    71: ("Tuyết", "snow"), 73: ("Tuyết", "snow"), 75: ("Tuyết", "snow"),
    77: ("Tuyết", "snow"), 80: ("Mưa rào", "rain"), 81: ("Mưa rào", "rain"),
    82: ("Mưa rào", "rain"), 95: ("Dông", "storm"), 96: ("Dông", "storm"),
    99: ("Dông", "storm"),
}

_cache: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _stale_weather(key: tuple[float, float], city: str | None) -> dict[str, Any] | None:
    """Dùng bản cache gần nhất nếu dịch vụ ngoài lỗi hoặc trả dữ liệu hỏng."""
    with _cache_lock:
        previous = _cache.get(key)
    if not previous:
        return None
    stale = dict(previous[1])
    if city:
        stale["city"] = city
    stale["stale"] = True
    return stale


def _request_json(url: str, *, params: dict[str, Any]) -> dict[str, Any]:
    try:
        # This environment injects HTTP(S)_PROXY=127.0.0.1:9, a dead deny proxy.
        # Do not inherit generic machine proxy variables; allow an explicit
        # working proxy through OPEN_METEO_PROXY when a deployment requires one.
        with requests.Session() as client:
            client.trust_env = False
            configured_proxy = os.environ.get("OPEN_METEO_PROXY", "").strip()
            if configured_proxy:
                client.proxies.update({"http": configured_proxy, "https": configured_proxy})
            response = client.get(url, params=params, timeout=WEATHER_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object")
        return payload
    except (requests.RequestException, ValueError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Không lấy được thời tiết từ Open-Meteo. Vui lòng thử lại hoặc nhập nhiệt độ thủ công.") from exc


def geocode(query: str) -> list[dict[str, Any]]:
    query = " ".join(query.split())
    if not query:
        raise HTTPException(status_code=422, detail="Vui lòng nhập tên thành phố cần tìm.")
    payload = _request_json(GEOCODING_URL, params={"name": query, "count": 5, "language": "vi", "format": "json"})
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise HTTPException(status_code=503, detail="Dữ liệu địa điểm từ Open-Meteo không hợp lệ.")
    return [{key: row.get(key) for key in ("name", "admin1", "country", "latitude", "longitude")} |
            {"lat": row.get("latitude"), "lon": row.get("longitude")} for row in results if isinstance(row, dict)]


def _condition(code: Any) -> tuple[str, str]:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Không rõ trạng thái", "cloud"
    if code in WEATHER_CODES:
        return WEATHER_CODES[code]
    for codes, result in (
        (range(51, 58), ("Mưa phùn", "rain")), (range(61, 68), ("Mưa", "rain")),
        (range(71, 78), ("Tuyết", "snow")), (range(80, 83), ("Mưa rào", "rain")),
        (range(95, 100), ("Dông", "storm")),
    ):
        if code in codes:
            return result
    return "Không rõ trạng thái", "cloud"


def _rain_alert(hourly: dict[str, Any], local_time: datetime) -> dict[str, Any] | None:
    times, probabilities = hourly.get("time"), hourly.get("precipitation_probability")
    if not isinstance(times, list) or not isinstance(probabilities, list):
        return None
    buckets: dict[str, list[float]] = {"sáng": [], "chiều": [], "tối": []}
    for stamp, chance in zip(times, probabilities):
        try:
            hour = datetime.fromisoformat(str(stamp)).hour
            if datetime.fromisoformat(str(stamp)).date() != local_time.date() or hour < local_time.hour:
                continue
            period = "sáng" if 5 <= hour <= 11 else "chiều" if 12 <= hour <= 17 else "tối" if 18 <= hour <= 23 else None
            if period is not None and chance is not None:
                buckets[period].append(float(chance))
        except (TypeError, ValueError):
            continue
    peaks = [(period, max(values)) for period, values in buckets.items() if values]
    if not peaks:
        return None
    period, probability = max(peaks, key=lambda entry: entry[1])
    if probability < RAIN_ALERT_THRESHOLD:
        return None
    rounded = min(100, int(5 * round(probability / 5)))
    return {"period": period, "probability": rounded, "text": f"{period.capitalize()} có khả năng mưa ({rounded}%)"}


def get_weather(lat: float | None = None, lon: float | None = None, city: str | None = None) -> dict[str, Any]:
    if (lat is None) != (lon is None):
        raise HTTPException(status_code=422, detail="Cần cung cấp đồng thời lat và lon.")
    if lat is not None and not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Tọa độ lat/lon nằm ngoài giới hạn hợp lệ.")
    city = " ".join((city or "").split())[:80]
    if lat is None and city and city.casefold() != DEFAULT_CITY.casefold():
        found = geocode(city)
        if not found:
            raise HTTPException(status_code=404, detail="Không tìm thấy thành phố. Hãy thử tên địa điểm khác.")
        location = found[0]
        lat, lon = float(location["lat"]), float(location["lon"])
        city = str(location.get("name") or city)
    elif lat is None:
        lat, lon = DEFAULT_LAT, DEFAULT_LON
        city = city or DEFAULT_CITY

    key = (round(float(lat), 2), round(float(lon), 2))
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and now - entry[0] < CACHE_TTL_SECONDS:
            cached = dict(entry[1])
            if city:
                cached["city"] = city
            cached["stale"] = False
            return cached

    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code,is_day",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "hourly": "precipitation_probability",
        "forecast_days": 1,
    }
    try:
        payload = _request_json(FORECAST_URL, params=params)
        current, daily, hourly = payload["current"], payload["daily"], payload["hourly"]
        timezone_name = payload["timezone"]
        tz = ZoneInfo(timezone_name)
        local_time = datetime.fromisoformat(str(current["time"])).replace(tzinfo=tz)
        code = current["weather_code"]
        condition, icon = _condition(code)
        local_date = local_time.date()
        temp_mode = os.environ.get("TEMP_MODE", "current").strip().lower()
        if temp_mode not in {"current", "daily_mean"}:
            temp_mode = "current"
        tmax = float(daily["temperature_2m_max"][0])
        tmin = float(daily["temperature_2m_min"][0])
        temperature = float(current["temperature_2m"]) if temp_mode == "current" else (tmax + tmin) / 2
        probability = daily.get("precipitation_probability_max", [None])[0]
        result = {
            "temp_current": float(current["temperature_2m"]), "temperature_for_model": temperature,
            "temperature_mode": temp_mode, "feels_like": float(current["apparent_temperature"]),
            "humidity": int(current["relative_humidity_2m"]), "wind_kmh": float(current["wind_speed_10m"]),
            "weather_code": int(code), "is_day": bool(current["is_day"]), "condition_text": condition,
            "icon": icon, "tmax": tmax, "tmin": tmin,
            "precipitation_probability_max": None if probability is None else int(probability),
            "city": city or "Vị trí đã chọn", "timezone": timezone_name,
            "local_date": local_date.isoformat(), "weekday": local_date.weekday(),
            "observed_at": str(current["time"]), "source": "open-meteo",
            "rain_alert": _rain_alert(hourly, local_time), "stale": False,
        }
        with _cache_lock:
            _cache[key] = (now, result)
        return result
    except HTTPException:
        stale = _stale_weather(key, city)
        if stale:
            return stale
        raise
    except (KeyError, IndexError, TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        stale = _stale_weather(key, city)
        if stale:
            return stale
        raise HTTPException(status_code=503, detail="Open-Meteo trả dữ liệu thời tiết không đầy đủ. Vui lòng thử lại hoặc nhập nhiệt độ thủ công.") from exc
