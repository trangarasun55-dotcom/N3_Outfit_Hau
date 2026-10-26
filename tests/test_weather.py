"""Open-Meteo adapter and FastAPI weather endpoint tests."""
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api import main, weather


def fake_response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


FORECAST = {
    "timezone": "Asia/Ho_Chi_Minh",
    "current": {"time": "2026-10-10T08:00", "temperature_2m": 29.2, "apparent_temperature": 33.1,
                "relative_humidity_2m": 75, "wind_speed_10m": 8.4, "weather_code": 3, "is_day": 1},
    "daily": {"temperature_2m_max": [32.0], "temperature_2m_min": [25.0], "precipitation_probability_max": [55]},
    "hourly": {"time": ["2026-10-10T08:00", "2026-10-10T13:00", "2026-10-10T16:00"],
               "precipitation_probability": [10, 45, 53]},
}


class WeatherApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        with weather._cache_lock:
            weather._cache.clear()

    def test_weather_payload_and_cache(self):
        with patch("api.weather.requests.Session") as session_factory:
            client = session_factory.return_value.__enter__.return_value
            client.get.return_value = fake_response(FORECAST)
            response = self.client.get("/weather")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["source"], "open-meteo")
            self.assertEqual(body["condition_text"], "Có mây")
            self.assertEqual(body["rain_alert"]["probability"], 55)
            self.assertEqual(body["city"], "Hà Nội")
            self.client.get("/weather")
            self.assertEqual(client.get.call_count, 1)
            self.assertFalse(client.trust_env)

    def test_city_geocoding_and_invalid_query(self):
        response = fake_response({"results": [{"name": "Đà Nẵng", "admin1": "Đà Nẵng", "country": "Việt Nam", "latitude": 16.05, "longitude": 108.2}]})
        with patch("api.weather.requests.Session") as session_factory:
            session_factory.return_value.__enter__.return_value.get.return_value = response
            found = self.client.get("/geocode", params={"q": "Đà Nẵng"})
        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.json()["results"][0]["lat"], 16.05)
        self.assertEqual(self.client.get("/geocode", params={"q": ""}).status_code, 422)
        self.assertEqual(self.client.get("/geocode", params={"q": "x" * 81}).status_code, 422)

    def test_network_error_returns_503_without_fake_temperature(self):
        with patch("api.weather.requests.Session") as session_factory:
            session_factory.return_value.__enter__.return_value.get.side_effect = weather.requests.Timeout
            response = self.client.get("/weather?lat=21.03&lon=105.85")
        self.assertEqual(response.status_code, 503)
        self.assertIn("nhập nhiệt độ thủ công", response.json()["detail"])

    def test_malformed_weather_uses_cached_result_as_stale(self):
        query = {"lat": 21.03, "lon": 105.85}
        with patch("api.weather.requests.Session") as session_factory:
            session_factory.return_value.__enter__.return_value.get.return_value = fake_response(FORECAST)
            fresh = self.client.get("/weather", params=query)
        self.assertEqual(fresh.status_code, 200)
        cache_key = (round(query["lat"], 2), round(query["lon"], 2))
        with weather._cache_lock:
            cached_at, cached_payload = weather._cache[cache_key]
            weather._cache[cache_key] = (
                cached_at - weather.CACHE_TTL_SECONDS - 1,
                cached_payload,
            )

        with patch("api.weather.requests.Session") as session_factory:
            session_factory.return_value.__enter__.return_value.get.return_value = fake_response(
                {"timezone": "Asia/Ho_Chi_Minh", "current": {}}
            )
            stale = self.client.get("/weather", params=query)

        self.assertEqual(stale.status_code, 200)
        self.assertTrue(stale.json()["stale"])
        self.assertEqual(
            stale.json()["temperature_for_model"],
            fresh.json()["temperature_for_model"],
        )

    def test_coordinate_validation(self):
        self.assertEqual(self.client.get("/weather?lat=91&lon=0").status_code, 422)
        self.assertEqual(self.client.get("/weather?lat=21").status_code, 422)


if __name__ == "__main__":
    unittest.main()
