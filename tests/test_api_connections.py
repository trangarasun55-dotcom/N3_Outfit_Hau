"""Connection diagnostics and CORS behavior tests."""

import unittest
from unittest.mock import Mock, patch

import requests
from fastapi.middleware.cors import CORSMiddleware

from api import main
from scripts import check_connections


class ConnectionTests(unittest.TestCase):
    def test_health_check_validates_http_status_and_json_shape(self):
        response = Mock()
        response.json.return_value = {"status": "ok", "model_version": "test-version"}
        with (
            patch.object(check_connections.requests, "get", return_value=response) as get,
            patch.object(check_connections, "report") as report,
        ):
            self.assertTrue(check_connections.check_api())
        response.raise_for_status.assert_called_once_with()
        get.assert_called_once_with(
            f"{check_connections.API_URL}/health", timeout=check_connections.API_TIMEOUT
        )
        self.assertTrue(report.call_args.args[1])

    def test_health_check_reports_http_and_connection_failures(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("private response body")
        with (
            patch.object(check_connections.requests, "get", return_value=response),
            patch.object(check_connections, "report") as report,
        ):
            self.assertFalse(check_connections.check_api())
        self.assertEqual(report.call_args.args[2], "HTTPError")
        self.assertNotIn("private response body", str(report.call_args))

        with (
            patch.object(check_connections.requests, "get", side_effect=requests.Timeout),
            patch.object(check_connections, "report") as report,
        ):
            self.assertFalse(check_connections.check_api())
        self.assertEqual(report.call_args.args[2], "Timeout")

    def test_weather_check_validates_http_status_without_exposing_error_body(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("private body")
        with (
            patch.object(check_connections.requests, "get", return_value=response) as get,
            patch.object(check_connections, "report") as report,
        ):
            check_connections.check_weather()
        get.assert_called_once_with(
            f"{check_connections.API_URL}/weather",
            params={"city": "Hà Nội"},
            timeout=check_connections.WEATHER_TIMEOUT,
        )
        self.assertEqual(report.call_args.args[2], "HTTPError")
        self.assertNotIn("private body", str(report.call_args))

    def test_cors_allows_configured_origin_and_rejects_other_origin(self):
        cors_config = next(
            middleware.kwargs
            for middleware in main.app.user_middleware
            if middleware.cls is CORSMiddleware
        )
        configured_origins = cors_config["allow_origins"]
        self.assertTrue(configured_origins, "CORS chưa cấu hình origin nào.")
        allowed_origin = configured_origins[0]
        middleware = CORSMiddleware(lambda *_args: None, **cors_config)
        self.assertTrue(middleware.is_allowed_origin(allowed_origin))
        self.assertFalse(middleware.is_allowed_origin("https://cors-test.invalid"))


if __name__ == "__main__":
    unittest.main()
