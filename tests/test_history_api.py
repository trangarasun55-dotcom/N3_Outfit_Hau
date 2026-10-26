"""HTTP validation and ownership checks for persisted recommendations."""

import os
import sqlite3
from contextlib import closing
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from api import main
from test_db_utils import new_test_db_path, remove_test_db


class HistoryApiTests(unittest.TestCase):
    def setUp(self):
        self.db_path = new_test_db_path("history")
        self.addCleanup(remove_test_db, self.db_path)
        self.env_patch = patch.dict(
            os.environ,
            {
                "DB_PATH": str(self.db_path),
                "GEMINI_API_KEY": "",
                "SECRET_KEY": "history-test-secret",
            },
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.client = TestClient(main.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        registered = self.client.post(
            "/auth/register",
            json={
                "email": "history-user@example.test",
                "password": "History-User-Password-42!",
                "display_name": "History User",
            },
        )
        self.user_id = registered.json()["user"]["id"]
        self.result = {
            "outfits": [
                {
                    "id": 1,
                    "name": "Casual – Đi học",
                    "score": 8.2,
                    "match_score": 82,
                    "items": [{"id": "shirt-1", "name": "Áo sơ mi"}],
                    "explanation": "Giải thích theo quy tắc.",
                }
            ]
        }

    def test_recommendation_is_saved_and_scoped_to_user(self):
        with (
            patch("api.main.build_recommendations", return_value=self.result.copy()),
            patch("api.main.add_explanations", wraps=main.add_explanations) as explain_mock,
        ):
            response = self.client.post(
                "/recommend",
                headers={"X-User-Id": self.user_id},
                json={
                    "gender": "Nữ",
                    "activity": "Đi học",
                    "temperature": 22,
                },
            )
        explain_mock.assert_called_once()
        self.assertFalse(explain_mock.call_args.kwargs["use_gemini"])
        self.assertEqual(response.status_code, 200)
        request_id = response.json()["request_id"]
        self.assertEqual(
            response.json()["outfits"][0]["explanation"],
            self.result["outfits"][0]["explanation"],
        )
        self.assertEqual(response.json()["outfits"][0]["explanation_source"], "rules")
        self.assertEqual(
            response.json()["outfits"][0]["explanation_llm"],
            self.result["outfits"][0]["explanation"],
        )
        history = self.client.get("/history")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(
            history.json()["items"],
            [],
        )
        self.assertEqual(
            self.client.get("/history?type=all").status_code,
            422,
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM recommendation_requests WHERE id = ?",
                    (request_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM recommendation_items WHERE request_id = ?",
                    (request_id,),
                ).fetchone()[0],
                1,
            )
            self.assertIn(
                3,
                {
                    row[0]
                    for row in connection.execute("SELECT version FROM schema_migrations")
                },
            )

    def test_history_requires_authenticated_session(self):
        self.client.post("/auth/logout")
        self.assertEqual(self.client.get("/history").status_code, 401)
        self.assertEqual(
            self.client.get("/history", headers={"X-User-Id": str(uuid4())}).status_code,
            401,
        )
        self.client.post(
            "/auth/login",
            json={
                "email": "history-user@example.test",
                "password": "History-User-Password-42!",
            },
        )
        recommendation = self.client.post(
            "/recommend",
            json={"gender": "Nữ", "activity": "Đi học", "temperature": 22},
        )
        self.assertEqual(recommendation.status_code, 200)
        self.assertIn("request_id", recommendation.json())

    def test_explanation_integration_failure_keeps_recommendation_response(self):
        with (
            patch("api.main.build_recommendations", return_value=self.result.copy()),
            patch("api.main.add_explanations", side_effect=RuntimeError("test-only private detail")),
        ):
            response = self.client.post(
                "/recommend",
                headers={"X-User-Id": self.user_id},
                json={"gender": "Nữ", "activity": "Đi học", "temperature": 22},
            )
        self.assertEqual(response.status_code, 200)
        outfit = response.json()["outfits"][0]
        self.assertEqual(outfit["explanation_source"], "rules")
        self.assertTrue(outfit["explanation_llm"])
        self.assertNotIn("test-only private detail", response.text)

    def test_explanation_endpoint_runs_after_recommend_and_persists_only_explanations(self):
        recommendation = self.client.post(
            "/recommend",
            headers={"X-User-Id": self.user_id},
            json={"gender": "Nữ", "activity": "Đi học", "temperature": 22},
        )
        self.assertEqual(recommendation.status_code, 200)
        initial = recommendation.json()["outfits"][0]
        self.assertEqual(initial["explanation_source"], "rules")

        def fake_gemini(result, request_data, *, use_gemini=True):
            self.assertTrue(use_gemini)
            for outfit in result["outfits"]:
                outfit["explanation_llm"] = "**Áo sơ mi** hợp 22°C."
                outfit["explanation_structured"] = {
                    "headline": "Lựa chọn gọn gàng",
                    "why_suitable": "**Áo sơ mi** hợp 22°C.",
                    "weather_fit": "Nhiệt độ đầu vào là 22°C.",
                    "caution": "Khoảng điểm model ổn định.",
                    "tip": "Bắt đầu với áo sơ mi.",
                }
                outfit["explanation_source"] = "gemini"
                outfit["comment_source"] = "gemini"
                outfit["explanation_notice"] = None
            return result

        with patch("api.main.add_explanations", side_effect=fake_gemini):
            response = self.client.post(
                f"/recommendations/{recommendation.json()['request_id']}/explanations"
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(all(item["explanation_source"] == "gemini" for item in response.json()["outfits"]))
        stored = main.storage.get_recommendation_for_explanation(
            UUID(self.user_id), UUID(recommendation.json()["request_id"])
        )
        self.assertEqual(stored[1]["outfits"][0]["score"], initial["score"])
        self.assertEqual(stored[1]["outfits"][0]["comment_source"], "gemini")
        self.assertEqual(stored[1]["outfits"][0]["explanation_llm"], "**Áo sơ mi** hợp 22°C.")

    def test_explanation_endpoint_is_owner_scoped(self):
        recommendation = self.client.post(
            "/recommend",
            headers={"X-User-Id": self.user_id},
            json={"gender": "Nữ", "activity": "Đi học", "temperature": 22},
        )
        second = TestClient(main.app)
        second.__enter__()
        self.addCleanup(second.__exit__, None, None, None)
        second.post(
            "/auth/register",
            json={
                "email": "other-explanation-user@example.test",
                "password": "Other-User-Password-42!",
                "display_name": "Other User",
            },
        )
        response = second.post(
            f"/recommendations/{recommendation.json()['request_id']}/explanations"
        )
        self.assertEqual(response.status_code, 404)

    def test_feedback_is_validated_and_owner_scoped(self):
        request_id = main.save_recommendation(
            UUID(self.user_id),
            {"activity": "Đi học", "temperature": 22},
            self.result.copy(),
        )
        payload = {
            "request_id": request_id,
            "outfit_id": 1,
            "action": "worn",
            "note": "  Đã mặc hôm nay  ",
        }
        saved = self.client.post(
            "/feedback", headers={"X-User-Id": self.user_id}, json=payload
        )
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["status"], "saved")
        self.assertEqual(
            self.client.post(
                "/feedback",
                headers={"X-User-Id": str(uuid4())},
                json=payload,
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                "/feedback",
                headers={"X-User-Id": self.user_id},
                json={**payload, "action": "invalid"},
            ).status_code,
            422,
        )


if __name__ == "__main__":
    unittest.main()
