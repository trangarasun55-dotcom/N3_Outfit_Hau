"""Regression test: live previews return outfits without creating history rows."""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import main
from test_db_utils import isolated_test_db


class RecommendationPreviewTests(unittest.TestCase):
    def test_preview_is_authenticated_and_does_not_persist_history(self):
        with isolated_test_db("preview") as db_path, patch.dict(
            os.environ,
            {
                "DB_PATH": str(db_path),
                "SECRET_KEY": "preview-test-secret",
                "ADMIN_EMAIL": "preview-admin@example.test",
                "ADMIN_PASSWORD": "Preview-Admin-Password-123!",
                "GEMINI_API_KEY": "",
            },
        ):
            with TestClient(main.app) as client:
                unauthenticated = client.post(
                    "/recommend/preview",
                    json={
                        "gender": "Nữ", "activity": "Đi học", "temperature": 24,
                        "styles": ["Casual"], "colors": ["Trắng"],
                        "formality": 35, "top_n": 3, "temperature_source": "manual",
                    },
                )
                self.assertEqual(unauthenticated.status_code, 401)
                registered = client.post(
                    "/auth/register",
                    json={
                        "email": "preview-user@example.test",
                        "password": "Preview-User-Password-123!",
                        "display_name": "Preview User",
                    },
                )
                self.assertEqual(registered.status_code, 201)
                preview = client.post(
                    "/recommend/preview",
                    json={
                        "gender": "Nữ", "activity": "Đi học", "temperature": 24,
                        "styles": ["Casual"], "colors": ["Trắng"],
                        "formality": 35, "top_n": 3, "temperature_source": "manual",
                        "notes": "Ưu tiên dễ vận động",
                    },
                )
                self.assertEqual(preview.status_code, 200, preview.text)
                self.assertTrue(preview.json().get("preview"))
                self.assertEqual(len(preview.json().get("outfits", [])), 3)
                history = client.get("/history?type=viewed&range=all")
                self.assertEqual(history.status_code, 200, history.text)
                self.assertEqual(history.json().get("total"), 0)


if __name__ == "__main__":
    unittest.main()
