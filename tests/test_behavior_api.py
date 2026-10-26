"""Behavior tracking, private history, time ranges and admin analytics tests."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing
from datetime import UTC, datetime, time, timedelta
from unittest.mock import patch
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from api import main, storage
from api.history_time import history_bounds, to_utc_iso
from test_db_utils import new_test_db_path, remove_test_db

VN = ZoneInfo("Asia/Ho_Chi_Minh")


class HistoryTimeTests(unittest.TestCase):
    def test_midnight_boundary_uses_vietnam_calendar_day(self) -> None:
        now = datetime(2026, 10, 10, 5, 0, tzinfo=UTC)
        start, end = history_bounds("day", now=now)
        assert start is not None and end is not None
        at_previous_2359 = datetime(2026, 10, 9, 16, 59, tzinfo=UTC)
        at_current_0001 = datetime(2026, 10, 9, 17, 1, tzinfo=UTC)
        self.assertFalse(start <= to_utc_iso(at_previous_2359) < end)
        self.assertTrue(start <= to_utc_iso(at_current_0001) < end)

    def test_week_starts_monday_in_vietnam(self) -> None:
        start, end = history_bounds(
            "week", now=datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
        )
        self.assertEqual(start, "2026-10-04T17:00:00.000000Z")
        self.assertEqual(end, "2026-10-11T17:00:00.000000Z")

    def test_month_boundaries_include_day_one_and_exclude_prior_month(self) -> None:
        start, end = history_bounds(
            "month", now=datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
        )
        self.assertEqual(start, "2026-09-30T17:00:00.000000Z")
        self.assertEqual(end, "2026-10-31T17:00:00.000000Z")

    def test_single_day_custom_range_handles_leap_day(self) -> None:
        start, end = history_bounds(
            "from-to", from_date="2024-02-29", to_date="2024-02-29"
        )
        self.assertEqual(start, "2024-02-28T17:00:00.000000Z")
        self.assertEqual(end, "2024-02-29T17:00:00.000000Z")
        self.assertEqual(
            history_bounds(
                "month",
                now=datetime(2024, 2, 29, 5, 0, tzinfo=UTC),
            ),
            ("2024-01-31T17:00:00.000000Z", "2024-02-29T17:00:00.000000Z"),
        )


class BehaviorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = new_test_db_path("behavior")
        self.addCleanup(remove_test_db, self.db_path)
        self.env_patch = patch.dict(
            os.environ,
            {
                "DB_PATH": str(self.db_path),
                "SECRET_KEY": "behavior-test-secret",
                "ADMIN_EMAIL": "admin-behavior@example.test",
                "ADMIN_PASSWORD": "Admin-Behavior-Password-42!",
                "GEMINI_API_KEY": "",
            },
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

        self.client_a = TestClient(main.app)
        self.client_a.__enter__()
        self.addCleanup(self.client_a.__exit__, None, None, None)
        self.user_a = self._register(
            self.client_a, "person-a@example.test", "Person A"
        )

        self.client_b = TestClient(main.app)
        self.client_b.__enter__()
        self.addCleanup(self.client_b.__exit__, None, None, None)
        self.user_b = self._register(
            self.client_b, "person-b@example.test", "Person B"
        )

        self.admin_client = TestClient(main.app)
        self.admin_client.__enter__()
        self.addCleanup(self.admin_client.__exit__, None, None, None)
        login = self.admin_client.post(
            "/auth/login",
            json={
                "email": "admin-behavior@example.test",
                "password": "Admin-Behavior-Password-42!",
            },
        )
        self.assertEqual(login.status_code, 200)
        self.item_a = self._save_item(self.user_a)
        self.item_b = self._save_item(self.user_b)

    @staticmethod
    def _register(client: TestClient, email: str, name: str) -> dict[str, str]:
        response = client.post(
            "/auth/register",
            json={
                "email": email,
                "password": "User-Behavior-Password-42!",
                "display_name": name,
            },
        )
        if response.status_code != 201:
            raise AssertionError(response.text)
        return response.json()["user"]

    @staticmethod
    def _save_item(user: dict[str, str]) -> int:
        result = {
            "outfits": [
                {
                    "id": 1,
                    "name": "Bộ đi học mẫu",
                    "score": 8.2,
                    "score_low": 7.7,
                    "score_high": 8.7,
                    "confidence_level": "Trung bình",
                    "items": [{"id": "shirt-1", "name": "Áo sơ mi"}],
                    "explanation": "Lý do mẫu",
                    "explanation_source": "rules",
                }
            ]
        }
        storage.save_recommendation(
            UUID(user["id"]),
            {
                "gender": "Nữ",
                "activity": "Đi học",
                "temperature": 22,
                "styles": ["Casual"],
            },
            result,
        )
        return int(result["outfits"][0]["request_item_id"])

    def test_user_cannot_read_or_modify_another_users_outfit(self) -> None:
        self.assertEqual(self.client_b.get("/history?type=viewed").json()["items"], [])
        self.assertEqual(
            self.client_b.post(
                f"/outfits/{self.item_a}/finalize", json={"note": "private"}
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client_b.post(
                f"/outfits/{self.item_a}/feedback", json={"value": 1}
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client_b.post(
                "/events/view",
                json={"action": "start", "request_item_id": self.item_a},
            ).status_code,
            404,
        )
        self.assertEqual(self.client_b.get("/admin/stats").status_code, 403)

    def test_view_start_heartbeat_end_has_utc_z_and_thirty_minute_cap(self) -> None:
        started = self.client_a.post(
            "/events/view",
            json={"action": "start", "request_item_id": self.item_a},
        )
        self.assertEqual(started.status_code, 200)
        view = started.json()
        self.assertTrue(view["started_at"].endswith("Z"))
        old = to_utc_iso(datetime.now(UTC) - timedelta(seconds=storage.MAX_VIEW_SECONDS + 60))
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                "UPDATE outfit_views SET started_at = ?, last_seen_at = ? WHERE id = ?",
                (old, old, view["view_id"]),
            )
        heartbeat = self.client_a.post(
            "/events/view",
            json={"action": "heartbeat", "view_id": view["view_id"]},
        )
        self.assertEqual(heartbeat.status_code, 200)
        self.assertEqual(heartbeat.json()["duration_seconds"], 1800)
        self.assertTrue(heartbeat.json()["ended"])

        next_view = self.client_a.post(
            "/events/view",
            json={"action": "start", "request_item_id": self.item_a},
        ).json()
        ended = self.client_a.post(
            "/events/view",
            json={"action": "end", "view_id": next_view["view_id"]},
        )
        self.assertEqual(ended.status_code, 200)
        self.assertTrue(ended.json()["ended"])
        self.assertTrue(ended.json()["ended_at"].endswith("Z"))
        with closing(sqlite3.connect(self.db_path)) as connection:
            durations = connection.execute(
                "SELECT duration_seconds FROM outfit_views WHERE user_id = ?",
                (self.user_a["id"],),
            ).fetchall()
        self.assertTrue(all(0 <= row[0] <= storage.MAX_VIEW_SECONDS for row in durations))

    def test_finalize_can_be_cancelled_and_rating_is_upserted(self) -> None:
        first_finalize = self.client_a.post(
            f"/outfits/{self.item_a}/finalize", json={"note": "Mặc hôm nay"}
        )
        self.assertEqual(first_finalize.status_code, 200)
        self.assertTrue(first_finalize.json()["finalization"]["finalized_at"].endswith("Z"))
        self.assertEqual(
            self.client_a.delete(f"/outfits/{self.item_a}/finalize").json()["status"],
            "unfinalized",
        )
        self.assertEqual(
            self.client_a.post(
                f"/outfits/{self.item_a}/finalize", json={}
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client_a.post(
                f"/outfits/{self.item_a}/feedback",
                json={"value": 1, "reason": "Phù hợp"},
            ).json()["feedback"]["value"],
            1,
        )
        changed = self.client_a.post(
            f"/outfits/{self.item_a}/feedback",
            json={"value": -1, "reason": "Đổi đánh giá"},
        )
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.json()["feedback"]["value"], -1)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM outfit_ratings WHERE user_id = ? AND request_item_id = ?",
                    (self.user_a["id"], self.item_a),
                ).fetchone()[0],
                1,
            )
        self.assertEqual(
            self.client_a.get("/history?type=finalized").json()["total"],
            1,
        )
        self.assertEqual(
            self.client_a.get("/history?type=feedback").json()["items"][0]["feedback"]["value"],
            -1,
        )

    def test_day_filter_excludes_previous_2359_and_includes_today_0001(self) -> None:
        today = datetime.now(VN).date()
        previous_2359 = to_utc_iso(
            datetime.combine(today - timedelta(days=1), time(23, 59), tzinfo=VN)
        )
        today_0001 = to_utc_iso(datetime.combine(today, time(0, 1), tzinfo=VN))
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            for stamp in (previous_2359, today_0001):
                connection.execute(
                    """
                    INSERT INTO outfit_views
                        (id, user_id, request_item_id, started_at, last_seen_at, ended_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid4()), self.user_a["id"], self.item_a, stamp, stamp, stamp),
                )
        filtered = self.client_a.get("/history?type=viewed&range=day")
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(filtered.json()["total"], 1)
        self.assertEqual(filtered.json()["items"][0]["created_at"], today_0001)

    def test_admin_stats_and_admin_views_match_seeded_activity(self) -> None:
        started_at = to_utc_iso(datetime.now(UTC) - timedelta(seconds=45))
        ended_at = to_utc_iso(datetime.now(UTC) - timedelta(seconds=10))
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO outfit_views
                    (id, user_id, request_item_id, started_at, last_seen_at,
                     ended_at, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    self.user_a["id"],
                    self.item_a,
                    started_at,
                    ended_at,
                    ended_at,
                    35,
                ),
            )
        self.client_a.post(f"/outfits/{self.item_a}/finalize", json={})
        self.client_a.post(
            f"/outfits/{self.item_a}/feedback", json={"value": 1}
        )

        stats_response = self.admin_client.get("/admin/stats")
        self.assertEqual(stats_response.status_code, 200)
        self.assertEqual(
            stats_response.json()["stats"],
            {
                "total_users": 2,
                "total_recommendations": 2,
                "total_views": 1,
                "total_view_seconds": 35,
                "average_view_seconds": 35.0,
                "likes": 1,
                "dislikes": 0,
                "finalized_count": 1,
            },
        )
        self.assertEqual(len(self.admin_client.get("/admin/users").json()["items"]), 3)
        outfit_rows = self.admin_client.get("/admin/outfits").json()["items"]
        self.assertEqual(outfit_rows[0]["view_count"], 1)
        self.assertEqual(outfit_rows[0]["view_seconds"], 35)
        export = self.admin_client.get("/admin/export.csv")
        self.assertEqual(export.status_code, 200)
        self.assertIn("view_count", export.text)
        self.assertEqual(self.client_a.get("/admin/outfits").status_code, 403)


if __name__ == "__main__":
    unittest.main()
