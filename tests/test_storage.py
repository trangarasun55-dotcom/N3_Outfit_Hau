"""SQLite history and feedback isolation tests."""

import os
import sqlite3
import unittest
from contextlib import closing
from uuid import UUID, uuid4
from unittest.mock import patch

from api import storage
from test_db_utils import new_test_db_path, remove_test_db


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.db_path = new_test_db_path("storage")
        self.addCleanup(remove_test_db, self.db_path)
        self.env_patch = patch.dict(os.environ, {"DB_PATH": str(self.db_path)})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.user_id = uuid4()
        self.request_data = {"activity": "Đi học", "temperature": 22}
        self.result = {
            "outfits": [
                {
                    "id": 1,
                    "name": "Casual – Đi học",
                    "score": 8.1,
                    "match_score": 81,
                    "items": [{"id": "item-1", "name": "Áo"}],
                }
            ]
        }

    def test_history_is_persisted_and_isolated_by_user(self):
        request_id = storage.save_recommendation(self.user_id, self.request_data, self.result)
        self.assertEqual(self.result["request_id"], request_id)
        entries = storage.get_history(self.user_id, 20)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["request_id"], request_id)
        self.assertEqual(entries[0]["result"]["outfits"][0]["name"], "Casual – Đi học")
        self.assertEqual(storage.get_history(uuid4(), 20), [])

    def test_feedback_requires_owned_request_and_valid_outfit(self):
        request_id = storage.save_recommendation(self.user_id, self.request_data, self.result)
        feedback_id = storage.add_feedback(
            user_id=self.user_id,
            request_id=UUID(request_id),
            outfit_id=1,
            action="worn",
            note="  Mặc hôm nay  ",
        )
        self.assertNotIn(feedback_id, {"not_found", "outfit_not_found"})
        with closing(storage._connect()) as connection:
            feedback = connection.execute(
                "SELECT action, note FROM feedback WHERE id = ?", (feedback_id,)
            ).fetchone()
        self.assertEqual(tuple(feedback), ("worn", "  Mặc hôm nay  "))
        self.assertEqual(
            storage.add_feedback(
                user_id=uuid4(),
                request_id=UUID(request_id),
                outfit_id=1,
                action="like",
                note=None,
            ),
            "not_found",
        )
        self.assertEqual(
            storage.add_feedback(
                user_id=self.user_id,
                request_id=UUID(request_id),
                outfit_id=2,
                action="like",
                note=None,
            ),
            "outfit_not_found",
        )

    def test_legacy_requests_are_preserved_and_remain_writable_after_migration(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    item_ids_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL
                );
                CREATE TABLE feedback (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    outfit_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "legacy-request",
                    str(self.user_id),
                    "2026-10-10T00:00:00Z",
                    '{"activity":"Đi học"}',
                    '[]',
                    '[]',
                ),
            )
            connection.commit()

        storage.initialize_storage()
        saved_id = storage.save_recommendation(
            self.user_id,
            self.request_data,
            {
                "outfits": [
                    {
                        "id": 1,
                        "name": "Bộ mới",
                        "score": 8,
                        "items": [{"id": "item-new", "name": "Áo mới"}],
                    }
                ]
            },
        )
        with closing(storage._connect()) as connection:
            legacy = connection.execute(
                "SELECT user_id, input_json, response_json FROM requests WHERE id = ?",
                ("legacy-request",),
            ).fetchone()
            recommendation_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(recommendation_requests)"
                )
            }
            new_request = connection.execute(
                "SELECT response_json FROM requests WHERE id = ?",
                (saved_id,),
            ).fetchone()
            persisted_recommendation = connection.execute(
                "SELECT response_json FROM recommendation_requests WHERE id = ?",
                (saved_id,),
            ).fetchone()
        self.assertEqual(legacy["user_id"], str(self.user_id))
        self.assertEqual(legacy["input_json"], '{"activity":"Đi học"}')
        self.assertEqual(legacy["response_json"], "{}")
        self.assertIn("response_json", recommendation_columns)
        self.assertIn('"outfits"', new_request["response_json"])
        self.assertIn('"outfits"', persisted_recommendation["response_json"])

    def test_migration_repairs_databases_already_at_previous_integrity_version(self):
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                INSERT INTO schema_migrations VALUES
                    (1, '2026-01-01T00:00:00Z'),
                    (2, '2026-01-01T00:00:00Z'),
                    (3, '2026-01-01T00:00:00Z'),
                    (4, '2026-01-01T00:00:00Z');
                CREATE TABLE requests (
                    id TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL
                );
                CREATE TABLE recommendation_requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL
                );
                INSERT INTO requests VALUES ('legacy-1', '{"outfits":[]}');
                INSERT INTO recommendation_requests VALUES
                    ('legacy-1', 'user-1', '2026-01-01T00:00:00Z', '{}', '[]');
                """
            )

        storage.initialize_storage()

        with closing(sqlite3.connect(self.db_path)) as connection:
            saved_payload = connection.execute(
                "SELECT response_json FROM recommendation_requests WHERE id = ?",
                ("legacy-1",),
            ).fetchone()[0]
            latest_version = connection.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()[0]
        self.assertEqual(saved_payload, '{"outfits":[]}')
        self.assertEqual(latest_version, 5)

    def test_admin_outfits_do_not_merge_different_item_compositions(self):
        for item_id, item_name in (("shirt-a", "Áo A"), ("shirt-b", "Áo B")):
            storage.save_recommendation(
                self.user_id,
                self.request_data,
                {
                    "outfits": [
                        {
                            "id": 1,
                            "name": "Casual – Đi học #1",
                            "score": 8,
                            "items": [{"id": item_id, "name": item_name}],
                        }
                    ]
                },
            )

        outfits = storage.admin_outfits()
        self.assertEqual(len(outfits), 2)
        self.assertEqual(
            {tuple(outfit["items"]) for outfit in outfits},
            {("Áo A",), ("Áo B",)},
        )
        self.assertEqual(
            {outfit["recommendation_count"] for outfit in outfits},
            {1},
        )


if __name__ == "__main__":
    unittest.main()
