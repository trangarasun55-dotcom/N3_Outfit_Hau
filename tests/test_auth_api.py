"""Authentication, role authorization, migrations and profile API tests."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import main
from api.auth import verify_password
from api import storage
from test_db_utils import new_test_db_path, remove_test_db


class AuthApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = new_test_db_path("auth")
        self.addCleanup(remove_test_db, self.db_path)
        self.env_patch = patch.dict(
            os.environ,
            {
                "DB_PATH": str(self.db_path),
                "SECRET_KEY": "test-only-session-secret-key",
                "ADMIN_EMAIL": "admin@outfitai.test",
                "ADMIN_PASSWORD": "Test-Admin-Password-42!",
                "COOKIE_SECURE": "0",
            },
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.client = TestClient(main.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def register_user(self, email: str = "student@example.test"):
        return self.client.post(
            "/auth/register",
            json={
                "email": email,
                "password": "Student-Password-42!",
                "display_name": "  Mai   Linh ",
            },
        )

    def login_admin(self):
        return self.client.post(
            "/auth/login",
            json={
                "email": "admin@outfitai.test",
                "password": "Test-Admin-Password-42!",
            },
        )

    def test_admin_is_seeded_once_from_environment_and_can_access_admin_route(self) -> None:
        admin = storage.get_user_by_email("admin@outfitai.test")
        self.assertIsNotNone(admin)
        assert admin is not None
        self.assertEqual(admin["role"], "admin")
        self.assertTrue(admin["is_active"])
        self.assertNotEqual(admin["password_hash"], "Test-Admin-Password-42!")
        self.assertTrue(
            verify_password("Test-Admin-Password-42!", admin["password_hash"])
        )
        logged_in = self.login_admin()
        self.assertEqual(logged_in.status_code, 200)
        self.assertIn("httponly", logged_in.headers["set-cookie"].casefold())
        self.assertIn("samesite=lax", logged_in.headers["set-cookie"].casefold())
        self.assertEqual(self.client.get("/admin/users").status_code, 200)

    def test_duplicate_email_is_rejected_case_insensitively(self) -> None:
        first = self.register_user()
        self.assertEqual(first.status_code, 201)
        duplicate = self.client.post(
            "/auth/register",
            json={
                "email": "STUDENT@example.test",
                "password": "Different-Password-42!",
                "display_name": "Duplicate",
            },
        )
        self.assertEqual(duplicate.status_code, 409)

    def test_wrong_password_returns_401(self) -> None:
        self.assertEqual(self.register_user().status_code, 201)
        response = self.client.post(
            "/auth/login",
            json={"email": "student@example.test", "password": "Incorrect-Password"},
        )
        self.assertEqual(response.status_code, 401)

    def test_profile_requires_login_and_user_cannot_access_admin(self) -> None:
        self.assertEqual(self.client.get("/profile").status_code, 401)
        registered = self.register_user()
        self.assertEqual(registered.status_code, 201)
        self.assertEqual(self.client.get("/admin/users").status_code, 403)
        profile = self.client.get("/profile")
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json()["profile"]["display_name"], "Mai Linh")

    def test_auth_me_profile_patch_and_logout(self) -> None:
        self.assertEqual(self.register_user().status_code, 201)
        self.assertEqual(self.client.get("/auth/me").status_code, 200)
        changed = self.client.patch("/profile", json={"gender": "Nữ"})
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.json()["profile"]["gender"], "Nữ")
        logout = self.client.post("/auth/logout")
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(self.client.get("/profile").status_code, 401)

    def test_password_change_verifies_old_password_and_hashes_new_password(self) -> None:
        self.assertEqual(self.register_user().status_code, 201)
        changed = self.client.post(
            "/profile/password",
            json={
                "current_password": "Student-Password-42!",
                "new_password": "New-Student-Password-52!",
            },
        )
        self.assertEqual(changed.status_code, 200)
        bad_current = self.client.post(
            "/profile/password",
            json={
                "current_password": "Wrong-Student-Password-42!",
                "new_password": "Another-Student-Password-62!",
            },
        )
        self.assertEqual(bad_current.status_code, 401)
        self.client.post("/auth/logout")
        old_login = self.client.post(
            "/auth/login",
            json={
                "email": "student@example.test",
                "password": "Student-Password-42!",
            },
        )
        self.assertEqual(old_login.status_code, 401)
        new_login = self.client.post(
            "/auth/login",
            json={
                "email": "student@example.test",
                "password": "New-Student-Password-52!",
            },
        )
        self.assertEqual(new_login.status_code, 200)

    def test_existing_legacy_rows_survive_schema_migration(self) -> None:
        legacy_path = new_test_db_path("legacy")
        self.addCleanup(remove_test_db, legacy_path)
        with closing(sqlite3.connect(legacy_path)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    item_ids_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    response_json TEXT NOT NULL
                );
                CREATE TABLE feedback (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    outfit_id INTEGER NOT NULL CHECK(outfit_id BETWEEN 1 AND 5),
                    action TEXT NOT NULL CHECK(action IN ('worn', 'like', 'dislike')),
                    note TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES requests(id) ON DELETE CASCADE
                );
                INSERT INTO requests VALUES
                    ('request-legacy', 'legacy-user', '2026-01-01T00:00:00+00:00',
                     '{}', '[]', '[]', '{}');
                INSERT INTO feedback VALUES
                    ('feedback-legacy', 'request-legacy', 'legacy-user', 1,
                     'like', 'keep me', '2026-01-01T00:01:00+00:00');
                """
            )

        with patch.dict(os.environ, {"DB_PATH": str(legacy_path)}):
            storage.initialize_storage()
            with closing(sqlite3.connect(legacy_path)) as connection:
                request_count = connection.execute(
                    "SELECT COUNT(*) FROM requests WHERE id = 'request-legacy'"
                ).fetchone()[0]
                feedback_count = connection.execute(
                    "SELECT COUNT(*) FROM feedback WHERE id = 'feedback-legacy'"
                ).fetchone()[0]
                migration_version = connection.execute(
                    "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()[0]
                user_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(users)")
                }
                recommendation_request_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(recommendation_requests)"
                    )
                }
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
        self.assertEqual(request_count, 1)
        self.assertEqual(feedback_count, 1)
        self.assertEqual(migration_version, 5)
        self.assertTrue({"recommendation_requests", "recommendation_items"} <= tables)
        self.assertIn("response_json", recommendation_request_columns)
        self.assertTrue(
            {"email", "password_hash", "role", "is_active", "last_login_at"}
            .issubset(user_columns)
        )

    def test_system_endpoints_are_available(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/model-info").status_code, 200)


if __name__ == "__main__":
    unittest.main()
