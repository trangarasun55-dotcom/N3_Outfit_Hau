"""SQLite persistence for recommendation history and user feedback."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

try:
    from .config import ROOT
    from .history_time import (
        history_bounds,
        parse_utc_iso,
        to_utc_iso,
        utc_now_iso,
    )
except ImportError:  # `uvicorn main:app` from api/
    from config import ROOT
    from history_time import history_bounds, parse_utc_iso, to_utc_iso, utc_now_iso

MAX_VIEW_SECONDS = 30 * 60


def _database_path() -> Path:
    configured = os.environ.get("DB_PATH", "").strip() or "data/app.db"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else ROOT / path


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_storage() -> None:
    with closing(_connect()) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            applied = {
                int(row["version"])
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            if 1 not in applied:
                _apply_initial_schema(connection)
                connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (1, utc_now_iso()),
                )
            if 2 not in applied:
                _apply_recommendation_schema(connection)
                connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (2, utc_now_iso()),
                )
            if 3 not in applied:
                _apply_behavior_schema(connection)
                connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (3, utc_now_iso()),
                )
            if 4 not in applied:
                _apply_data_integrity_schema(connection)
                connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (4, utc_now_iso()),
                )
            if 5 not in applied:
                _apply_history_response_schema(connection)
                connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (5, utc_now_iso()),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _apply_data_integrity_schema(connection: sqlite3.Connection) -> None:
    for table in ("requests", "recommendation_requests"):
        request_columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if "response_json" not in request_columns:
            connection.execute(
                f"ALTER TABLE {table} "
                "ADD COLUMN response_json TEXT NOT NULL DEFAULT '{}'"
            )
    if "response_json" in {
        row["name"] for row in connection.execute("PRAGMA table_info(requests)")
    }:
        connection.execute(
            """
            UPDATE recommendation_requests
            SET response_json = (
                SELECT requests.response_json
                FROM requests
                WHERE requests.id = recommendation_requests.id
            )
            WHERE response_json = '{}'
              AND EXISTS (
                  SELECT 1 FROM requests
                  WHERE requests.id = recommendation_requests.id
                    AND requests.response_json != '{}'
              )
            """
        )

    item_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(recommendation_items)")
    }
    if "composition_key" not in item_columns:
        connection.execute(
            "ALTER TABLE recommendation_items "
            "ADD COLUMN composition_key TEXT NOT NULL DEFAULT '[]'"
        )
    rows = connection.execute(
        "SELECT id, items_json FROM recommendation_items WHERE composition_key = '[]'"
    ).fetchall()
    for row in rows:
        try:
            items = json.loads(row["items_json"])
        except (TypeError, json.JSONDecodeError):
            items = []
        key = _composition_key(items) or json.dumps(
            [f"legacy-recommendation-{row['id']}"], separators=(",", ":")
        )
        connection.execute(
            "UPDATE recommendation_items SET composition_key = ? WHERE id = ?",
            (key, row["id"]),
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_recommendation_items_composition "
        "ON recommendation_items(composition_key)"
    )


def _apply_history_response_schema(connection: sqlite3.Connection) -> None:
    request_columns = {
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(recommendation_requests)"
        )
    }
    if "response_json" not in request_columns:
        connection.execute(
            "ALTER TABLE recommendation_requests "
            "ADD COLUMN response_json TEXT NOT NULL DEFAULT '{}'"
        )
    legacy_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(requests)")
    }
    if "response_json" in legacy_columns:
        connection.execute(
            """
            UPDATE recommendation_requests
            SET response_json = (
                SELECT requests.response_json
                FROM requests
                WHERE requests.id = recommendation_requests.id
            )
            WHERE response_json = '{}'
              AND EXISTS (
                  SELECT 1 FROM requests
                  WHERE requests.id = recommendation_requests.id
                    AND requests.response_json != '{}'
              )
            """
        )


def _apply_initial_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'admin')),
            gender TEXT CHECK(gender IS NULL OR gender IN ('Nam', 'Nữ')),
            preferred_style TEXT,
            default_context TEXT,
            is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
            created_at TEXT NOT NULL,
            last_login_at TEXT
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_created ON users(created_at DESC)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            input_json TEXT NOT NULL,
            item_ids_json TEXT NOT NULL,
            scores_json TEXT NOT NULL,
            response_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_user_created "
        "ON requests(user_id, created_at DESC)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            outfit_id INTEGER NOT NULL CHECK(outfit_id BETWEEN 1 AND 5),
            action TEXT NOT NULL CHECK(action IN ('worn', 'like', 'dislike')),
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(request_id) REFERENCES requests(id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_user_created "
        "ON feedback(user_id, created_at DESC)"
    )


def _apply_recommendation_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_requests (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            input_json TEXT NOT NULL,
            warnings_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_recommendation_requests_user_created "
        "ON recommendation_requests(user_id, created_at DESC)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            rank INTEGER NOT NULL CHECK(rank BETWEEN 1 AND 5),
            outfit_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            score REAL NOT NULL CHECK(score BETWEEN 0 AND 10),
            score_low REAL NOT NULL CHECK(score_low BETWEEN 0 AND 10),
            score_high REAL NOT NULL CHECK(score_high BETWEEN 0 AND 10),
            confidence_level TEXT NOT NULL
                CHECK(confidence_level IN ('Cao', 'Trung bình', 'Thấp')),
            items_json TEXT NOT NULL,
            explanation TEXT NOT NULL,
            comment_source TEXT NOT NULL
                CHECK(comment_source IN ('gemini', 'rules')),
            created_at TEXT NOT NULL,
            UNIQUE(request_id, rank),
            FOREIGN KEY(request_id) REFERENCES recommendation_requests(id)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_recommendation_items_user_created "
        "ON recommendation_items(user_id, created_at DESC)"
    )


def _apply_behavior_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS outfit_views (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            request_item_id INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            ended_at TEXT,
            duration_seconds INTEGER NOT NULL DEFAULT 0
                CHECK(duration_seconds BETWEEN 0 AND 1800),
            FOREIGN KEY(request_item_id) REFERENCES recommendation_items(id)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_outfit_views_user_started "
        "ON outfit_views(user_id, started_at DESC)"
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_outfit_views_one_active
        ON outfit_views(user_id, request_item_id)
        WHERE ended_at IS NULL
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS finalized_outfits (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            request_item_id INTEGER NOT NULL,
            finalized_at TEXT NOT NULL,
            note TEXT,
            UNIQUE(user_id, request_item_id),
            FOREIGN KEY(request_item_id) REFERENCES recommendation_items(id)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_finalized_user_time "
        "ON finalized_outfits(user_id, finalized_at DESC)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS outfit_ratings (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            request_item_id INTEGER NOT NULL,
            value INTEGER NOT NULL CHECK(value IN (-1, 1)),
            reason TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, request_item_id),
            FOREIGN KEY(request_item_id) REFERENCES recommendation_items(id)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_outfit_ratings_user_time "
        "ON outfit_ratings(user_id, created_at DESC)"
    )


def _user_public_fields(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "display_name": row["display_name"],
        "role": row["role"],
        "gender": row["gender"],
        "preferred_style": row["preferred_style"],
        "default_context": row["default_context"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def create_user(
    *,
    email: str,
    display_name: str,
    password_hash: str,
    role: str = "user",
) -> dict[str, Any]:
    initialize_storage()
    user_id = str(uuid4())
    created_at = utc_now_iso()
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO users (id, email, display_name, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, email, display_name, password_hash, role, created_at),
        )
        row = connection.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        raise RuntimeError("Không thể đọc lại người dùng vừa tạo.")
    return _user_public_fields(row)


def get_user_by_email(email: str) -> dict[str, Any] | None:
    initialize_storage()
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
        ).fetchone()
    if row is None:
        return None
    user = _user_public_fields(row)
    user["password_hash"] = row["password_hash"]
    return user


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    initialize_storage()
    with closing(_connect()) as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _user_public_fields(row) if row is not None else None


def update_last_login(user_id: str) -> None:
    initialize_storage()
    with closing(_connect()) as connection, connection:
        connection.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (utc_now_iso(), user_id),
        )


def update_password_hash(user_id: str, password_hash: str) -> None:
    initialize_storage()
    with closing(_connect()) as connection, connection:
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )


def update_user_profile(user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    allowed_columns = {
        "display_name",
        "gender",
        "preferred_style",
        "default_context",
    }
    if not updates or not set(updates).issubset(allowed_columns):
        raise ValueError("Cập nhật hồ sơ không hợp lệ.")
    initialize_storage()
    assignments = ", ".join(f"{column} = ?" for column in updates)
    values = [*updates.values(), user_id]
    with closing(_connect()) as connection, connection:
        connection.execute(
            f"UPDATE users SET {assignments} WHERE id = ?",
            values,
        )
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _user_public_fields(row) if row is not None else None


def seed_admin_user(email: str, password_hash: str, display_name: str = "Quản trị viên") -> dict[str, Any]:
    initialize_storage()
    existing = get_user_by_email(email)
    if existing is None:
        try:
            return create_user(
                email=email,
                display_name=display_name,
                password_hash=password_hash,
                role="admin",
            )
        except sqlite3.IntegrityError:
            existing = get_user_by_email(email)
            if existing is None:
                raise
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            UPDATE users
            SET role = 'admin', is_active = 1, password_hash = ?
            WHERE id = ?
            """,
            (password_hash, existing["id"]),
        )
    seeded = get_user_by_id(existing["id"])
    if seeded is None:
        raise RuntimeError("Không thể đọc lại tài khoản admin vừa seed.")
    return seeded


def list_users(limit: int = 100) -> list[dict[str, Any]]:
    initialize_storage()
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT * FROM users
            ORDER BY created_at DESC, id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_user_public_fields(row) for row in rows]


def save_recommendation(
    user_id: UUID, request_data: dict[str, Any], result: dict[str, Any]
) -> str:
    initialize_storage()
    request_id = str(uuid4())
    result["request_id"] = request_id
    outfits = result.get("outfits", [])
    item_ids = [
        [str(item["id"]) for item in outfit.get("items", []) if item.get("id") is not None]
        for outfit in outfits
    ]
    scores = [
        {
            "outfit_id": int(outfit["id"]),
            "score": outfit.get("score"),
            "match_score": outfit.get("match_score"),
        }
        for outfit in outfits
    ]
    created_at = utc_now_iso()
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO recommendation_requests
                (id, user_id, created_at, input_json, warnings_json, response_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                str(user_id),
                created_at,
                json.dumps(request_data, ensure_ascii=False, separators=(",", ":")),
                json.dumps(result.get("warnings", []), ensure_ascii=False, separators=(",", ":")),
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        connection.execute(
            """
            INSERT INTO requests
                (id, user_id, created_at, input_json, item_ids_json, scores_json, response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                str(user_id),
                created_at,
                json.dumps(request_data, ensure_ascii=False, separators=(",", ":")),
                json.dumps(item_ids, ensure_ascii=False, separators=(",", ":")),
                json.dumps(scores, ensure_ascii=False, separators=(",", ":")),
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        for rank, outfit in enumerate(outfits, start=1):
            cursor = connection.execute(
                """
                INSERT INTO recommendation_items
                    (request_id, user_id, rank, outfit_id, name, score, score_low,
                     score_high, confidence_level, items_json, explanation,
                     comment_source, created_at, composition_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    str(user_id),
                    rank,
                    int(outfit["id"]),
                    str(outfit.get("name") or f"Outfit {rank}"),
                    float(outfit.get("score", 0)),
                    float(outfit.get("score_low", 0)),
                    float(outfit.get("score_high", 0)),
                    str(outfit.get("confidence_level") or "Thấp"),
                    json.dumps(
                        outfit.get("items", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    str(outfit.get("explanation_llm") or outfit.get("explanation") or ""),
                    str(outfit.get("comment_source") or outfit.get("explanation_source") or "rules"),
                    created_at,
                    _composition_key(outfit.get("items", [])),
                ),
            )
            outfit["request_item_id"] = int(cursor.lastrowid)
        connection.execute(
            "UPDATE requests SET response_json = ? WHERE id = ?",
            (
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                request_id,
            ),
        )
        connection.execute(
            "UPDATE recommendation_requests SET response_json = ? WHERE id = ?",
            (
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                request_id,
            ),
        )
    return request_id


def get_recommendation_for_explanation(
    user_id: UUID, request_id: UUID, *, is_admin: bool = False
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load a saved recommendation only for its owner (or an administrator)."""
    initialize_storage()
    query = "SELECT input_json, response_json FROM recommendation_requests WHERE id = ?"
    parameters: tuple[Any, ...] = (str(request_id),)
    if not is_admin:
        query += " AND user_id = ?"
        parameters += (str(user_id),)
    with closing(_connect()) as connection:
        row = connection.execute(query, parameters).fetchone()
    if row is None:
        return None
    try:
        request_data = json.loads(row["input_json"])
        result = json.loads(row["response_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Dữ liệu gợi ý đã lưu không hợp lệ.") from exc
    if not isinstance(request_data, dict) or not isinstance(result, dict):
        raise ValueError("Dữ liệu gợi ý đã lưu không hợp lệ.")
    return request_data, result


def update_recommendation_explanations(
    user_id: UUID,
    request_id: UUID,
    outfits: list[dict[str, Any]],
    *,
    is_admin: bool = False,
) -> bool:
    """Persist only explanation fields while preserving scores, items, and history."""
    initialize_storage()
    query = "SELECT response_json FROM recommendation_requests WHERE id = ?"
    parameters: tuple[Any, ...] = (str(request_id),)
    if not is_admin:
        query += " AND user_id = ?"
        parameters += (str(user_id),)
    allowed_fields = (
        "explanation_llm",
        "explanation_structured",
        "explanation_source",
        "comment_source",
        "explanation_notice",
    )
    updates = {
        str(outfit["id"]): {key: outfit.get(key) for key in allowed_fields}
        for outfit in outfits
        if isinstance(outfit, dict) and outfit.get("id") is not None
    }
    with closing(_connect()) as connection, connection:
        row = connection.execute(query, parameters).fetchone()
        if row is None:
            return False
        try:
            result = json.loads(row["response_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Dữ liệu gợi ý đã lưu không hợp lệ.") from exc
        if not isinstance(result, dict) or not isinstance(result.get("outfits"), list):
            raise ValueError("Dữ liệu gợi ý đã lưu không hợp lệ.")
        for outfit in result["outfits"]:
            if not isinstance(outfit, dict):
                continue
            explanation = updates.get(str(outfit.get("id", "")))
            if explanation is not None:
                outfit.update(explanation)
        response_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        connection.execute(
            "UPDATE recommendation_requests SET response_json = ? WHERE id = ?",
            (response_json, str(request_id)),
        )
        connection.execute(
            "UPDATE requests SET response_json = ? WHERE id = ?",
            (response_json, str(request_id)),
        )
        for rank, outfit in enumerate(result["outfits"], start=1):
            if not isinstance(outfit, dict):
                continue
            item_query = """
                UPDATE recommendation_items
                SET explanation = ?, comment_source = ?
                WHERE request_id = ? AND rank = ?
            """
            item_parameters: tuple[Any, ...] = (
                str(outfit.get("explanation_llm") or outfit.get("explanation") or ""),
                str(outfit.get("comment_source") or outfit.get("explanation_source") or "rules"),
                str(request_id),
                rank,
            )
            if not is_admin:
                item_query += " AND user_id = ?"
                item_parameters += (str(user_id),)
            connection.execute(item_query, item_parameters)
    return True


def _composition_key(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    item_ids = sorted(
        {
            str(item["id"])
            for item in items
            if isinstance(item, dict) and item.get("id") is not None
        }
    )
    return json.dumps(item_ids, ensure_ascii=False, separators=(",", ":"))


def _item_is_accessible(
    connection: sqlite3.Connection,
    request_item_id: int,
    user_id: UUID,
    is_admin: bool,
) -> bool:
    query = "SELECT 1 FROM recommendation_items WHERE id = ?"
    parameters: tuple[Any, ...] = (request_item_id,)
    if not is_admin:
        query += " AND user_id = ?"
        parameters += (str(user_id),)
    return connection.execute(query, parameters).fetchone() is not None


def _view_payload(row: sqlite3.Row) -> dict[str, Any]:
    started = parse_utc_iso(row["started_at"])
    ended_text = row["ended_at"]
    ended = parse_utc_iso(ended_text) if ended_text else None
    return {
        "view_id": row["id"],
        "request_item_id": int(row["request_item_id"]),
        "started_at": row["started_at"],
        "last_seen_at": row["last_seen_at"],
        "ended_at": ended_text,
        "duration_seconds": int(row["duration_seconds"]),
        "duration_cap_seconds": MAX_VIEW_SECONDS,
        "ended": ended is not None,
        "elapsed_seconds": min(
            MAX_VIEW_SECONDS,
            int((ended - started).total_seconds())
            if ended
            else int((datetime.now(UTC) - started).total_seconds()),
        ),
    }


def start_outfit_view(
    user_id: UUID, request_item_id: int, *, is_admin: bool = False
) -> dict[str, Any] | None:
    initialize_storage()
    with closing(_connect()) as connection, connection:
        if not _item_is_accessible(connection, request_item_id, user_id, is_admin):
            return None
        active = connection.execute(
            """
            SELECT * FROM outfit_views
            WHERE user_id = ? AND request_item_id = ? AND ended_at IS NULL
            ORDER BY started_at DESC LIMIT 1
            """,
            (str(user_id), request_item_id),
        ).fetchone()
        if active is not None:
            return _view_payload(active)
        now = utc_now_iso()
        view_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO outfit_views
                (id, user_id, request_item_id, started_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (view_id, str(user_id), request_item_id, now, now),
        )
        row = connection.execute(
            "SELECT * FROM outfit_views WHERE id = ?", (view_id,)
        ).fetchone()
    if row is None:
        raise RuntimeError("Không thể đọc lại sự kiện xem vừa tạo.")
    return _view_payload(row)


def update_outfit_view(
    user_id: UUID,
    view_id: UUID,
    action: str,
    *,
    is_admin: bool = False,
) -> dict[str, Any] | None:
    initialize_storage()
    with closing(_connect()) as connection, connection:
        query = "SELECT * FROM outfit_views WHERE id = ?"
        parameters: tuple[Any, ...] = (str(view_id),)
        if not is_admin:
            query += " AND user_id = ?"
            parameters += (str(user_id),)
        row = connection.execute(query, parameters).fetchone()
        if row is None:
            return None
        if row["ended_at"] is None:
            started = parse_utc_iso(row["started_at"])
            now = datetime.now(UTC)
            capped_end = started + timedelta(seconds=MAX_VIEW_SECONDS)
            at_cap = now >= capped_end
            timestamp = capped_end if at_cap else now
            duration = min(
                MAX_VIEW_SECONDS,
                max(0, int((timestamp - started).total_seconds())),
            )
            ended_at = to_utc_iso(timestamp) if action == "end" or at_cap else None
            connection.execute(
                """
                UPDATE outfit_views
                SET last_seen_at = ?, ended_at = ?, duration_seconds = ?
                WHERE id = ?
                """,
                (
                    to_utc_iso(timestamp),
                    ended_at,
                    duration,
                    str(view_id),
                ),
            )
            row = connection.execute(
                "SELECT * FROM outfit_views WHERE id = ?", (str(view_id),)
            ).fetchone()
    if row is None:
        raise RuntimeError("Không thể đọc lại sự kiện xem vừa cập nhật.")
    return _view_payload(row)


def _require_accessible_item(
    connection: sqlite3.Connection,
    request_item_id: int,
    user_id: UUID,
    is_admin: bool,
) -> bool:
    return _item_is_accessible(connection, request_item_id, user_id, is_admin)


def finalize_outfit(
    user_id: UUID,
    request_item_id: int,
    note: str | None,
    *,
    is_admin: bool = False,
) -> dict[str, Any] | None:
    initialize_storage()
    with closing(_connect()) as connection, connection:
        if not _require_accessible_item(
            connection, request_item_id, user_id, is_admin
        ):
            return None
        finalized_at = utc_now_iso()
        connection.execute(
            """
            INSERT INTO finalized_outfits
                (id, user_id, request_item_id, finalized_at, note)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, request_item_id) DO UPDATE SET
                finalized_at = excluded.finalized_at,
                note = excluded.note
            """,
            (str(uuid4()), str(user_id), request_item_id, finalized_at, note),
        )
        row = connection.execute(
            """
            SELECT id, user_id, request_item_id, finalized_at, note
            FROM finalized_outfits
            WHERE user_id = ? AND request_item_id = ?
            """,
            (str(user_id), request_item_id),
        ).fetchone()
    if row is None:
        raise RuntimeError("Không thể đọc lại trạng thái chốt trang phục.")
    return dict(row)


def cancel_finalization(
    user_id: UUID, request_item_id: int, *, is_admin: bool = False
) -> bool:
    initialize_storage()
    with closing(_connect()) as connection, connection:
        if not _require_accessible_item(
            connection, request_item_id, user_id, is_admin
        ):
            return False
        connection.execute(
            "DELETE FROM finalized_outfits WHERE user_id = ? AND request_item_id = ?",
            (str(user_id), request_item_id),
        )
        return True


def rate_outfit(
    user_id: UUID,
    request_item_id: int,
    value: int,
    reason: str | None,
    *,
    is_admin: bool = False,
) -> dict[str, Any] | None:
    initialize_storage()
    with closing(_connect()) as connection, connection:
        if not _require_accessible_item(
            connection, request_item_id, user_id, is_admin
        ):
            return None
        created_at = utc_now_iso()
        connection.execute(
            """
            INSERT INTO outfit_ratings
                (id, user_id, request_item_id, value, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, request_item_id) DO UPDATE SET
                value = excluded.value,
                reason = excluded.reason,
                created_at = excluded.created_at
            """,
            (
                str(uuid4()),
                str(user_id),
                request_item_id,
                value,
                reason,
                created_at,
            ),
        )
        row = connection.execute(
            """
            SELECT id, user_id, request_item_id, value, reason, created_at
            FROM outfit_ratings
            WHERE user_id = ? AND request_item_id = ?
            """,
            (str(user_id), request_item_id),
        ).fetchone()
    if row is None:
        raise RuntimeError("Không thể đọc lại đánh giá trang phục.")
    return dict(row)


def get_history_page(
    *,
    user_id: UUID,
    is_admin: bool,
    history_type: str,
    range_name: str,
    from_date: str | None,
    to_date: str | None,
    page: int,
    page_size: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    start_at, end_at = history_bounds(
        range_name, from_date=from_date, to_date=to_date, now=now
    )
    table, timestamp_column = {
        "viewed": ("outfit_views", "started_at"),
        "finalized": ("finalized_outfits", "finalized_at"),
        "feedback": ("outfit_ratings", "created_at"),
    }[history_type]
    alias = "e"
    where: list[str] = []
    parameters: list[Any] = []
    if not is_admin:
        where.append(f"{alias}.user_id = ?")
        parameters.append(str(user_id))
    if start_at is not None:
        where.append(f"{alias}.{timestamp_column} >= ?")
        parameters.append(start_at)
    if end_at is not None:
        where.append(f"{alias}.{timestamp_column} < ?")
        parameters.append(end_at)
    where_sql = " WHERE " + " AND ".join(where) if where else ""
    joins = (
        f"""
        FROM {table} e
        JOIN recommendation_items ri ON ri.id = e.request_item_id
        JOIN recommendation_requests rr ON rr.id = ri.request_id
        LEFT JOIN finalized_outfits f_status
            ON f_status.user_id = e.user_id AND f_status.request_item_id = ri.id
        LEFT JOIN outfit_ratings r_status
            ON r_status.user_id = e.user_id AND r_status.request_item_id = ri.id
        """
    )
    initialize_storage()
    offset = (page - 1) * page_size
    with closing(_connect()) as connection:
        total = int(
            connection.execute(
                f"SELECT COUNT(*) {joins}{where_sql}", parameters
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"""
            SELECT e.id AS event_id,
                   e.request_item_id,
                   e.{timestamp_column} AS occurred_at,
                   ri.request_id, ri.outfit_id, ri.rank, ri.name,
                   ri.score, ri.score_low, ri.score_high,
                   ri.confidence_level, ri.items_json, rr.input_json,
                   rr.response_json,
                   f_status.finalized_at,
                   r_status.value AS feedback_value,
                   r_status.reason AS feedback_reason
            {joins}{where_sql}
            ORDER BY e.{timestamp_column} DESC, e.id DESC
            LIMIT ? OFFSET ?
            """,
            (*parameters, page_size, offset),
        ).fetchall()
    items = []
    for row in rows:
        request_input = json.loads(row["input_json"])
        response_payload = json.loads(row["response_json"] or "{}")
        original_outfit = next(
            (
                outfit
                for outfit in response_payload.get("outfits", [])
                if isinstance(outfit, dict)
                and outfit.get("id") == row["outfit_id"]
            ),
            None,
        )
        items.append(
            {
                "event_id": row["event_id"],
                "type": history_type,
                "request_item_id": int(row["request_item_id"]),
                "request_id": row["request_id"],
                "name": row["name"],
                "activity": request_input.get("activity"),
                "temperature": request_input.get("temperature"),
                "score": float(row["score"]),
                "score_low": float(row["score_low"]),
                "score_high": float(row["score_high"]),
                "confidence_level": row["confidence_level"],
                "items": json.loads(row["items_json"]),
                "outfit": original_outfit,
                "rank": int(row["rank"]),
                "created_at": row["occurred_at"],
                "finalized_at": row["finalized_at"],
                "feedback": (
                    None
                    if row["feedback_value"] is None
                    else {
                        "value": int(row["feedback_value"]),
                        "reason": row["feedback_reason"],
                    }
                ),
            }
        )
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "range": range_name,
        "timezone": "Asia/Ho_Chi_Minh",
    }


def admin_stats() -> dict[str, Any]:
    initialize_storage()
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM users WHERE role = 'user') AS total_users,
                (SELECT COUNT(*) FROM recommendation_requests) AS total_recommendations,
                (SELECT COUNT(*) FROM outfit_views) AS total_views,
                (SELECT COALESCE(SUM(duration_seconds), 0) FROM outfit_views)
                    AS total_view_seconds,
                (SELECT COALESCE(AVG(duration_seconds), 0) FROM outfit_views)
                    AS average_view_seconds,
                (SELECT COUNT(*) FROM outfit_ratings WHERE value = 1) AS likes,
                (SELECT COUNT(*) FROM outfit_ratings WHERE value = -1) AS dislikes,
                (SELECT COUNT(*) FROM finalized_outfits) AS finalized_count
            """
        ).fetchone()
    return {
        key: (round(float(row[key]), 2) if key == "average_view_seconds" else int(row[key]))
        for key in row.keys()
    }


def admin_outfits(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    initialize_storage()
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            WITH item_metrics AS (
                SELECT ri.id, ri.outfit_id, ri.name, ri.composition_key,
                       ri.items_json,
                       (SELECT COUNT(*) FROM outfit_views v
                        WHERE v.request_item_id = ri.id) AS view_count,
                       (SELECT COALESCE(SUM(v.duration_seconds), 0) FROM outfit_views v
                        WHERE v.request_item_id = ri.id) AS view_seconds,
                       (SELECT COUNT(*) FROM finalized_outfits f
                        WHERE f.request_item_id = ri.id) AS finalize_count,
                       (SELECT COUNT(*) FROM outfit_ratings r
                        WHERE r.request_item_id = ri.id AND r.value = 1) AS likes,
                       (SELECT COUNT(*) FROM outfit_ratings r
                        WHERE r.request_item_id = ri.id AND r.value = -1) AS dislikes
                FROM recommendation_items ri
            )
            SELECT MIN(outfit_id) AS outfit_id, composition_key, name,
                   MIN(items_json) AS items_json,
                   COUNT(*) AS recommendation_count,
                   SUM(view_count) AS view_count,
                   SUM(view_seconds) AS view_seconds,
                   SUM(finalize_count) AS finalize_count,
                   SUM(likes) AS likes,
                   SUM(dislikes) AS dislikes
            FROM item_metrics
            GROUP BY composition_key, name
            ORDER BY view_count DESC, finalize_count DESC, likes DESC,
                     recommendation_count DESC, name
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    results = []
    for row in rows:
        outfit = dict(row)
        try:
            saved_items = json.loads(outfit.pop("items_json"))
        except (TypeError, json.JSONDecodeError):
            saved_items = []
        outfit["items"] = [
            str(item["name"])
            for item in saved_items
            if isinstance(item, dict) and item.get("name")
        ]
        results.append(outfit)
    return results


def admin_users(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    initialize_storage()
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT u.*, COUNT(DISTINCT rr.id) AS recommendation_count,
                   MAX(activity.occurred_at) AS last_activity_at
            FROM users u
            LEFT JOIN recommendation_requests rr ON rr.user_id = u.id
            LEFT JOIN (
                SELECT user_id, started_at AS occurred_at FROM outfit_views
                UNION ALL
                SELECT user_id, finalized_at FROM finalized_outfits
                UNION ALL
                SELECT user_id, created_at FROM outfit_ratings
            ) activity ON activity.user_id = u.id
            GROUP BY u.id
            ORDER BY u.created_at DESC, u.id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [
        {
            **_user_public_fields(row),
            "recommendation_count": int(row["recommendation_count"]),
            "last_activity_at": row["last_activity_at"],
        }
        for row in rows
    ]


def get_history(user_id: UUID, limit: int) -> list[dict[str, Any]]:
    initialize_storage()
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, input_json, response_json
            FROM requests
            WHERE user_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (str(user_id), limit),
        ).fetchall()
    return [
        {
            "request_id": row["id"],
            "created_at": row["created_at"],
            "input": json.loads(row["input_json"]),
            "result": json.loads(row["response_json"]),
        }
        for row in rows
    ]


def add_feedback(
    *,
    user_id: UUID,
    request_id: UUID,
    outfit_id: int,
    action: str,
    note: str | None,
) -> str:
    initialize_storage()
    with closing(_connect()) as connection, connection:
        row = connection.execute(
            "SELECT response_json FROM requests WHERE id = ? AND user_id = ?",
            (str(request_id), str(user_id)),
        ).fetchone()
        if row is None:
            return "not_found"
        response = json.loads(row["response_json"])
        selected = next((outfit for outfit in response.get("outfits", [])
                         if int(outfit.get("id", -1)) == outfit_id), None)
        if selected is None:
            return "outfit_not_found"
        feedback_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO feedback
                (id, request_id, user_id, outfit_id, action, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                str(request_id),
                str(user_id),
                outfit_id,
                action,
                note,
                utc_now_iso(),
            ),
        )
        if action in {"like", "worn"}:
            styles = [str(item.get("style", "")).strip()
                      for item in selected.get("items", []) if item.get("style")]
            if styles:
                connection.execute(
                    "UPDATE users SET preferred_style = ? WHERE id = ?",
                    (styles[0], str(user_id)),
                )
    return feedback_id
