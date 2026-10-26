"""Password hashing and signed-cookie authentication for the REST API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request

try:
    from . import storage
except ImportError:  # `uvicorn main:app` from api/
    import storage

LOGGER = logging.getLogger(__name__)
COOKIE_NAME = "outfit_session"
PASSWORD_HASH_ITERATIONS = 600_000
_PROCESS_SECRET = secrets.token_bytes(32)
_PERSISTED_SECRET_CACHE: list[bytes] = []
_EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s.]+(?:\.[^@\s.]+)+$")


def normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if len(normalized) > 254 or not _EMAIL_PATTERN.fullmatch(normalized):
        raise ValueError("Địa chỉ email không hợp lệ.")
    return normalized


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS
    )
    return "$".join(
        (
            "pbkdf2_sha256",
            str(PASSWORD_HASH_ITERATIONS),
            base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
            base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
        )
    )


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, iteration_text, salt_text, expected_text = encoded_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iteration_text)
        if not 100_000 <= iterations <= 2_000_000:
            return False
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(
            expected_text + "=" * (-len(expected_text) % 4)
        )
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
    except (ValueError, TypeError, UnicodeEncodeError):
        return False
    return hmac.compare_digest(actual, expected)


def _session_secret() -> bytes:
    """SECRET_KEY từ .env; nếu trống thì dùng khóa lưu ở data/.secret_key (tạo một lần)."""
    configured = os.environ.get("SECRET_KEY", "").strip()
    if configured:
        return configured.encode("utf-8")
    if _PERSISTED_SECRET_CACHE:
        return _PERSISTED_SECRET_CACHE[0]
    key_file = Path(__file__).resolve().parents[1] / "data" / ".secret_key"
    try:
        if key_file.is_file() and key_file.read_text(encoding="utf-8").strip():
            secret = key_file.read_text(encoding="utf-8").strip().encode("utf-8")
        else:
            key_file.parent.mkdir(parents=True, exist_ok=True)
            generated = secrets.token_hex(32)
            key_file.write_text(generated, encoding="utf-8")
            secret = generated.encode("utf-8")
    except OSError:
        LOGGER.warning("Không ghi được data/.secret_key; dùng khóa tạm thời trong tiến trình.")
        secret = _PROCESS_SECRET
    _PERSISTED_SECRET_CACHE.append(secret)
    return secret


def _session_max_age() -> int:
    raw_value = os.environ.get("SESSION_MAX_AGE_SECONDS", "43200")
    try:
        max_age = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("SESSION_MAX_AGE_SECONDS phải là số nguyên.") from exc
    if not 60 <= max_age <= 2_592_000:
        raise RuntimeError("SESSION_MAX_AGE_SECONDS phải nằm trong 60..2592000.")
    return max_age


def issue_session(user_id: str) -> tuple[str, int]:
    max_age = _session_max_age()
    payload = json.dumps(
        {"sub": user_id, "exp": int(time.time()) + max_age},
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_payload = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    signature = hmac.new(
        _session_secret(),
        b"outfitai-session-v1." + encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{encoded_payload}.{encoded_signature}", max_age


def _read_session(token: str) -> str | None:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        expected_signature = hmac.new(
            _session_secret(),
            b"outfitai-session-v1." + encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual_signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        if not hmac.compare_digest(actual_signature, expected_signature):
            return None
        payload_bytes = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        )
        payload = json.loads(payload_bytes)
        user_id = payload.get("sub")
        expires_at = payload.get("exp")
        if not isinstance(user_id, str) or not isinstance(expires_at, int):
            return None
        if expires_at <= int(time.time()):
            return None
        return user_id
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def get_current_user(request: Request) -> dict[str, Any]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Bạn cần đăng nhập.")
    user_id = _read_session(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn.")
    user = storage.get_user_by_id(user_id)
    if user is None or not user["is_active"]:
        raise HTTPException(status_code=401, detail="Tài khoản không còn hoạt động.")
    return user


def require_admin(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Bạn không có quyền quản trị.")
    return user


def seed_admin_from_environment() -> dict[str, Any] | None:
    email_value = os.environ.get("ADMIN_EMAIL", "").strip()
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not email_value and not password:
        LOGGER.warning("Chưa cấu hình ADMIN_EMAIL/ADMIN_PASSWORD; bỏ qua seed admin.")
        return None
    if not email_value or not password:
        raise RuntimeError("Cần cấu hình đồng thời ADMIN_EMAIL và ADMIN_PASSWORD.")
    try:
        email = normalize_email(email_value)
    except ValueError as exc:
        raise RuntimeError("ADMIN_EMAIL không hợp lệ.") from exc
    if len(password) < 12 or len(password) > 128:
        raise RuntimeError("ADMIN_PASSWORD phải có độ dài từ 12 đến 128 ký tự.")
    return storage.seed_admin_user(email, hash_password(password))
