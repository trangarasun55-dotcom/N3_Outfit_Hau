"""Validated request bodies for account authentication and profiles."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s.]+(?:\.[^@\s.]+)+$")


class StrictAuthModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterRequest(StrictAuthModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)
    display_name: str | None = Field(default=None, min_length=1, max_length=60)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("Địa chỉ email không hợp lệ.")
        return normalized

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Tên hiển thị không được để trống.")
        return normalized


class LoginRequest(StrictAuthModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("Địa chỉ email không hợp lệ.")
        return normalized


class PasswordChangeRequest(StrictAuthModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)

    @model_validator(mode="after")
    def password_must_change(self) -> "PasswordChangeRequest":
        if self.current_password == self.new_password:
            raise ValueError("Mật khẩu mới phải khác mật khẩu hiện tại.")
        return self


class ProfilePatch(StrictAuthModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=60)
    gender: Literal["Nam", "Nữ"] | None = None
    preferred_style: str | None = Field(default=None, max_length=40)
    default_context: str | None = Field(default=None, max_length=60)

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Tên hiển thị không được để trống.")
        return normalized

    @field_validator("preferred_style", "default_context")
    @classmethod
    def trim_optional_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @model_validator(mode="after")
    def require_patch_field(self) -> "ProfilePatch":
        if not self.model_fields_set:
            raise ValueError("Cần cung cấp ít nhất một trường để cập nhật.")
        return self
