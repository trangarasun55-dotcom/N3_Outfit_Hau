"""Pydantic request/response schemas; lựa chọn được kiểm tra từ metadata."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

try:
    from .config import ACTIVITIES, COLOR_CHOICES, DEFAULT_FORMALITY, GENDERS, MODEL_METADATA, STYLES
except ImportError:  # chạy `uvicorn main:app` từ thư mục api
    from config import ACTIVITIES, COLOR_CHOICES, DEFAULT_FORMALITY, GENDERS, MODEL_METADATA, STYLES


def _choice_validator(choices: tuple[str, ...], field_name: str):
    def validate(value: str) -> str:
        if value not in choices:
            raise ValueError(f"{field_name} không nằm trong danh sách cho phép.")
        return value

    return validate


Activity: TypeAlias = Annotated[
    str,
    Field(json_schema_extra={"enum": list(ACTIVITIES)}),
    AfterValidator(_choice_validator(ACTIVITIES, "Hoàn cảnh")),
]
Style: TypeAlias = Annotated[
    str,
    Field(json_schema_extra={"enum": list(STYLES)}),
    AfterValidator(_choice_validator(STYLES, "Phong cách")),
]
Gender: TypeAlias = Annotated[
    str,
    Field(json_schema_extra={"enum": list(GENDERS)}),
    AfterValidator(_choice_validator(GENDERS, "Giới tính")),
]
Color: TypeAlias = Annotated[
    str,
    Field(json_schema_extra={"enum": list(COLOR_CHOICES)}),
    AfterValidator(_choice_validator(COLOR_CHOICES, "Màu sắc")),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictRequest(StrictModel):
    activity: Activity = Field(alias="Hoàn cảnh")
    style: Style = Field(alias="Phong cách")
    tmin: float = Field(
        alias="Tmin (°C)",
        ge=MODEL_METADATA["feature_ranges"]["Tmin (°C)"]["min"],
        le=MODEL_METADATA["feature_ranges"]["Tmin (°C)"]["max"],
    )
    tmax: float = Field(
        alias="Tmax (°C)",
        ge=MODEL_METADATA["feature_ranges"]["Tmax (°C)"]["min"],
        le=MODEL_METADATA["feature_ranges"]["Tmax (°C)"]["max"],
    )
    temperature: float = Field(alias="Nhiệt độ ngày (°C)", ge=-10, le=50)
    warmth: float = Field(
        alias="Giữ ấm",
        ge=MODEL_METADATA["feature_ranges"]["Giữ ấm"]["min"],
        le=MODEL_METADATA["feature_ranges"]["Giữ ấm"]["max"],
    )
    breathability: float = Field(
        alias="Thoáng khí",
        ge=MODEL_METADATA["feature_ranges"]["Thoáng khí"]["min"],
        le=MODEL_METADATA["feature_ranges"]["Thoáng khí"]["max"],
    )
    mobility: float = Field(
        alias="Vận động",
        ge=MODEL_METADATA["feature_ranges"]["Vận động"]["min"],
        le=MODEL_METADATA["feature_ranges"]["Vận động"]["max"],
    )
    neatness: float = Field(
        alias="Chỉn chu",
        ge=MODEL_METADATA["feature_ranges"]["Chỉn chu"]["min"],
        le=MODEL_METADATA["feature_ranges"]["Chỉn chu"]["max"],
    )

    def as_features(self) -> dict[str, object]:
        return {
            "Hoàn cảnh": self.activity,
            "Phong cách": self.style,
            "Tmin (°C)": self.tmin,
            "Tmax (°C)": self.tmax,
            "Nhiệt độ ngày (°C)": self.temperature,
            "Giữ ấm": self.warmth,
            "Thoáng khí": self.breathability,
            "Vận động": self.mobility,
            "Chỉn chu": self.neatness,
        }


class RecommendRequest(StrictModel):
    gender: Gender
    activity: Activity
    temperature: float = Field(ge=-10, le=50)
    styles: list[Style] = Field(default_factory=list, max_length=len(STYLES))
    colors: list[Color] = Field(default_factory=list, max_length=len(COLOR_CHOICES))
    formality: int = Field(default=DEFAULT_FORMALITY, ge=0, le=100)
    top_n: int = Field(default=3, ge=1, le=3)
    temperature_source: Literal["live", "manual"] = "live"
    rain_probability: int | None = Field(default=None, ge=0, le=100)
    schedule_summary: str | None = Field(default=None, max_length=300)
    notes: str | None = Field(default=None, max_length=300)

    @field_validator("notes")
    @classmethod
    def trim_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if len(normalized) > 300:
            raise ValueError("notes tối đa 300 ký tự.")
        return normalized or None

    @field_validator("schedule_summary")
    @classmethod
    def trim_schedule_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if len(normalized) > 300:
            raise ValueError("schedule_summary tối đa 300 ký tự.")
        return normalized or None


class FeedbackRequest(StrictModel):
    request_id: UUID
    outfit_id: int = Field(ge=1, le=5)
    action: Literal["worn", "like", "dislike"]
    note: str | None = Field(default=None, max_length=300)

    @field_validator("note")
    @classmethod
    def trim_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None


class ViewEventRequest(StrictModel):
    action: Literal["start", "heartbeat", "end"]
    request_item_id: int | None = Field(default=None, ge=1)
    view_id: UUID | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "ViewEventRequest":
        if self.action == "start" and self.request_item_id is None:
            raise ValueError("request_item_id là bắt buộc khi bắt đầu xem.")
        if self.action != "start" and self.view_id is None:
            raise ValueError("view_id là bắt buộc khi heartbeat hoặc kết thúc.")
        return self


class FinalizationRequest(StrictModel):
    note: str | None = Field(default=None, max_length=500)

    @field_validator("note")
    @classmethod
    def trim_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.split()) or None


class OutfitRatingRequest(StrictModel):
    value: Literal[1, -1]
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("reason")
    @classmethod
    def trim_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.split()) or None


def build_predict_request_from_frame():
    """Giữ tham chiếu metadata ở đây để dễ kiểm tra khi build OpenAPI."""
    return MODEL_METADATA["features"]
