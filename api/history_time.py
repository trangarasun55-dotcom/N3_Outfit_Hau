"""UTC persistence and Vietnam-local history range boundaries."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

HISTORY_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def to_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Thời điểm cần có múi giờ.")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def parse_utc_iso(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def history_bounds(
    range_name: str,
    *,
    from_date: str | None = None,
    to_date: str | None = None,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Return a half-open UTC interval for a local Vietnam calendar range."""
    if range_name == "all":
        if from_date is not None or to_date is not None:
            raise ValueError("Không truyền from/to khi range=all.")
        return None, None

    local_now = (now or datetime.now(UTC)).astimezone(HISTORY_TIMEZONE)
    if range_name == "day":
        start_day = local_now.date()
        end_day = start_day + timedelta(days=1)
    elif range_name == "week":
        start_day = local_now.date() - timedelta(days=local_now.weekday())
        end_day = start_day + timedelta(days=7)
    elif range_name == "month":
        start_day = local_now.date().replace(day=1)
        if start_day.month == 12:
            end_day = date(start_day.year + 1, 1, 1)
        else:
            end_day = date(start_day.year, start_day.month + 1, 1)
    elif range_name == "from-to":
        if from_date is None or to_date is None:
            raise ValueError("range=from-to cần cả from và to theo định dạng YYYY-MM-DD.")
        try:
            start_day = date.fromisoformat(from_date)
            inclusive_end = date.fromisoformat(to_date)
        except ValueError as exc:
            raise ValueError("from/to phải là ngày hợp lệ theo định dạng YYYY-MM-DD.") from exc
        if inclusive_end < start_day:
            raise ValueError("to phải bằng hoặc sau from.")
        end_day = inclusive_end + timedelta(days=1)
    else:
        raise ValueError("range chỉ nhận day, week, month, all hoặc from-to.")

    start_local = datetime.combine(start_day, time.min, tzinfo=HISTORY_TIMEZONE)
    end_local = datetime.combine(end_day, time.min, tzinfo=HISTORY_TIMEZONE)
    return to_utc_iso(start_local), to_utc_iso(end_local)
