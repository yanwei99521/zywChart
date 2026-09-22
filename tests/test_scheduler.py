import asyncio
from datetime import datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

from chart_service.scheduler import (
    REFRESH_WEEKDAY,
    RUN_TIMES,
    SHANGHAI_TZ,
    RefreshState,
    _run_refresh,
    next_scheduled_time,
    schedule_description,
)


def local_time(day: int, hour: int, minute: int = 0) -> datetime:
    # 2026-09-20 is a Sunday, 2026-09-21 a Monday.
    return datetime(2026, 9, day, hour, minute, tzinfo=ZoneInfo(SHANGHAI_TZ))


def test_weekly_schedule_defaults_to_monday_morning() -> None:
    assert REFRESH_WEEKDAY == 0
    assert RUN_TIMES == (dt_time(7, 0),)
    assert schedule_description() == "每周一 07:00"


def test_sunday_evening_schedules_next_monday() -> None:
    assert next_scheduled_time(local_time(20, 23, 30)) == local_time(21, 7)


def test_before_the_slot_schedules_same_monday() -> None:
    assert next_scheduled_time(local_time(21, 6, 59)) == local_time(21, 7)


def test_after_the_slot_schedules_next_monday() -> None:
    assert next_scheduled_time(local_time(21, 7, 1)) == local_time(28, 7)


def test_exact_schedule_time_moves_to_next_week() -> None:
    assert next_scheduled_time(local_time(21, 7)) == local_time(28, 7)


def test_midweek_schedules_next_monday() -> None:
    assert next_scheduled_time(local_time(23, 15, 30)) == local_time(28, 7)


def test_utc_input_is_converted_to_shanghai() -> None:
    # 2026-09-20 22:00Z == 2026-09-21 06:00 Shanghai (Monday, before the slot).
    utc_now = datetime(2026, 9, 20, 22, 0, tzinfo=ZoneInfo("UTC"))

    assert next_scheduled_time(utc_now) == local_time(21, 7)


def test_refresh_state_records_success() -> None:
    state = RefreshState()

    asyncio.run(_run_refresh(lambda: {"as_of": "2026-09-20"}, state))

    assert state.last_attempt_at is not None
    assert state.last_success_at is not None
    assert state.last_error is None
    assert state.as_dict()["last_success_at"] == state.last_success_at


def test_refresh_state_records_failure_without_raising() -> None:
    state = RefreshState()

    def fail() -> dict[str, object]:
        raise RuntimeError("upstream unavailable")

    asyncio.run(_run_refresh(fail, state))

    assert state.last_success_at is None
    assert state.last_error == "RuntimeError: upstream unavailable"
