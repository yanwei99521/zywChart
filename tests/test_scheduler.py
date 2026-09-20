import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from chart_service.scheduler import (
    SHANGHAI_TZ,
    RefreshState,
    _run_refresh,
    next_scheduled_time,
)


def local_time(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 20, hour, minute, tzinfo=ZoneInfo(SHANGHAI_TZ))


def test_before_morning_run_schedules_0700() -> None:
    assert next_scheduled_time(local_time(6, 59)) == local_time(7)


def test_after_morning_run_schedules_1800() -> None:
    assert next_scheduled_time(local_time(7, 1)) == local_time(18)


def test_exact_schedule_time_moves_to_next_slot() -> None:
    assert next_scheduled_time(local_time(7)) == local_time(18)


def test_after_evening_run_schedules_next_day_morning() -> None:
    assert next_scheduled_time(local_time(18, 1)) == datetime(
        2026, 9, 21, 7, 0, tzinfo=ZoneInfo(SHANGHAI_TZ)
    )


def test_utc_input_is_converted_to_shanghai() -> None:
    utc_now = datetime(2026, 9, 19, 22, 0, tzinfo=ZoneInfo("UTC"))

    assert next_scheduled_time(utc_now) == local_time(7)


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
