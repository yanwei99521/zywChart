"""A small timezone-aware scheduler for weekly chart refreshes."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

SHANGHAI_TZ = "Asia/Shanghai"
LOGGER = logging.getLogger(__name__)

# Weekly refresh: every Monday at 07:00 Shanghai time by default. Both the
# weekday and the time of day can be overridden without touching the code via
# CHART_REFRESH_WEEKDAY (0 = Monday ... 6 = Sunday) and CHART_REFRESH_TIME
# ("HH:MM"), e.g. to refresh every Friday at 18:30.


def _weekday_from_env() -> int:
    raw = os.getenv("CHART_REFRESH_WEEKDAY", "0").strip()
    try:
        value = int(raw)
    except ValueError:
        return 0
    return value if 0 <= value <= 6 else 0


def _run_time_from_env() -> time:
    raw = os.getenv("CHART_REFRESH_TIME", "07:00").strip()
    try:
        hour_text, minute_text = raw.split(":", 1)
        return time(int(hour_text) % 24, int(minute_text) % 60)
    except (ValueError, AttributeError):
        return time(7, 0)


REFRESH_WEEKDAY = _weekday_from_env()
RUN_TIMES = (_run_time_from_env(),)
WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def schedule_description() -> str:
    """Human readable description of the current schedule, e.g. 每周一 07:00."""
    run_time = RUN_TIMES[0]
    return (
        f"每{WEEKDAY_NAMES[REFRESH_WEEKDAY]} "
        f"{run_time.hour:02d}:{run_time.minute:02d}"
    )


@dataclass
class RefreshState:
    """Observable scheduler state returned by the public status endpoint."""

    last_attempt_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_error: Optional[str] = None
    gold_stale: Optional[bool] = None
    gold_source: Optional[str] = None
    next_run_at: Optional[str] = None

    def as_dict(self) -> dict[str, Optional[str]]:
        return asdict(self)


def next_scheduled_time(now: datetime) -> datetime:
    """Return the next weekly slot (Monday 07:00 Asia/Shanghai by default)."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(ZoneInfo(SHANGHAI_TZ))

    today_slot = datetime.combine(local_now.date(), RUN_TIMES[0]).replace(
        tzinfo=ZoneInfo(SHANGHAI_TZ)
    )
    # Walk forward one day at a time until we land on the configured weekday.
    # A candidate exactly equal to "now" is skipped, so a refresh that fires on
    # the slot immediately re-arms for the same weekday next week.
    for offset in range(8):
        candidate = today_slot + timedelta(days=offset)
        if candidate.weekday() == REFRESH_WEEKDAY and candidate > local_now:
            return candidate

    return today_slot + timedelta(days=7)


async def _run_refresh(
    refresh: Callable[[], dict[str, object]], state: RefreshState
) -> None:
    attempted_at = datetime.now(ZoneInfo(SHANGHAI_TZ)).isoformat()
    state.last_attempt_at = attempted_at
    try:
        await asyncio.to_thread(refresh)
    except asyncio.CancelledError:
        raise
    except Exception as error:  # The scheduler must survive upstream failures.
        state.last_error = f"{type(error).__name__}: {error}"
        LOGGER.exception("Scheduled chart refresh failed")
    else:
        state.last_success_at = datetime.now(ZoneInfo(SHANGHAI_TZ)).isoformat()
        state.last_error = None
        LOGGER.info("Scheduled chart refresh completed")


async def run_scheduler(
    refresh: Callable[[], dict[str, object]],
    state: RefreshState,
    *,
    run_immediately: bool = False,
) -> None:
    """Run forever at the configured weekly Shanghai-time slot."""
    if run_immediately:
        await _run_refresh(refresh, state)

    while True:
        now = datetime.now(timezone.utc)
        next_run = next_scheduled_time(now)
        state.next_run_at = next_run.isoformat()
        delay = max((next_run - now.astimezone(next_run.tzinfo)).total_seconds(), 0)
        await asyncio.sleep(delay)
        state.next_run_at = None
        await _run_refresh(refresh, state)
