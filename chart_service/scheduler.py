"""A small timezone-aware scheduler for twice-daily chart refreshes."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

SHANGHAI_TZ = "Asia/Shanghai"
RUN_TIMES = (time(7, 0), time(18, 0))
LOGGER = logging.getLogger(__name__)


@dataclass
class RefreshState:
    """Observable scheduler state returned by the public status endpoint."""

    last_attempt_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_error: Optional[str] = None
    next_run_at: Optional[str] = None

    def as_dict(self) -> dict[str, Optional[str]]:
        return asdict(self)


def next_scheduled_time(now: datetime) -> datetime:
    """Return the next 07:00 or 18:00 occurrence in Asia/Shanghai."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(ZoneInfo(SHANGHAI_TZ))

    for run_time in RUN_TIMES:
        candidate = datetime.combine(local_now.date(), run_time).replace(
            tzinfo=ZoneInfo(SHANGHAI_TZ)
        )
        if candidate > local_now:
            return candidate

    tomorrow = local_now.date() + timedelta(days=1)
    return datetime.combine(tomorrow, RUN_TIMES[0]).replace(
        tzinfo=ZoneInfo(SHANGHAI_TZ)
    )


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
    """Run forever at the two configured Beijing-time slots."""
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
