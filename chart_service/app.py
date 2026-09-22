"""FastAPI application exposing the generated chart without authentication."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys

LOGGER = logging.getLogger(__name__)
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from chart_service.dataset import build_dataset
from chart_service.refresh import gold_source, refresh_chart, was_gold_stale
from chart_service.scheduler import (
    REFRESH_WEEKDAY,
    RUN_TIMES,
    SHANGHAI_TZ,
    RefreshState,
    run_scheduler,
    schedule_description,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATED_CHART_RE = re.compile(r"bitcoin-power-law-\d{4}-\d{2}-\d{2}")
CACHE_HEADERS = {"Cache-Control": "public, max-age=300"}


class ChartRepository:
    """Resolve the latest complete chart files in an output directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def latest_chart(self, suffix: str) -> Optional[Path]:
        candidates = [
            path
            for path in self.root.glob(f"bitcoin-power-law-*{suffix}")
            if DATED_CHART_RE.fullmatch(path.stem)
        ]
        return max(candidates, key=lambda path: path.name) if candidates else None

    @property
    def summary(self) -> Path:
        return self.root / "model-summary.json"

    @property
    def data(self) -> Path:
        return self.root / "data" / "weekly_model_data.csv"


def _needs_refresh(repository: "ChartRepository", max_age_hours: int = 6) -> bool:
    """Refresh on startup when no chart exists or the cached one is stale.

    The scheduler only ticks at 07:00/18:00 Shanghai time, so a service that
    starts between slots (or after a reboot) would otherwise serve an
    arbitrarily old PNG until the next slot. This forces a fresh pull on boot
    whenever the published chart is older than ``max_age_hours``.
    """
    latest = repository.latest_chart(".png")
    if latest is None:
        return True
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)
    return (datetime.now() - mtime) > timedelta(hours=max_age_hours)


def unavailable(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": {"code": "chart_not_ready", "message": message}},
    )


def create_app(
    output_dir: Path = PROJECT_ROOT, *, scheduler_enabled: bool = True
) -> FastAPI:
    repository = ChartRepository(output_dir.resolve())
    refresh_state = RefreshState()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        task: Optional[asyncio.Task[None]] = None
        if scheduler_enabled:

            def _scheduled_refresh() -> dict[str, object]:
                result = refresh_chart(repository.root)
                refresh_state.gold_stale = was_gold_stale()
                refresh_state.gold_source = gold_source()
                return result

            task = asyncio.create_task(
                run_scheduler(
                    _scheduled_refresh,
                    refresh_state,
                    run_immediately=_needs_refresh(repository),
                )
            )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    application = FastAPI(
        title="Bitcoin Power Law Chart API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "OPTIONS", "POST"],
        allow_headers=["*"],
    )

    def chart_response(suffix: str, media_type: str) -> Response:
        path = repository.latest_chart(suffix)
        if path is None:
            return unavailable("Chart has not been generated yet")
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
            content_disposition_type="inline",
            headers=CACHE_HEADERS,
        )

    @application.get("/", include_in_schema=False)
    def index() -> dict[str, object]:
        return {
            "name": "Bitcoin Power Law Chart API",
            "documentation": "/docs",
            "admin": "/admin",
            "latest_png": "/api/v1/charts/bitcoin-power-law/latest.png",
        }

    @application.get("/admin", include_in_schema=False)
    def admin_page() -> HTMLResponse:
        """Lightweight operations console: preview the current chart and trigger
        an on-demand refresh from the browser without touching the API docs."""
        html_path = Path(__file__).resolve().parent / "admin_page.html"
        html = html_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html)

    @application.get("/api/v1/health")
    def health() -> dict[str, object]:
        latest_png = repository.latest_chart(".png")
        chart_updated_at = None
        if latest_png is not None:
            chart_updated_at = datetime.fromtimestamp(
                latest_png.stat().st_mtime
            ).isoformat(timespec="seconds")
        return {
            "data": {
                "status": "ok",
                "chart_ready": latest_png is not None,
                "chart_updated_at": chart_updated_at,
                "scheduler": refresh_state.as_dict(),
                "schedule_timezone": SHANGHAI_TZ,
                "schedule_times": [f"{t.hour:02d}:{t.minute:02d}" for t in RUN_TIMES],
                "schedule_weekday": REFRESH_WEEKDAY,
                "schedule_description": schedule_description(),
            }
        }

    @application.get("/api/v1/charts/bitcoin-power-law/latest")
    @application.get("/api/v1/charts/bitcoin-power-law/latest.png")
    def latest_png() -> Response:
        return chart_response(".png", "image/png")

    @application.get("/api/v1/charts/bitcoin-power-law/latest.svg")
    def latest_svg() -> Response:
        return chart_response(".svg", "image/svg+xml")

    @application.get("/api/v1/charts/bitcoin-power-law/summary")
    def summary() -> JSONResponse:
        try:
            content = json.loads(repository.summary.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return unavailable("Chart summary has not been generated yet")
        return JSONResponse(content={"data": content}, headers=CACHE_HEADERS)

    @application.get("/api/v1/charts/bitcoin-power-law/data.csv")
    def data_csv() -> Response:
        if not repository.data.is_file():
            return unavailable("Chart data has not been generated yet")
        return FileResponse(
            repository.data,
            media_type="text/csv; charset=utf-8",
            filename="bitcoin-power-law-weekly-data.csv",
            content_disposition_type="attachment",
            headers=CACHE_HEADERS,
        )

    @application.get("/api/v1/charts/bitcoin-power-law/recent")
    def recent_prices(limit: int = 50) -> JSONResponse:
        """Return the most recent weekly rows (BTC + gold) newest first.

        Each row also carries week-over-week percentage changes so an operator
        can spot a frozen/stale feed (gold repeating an identical value because
        Yahoo was rate-limited and we fell back to the cached series) or an
        outlier BTC print at a glance.
        """
        limit = max(1, min(int(limit), 500))
        if not repository.data.is_file():
            return unavailable("Chart data has not been generated yet")
        try:
            frame = pd.read_csv(repository.data, parse_dates=["week_ending"])
        except Exception as exc:  # corrupt or unreadable CSV should not 500
            LOGGER.warning("Could not read weekly data: %s", exc)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "code": "data_unreadable",
                        "message": f"Could not read weekly data: {exc}",
                    }
                },
            )
        if frame.empty:
            return unavailable("Chart data has not been generated yet")

        frame = frame.sort_values("week_ending")
        btc = pd.to_numeric(frame.get("btc_usd"), errors="coerce")
        gold = pd.to_numeric(frame.get("gold_usd_oz"), errors="coerce")
        btc_wow = btc.pct_change() * 100.0
        gold_wow = gold.pct_change() * 100.0

        def cell(value: object) -> Optional[float]:
            if value is None or pd.isna(value):
                return None
            return float(value)

        rows: list[dict[str, object]] = []
        for position in range(len(frame) - 1, -1, -1):
            if len(rows) >= limit:
                break
            record = frame.iloc[position]
            week = record["week_ending"]
            rows.append(
                {
                    "week_ending": pd.Timestamp(week).date().isoformat(),
                    "btc_usd": cell(record.get("btc_usd")),
                    "btc_wow_pct": cell(btc_wow.iloc[position]),
                    "gold_usd_oz": cell(record.get("gold_usd_oz")),
                    "gold_wow_pct": cell(gold_wow.iloc[position]),
                    "btc_gold_ratio": cell(record.get("btc_gold_ratio")),
                    "btc_gold_52w_zscore_pct": cell(
                        record.get("btc_gold_52w_zscore_pct")
                    ),
                }
            )

        return JSONResponse(
            content={
                "data": {
                    "count": len(rows),
                    "limit": limit,
                    "total_weeks": int(len(frame)),
                    "gold_source": gold_source(),
                    "gold_stale": was_gold_stale(),
                    "updated_at": datetime.fromtimestamp(
                        repository.data.stat().st_mtime
                    ).isoformat(timespec="seconds"),
                    "rows": rows,
                }
            },
            headers=CACHE_HEADERS,
        )

    @application.get("/api/v1/charts/bitcoin-power-law/dataset")
    @application.get("/api/v1/charts/bitcoin-power-law/dataset.json")
    def dataset(pretty: bool = False, download: bool = False) -> Response:
        """Everything needed to redraw the chart elsewhere, as one JSON document.

        Contains the weekly inputs (BTC, gold, ratio, z-score, power-law
        deviation), the fitted power-law parameters and the pre-computed trend /
        support / resistance curves out to 2030. Timestamps are UNIX seconds in
        UTC - the unit TradingView expects - so a consumer can plot the same
        picture without calling our providers. See ``tradingview/`` for a Pine
        Script template that consumes this payload.
        """
        try:
            payload = build_dataset(repository.root)
        except FileNotFoundError:
            return unavailable("Chart data has not been generated yet")
        except ValueError as error:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {"code": "dataset_unavailable", "message": str(error)}
                },
            )
        headers = dict(CACHE_HEADERS)
        if download:
            headers["Content-Disposition"] = (
                'attachment; filename="bitcoin-power-law-dataset.json"'
            )
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None),
            media_type="application/json; charset=utf-8",
            headers=headers,
        )

    @application.post("/api/v1/charts/bitcoin-power-law/refresh")
    async def trigger_refresh() -> JSONResponse:
        """Manually fetch source data and regenerate the chart on demand.

        Runs in a worker thread so the event loop stays responsive. State is
        recorded in ``refresh_state`` so /health reflects the manual run. Use
        this when you don't want to wait for the next hourly slot.
        """
        attempted_at = datetime.now(ZoneInfo(SHANGHAI_TZ)).isoformat()
        refresh_state.last_attempt_at = attempted_at
        try:
            result = await asyncio.to_thread(refresh_chart, repository.root)
        except Exception as error:  # Keep the service alive on upstream failure.
            refresh_state.last_error = f"{type(error).__name__}: {error}"
            LOGGER.warning("Manual chart refresh failed: %s", error)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "code": "refresh_failed",
                        "message": str(error),
                        "attempted_at": attempted_at,
                    }
                },
                headers=CACHE_HEADERS,
            )
        refresh_state.last_success_at = datetime.now(ZoneInfo(SHANGHAI_TZ)).isoformat()
        refresh_state.last_error = None
        refresh_state.gold_stale = was_gold_stale()
        refresh_state.gold_source = gold_source()
        return JSONResponse(
            content={
                "data": {
                    "refreshed": True,
                    "gold_stale": was_gold_stale(),
                    "gold_source": gold_source(),
                    "summary": result,
                }
            },
            headers=CACHE_HEADERS,
        )

    @application.post("/api/v1/admin/restart", include_in_schema=False)
    async def restart_service() -> JSONResponse:
        """Restart the Windows service hosting this app (self-restart).

        The service runs as LocalSystem, which is permitted to talk to the SCM,
        so we can ask nssm to restart the service from inside the worker. We
        spawn nssm *detached* and do not wait: the restart stops this very
        process, so blocking would hang the request. The caller should poll
        /health until the service returns. Returns 501 on non-Windows hosts.
        """
        if sys.platform != "win32":
            return JSONResponse(
                status_code=501,
                content={
                    "error": {
                        "code": "unsupported_platform",
                        "message": "Service restart is only available on Windows",
                    }
                },
                headers=CACHE_HEADERS,
            )
        nssm = os.getenv(
            "CHART_NSSM_PATH", r"D:\akshare\nssm\nssm-2.24\win64\nssm.exe"
        )
        service = os.getenv("CHART_SERVICE_NAME", "zywChart")
        try:
            # Stop, wait for the old process to release its log-file handles,
            # then start. A bare `nssm restart` stops and starts back-to-back;
            # if the old worker hasn't freed C:\Users\...\zywChart_svc.log yet
            # the new instance fails with ERROR_SHARING_VIOLATION
            # ("另一个程序正在使用此文件"). Splitting with a short pause avoids
            # that overlap. The detached cmd survives this worker being stopped.
            cmd = (
                f'"{nssm}" stop {service} '
                f"&& timeout /t 2 /nobreak >nul "
                f'&& "{nssm}" start {service}'
            )
            subprocess.Popen(
                ["cmd.exe", "/c", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        except FileNotFoundError:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "nssm_missing",
                        "message": f"nssm executable not found at {nssm}",
                    }
                },
                headers=CACHE_HEADERS,
            )
        LOGGER.info("Service restart requested via /admin (nssm restart %s)", service)
        return JSONResponse(
            content={"data": {"restarting": True, "service": service}},
            headers=CACHE_HEADERS,
        )

    return application


def _scheduler_enabled_from_env() -> bool:
    return os.getenv("CHART_SCHEDULER_ENABLED", "1").lower() not in {
        "0",
        "false",
        "no",
    }


app = create_app(
    Path(os.getenv("CHART_OUTPUT_DIR", str(PROJECT_ROOT))),
    scheduler_enabled=_scheduler_enabled_from_env(),
)
