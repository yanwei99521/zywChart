"""FastAPI application exposing the generated chart without authentication."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from chart_service.refresh import refresh_chart
from chart_service.scheduler import RefreshState, run_scheduler

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
            task = asyncio.create_task(
                run_scheduler(
                    lambda: refresh_chart(repository.root),
                    refresh_state,
                    run_immediately=repository.latest_chart(".png") is None,
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
        allow_methods=["GET", "OPTIONS"],
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
            "latest_png": "/api/v1/charts/bitcoin-power-law/latest.png",
        }

    @application.get("/api/v1/health")
    def health() -> dict[str, object]:
        return {
            "data": {
                "status": "ok",
                "chart_ready": repository.latest_chart(".png") is not None,
                "scheduler": refresh_state.as_dict(),
                "schedule_timezone": "Asia/Shanghai",
                "schedule_times": ["07:00", "18:00"],
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
