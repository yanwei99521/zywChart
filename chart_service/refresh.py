"""Download source data and publish a newly rendered chart."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Callable, Optional

import httpx
import pandas as pd
import yfinance as yf

from build_chart import draw_chart

COIN_METRICS_URL = (
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
)


class RefreshError(RuntimeError):
    """Raised when fresh source data cannot produce a usable chart."""


HttpClientFactory = Callable[[], AbstractContextManager[httpx.Client]]


def _default_client_factory() -> AbstractContextManager[httpx.Client]:
    return httpx.Client(timeout=45.0, follow_redirects=True)


def fetch_btc_series(
    client_factory: HttpClientFactory = _default_client_factory,
) -> pd.Series:
    """Fetch all available daily BTC/USD observations from Coin Metrics."""
    rows: list[dict[str, object]] = []
    url = COIN_METRICS_URL
    params: Optional[dict[str, object]] = {
        "assets": "btc",
        "metrics": "PriceUSD",
        "frequency": "1d",
        "start_time": "2009-01-03",
        "page_size": 10000,
    }

    with client_factory() as client:
        for _page in range(10):
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            page_rows = payload.get("data", [])
            if not isinstance(page_rows, list):
                raise RefreshError("Coin Metrics returned malformed data")
            rows.extend(page_rows)

            next_page_url = payload.get("next_page_url")
            if not next_page_url:
                break
            if not isinstance(next_page_url, str):
                raise RefreshError("Coin Metrics returned an invalid next page URL")
            url = next_page_url
            params = None
        else:
            raise RefreshError("Coin Metrics pagination exceeded the safety limit")

    frame = pd.DataFrame(rows)
    if "time" not in frame.columns or "PriceUSD" not in frame.columns:
        raise RefreshError("Coin Metrics returned no valid BTC prices")

    frame["date"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    frame["btc_usd"] = pd.to_numeric(frame["PriceUSD"], errors="coerce")
    valid = frame.dropna(subset=["date", "btc_usd"])
    if valid.empty:
        raise RefreshError("Coin Metrics returned no valid BTC prices")

    valid = valid.assign(date=valid["date"].dt.tz_convert(None))
    return (
        valid.set_index("date")["btc_usd"]
        .sort_index()
        .resample("W-SUN")
        .last()
        .dropna()
    )


def fetch_gold_series(as_of: pd.Timestamp) -> pd.Series:
    """Fetch COMEX gold futures daily closes and convert them to weekly data."""
    end = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frame = yf.download(
        "GC=F",
        start="2009-01-01",
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=30,
        multi_level_index=False,
    )
    if frame.empty or "Close" not in frame.columns:
        raise RefreshError("Yahoo Finance returned no valid gold prices")

    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        raise RefreshError("Yahoo Finance returned no valid gold prices")

    index = pd.DatetimeIndex(pd.to_datetime(close.index))
    if index.tz is not None:
        index = index.tz_convert(None)
    close.index = index
    close.name = "gold_usd_oz"
    return close.sort_index().resample("W-SUN").last().dropna()


Renderer = Callable[
    [pd.Series, pd.Series, Path, pd.Timestamp], dict[str, object]
]


def refresh_chart(
    output_dir: Path,
    *,
    as_of: Optional[pd.Timestamp] = None,
    btc_loader: Callable[[], pd.Series] = fetch_btc_series,
    gold_loader: Callable[[pd.Timestamp], pd.Series] = fetch_gold_series,
    renderer: Renderer = draw_chart,
) -> dict[str, object]:
    """Render in a temporary directory, then publish a complete snapshot."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    effective_as_of = as_of or pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)

    btc = btc_loader()
    gold = gold_loader(effective_as_of)

    with tempfile.TemporaryDirectory(
        prefix=".chart-refresh-", dir=output_dir
    ) as temporary:
        staging = Path(temporary)
        (staging / "data").mkdir()
        rendered_summary = renderer(btc, gold, staging, effective_as_of)
        (staging / "model-summary.json").write_text(
            json.dumps(rendered_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        png_files = sorted(staging.glob("bitcoin-power-law-*.png"))
        svg_files = sorted(staging.glob("bitcoin-power-law-*.svg"))
        required = [
            staging / "model-summary.json",
            staging / "data" / "weekly_model_data.csv",
        ]
        if len(png_files) != 1 or len(svg_files) != 1:
            raise RefreshError("Renderer did not create exactly one PNG and SVG")
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RefreshError("Renderer did not create a complete data snapshot")
        if png_files[0].stat().st_size == 0 or svg_files[0].stat().st_size == 0:
            raise RefreshError("Renderer created an empty chart")

        # Publish metadata first and the primary PNG last. A failed render never
        # touches the prior public files.
        os.replace(
            staging / "data" / "weekly_model_data.csv",
            output_dir / "data" / "weekly_model_data.csv",
        )
        os.replace(
            staging / "model-summary.json", output_dir / "model-summary.json"
        )
        os.replace(svg_files[0], output_dir / svg_files[0].name)
        os.replace(png_files[0], output_dir / png_files[0].name)

    # Re-read the published file so the return value is the exact public state.
    return json.loads((output_dir / "model-summary.json").read_text(encoding="utf-8"))
