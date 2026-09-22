"""Download source data and publish a newly rendered chart."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Callable, Optional

import httpx
import pandas as pd
import yfinance as yf

from build_chart import draw_chart

LOGGER = logging.getLogger(__name__)

COIN_METRICS_URL = (
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
)

LBMA_GOLD_URL = "https://prices.lbma.org.uk/json/gold_pm.json"

# Labels used on the rendered chart footnote, keyed by gold data source.
GOLD_LABELS = {
    "yahoo": "Yahoo Finance COMEX Gold (GC=F)",
    "lbma": "LBMA Gold Price PM (spot)",
}

# Last successfully fetched weekly gold series, used as a fallback when Yahoo
# rate-limits or is unreachable so an hourly refresh can still publish a chart
# (with fresh BTC and the most recent good gold value) instead of failing hard.
GOLD_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_cache.csv"

# Source of the gold series used by the most recent fetch:
# "yahoo" (futures GC=F, primary), "lbma" (spot fallback), "cache" (stale
# last-known-good), or "" (no fetch has happened yet).
GOLD_SOURCE = ""

# Whether the most recent fetch fell back to the cached gold series. Surfaced
# via the health endpoint so operators know the gold value may be stale.
GOLD_STALE = False


def was_gold_stale() -> bool:
    """Return True if the last successful refresh used cached gold data."""
    return GOLD_STALE


def gold_source() -> str:
    """Return the data source used by the most recent gold fetch."""
    return GOLD_SOURCE


def gold_label() -> str:
    """Chart footnote label describing the gold data source in use."""
    return GOLD_LABELS.get(GOLD_SOURCE, GOLD_LABELS["yahoo"])


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


def _save_gold_cache(series: pd.Series) -> None:
    """Persist the last good weekly gold series so refreshes can fall back to it."""
    try:
        GOLD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(GOLD_CACHE_PATH, header=True)
    except Exception as exc:  # pragma: no cover - best effort only
        LOGGER.warning("Could not persist gold cache: %s", exc)


def _load_gold_cache() -> Optional[pd.Series]:
    try:
        if GOLD_CACHE_PATH.is_file():
            frame = pd.read_csv(GOLD_CACHE_PATH, index_col=0, parse_dates=True)
            series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
            series.name = "gold_usd_oz"
            if not series.empty:
                return series.sort_index()
    except Exception as exc:  # pragma: no cover - best effort only
        LOGGER.warning("Could not read gold cache: %s", exc)
    return None


def fetch_gold_series_lbma(
    client_factory: HttpClientFactory = _default_client_factory,
) -> pd.Series:
    """Fetch the LBMA Gold Price PM (USD) fixings as a weekly series.

    Official, keyless JSON feed dating back to 1968; used as the fallback when
    Yahoo Finance (GC=F futures) is rate-limited or unreachable. Values are
    spot fixings, so they track GC=F closely but are not identical.
    """
    with client_factory() as client:
        response = client.get(LBMA_GOLD_URL)
        response.raise_for_status()
        rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise RefreshError("LBMA returned no gold prices")

    records = []
    for row in rows:
        date = row.get("d")
        values = row.get("v") or []
        if not date or not values or values[0] is None:
            continue
        records.append((pd.Timestamp(date), float(values[0])))
    if not records:
        raise RefreshError("LBMA returned no gold prices")

    close = pd.Series(
        pd.to_numeric([value for _, value in records], errors="coerce"),
        index=pd.DatetimeIndex([date for date, _ in records]),
    ).dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.name = "gold_usd_oz"
    return close.resample("W-SUN").last().dropna()


def fetch_gold_series(as_of: pd.Timestamp) -> pd.Series:
    """Fetch weekly gold prices, preferring COMEX futures with LBMA fallback.

    Source chain: Yahoo Finance GC=F futures (primary) -> LBMA Gold Price PM
    spot (fallback) -> last-known-good cache (last resort). Yahoo frequently
    rate-limits automated requests (HTTP 429), which makes yfinance return an
    empty frame; LBMA is an official keyless feed that is tracked closely.
    """
    end = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_error: Optional[str] = None
    global GOLD_SOURCE, GOLD_STALE
    GOLD_STALE = False
    GOLD_SOURCE = ""

    for attempt in range(3):
        try:
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
            if frame is not None and not frame.empty and "Close" in frame.columns:
                close = frame["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = pd.to_numeric(close, errors="coerce").dropna()
                if not close.empty:
                    index = pd.DatetimeIndex(pd.to_datetime(close.index))
                    if index.tz is not None:
                        index = index.tz_convert(None)
                    close.index = index
                    close.name = "gold_usd_oz"
                    weekly = close.sort_index().resample("W-SUN").last().dropna()
                    if not weekly.empty:
                        GOLD_SOURCE = "yahoo"
                        _save_gold_cache(weekly)
                        return weekly
                last_error = "Yahoo returned a frame with no usable Close prices"
            else:
                last_error = "Yahoo returned an empty frame"
        except Exception as exc:  # yfinance can raise on rate-limit / network
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < 2:
            time.sleep(10 * (attempt + 1))  # 10s, then 20s backoff

    LOGGER.warning(
        "Yahoo gold fetch failed after retries (last error: %s); trying LBMA "
        "spot feed before falling back to the cache.",
        last_error,
    )
    try:
        lbma_weekly = fetch_gold_series_lbma()
    except Exception as exc:
        LOGGER.warning("LBMA gold fetch also failed: %s", exc)
        lbma_weekly = None

    if lbma_weekly is not None and not lbma_weekly.empty:
        GOLD_SOURCE = "lbma"
        _save_gold_cache(lbma_weekly)
        return lbma_weekly

    cached = _load_gold_cache()
    if cached is not None:
        GOLD_SOURCE = "cache"
        GOLD_STALE = True
        LOGGER.warning(
            "All gold fetches failed (Yahoo last error: %s; see log for LBMA); "
            "using cached gold series (most recent good value) so the chart "
            "still refreshes.",
            last_error,
        )
        return cached

    raise RefreshError(
        f"Yahoo Finance returned no valid gold prices (last attempt: {last_error})"
    )


Renderer = Callable[..., dict[str, object]]


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
        rendered_summary = renderer(
            btc, gold, staging, effective_as_of, gold_label=gold_label()
        )
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
