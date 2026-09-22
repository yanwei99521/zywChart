"""Export everything needed to redraw the chart elsewhere (e.g. TradingView).

The published chart is derived from a handful of series plus the fitted power
law model. ``build_dataset`` packages all of it into one self-describing JSON
document so a third party can reproduce the exact same picture without access
to our data providers:

* raw weekly inputs (BTC, gold, ratio, z-score, power-law deviation)
* the fitted model parameters (slope / intercept / genesis date), which are
  enough to recompute the trend corridor on any future bar
* the pre-computed trend / support / resistance curves extended to 2030, so a
  consumer that cannot evaluate the formula itself can just plot the points

Timestamps are UNIX **seconds in UTC**, the unit TradingView expects.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from build_chart import GENESIS, model_values, power_law_fit

SCHEMA = "zywchart/bitcoin-power-law-dataset@1"
FUTURE_END = pd.Timestamp("2030-12-31")
EULER = math.e

# Week bars end on Sunday (pandas "W-SUN"): the label is the last day included.
WEEK_RULE = "W-SUN"

SERIES_UNITS = {
    "btc": "USD per BTC",
    "gold": "USD per troy ounce",
    "btc_gold_ratio": "ounces of gold per BTC",
    "btc_gold_ratio_52w_zscore_pct": "z-score of the ratio vs its own 52 week mean, x100",
    "btc_vs_power_law_pct": "100 * ln(btc / power_law_trend), percent",
    "power_law_trend": "USD per BTC (fitted power law)",
    "power_law_support": "USD per BTC (trend / e)",
    "power_law_resistance": "USD per BTC (trend * e)",
}


def _points(series: pd.Series) -> list[list[float]]:
    """Convert a dated series to [[unix_seconds_utc, value], ...]."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    stamps = pd.DatetimeIndex(numeric.index)
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    seconds = ((stamps - epoch) // pd.Timedelta("1s")).astype(int)
    return [[int(sec), float(value)] for sec, value in zip(seconds, numeric.to_numpy())]


def _weekly_frame(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "data" / "weekly_model_data.csv", parse_dates=["week_ending"])
    return frame.sort_values("week_ending").set_index("week_ending")


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(dtype="float64")


def build_dataset(root: Path) -> dict[str, object]:
    """Assemble the full dataset used to render the chart."""
    frame = _weekly_frame(root)
    if frame.empty or "btc_usd" not in frame.columns:
        raise ValueError("weekly model data is missing or empty")

    btc = pd.to_numeric(frame["btc_usd"], errors="coerce").dropna()
    if len(btc) < 2:
        raise ValueError("at least two BTC observations are required to fit the model")
    slope, intercept = power_law_fit(btc)

    # The chart draws the corridor from the fitted start out to 2030, so ship
    # the computed curves (history + future) instead of only the raw inputs.
    curve_index = pd.date_range(btc.index.min(), FUTURE_END, freq=WEEK_RULE)
    trend_curve = pd.Series(model_values(curve_index, slope, intercept), index=curve_index)

    series: dict[str, list[list[float]]] = {
        "btc": _points(btc),
        "gold": _points(_column(frame, "gold_usd_oz")),
        "btc_gold_ratio": _points(_column(frame, "btc_gold_ratio")),
        "btc_gold_ratio_52w_zscore_pct": _points(
            _column(frame, "btc_gold_52w_zscore_pct")
        ),
        "btc_vs_power_law_pct": _points(_column(frame, "btc_vs_power_law_pct")),
        "power_law_trend": _points(trend_curve),
        "power_law_support": _points(trend_curve / EULER),
        "power_law_resistance": _points(trend_curve * EULER),
    }

    summary: dict[str, object] = {}
    summary_path = root / "model-summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = {}

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": pd.Timestamp(btc.index.max()).date().isoformat(),
        "timezone": "UTC",
        "frequency": "weekly",
        "week_rule": WEEK_RULE,
        "timestamp_unit": "seconds",
        "sources": {
            "btc": "Coin Metrics community API (PriceUSD, daily) resampled to weekly",
            "gold": summary.get("gold_label")
            or "gold weekly close (source recorded in gold_source)",
            "gold_source": summary.get("gold_source"),
        },
        "model": {
            "type": "power_law",
            "genesis": pd.Timestamp(GENESIS).date().isoformat(),
            "formula": "price = exp(intercept) * days_since_genesis ** slope",
            "slope": slope,
            "intercept": intercept,
            "support": "trend / e",
            "resistance": "trend * e",
            "corridor_levels": [round(x, 4) for x in _corridor_levels()],
            "corridor_formula": "trend * exp(level)",
        },
        "units": SERIES_UNITS,
        "series": series,
        "counts": {name: len(points) for name, points in series.items()},
        "latest": {
            "btc_usd": float(btc.iloc[-1]) if len(btc) else None,
            "gold_usd_oz": _last_value(_column(frame, "gold_usd_oz")),
            "btc_gold_ratio": _last_value(_column(frame, "btc_gold_ratio")),
            "btc_gold_ratio_52w_zscore_pct": _last_value(
                _column(frame, "btc_gold_52w_zscore_pct")
            ),
            "btc_vs_power_law_pct": _last_value(_column(frame, "btc_vs_power_law_pct")),
            "power_law_trend": float(trend_curve.loc[btc.index.max()])
            if btc.index.max() in trend_curve.index
            else None,
        },
        "tradingview": {
            "timeframe": "1W",
            "timestamp_unit": "seconds_utc",
            "pine_model": {
                "genesis": pd.Timestamp(GENESIS).date().isoformat(),
                "slope": slope,
                "intercept": intercept,
            },
            "notes": (
                "TradingView cannot import JSON directly. Use the model block "
                "above to plot the corridor in Pine Script (see "
                "tradingview/bitcoin_power_law.pine), and let TradingView "
                "supply BTC/gold prices from its own feeds."
            ),
        },
    }


def _corridor_levels() -> list[float]:
    import numpy as np

    return [float(x) for x in np.linspace(-1.0, 1.0, 17)]


def _last_value(series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[-1])


def dataset_json(root: Path, *, indent: Optional[int] = None) -> str:
    return json.dumps(build_dataset(root), ensure_ascii=False, indent=indent)
