#!/usr/bin/env python3
"""Build a data-driven recreation of the Bitcoin power-law chart.

The rules implemented here follow the reference front-end implementation
(``bitcoin-power-law.ts``):

*weekly bars*   daily closes are bucketed into weeks that end on Sunday
                (pandas ``W-SUN``), keeping the last observation of the week.
*power law*     ordinary least squares on ``ln(days since 2009-01-03)`` vs
                ``ln(price)`` over the full BTC history.
*corridor*      trend = ``exp(intercept) * days ** slope``; support is
                ``trend / e`` and resistance ``trend * e``.
*deviation*     ``100 * ln(price / trend)`` (log deviation, in percent).
*z-score*       the BTC/gold ratio against its own trailing 52 week mean,
                using the **sample** standard deviation (ddof=1), scaled by 100.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


GENESIS = pd.Timestamp("2009-01-03")

# Reference layout: the price panel spans 2009-01-01 .. 2030 and the oscillator
# panel is clipped to +/-140, mirroring the SVG the front-end renders.
CHART_START = pd.Timestamp("2009-01-01")
CHART_END = pd.Timestamp("2030-12-31")
PRICE_FLOOR = 0.1
PRICE_CEILING_MIN = 1_000_000.0
OSC_LIMIT = 140.0
ZSCORE_WINDOW = 52

# Palette taken from the reference SVG so both renderings look like one chart.
COLORS = {
    "bg": "#ffffff",
    "grid": "#edf1f3",
    "btc": "#17242b",
    "trend": "#20cbd3",
    "support": "#d7b46b",
    "band": "#42e9ef",
    "deviation": "#28ef4e",
    "zscore": "#d91bd1",
    "zero": "#aeb8bf",
    "title": "#60666c",
    "subtitle": "#76808a",
    "axis": "#70777d",
    "foot": "#8a9298",
    "legend": "#4e5962",
}

# Fonts able to render the Chinese labels used throughout the chart, most
# preferred first. The previous list only held macOS fonts, so on Windows
# matplotlib fell back to DejaVu Sans (no CJK glyphs) and every Chinese label
# came out as tofu boxes / mojibake.
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "DengXian",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Noto Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Arial Unicode MS",
    "SimSun",
    "DejaVu Sans",
)


def resolve_cjk_font() -> str | None:
    """Return the first installed font that can render Chinese glyphs."""
    try:
        from matplotlib import font_manager
    except Exception:  # pragma: no cover - matplotlib is always available here
        return None
    try:
        available = {entry.name for entry in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover - defensive
        return None
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            return name
    return None


def read_btc_coinmetrics(path: Path) -> pd.Series:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["data"])
    frame["date"] = pd.to_datetime(frame["time"]).dt.tz_localize(None)
    frame["btc_usd"] = pd.to_numeric(frame["PriceUSD"], errors="coerce")
    return (
        frame.dropna(subset=["date", "btc_usd"])
        .set_index("date")["btc_usd"]
        .sort_index()
        .resample("W-SUN")
        .last()
        .dropna()
    )


def read_gold_yfinance(path: Path) -> pd.Series:
    # yfinance emits a 3-row header when saving a one-ticker MultiIndex frame.
    frame = pd.read_csv(path, skiprows=[1, 2])
    date_col = "Price" if "Price" in frame.columns else "Date"
    frame["date"] = pd.to_datetime(frame[date_col])
    frame["gold_usd_oz"] = pd.to_numeric(frame["Close"], errors="coerce")
    return (
        frame.dropna(subset=["date", "gold_usd_oz"])
        .set_index("date")["gold_usd_oz"]
        .sort_index()
        .resample("W-SUN")
        .last()
        .dropna()
    )


def power_law_fit(btc: pd.Series) -> tuple[float, float]:
    """OLS of ln(price) on ln(days since genesis); days are floored at 1."""
    days = np.maximum((btc.index - GENESIS).days.to_numpy(dtype=float), 1.0)
    slope, intercept = np.polyfit(np.log(days), np.log(btc.to_numpy()), 1)
    return float(slope), float(intercept)


def model_values(index: pd.DatetimeIndex, slope: float, intercept: float) -> np.ndarray:
    days = np.maximum((index - GENESIS).days.to_numpy(dtype=float), 1.0)
    return np.exp(intercept) * np.power(days, slope)


def ratio_zscore(ratio: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Z-score of the BTC/gold ratio vs its trailing window (x100).

    Uses the sample standard deviation (ddof=1) and starts as soon as two
    observations exist, matching the reference implementation.
    """
    rolling = ratio.rolling(window, min_periods=2)
    z = (ratio - rolling.mean()) / rolling.std(ddof=1)
    return (z * 100.0).replace([np.inf, -np.inf], np.nan)


def money(value: float, _position: float | None = None) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    if value >= 1:
        return f"${value:,.0f}"
    return f"${value:.2f}"


def price_tick(value: float, _position: float | None = None) -> str:
    """Axis-tick label: whole dollars above $1, decimals below."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:g}M"
    if value >= 1:
        return f"${value:,.0f}"
    return f"${value:g}"


def draw_chart(
    btc: pd.Series,
    gold: pd.Series,
    output_dir: Path,
    as_of: pd.Timestamp,
    gold_label: str = "LBMA Gold Price PM（USD/盎司）",
) -> dict[str, float]:
    btc = btc.loc[:as_of].dropna().sort_index()
    gold = gold.loc[:as_of].dropna().sort_index()
    if len(btc) < 2:
        raise ValueError("at least two BTC observations are required to fit the model")

    slope, intercept = power_law_fit(btc)

    # Corridor: draw from the first observation out to 2030 like the reference.
    future_index = pd.date_range(btc.index.min(), CHART_END, freq="W-SUN")
    trend_future = model_values(future_index, slope, intercept)
    support_future = trend_future / np.e
    upper_future = trend_future * np.e

    trend_hist = pd.Series(model_values(btc.index, slope, intercept), index=btc.index)
    oscillator = 100.0 * np.log(btc / trend_hist)

    # Gold is aligned on the exact week-ending Sunday; weeks without a gold
    # print simply have no ratio (no forward filling), as in the reference.
    aligned = pd.concat([btc.rename("btc_usd"), gold.rename("gold_usd_oz")], axis=1).sort_index()
    aligned = aligned.dropna()
    ratio = aligned["btc_usd"] / aligned["gold_usd_oz"]
    ratio_z = ratio_zscore(ratio)

    data = pd.concat(
        [
            btc.rename("btc_usd"),
            trend_hist.rename("power_law_trend"),
            (trend_hist / np.e).rename("power_law_support"),
            oscillator.rename("btc_vs_power_law_pct"),
            aligned["gold_usd_oz"],
            ratio.rename("btc_gold_ratio"),
            ratio_z.rename("btc_gold_52w_zscore_pct"),
        ],
        axis=1,
    )
    data.index.name = "week_ending"
    data.to_csv(output_dir / "data" / "weekly_model_data.csv", float_format="%.8f")

    chosen_font = resolve_cjk_font()
    font_stack = ([chosen_font] if chosen_font else []) + [
        name for name in CJK_FONT_CANDIDATES if name != chosen_font
    ]
    if chosen_font is None:
        print("warning: no CJK-capable font found; Chinese labels may render as boxes")
    plt.rcParams.update(
        {
            "font.family": font_stack,
            "font.sans-serif": list(CJK_FONT_CANDIDATES),
            "axes.unicode_minus": False,
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "savefig.facecolor": COLORS["bg"],
        }
    )

    fig = plt.figure(figsize=(13.6, 9.0), dpi=160)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.09)
    ax = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1], sharex=ax)

    ceiling = max(PRICE_CEILING_MIN, float(np.max(upper_future)) * 1.05)
    ax.set_yscale("log")
    ax.set_ylim(PRICE_FLOOR, ceiling)
    ax.set_xlim(CHART_START, CHART_END)

    # Corridor: one vertical gradient from trend*e (top, .34) to trend/e (.04).
    levels = np.linspace(1.0, -1.0, 25)
    for high, low in zip(levels[:-1], levels[1:]):
        middle = (high + low) / 2.0
        alpha = 0.04 + 0.30 * (middle + 1.0) / 2.0
        ax.fill_between(
            future_index,
            trend_future * np.exp(low),
            trend_future * np.exp(high),
            color=COLORS["band"],
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )

    ax.plot(
        future_index,
        trend_future,
        color=COLORS["trend"],
        lw=1.5,
        ls=(0, (3, 3)),
        label="幂律趋势线",
        zorder=3,
    )
    ax.plot(
        future_index,
        support_future,
        color=COLORS["support"],
        lw=1.2,
        label="幂律支撑线（中枢 ÷ e）",
        zorder=3,
    )
    ax.plot(
        btc.index,
        btc.values,
        color=COLORS["btc"],
        lw=1.45,
        label="比特币周价",
        zorder=5,
    )

    ticks = [0.1, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    ticks = [tick for tick in ticks if PRICE_FLOOR <= tick <= ceiling]
    ax.set_yticks(ticks)
    ax.set_yticklabels([price_tick(tick) for tick in ticks], color=COLORS["axis"], fontsize=10)
    ax.grid(which="major", color=COLORS["grid"], lw=0.8)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", colors=COLORS["axis"], labelsize=10)
    ax.set_ylabel("比特币价格（美元，对数刻度）", color="#4e5962", fontsize=10)

    current_date = btc.index[-1]
    current_price = float(btc.iloc[-1])
    current_trend = float(trend_hist.iloc[-1])
    current_support = current_trend / np.e
    current_osc = float(oscillator.iloc[-1])
    current_gold = float(aligned["gold_usd_oz"].iloc[-1]) if len(aligned) else float("nan")
    ratio_z_clean = ratio_z.dropna()
    current_z = float(ratio_z_clean.iloc[-1]) if len(ratio_z_clean) else float("nan")

    ax.annotate(
        money(current_price),
        xy=(current_date, current_price),
        xytext=(8, -4),
        textcoords="offset points",
        fontsize=11,
        color="#111111",
        weight="bold",
    )
    ax.annotate(
        money(current_trend),
        xy=(current_date, current_trend),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=10,
        color="#12aeb6",
    )
    ax.annotate(
        money(current_support),
        xy=(current_date, current_support),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=10,
        color="#b8862e",
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        bbox_to_anchor=(0.012, 0.965),
        frameon=False,
        fontsize=9.5,
        labelcolor=COLORS["legend"],
    )

    ax_bottom.fill_between(
        oscillator.index,
        0,
        oscillator.values,
        color=COLORS["deviation"],
        alpha=0.82,
        linewidth=0,
        label="BTC vs 幂律",
    )
    ax_bottom.fill_between(
        ratio_z.index,
        0,
        ratio_z.values,
        color=COLORS["zscore"],
        alpha=0.78,
        linewidth=0,
        label="BTC/黄金 52周 Z-score",
    )
    ax_bottom.axhline(0, color=COLORS["zero"], lw=0.9, ls=(0, (3, 3)))
    ax_bottom.set_ylim(-OSC_LIMIT, OSC_LIMIT)
    ax_bottom.set_yticks([-140, -70, 0, 70, 140])
    ax_bottom.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}"))
    ax_bottom.yaxis.tick_right()
    ax_bottom.yaxis.set_label_position("right")
    ax_bottom.tick_params(axis="y", colors=COLORS["axis"], labelsize=9)
    ax_bottom.set_ylabel("偏离 / Z-score", color="#4e5962", fontsize=9)
    ax_bottom.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bottom.tick_params(axis="x", colors=COLORS["axis"], labelsize=9)
    ax_bottom.grid(which="major", color=COLORS["grid"], lw=0.8)
    ax_bottom.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.97),
        frameon=False,
        fontsize=9.5,
        labelcolor=COLORS["legend"],
    )

    for axis in (ax, ax_bottom):
        axis.spines["top"].set_visible(False)
        axis.spines["left"].set_color(COLORS["grid"])
        axis.spines["right"].set_color(COLORS["grid"])
        axis.spines["bottom"].set_color(COLORS["grid"])

    fig.suptitle("Bitcoin's Power Law", y=0.975, fontsize=22, color=COLORS["title"], weight="bold")
    fig.text(
        0.5,
        0.936,
        f"比特币幂律与 BTC/黄金 52 周 Z-score · 数据截至 {current_date:%Y-%m-%d}",
        ha="center",
        fontsize=11,
        color=COLORS["subtitle"],
    )
    fig.text(
        0.075,
        0.018,
        (
            f"数据源：Coin Metrics / Yahoo（BTC）、{gold_label}。周频数据（周日收周），本地缓存。"
            f"最新：BTC {money(current_price)}｜中枢 {money(current_trend)}｜支撑 {money(current_support)}"
            f"｜偏离 {current_osc:.0f}%｜Z-score {current_z:.0f}"
        ),
        fontsize=9,
        color=COLORS["foot"],
    )
    fig.subplots_adjust(left=0.075, right=0.945, top=0.90, bottom=0.075)

    png_path = output_dir / f"bitcoin-power-law-{current_date:%Y-%m-%d}.png"
    svg_path = output_dir / f"bitcoin-power-law-{current_date:%Y-%m-%d}.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)

    return {
        "as_of": current_date.strftime("%Y-%m-%d"),
        "btc_usd": current_price,
        "gold_usd_oz": current_gold,
        "power_law_slope": slope,
        "power_law_intercept_ln": intercept,
        "power_law_trend": current_trend,
        "power_law_support": current_support,
        "btc_vs_power_law_pct": current_osc,
        "btc_gold_52w_zscore_pct": current_z,
        "gold_label": gold_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-json", type=Path, required=True)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", default="2026-09-20")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data").mkdir(parents=True, exist_ok=True)
    summary = draw_chart(
        read_btc_coinmetrics(args.btc_json),
        read_gold_yfinance(args.gold_csv),
        args.output_dir,
        pd.Timestamp(args.as_of),
    )
    (args.output_dir / "model-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
