#!/usr/bin/env python3
"""Build a data-driven recreation of the Bitcoin power-law chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator
import numpy as np
import pandas as pd


GENESIS = pd.Timestamp("2009-01-03")

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
    days = (btc.index - GENESIS).days.to_numpy(dtype=float)
    slope, intercept = np.polyfit(np.log(days), np.log(btc.to_numpy()), 1)
    return float(slope), float(intercept)


def model_values(index: pd.DatetimeIndex, slope: float, intercept: float) -> np.ndarray:
    days = np.maximum((index - GENESIS).days.to_numpy(dtype=float), 1.0)
    return np.exp(intercept) * np.power(days, slope)


def money(value: float, _position: float | None = None) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    if value >= 1_000:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.0f}"
    return f"${value:.2f}"


def cycle_extreme(series: pd.Series, start: str, end: str, kind: str) -> tuple[pd.Timestamp, float]:
    window = series.loc[start:end]
    when = window.idxmax() if kind == "max" else window.idxmin()
    return when, float(window.loc[when])


def draw_chart(
    btc: pd.Series,
    gold: pd.Series,
    output_dir: Path,
    as_of: pd.Timestamp,
    gold_label: str = "Yahoo Finance COMEX Gold (GC=F)",
) -> dict[str, float]:
    btc = btc.loc[:as_of]
    gold = gold.loc[:as_of]
    slope, intercept = power_law_fit(btc)

    future_index = pd.date_range(GENESIS + pd.Timedelta(days=120), "2030-12-31", freq="W-SUN")
    trend_future = model_values(future_index, slope, intercept)
    support_future = trend_future / np.e
    upper_future = trend_future * np.e

    trend_hist = pd.Series(model_values(btc.index, slope, intercept), index=btc.index)
    oscillator = 100.0 * np.log(btc / trend_hist)

    aligned = pd.concat([btc.rename("btc_usd"), gold.rename("gold_usd_oz")], axis=1).sort_index()
    aligned["gold_usd_oz"] = aligned["gold_usd_oz"].ffill(limit=2)
    aligned = aligned.dropna()
    ratio = aligned["btc_usd"] / aligned["gold_usd_oz"]
    ratio_z = (ratio - ratio.rolling(52).mean()) / ratio.rolling(52).std(ddof=0)
    ratio_z = ratio_z * 100.0

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
            "figure.facecolor": "#fbfbfa",
            "axes.facecolor": "#fbfbfa",
            "savefig.facecolor": "#fbfbfa",
        }
    )

    fig = plt.figure(figsize=(18, 10.5), dpi=150)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.65, 1.0], hspace=0.0)
    ax = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1], sharex=ax)

    # A layered cyan corridor approximates the soft model glow in the reference.
    band_levels = np.linspace(-1.0, 1.0, 17)
    for low, high in zip(band_levels[:-1], band_levels[1:]):
        alpha = 0.018 + 0.022 * (1.0 - abs((low + high) / 2.0))
        ax.fill_between(
            future_index,
            trend_future * np.exp(low),
            trend_future * np.exp(high),
            color="#39e8e4",
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )

    ax.plot(future_index, trend_future, color="#28d8dc", lw=1.4, ls=(0, (2, 2)), label="幂律中枢线")
    ax.plot(future_index, support_future, color="#c9974b", lw=1.3, alpha=0.9, label="幂律支撑线（中枢 ÷ e）")
    ax.plot(btc.index, btc.values, color="#171717", lw=1.35, label="比特币周价", zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(0.02, 4_000_000)
    ax.set_xlim(pd.Timestamp("2009-01-01"), pd.Timestamp("2030-12-31"))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.grid(which="major", color="#d7d7d4", lw=0.7, alpha=0.5)
    ax.grid(which="minor", color="#e8e8e5", lw=0.45, alpha=0.35)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", colors="#686866", labelsize=10)
    ax.set_ylabel("比特币价格（美元，对数刻度）", color="#4d4d4b", fontsize=11)

    current_date = btc.index[-1]
    current_price = float(btc.iloc[-1])
    current_trend = float(trend_hist.iloc[-1])
    current_support = current_trend / np.e
    current_osc = float(oscillator.iloc[-1])
    current_z = float(ratio_z.dropna().iloc[-1])
    current_gold = float(aligned["gold_usd_oz"].iloc[-1])

    ax.axhline(60_000, color="#202020", lw=0.9, ls=(0, (3, 2)), alpha=0.75)
    ax.text(pd.Timestamp("2011-08-01"), 70_000, "关键观察位：$60k", fontsize=10, color="#202020")

    ax.annotate(
        f"现价  {money(current_price)}",
        xy=(current_date, current_price),
        xytext=(24, 12),
        textcoords="offset points",
        fontsize=11,
        color="#111111",
        arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 0.8},
    )
    ax.annotate(
        f"幂律中枢  {money(current_trend)}",
        xy=(current_date, current_trend),
        xytext=(24, 6),
        textcoords="offset points",
        fontsize=10,
        color="#149da1",
    )
    ax.annotate(
        f"模型支撑  {money(current_support)}",
        xy=(current_date, current_support),
        xytext=(24, -17),
        textcoords="offset points",
        fontsize=10,
        color="#a66f24",
    )

    cycles = [
        ("2011-01-01", "2011-07-01", "2011-07-01", "2012-01-31"),
        ("2013-01-01", "2014-01-31", "2014-01-01", "2015-06-30"),
        ("2016-01-01", "2018-01-31", "2018-01-01", "2019-06-30"),
        ("2020-01-01", "2022-01-31", "2022-01-01", "2023-06-30"),
        ("2024-01-01", "2026-01-31", "2025-10-01", str(as_of.date())),
    ]
    for peak_start, peak_end, trough_start, trough_end in cycles:
        peak_date, peak = cycle_extreme(btc, peak_start, peak_end, "max")
        trough_date, trough = cycle_extreme(btc, trough_start, trough_end, "min")
        drawdown = (trough / peak - 1.0) * 100.0
        ax.annotate(
            f"{money(peak)}",
            xy=(peak_date, peak),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color="#111111",
        )
        ax.annotate(
            f"{drawdown:.0f}%",
            xy=(trough_date, trough),
            xytext=(0, -23),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color="#111111",
            arrowprops={"arrowstyle": "-|>", "color": "#303030", "lw": 0.8},
        )

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.04),
        frameon=False,
        fontsize=10.5,
    )

    green = oscillator.reindex(btc.index)
    magenta = ratio_z.reindex(btc.index).interpolate(limit=2)
    ax_bottom.fill_between(green.index, 0, green.values, color="#35ef4f", alpha=0.92, linewidth=0, label="BTC 相对幂律偏离")
    ax_bottom.fill_between(magenta.index, 0, magenta.values, color="#d92acf", alpha=0.82, linewidth=0, label="BTC/黄金 52周 Z-score")
    ax_bottom.axhline(0, color="#8a8a87", lw=0.8, alpha=0.8)
    ax_bottom.set_ylim(-140, 120)
    ax_bottom.yaxis.set_major_locator(MultipleLocator(20))
    ax_bottom.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    ax_bottom.yaxis.tick_right()
    ax_bottom.yaxis.set_label_position("right")
    ax_bottom.tick_params(axis="y", colors="#686866", labelsize=9)
    ax_bottom.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bottom.tick_params(axis="x", colors="#686866", labelsize=9)
    ax_bottom.grid(which="major", color="#d7d7d4", lw=0.7, alpha=0.45)
    ax_bottom.legend(loc="upper right", bbox_to_anchor=(0.985, 0.98), frameon=False, fontsize=10.5)

    ax_bottom.annotate(
        f"最新：幂律偏离 {current_osc:.0f}%  |  BTC/黄金 Z-score {current_z:.0f}%",
        xy=(current_date, current_osc),
        xytext=(-8, -30),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        color="#4c4c49",
    )

    for axis in (ax, ax_bottom):
        axis.spines["top"].set_visible(False)
        axis.spines["left"].set_color("#c9c9c6")
        axis.spines["right"].set_color("#c9c9c6")
        axis.spines["bottom"].set_color("#c9c9c6")

    fig.suptitle("比特币幂律与黄金相对强弱", y=0.965, fontsize=22, color="#595957", weight="semibold")
    fig.text(
        0.5,
        0.93,
        f"Coin Metrics BTC/USD · {gold_label} · 周度数据",
        ha="center",
        fontsize=11,
        color="#777774",
    )
    fig.text(
        0.065,
        0.025,
        f"数据截至 {current_date:%Y-%m-%d}。模型为历史拟合，不构成价格保证或投资建议。黄金最新周收盘：{money(current_gold)} / 盎司。",
        fontsize=9.5,
        color="#777774",
    )
    fig.subplots_adjust(left=0.07, right=0.93, top=0.90, bottom=0.08)

    png_path = output_dir / f"bitcoin-power-law-{current_date:%Y-%m-%d}.png"
    svg_path = output_dir / f"bitcoin-power-law-{current_date:%Y-%m-%d}.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
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
