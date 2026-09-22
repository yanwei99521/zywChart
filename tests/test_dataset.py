import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from chart_service.app import create_app
from chart_service.dataset import build_dataset

START = date(2010, 1, 3)  # a Sunday, matching the W-SUN weekly rule


def seed_weekly(root: Path, weeks: int = 80) -> None:
    lines = [
        "week_ending,btc_usd,power_law_trend,power_law_support,btc_vs_power_law_pct,"
        "gold_usd_oz,btc_gold_ratio,btc_gold_52w_zscore_pct"
    ]
    price = 100.0
    gold = 1200.0
    for index in range(weeks):
        stamp = (START + timedelta(weeks=index)).isoformat()
        price *= 1.02
        gold *= 1.001
        lines.append(
            f"{stamp},{price:.8f},{price * 0.9:.8f},{price * 0.33:.8f},5.0,{gold:.8f},"
            f"{price / gold:.8f},{index % 7:.8f}"
        )
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "weekly_model_data.csv").write_text("\n".join(lines), encoding="utf-8")


def test_dataset_exposes_model_and_all_series(tmp_path: Path) -> None:
    seed_weekly(tmp_path)
    dataset = build_dataset(tmp_path)

    assert dataset["schema"].startswith("zywchart/bitcoin-power-law-dataset@")
    assert dataset["timestamp_unit"] == "seconds"
    assert dataset["week_rule"] == "W-SUN"

    series = dataset["series"]
    for name in (
        "btc",
        "gold",
        "btc_gold_ratio",
        "btc_gold_ratio_52w_zscore_pct",
        "btc_vs_power_law_pct",
        "power_law_trend",
        "power_law_support",
        "power_law_resistance",
    ):
        assert name in series
        assert len(series[name]) > 0
        for point in series[name]:
            assert len(point) == 2
            assert isinstance(point[0], int)
            assert isinstance(point[1], float)

    model = dataset["model"]
    assert model["type"] == "power_law"
    assert model["genesis"] == "2009-01-03"
    assert isinstance(model["slope"], float)
    assert isinstance(model["intercept"], float)


def test_dataset_timestamps_are_utc_seconds_and_ascending(tmp_path: Path) -> None:
    seed_weekly(tmp_path)
    points = build_dataset(tmp_path)["series"]["btc"]

    stamps = [point[0] for point in points]
    assert stamps == sorted(stamps)
    expected = int(datetime(START.year, START.month, START.day, tzinfo=timezone.utc).timestamp())
    assert stamps[0] == expected


def test_power_law_curves_extend_into_the_future(tmp_path: Path) -> None:
    seed_weekly(tmp_path)
    dataset = build_dataset(tmp_path)
    trend = dataset["series"]["power_law_trend"]

    last_stamp = trend[-1][0]
    assert last_stamp > dataset["series"]["btc"][-1][0]
    # The published chart draws the corridor out to the end of 2030.
    assert datetime.fromtimestamp(last_stamp, timezone.utc).year >= 2030


def test_support_and_resistance_are_trend_divided_and_times_e(tmp_path: Path) -> None:
    import math

    seed_weekly(tmp_path)
    dataset = build_dataset(tmp_path)
    trend = dataset["series"]["power_law_trend"]
    support = dataset["series"]["power_law_support"]
    resistance = dataset["series"]["power_law_resistance"]

    assert len(trend) == len(support) == len(resistance)
    for index in range(0, len(trend), 10):
        assert support[index][1] == trend[index][1] / math.e
        assert resistance[index][1] == trend[index][1] * math.e


def test_dataset_endpoint_returns_json(tmp_path: Path) -> None:
    seed_weekly(tmp_path)
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    response = client.get("/api/v1/charts/bitcoin-power-law/dataset.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["schema"].startswith("zywchart/bitcoin-power-law-dataset@")
    assert payload["tradingview"]["pine_model"]["genesis"] == "2009-01-03"


def test_dataset_endpoint_supports_download(tmp_path: Path) -> None:
    seed_weekly(tmp_path)
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    response = client.get("/api/v1/charts/bitcoin-power-law/dataset?download=true")

    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]


def test_dataset_endpoint_503_without_enough_data(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "weekly_model_data.csv").write_text(
        "week_ending,btc_usd\n2026-09-20,81262.34\n", encoding="utf-8"
    )
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    response = client.get("/api/v1/charts/bitcoin-power-law/dataset.json")

    assert response.status_code == 503
    assert response.json()["error"]["code"] in {"chart_not_ready", "dataset_unavailable"}
