import json
from pathlib import Path

import pandas as pd
import pytest

from chart_service.refresh import (
    RefreshError,
    fetch_btc_series,
    fetch_gold_series,
    refresh_chart,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self.payload


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = iter(responses)

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
        return next(self.responses)


def test_fetch_btc_series_follows_coinmetrics_pagination() -> None:
    client = FakeClient(
        [
            FakeResponse(
                {
                    "data": [
                        {"time": "2026-09-18T00:00:00Z", "PriceUSD": "100"}
                    ],
                    "next_page_url": "https://next-page",
                }
            ),
            FakeResponse(
                {
                    "data": [
                        {"time": "2026-09-19T00:00:00Z", "PriceUSD": "110"}
                    ]
                }
            ),
        ]
    )

    result = fetch_btc_series(client_factory=lambda: client)

    assert result.iloc[-1] == 110
    assert result.index.tz is None


def test_fetch_btc_series_rejects_empty_data() -> None:
    client = FakeClient([FakeResponse({"data": []})])

    with pytest.raises(RefreshError, match="no valid BTC prices"):
        fetch_btc_series(client_factory=lambda: client)


def test_fetch_gold_series_converts_daily_close_to_weekly(
    monkeypatch, tmp_path: Path
) -> None:
    daily = pd.DataFrame(
        {"Close": [100.0, 110.0]},
        index=pd.DatetimeIndex(["2026-09-18", "2026-09-19"], tz="UTC"),
    )
    monkeypatch.setattr("chart_service.refresh.yf.download", lambda *_a, **_k: daily)
    monkeypatch.setattr("chart_service.refresh.GOLD_CACHE_PATH", tmp_path / "cache.csv")

    result = fetch_gold_series(pd.Timestamp("2026-09-20"))

    assert result.iloc[-1] == 110.0
    assert result.index.tz is None


def test_fetch_gold_series_rejects_empty_download(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "chart_service.refresh.yf.download", lambda *_a, **_k: pd.DataFrame()
    )
    monkeypatch.setattr("chart_service.refresh.time.sleep", lambda *_a: None)
    monkeypatch.setattr(
        "chart_service.refresh.fetch_gold_series_lbma",
        lambda *a, **k: (_ for _ in ()).throw(RefreshError("LBMA down")),
    )
    monkeypatch.setattr("chart_service.refresh.GOLD_CACHE_PATH", tmp_path / "cache.csv")

    with pytest.raises(RefreshError, match="no valid gold prices"):
        fetch_gold_series(pd.Timestamp("2026-09-20"))


def test_fetch_gold_series_falls_back_to_lbma(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "chart_service.refresh.yf.download", lambda *_a, **_k: pd.DataFrame()
    )
    monkeypatch.setattr("chart_service.refresh.time.sleep", lambda *_a: None)
    monkeypatch.setattr(
        "chart_service.refresh.GOLD_CACHE_PATH", tmp_path / "cache.csv"
    )

    def fake_lbma() -> pd.Series:
        daily = pd.Series(
            [4000.0, 4100.0],
            index=pd.DatetimeIndex(["2026-09-18", "2026-09-19"]),
            name="gold_usd_oz",
        )
        return daily.resample("W-SUN").last().dropna()

    monkeypatch.setattr(
        "chart_service.refresh.fetch_gold_series_lbma", lambda *a, **k: fake_lbma()
    )

    result = fetch_gold_series(pd.Timestamp("2026-09-20"))

    assert result.iloc[-1] == 4100.0


def test_fetch_gold_series_falls_back_to_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "chart_service.refresh.yf.download", lambda *_a, **_k: pd.DataFrame()
    )
    monkeypatch.setattr("chart_service.refresh.time.sleep", lambda *_a: None)
    monkeypatch.setattr(
        "chart_service.refresh.fetch_gold_series_lbma",
        lambda *a, **k: (_ for _ in ()).throw(RefreshError("LBMA down")),
    )
    cache = tmp_path / "cache.csv"
    cached = pd.Series(
        [4200.0], index=pd.DatetimeIndex([pd.Timestamp("2026-09-20")]), name="gold_usd_oz"
    )
    cached.to_csv(cache, header=True)
    monkeypatch.setattr("chart_service.refresh.GOLD_CACHE_PATH", cache)

    result = fetch_gold_series(pd.Timestamp("2026-09-20"))

    assert result.iloc[-1] == 4200.0


def test_refresh_publishes_complete_output_only_after_render_succeeds(
    tmp_path: Path,
) -> None:
    old_png = tmp_path / "bitcoin-power-law-2026-09-19.png"
    old_png.write_bytes(b"old")

    series = pd.Series(
        [1.0], index=pd.DatetimeIndex([pd.Timestamp("2026-09-20")])
    )

    def fake_renderer(
        _btc: pd.Series,
        _gold: pd.Series,
        output_dir: Path,
        _as_of: pd.Timestamp,
        **_kwargs: object,
    ) -> dict[str, float]:
        (output_dir / "data").mkdir(exist_ok=True)
        (output_dir / "bitcoin-power-law-2026-09-20.png").write_bytes(b"new")
        (output_dir / "bitcoin-power-law-2026-09-20.svg").write_text(
            "<svg>new</svg>", encoding="utf-8"
        )
        (output_dir / "data" / "weekly_model_data.csv").write_text(
            "date,value\n", encoding="utf-8"
        )
        return {"as_of": "2026-09-20", "btc_usd": 1.0}

    result = refresh_chart(
        tmp_path,
        as_of=pd.Timestamp("2026-09-20"),
        btc_loader=lambda: series,
        gold_loader=lambda _as_of: series,
        renderer=fake_renderer,
    )

    assert result["as_of"] == "2026-09-20"
    assert (tmp_path / "bitcoin-power-law-2026-09-20.png").read_bytes() == b"new"
    assert old_png.read_bytes() == b"old"
    assert json.loads((tmp_path / "model-summary.json").read_text())["as_of"] == (
        "2026-09-20"
    )


def test_failed_refresh_keeps_previous_chart(tmp_path: Path) -> None:
    old_png = tmp_path / "bitcoin-power-law-2026-09-19.png"
    old_png.write_bytes(b"old")
    series = pd.Series(
        [1.0], index=pd.DatetimeIndex([pd.Timestamp("2026-09-20")])
    )

    def failing_renderer(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        refresh_chart(
            tmp_path,
            as_of=pd.Timestamp("2026-09-20"),
            btc_loader=lambda: series,
            gold_loader=lambda _as_of: series,
            renderer=failing_renderer,
        )

    assert old_png.read_bytes() == b"old"
    assert not (tmp_path / "bitcoin-power-law-2026-09-20.png").exists()
