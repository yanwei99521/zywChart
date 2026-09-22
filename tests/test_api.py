import json
from pathlib import Path

from fastapi.testclient import TestClient

from chart_service.app import create_app


def seed_chart_files(root: Path) -> None:
    (root / "bitcoin-power-law-2026-09-20.png").write_bytes(b"png-data")
    (root / "bitcoin-power-law-2026-09-20.svg").write_text(
        "<svg></svg>", encoding="utf-8"
    )
    (root / "model-summary.json").write_text(
        json.dumps({"as_of": "2026-09-20", "btc_usd": 81262.34}),
        encoding="utf-8",
    )
    (root / "data").mkdir()
    (root / "data" / "weekly_model_data.csv").write_text(
        "week_ending,btc_usd\n2026-09-20,81262.34\n", encoding="utf-8"
    )


def test_latest_png_is_publicly_downloadable(tmp_path: Path) -> None:
    seed_chart_files(tmp_path)
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    response = client.get("/api/v1/charts/bitcoin-power-law/latest.png")

    assert response.status_code == 200
    assert response.content == b"png-data"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "public, max-age=300"
    assert "bitcoin-power-law-2026-09-20.png" in response.headers[
        "content-disposition"
    ]


def test_extensionless_latest_route_returns_png(tmp_path: Path) -> None:
    seed_chart_files(tmp_path)
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    response = client.get("/api/v1/charts/bitcoin-power-law/latest")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_svg_summary_and_csv_are_public(tmp_path: Path) -> None:
    seed_chart_files(tmp_path)
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    svg_response = client.get("/api/v1/charts/bitcoin-power-law/latest.svg")
    summary_response = client.get("/api/v1/charts/bitcoin-power-law/summary")
    csv_response = client.get("/api/v1/charts/bitcoin-power-law/data.csv")

    assert svg_response.status_code == 200
    assert svg_response.headers["content-type"].startswith("image/svg+xml")
    assert summary_response.json()["data"]["as_of"] == "2026-09-20"
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")


def test_missing_chart_returns_service_unavailable(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    response = client.get("/api/v1/charts/bitcoin-power-law/latest.png")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "chart_not_ready",
            "message": "Chart has not been generated yet",
        }
    }


def test_cors_allows_public_browser_access(tmp_path: Path) -> None:
    seed_chart_files(tmp_path)
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    response = client.options(
        "/api/v1/charts/bitcoin-power-law/summary",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_index_and_health_publish_schedule(tmp_path: Path) -> None:
    seed_chart_files(tmp_path)
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    index_response = client.get("/")
    health_response = client.get("/api/v1/health")

    assert index_response.json()["documentation"] == "/docs"
    health = health_response.json()["data"]
    assert health["chart_ready"] is True
    assert health["schedule_timezone"] == "Asia/Shanghai"
    assert health["schedule_times"] == ["07:00"]
    assert health["schedule_weekday"] == 0
    assert health["schedule_description"] == "每周一 07:00"


def test_invalid_summary_and_missing_data_return_503(tmp_path: Path) -> None:
    (tmp_path / "model-summary.json").write_text("not json", encoding="utf-8")
    client = TestClient(create_app(tmp_path, scheduler_enabled=False))

    summary_response = client.get("/api/v1/charts/bitcoin-power-law/summary")
    data_response = client.get("/api/v1/charts/bitcoin-power-law/data.csv")

    assert summary_response.status_code == 503
    assert data_response.status_code == 503
