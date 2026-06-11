from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq
import pytest

from bot.backtest.cli import main
from bot.backtest.depth_table import LEAD_BUCKETS
from bot.backtest.forecast_replay import StationSpec
from bot.backtest.historical_open_meteo import (
    HistoricalOpenMeteoForecastReplay,
    SpreadCalibration,
)
from bot.backtest.ingest import ingest_weather_snapshots
from bot.forecast.cdf import EnsembleCDF
from scripts.build_spread_calibration import collect_points, sigma_table
from tests.fixtures.trevorjs_market_state_sample import make_market_state_table

DATA = Path(__file__).parent / "data"
HISTORICAL_PAYLOAD = json.loads((DATA / "historical_seamless_kden.json").read_text())
PINNED_SIGMA = 1.7888543819998317

KDEN = StationSpec(name="KDEN", latitude=39.8466, longitude=-104.6562, timezone="America/Denver")
VALID_DATE = date(2024, 11, 21)
AS_OF = datetime(2024, 11, 20, 12, 0, tzinfo=timezone.utc)


def _flat_calibration(sigma: float) -> SpreadCalibration:
    return SpreadCalibration({bucket: sigma for bucket in LEAD_BUCKETS})


def _historical_client(seen: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=HISTORICAL_PAYLOAD)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _offline_client() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_replay_members_centered_on_deterministic_high_with_calibrated_spread() -> None:
    async with _historical_client([]) as client:
        replay = HistoricalOpenMeteoForecastReplay(client, _flat_calibration(2.5))
        cdf = await replay.replay(KDEN, VALID_DATE, AS_OF)

    members = cdf.members
    assert members.shape == (31,)
    assert float(np.median(members)) == 70.0
    assert float(np.std(members, ddof=1)) == pytest.approx(2.5, rel=0.05)

    reference = EnsembleCDF.from_members(members.copy(), smoothing=1.0)
    assert cdf.prob_range(68.0, 72.0) == pytest.approx(reference.prob_range(68.0, 72.0))
    assert cdf.prob_range(72.0, 75.0) == pytest.approx(reference.prob_range(72.0, 75.0))


async def test_replay_is_deterministic_across_calls() -> None:
    async with _historical_client([]) as client:
        replay = HistoricalOpenMeteoForecastReplay(client, _flat_calibration(2.5))
        first = await replay.replay(KDEN, VALID_DATE, AS_OF)
        second = await replay.replay(KDEN, VALID_DATE, AS_OF)

    assert np.array_equal(first.members, second.members)


async def test_replay_window_covers_only_valid_date() -> None:
    seen: list[httpx.Request] = []
    async with _historical_client(seen) as client:
        replay = HistoricalOpenMeteoForecastReplay(client, _flat_calibration(2.5))
        await replay.replay(KDEN, VALID_DATE, AS_OF)

    assert len(seen) == 1
    request = seen[0]
    assert request.url.host == "historical-forecast-api.open-meteo.com"
    params = dict(request.url.params)
    assert params["start_date"] == "2024-11-21"
    assert params["end_date"] == "2024-11-22"
    assert set(params) == {
        "latitude",
        "longitude",
        "hourly",
        "models",
        "temperature_unit",
        "timezone",
        "start_date",
        "end_date",
    }


async def test_replay_missing_lead_bucket_raises_naming_bucket() -> None:
    async with _offline_client() as client:
        replay = HistoricalOpenMeteoForecastReplay(client, SpreadCalibration({"<=2h": 1.0}))
        with pytest.raises(ValueError, match=">72h"):
            await replay.replay(KDEN, VALID_DATE, datetime(2024, 11, 1, tzinfo=timezone.utc))


async def test_replay_naive_as_of_raises() -> None:
    async with _offline_client() as client:
        replay = HistoricalOpenMeteoForecastReplay(client, _flat_calibration(2.5))
        with pytest.raises(ValueError, match="timezone-aware"):
            await replay.replay(KDEN, VALID_DATE, datetime(2024, 11, 20, 12, 0))


def test_calibration_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "model": "gfs_seamless",
                "built_at": "2026-06-10T00:00:00+00:00",
                "buckets": [
                    {"bucket": "8-24h", "min_hours": 8, "max_hours": 24, "sigma": 1.5},
                    {"bucket": "24-72h", "min_hours": 24, "max_hours": 72, "sigma": 2.25},
                ],
            }
        )
    )
    calibration = SpreadCalibration.load(path)

    assert calibration.sigma_for(timedelta(hours=12)) == 1.5
    assert calibration.sigma_for(timedelta(hours=48)) == 2.25
    with pytest.raises(ValueError, match="2-8h"):
        calibration.sigma_for(timedelta(hours=4))


def test_builder_sigma_pinned_from_fixture_pair() -> None:
    ensemble = json.loads((DATA / "spread_cal_ensemble_kden.json").read_text())
    deterministic = json.loads((DATA / "spread_cal_deterministic_kden.json").read_text())
    issuance = datetime(2024, 11, 20, 0, 0, tzinfo=timezone.utc)

    rows = sigma_table(collect_points(ensemble, deterministic, "America/Denver", issuance))

    assert [row["bucket"] for row in rows] == ["24-72h", ">72h"]
    for row in rows:
        assert row["sigma"] == pytest.approx(PINNED_SIGMA, abs=1e-12)
        assert row["pairs"] == 1
        assert row["points"] == 31
    assert (rows[0]["min_hours"], rows[0]["max_hours"]) == (24, 72)
    assert (rows[1]["min_hours"], rows[1]["max_hours"]) == (72, None)


def test_builder_skips_days_already_ended() -> None:
    ensemble = json.loads((DATA / "spread_cal_ensemble_kden.json").read_text())
    deterministic = json.loads((DATA / "spread_cal_deterministic_kden.json").read_text())
    issuance = datetime(2024, 11, 25, 0, 0, tzinfo=timezone.utc)

    assert collect_points(ensemble, deterministic, "America/Denver", issuance) == []


def test_cli_source_resolves_and_dry_runs(monkeypatch, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        start = date.fromisoformat(params["start_date"])
        end = date.fromisoformat(params["end_date"])
        times: list[str] = []
        temps: list[float] = []
        day = start
        while day <= end:
            for hour in range(24):
                times.append(f"{day.isoformat()}T{hour:02d}:00")
                temps.append(48.0 + hour * 0.5)
            day += timedelta(days=1)
        payload = dict(HISTORICAL_PAYLOAD)
        payload["hourly"] = {"time": times, "temperature_2m": temps}
        return httpx.Response(200, json=payload)

    real_client = httpx.AsyncClient

    def patched(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)

    shard = tmp_path / "market_state.parquet"
    pq.write_table(make_market_state_table(), shard)
    parquet = tmp_path / "weather.parquet"
    ingest_weather_snapshots([shard], parquet, series=["KXHIGHDEN"])
    out_dir = tmp_path / "report"

    code = main(
        [
            "--weather-parquet",
            str(parquet),
            "--strategy",
            "tails",
            "--lead",
            "24h",
            "--out",
            str(out_dir),
            "--forecast-source",
            "bot.backtest.historical_open_meteo:cli_source",
            "--dry-run",
        ]
    )

    assert code == 0
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "orders.csv").exists()
