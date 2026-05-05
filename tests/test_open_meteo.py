from __future__ import annotations

from datetime import date
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pytest

from bot.forecast.open_meteo import OpenMeteoClient, StationForecast


def _build_payload(
    times: list[str],
    series: dict[str, list[float]],
) -> dict[str, object]:
    hourly: dict[str, object] = {"time": times}
    hourly.update(series)
    return {
        "latitude": 39.74,
        "longitude": -104.99,
        "generationtime_ms": 1.23,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "elevation": 1609.0,
        "hourly_units": {"time": "iso8601", "temperature_2m": "F"},
        "hourly": hourly,
    }


async def test_url_construction_hits_ensemble_endpoint_with_expected_params() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        payload = _build_payload(
            times=["2026-05-06T00:00", "2026-05-06T01:00"],
            series={
                "temperature_2m": [50.0, 51.0],
                "temperature_2m_member01": [49.0, 52.0],
            },
        )
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenMeteoClient(http_client=http)
        await client.fetch_station(
            station="KDEN",
            latitude=39.74,
            longitude=-104.99,
            timezone="America/Denver",
            forecast_days=7,
        )

    req = captured["req"]
    parsed = urlparse(str(req.url))
    assert parsed.netloc == "ensemble-api.open-meteo.com"
    assert parsed.path == "/v1/ensemble"
    qs = parse_qs(parsed.query)
    assert qs["latitude"] == ["39.74"]
    assert qs["longitude"] == ["-104.99"]
    assert qs["hourly"] == ["temperature_2m"]
    assert qs["models"] == ["gfs_seamless"]
    assert qs["temperature_unit"] == ["fahrenheit"]
    assert qs["timezone"] == ["UTC"]
    assert qs["forecast_days"] == ["7"]


async def test_parses_two_local_dates_with_per_member_daily_max() -> None:
    times = [f"2026-05-06T{h:02d}:00" for h in range(6, 24)] + [
        f"2026-05-07T{h:02d}:00" for h in range(0, 18)
    ]
    control = [60.0 + i * 0.1 for i in range(len(times))]
    m1 = [70.0] * len(times)
    m1[10] = 95.0
    m1[28] = 88.0
    m2 = [55.0] * len(times)
    m2[15] = 80.0
    m2[30] = 78.0

    payload = _build_payload(
        times=times,
        series={
            "temperature_2m": control,
            "temperature_2m_member01": m1,
            "temperature_2m_member02": m2,
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenMeteoClient(http_client=http)
        forecast = await client.fetch_station(
            station="KDEN",
            latitude=39.74,
            longitude=-104.99,
            timezone="America/Denver",
        )

    assert isinstance(forecast, StationForecast)
    assert forecast.station == "KDEN"
    assert forecast.timezone == "America/Denver"
    assert set(forecast.daily_highs.keys()) == {date(2026, 5, 6), date(2026, 5, 7)}

    for d, arr in forecast.daily_highs.items():
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert arr.dtype == np.float64

    day1 = forecast.daily_highs[date(2026, 5, 6)]
    assert day1[1] == pytest.approx(95.0)
    assert day1[2] == pytest.approx(80.0)
    day2 = forecast.daily_highs[date(2026, 5, 7)]
    assert day2[1] == pytest.approx(88.0)
    assert day2[2] == pytest.approx(78.0)


async def test_truncated_final_local_day_dropped() -> None:
    times = [f"2026-05-06T{h:02d}:00" for h in range(6, 24)] + [
        f"2026-05-07T{h:02d}:00" for h in range(0, 3)
    ]
    n = len(times)
    payload = _build_payload(
        times=times,
        series={
            "temperature_2m": [70.0] * n,
            "temperature_2m_member01": [71.0] * n,
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenMeteoClient(http_client=http)
        forecast = await client.fetch_station(
            station="KDEN",
            latitude=39.74,
            longitude=-104.99,
            timezone="America/Denver",
        )

    assert date(2026, 5, 7) not in forecast.daily_highs
    assert date(2026, 5, 6) in forecast.daily_highs


async def test_default_client_is_closed_by_aclose(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_build_payload(
                times=[f"2026-05-06T{h:02d}:00" for h in range(6, 24)],
                series={
                    "temperature_2m": [70.0] * 18,
                    "temperature_2m_member01": [71.0] * 18,
                },
            ),
        )

    transport = httpx.MockTransport(handler)
    real_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    client = OpenMeteoClient()
    forecast = await client.fetch_station(
        station="KDEN",
        latitude=39.74,
        longitude=-104.99,
        timezone="America/Denver",
    )
    assert isinstance(forecast, StationForecast)
    await client.aclose()
    assert client._http.is_closed


async def test_caller_owned_client_not_closed_by_aclose() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_build_payload(
                times=[f"2026-05-06T{h:02d}:00" for h in range(0, 24)],
                series={"temperature_2m": [70.0] * 24},
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenMeteoClient(http_client=http)
        await client.fetch_station(
            station="KDEN",
            latitude=39.74,
            longitude=-104.99,
            timezone="America/Denver",
        )
        await client.aclose()
        assert not http.is_closed
