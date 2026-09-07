from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import httpx


# The Single Runs route pins an exact run= but its retention cliff is 2026-04-02, which is later
# than the whole sample, so Class A is retrieved lead-anchored from the Previous Runs route instead.
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

CLASS_A_MEMBERS: tuple[str, ...] = (
    "ecmwf_ifs025",
    "icon_global",
    "ukmo_global_deterministic_10km",
)
PREVIOUS_DAY_VARIABLE: Mapping[int, str] = {
    24: "temperature_2m_previous_day1",
    36: "temperature_2m_previous_day2",
}
PREVIOUS_DAY_OFFSET: Mapping[str, timedelta] = {
    "temperature_2m_previous_day1": timedelta(hours=24),
    "temperature_2m_previous_day2": timedelta(hours=48),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class PreviousRunsSeries:
    member: str
    latitude: float
    longitude: float
    hourly: Mapping[str, Mapping[datetime, Decimal]]
    source_url: str


def parse_previous_runs(body: bytes, member: str, source_url: str) -> PreviousRunsSeries:
    payload = json.loads(body, parse_float=Decimal)
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError(f"previous-runs response for {member} carries no hourly.time")
    times = [datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc) for stamp in hourly["time"]]
    series = {}
    for variable in PREVIOUS_DAY_OFFSET:
        values = hourly.get(variable)
        if values is None:
            continue
        if len(values) != len(times):
            raise ValueError(
                f"{variable} for {member} carries {len(values)} values against {len(times)} times"
            )
        series[variable] = {
            stamp: value for stamp, value in zip(times, values) if value is not None
        }
    return PreviousRunsSeries(
        member=member,
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        hourly=series,
        source_url=source_url,
    )


async def fetch_previous_runs(
    *,
    latitude: float,
    longitude: float,
    member: str,
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient,
) -> PreviousRunsSeries:
    response = await client.get(
        PREVIOUS_RUNS_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(PREVIOUS_DAY_OFFSET),
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
            "models": member,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    response.raise_for_status()
    return parse_previous_runs(response.content, member, str(response.url))
