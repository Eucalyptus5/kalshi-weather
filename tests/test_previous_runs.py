from __future__ import annotations

import json
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from bot.backtest.previous_runs import (
    CLASS_A_MEMBERS,
    PREVIOUS_DAY_OFFSET,
    PREVIOUS_DAY_VARIABLE,
    PREVIOUS_RUNS_URL,
    SINGLE_RUNS_URL,
    fetch_previous_runs,
    parse_previous_runs,
)
from bot.lag.forecast_classes import (
    CLASS_A,
    lst_window_hours,
    window_basis_for,
)
from bot.lag.forecast_sample import SampleLeg, read_sample_freeze
from bot.markets.observation_window import observation_window
from scripts.freeze_f4_sample import DEFAULT_OUT, FREEZE_NAME


UTC = timezone.utc
ECMWF_FIXTURE = Path(__file__).parent / "data" / "previous_runs_knyc_ecmwf.json"
UKMO_GAP_FIXTURE = Path(__file__).parent / "data" / "previous_runs_knyc_ukmo_gap.json"
REFUSED_FIXTURE = Path(__file__).parent / "data" / "single_runs_refused_20260127.json"
FROZEN_SAMPLE = DEFAULT_OUT / FREEZE_NAME
KNYC = (40.7790, -73.9692)
SPAN = (date(2024, 10, 24), date(2026, 1, 29))
SAMPLE_LAST_DAY = date(2026, 1, 27)

LIVE_ONLY = pytest.mark.skipif(
    os.environ.get("KW_LIVE_F4") != "1",
    reason="set KW_LIVE_F4=1 to read the live forecast endpoints",
)
needs_tape = pytest.mark.skipif(
    not FROZEN_SAMPLE.exists(), reason="the recorded tape is not on this host"
)


@pytest.fixture(scope="module")
def legs() -> list[SampleLeg]:
    return read_sample_freeze(FROZEN_SAMPLE)


def triples(legs: list[SampleLeg], lead_hours: int) -> dict[tuple[str, date], SampleLeg]:
    return {(leg.station, leg.event_date): leg for leg in legs if leg.lead_hours == lead_hours}


def late_components(leg: SampleLeg, basis: str, offset: timedelta) -> int:
    hours = lst_window_hours(leg.timezone, leg.event_date, basis)
    return sum(1 for hour in hours if hour - offset > leg.as_of)


def test_the_lead_offsets_pair_day1_with_24h_and_day2_with_36h() -> None:
    assert PREVIOUS_DAY_VARIABLE[24] == "temperature_2m_previous_day1"
    assert PREVIOUS_DAY_VARIABLE[36] == "temperature_2m_previous_day2"
    assert PREVIOUS_DAY_OFFSET["temperature_2m_previous_day1"] == timedelta(hours=24)
    assert PREVIOUS_DAY_OFFSET["temperature_2m_previous_day2"] == timedelta(hours=48)
    assert CLASS_A_MEMBERS == (
        "ecmwf_ifs025",
        "icon_global",
        "ukmo_global_deterministic_10km",
    )


def test_parse_reads_both_variables_as_decimals_on_a_utc_clock() -> None:
    series = parse_previous_runs(ECMWF_FIXTURE.read_bytes(), "ecmwf_ifs025", PREVIOUS_RUNS_URL)

    day1 = series.hourly["temperature_2m_previous_day1"]
    day2 = series.hourly["temperature_2m_previous_day2"]
    assert series.latitude == pytest.approx(40.75)
    assert series.longitude == pytest.approx(-74.0)
    assert len(day1) == 48
    assert day1[datetime(2025, 1, 28, tzinfo=UTC)] == Decimal("30.3")
    assert day2[datetime(2025, 1, 28, tzinfo=UTC)] == Decimal("31.6")
    assert isinstance(day1[datetime(2025, 1, 28, 4, tzinfo=UTC)], Decimal)


def test_parse_drops_the_hours_the_model_did_not_run() -> None:
    series = parse_previous_runs(
        UKMO_GAP_FIXTURE.read_bytes(), "ukmo_global_deterministic_10km", PREVIOUS_RUNS_URL
    )

    day1 = series.hourly["temperature_2m_previous_day1"]
    assert series.latitude == pytest.approx(40.78125)
    assert series.longitude == pytest.approx(-73.96875)
    assert len(day1) == 10
    assert {stamp.date() for stamp in day1} == {date(2025, 4, 6)}


async def test_fetch_asks_the_previous_runs_route_for_both_variables() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=ECMWF_FIXTURE.read_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        series = await fetch_previous_runs(
            latitude=KNYC[0],
            longitude=KNYC[1],
            member="ecmwf_ifs025",
            start_date=date(2025, 1, 28),
            end_date=date(2025, 1, 29),
            client=client,
        )

    assert series.member == "ecmwf_ifs025"
    assert len(seen) == 1
    url = seen[0].url
    assert str(url).startswith(PREVIOUS_RUNS_URL)
    assert url.params["models"] == "ecmwf_ifs025"
    assert url.params["timezone"] == "UTC"
    assert url.params["temperature_unit"] == "fahrenheit"
    assert url.params["hourly"] == ("temperature_2m_previous_day1,temperature_2m_previous_day2")
    assert series.source_url == str(url)


@needs_tape
def test_reading_day1_at_the_36h_lead_leaks_and_day2_does_not(legs: list[SampleLeg]) -> None:
    basis = window_basis_for(CLASS_A, 36)
    day1 = Counter()
    day2 = Counter()
    worst = timedelta(0)
    slack = timedelta(days=99)
    for leg in triples(legs, 36).values():
        day1[late_components(leg, basis, timedelta(hours=24))] += 1
        day2[late_components(leg, basis, timedelta(hours=48))] += 1
        hours = lst_window_hours(leg.timezone, leg.event_date, basis)
        worst = max(worst, max(hour - timedelta(hours=24) for hour in hours) - leg.as_of)
        slack = min(slack, leg.as_of - max(hour - timedelta(hours=48) for hour in hours))

    assert dict(sorted(day1.items())) == {12: 898, 13: 1306, 14: 4}
    assert dict(day2) == {0: 2208}
    assert worst == timedelta(hours=13, minutes=1)
    assert slack == timedelta(hours=10, minutes=59)


@needs_tape
def test_dropping_the_last_hour_clears_all_but_the_seven_early_close_days(
    legs: list[SampleLeg],
) -> None:
    full = Counter()
    remedy = Counter()
    leaking_event_days = set()
    still_leaking = []
    for leg in triples(legs, 24).values():
        late_full = late_components(leg, "lst_full", timedelta(hours=24))
        full[late_full] += 1
        if late_full:
            leaking_event_days.add(leg.event_date)
        late_remedy = late_components(leg, window_basis_for(CLASS_A, 24), timedelta(hours=24))
        remedy[late_remedy] += 1
        if late_remedy:
            _, window_end = observation_window(leg.timezone, leg.event_date)
            still_leaking.append((leg.station, int((leg.close_time - window_end).total_seconds())))

    assert dict(sorted(full.items())) == {0: 942, 1: 1348, 2: 7}
    assert sum(count for late, count in full.items() if late) == 1355
    assert len(leaking_event_days) == 266
    assert dict(sorted(remedy.items())) == {0: 2290, 1: 7}
    assert sorted(set(still_leaking)) == [("KMDW", -7260)]


def single_runs_params(run: date) -> dict:
    return {
        "latitude": KNYC[0],
        "longitude": KNYC[1],
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "models": "ecmwf_ifs025",
        "run": f"{run.isoformat()}T00:00",
    }


def test_the_single_runs_route_refuses_the_last_day_of_the_sample() -> None:
    recorded = json.loads(REFUSED_FIXTURE.read_bytes())

    assert recorded["error"] is True
    assert "The requested model run is not available" in recorded["reason"]
    assert f"run: {SAMPLE_LAST_DAY.isoformat()}T00:00Z" in recorded["reason"]
    assert "ecmwf_ifs025" in recorded["reason"]
    assert SAMPLE_LAST_DAY <= SPAN[1]


@needs_tape
def test_the_recorded_refusal_names_the_last_day_the_sample_carries(legs: list[SampleLeg]) -> None:
    assert max(leg.event_date for leg in legs) == SAMPLE_LAST_DAY


@LIVE_ONLY
def test_the_single_runs_route_still_refuses_the_sample_and_serves_after_the_cliff() -> None:
    with httpx.Client(timeout=60.0) as client:
        refused_sample = client.get(SINGLE_RUNS_URL, params=single_runs_params(SAMPLE_LAST_DAY))
        refused = client.get(SINGLE_RUNS_URL, params=single_runs_params(date(2026, 4, 1)))
        served = client.get(SINGLE_RUNS_URL, params=single_runs_params(date(2026, 4, 2)))

    assert refused_sample.status_code == 400
    assert refused_sample.json()["reason"] == json.loads(REFUSED_FIXTURE.read_bytes())["reason"]
    assert refused.status_code == 400
    assert "The requested model run is not available" in refused.json()["reason"]
    assert served.status_code == 200
    values = served.json()["hourly"]["temperature_2m"]
    assert len(values) == 168
    assert sum(1 for value in values if value is not None) == 168


@LIVE_ONLY
async def test_the_previous_runs_route_covers_the_whole_sample_span() -> None:
    counts = {}
    grids = {}
    async with httpx.AsyncClient(timeout=180.0) as client:
        for member in CLASS_A_MEMBERS:
            series = await fetch_previous_runs(
                latitude=KNYC[0],
                longitude=KNYC[1],
                member=member,
                start_date=SPAN[0],
                end_date=SPAN[1],
                client=client,
            )
            counts[member] = {variable: len(values) for variable, values in series.hourly.items()}
            grids[member] = (series.latitude, series.longitude)

    day1 = "temperature_2m_previous_day1"
    day2 = "temperature_2m_previous_day2"
    assert counts["ecmwf_ifs025"] == {day1: 11112, day2: 11112}
    assert counts["icon_global"] == {day1: 11112, day2: 11112}
    assert counts["ukmo_global_deterministic_10km"] == {day1: 8320, day2: 9375}
    assert grids["ecmwf_ifs025"] == (pytest.approx(40.75), pytest.approx(-74.0))
    assert grids["icon_global"] == (pytest.approx(40.75), pytest.approx(-74.0))
    assert grids["ukmo_global_deterministic_10km"] == (
        pytest.approx(40.78125),
        pytest.approx(-73.96875),
    )


@LIVE_ONLY
async def test_ukmo_gaps_are_scattered_runs_rather_than_a_retention_start() -> None:
    async with httpx.AsyncClient(timeout=180.0) as client:
        series = await fetch_previous_runs(
            latitude=KNYC[0],
            longitude=KNYC[1],
            member="ukmo_global_deterministic_10km",
            start_date=SPAN[0],
            end_date=SPAN[1],
            client=client,
        )

    per_day = Counter(stamp.date() for stamp in series.hourly["temperature_2m_previous_day1"])
    span = [SPAN[0] + timedelta(days=offset) for offset in range((SPAN[1] - SPAN[0]).days + 1)]
    complete = [day for day in span if per_day[day] == 24]
    incomplete = [day for day in span if per_day[day] != 24]

    assert len(complete) == 338
    assert complete[0] == date(2024, 10, 28)
    assert len(incomplete) == 125
    assert sum(1 for day in incomplete if day > date(2025, 4, 7)) == 44
    assert per_day[date(2025, 4, 6)] == 10
