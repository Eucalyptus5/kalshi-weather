from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import numpy as np
import pytest

from bot.backtest.gefs_grib import decode_point, tmp2m_byte_range
from bot.backtest.hrrr import (
    EXTENDED_CYCLES,
    HRRR_BUCKET,
    PUBLICATION_ALLOWANCE,
    DecodedFieldCache,
    HrrrGrid,
    build_hrrr_url,
    choose_run,
    decode_points,
    fetch_tmp2m_field,
    fold_longitudes,
    leg_fields,
    pull_fields,
)
from bot.lag.forecast_classes import CLASS_C_MEMBER
from bot.lag.forecast_sample import SampleLeg, read_sample_freeze
from bot.main import STATIONS
from scripts.freeze_f4_sample import DEFAULT_OUT, FREEZE_NAME


UTC = timezone.utc
MESSAGE = (Path(__file__).parent / "data" / "hrrr_tmp2m_f024.grib2").read_bytes()
IDX_URL = f"{HRRR_BUCKET}/hrrr.20250615/conus/hrrr.t00z.wrfsfcf24.grib2.idx"
FROZEN_SAMPLE = DEFAULT_OUT / FREEZE_NAME
COUNTED_DAY = date(2025, 1, 15)
RECORD_BYTES = 1269328
EXPECTED = {
    "KNYC": ((698, 1553), Decimal("62.49")),
    "KLAX": ((433, 260), Decimal("71.72")),
    "KMIA": ((109, 1483), Decimal("83.08")),
}

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


@pytest.fixture(scope="module")
def grid() -> HrrrGrid:
    return HrrrGrid.from_message(MESSAGE)


def coordinates(station: str) -> tuple[float, float]:
    config = next(row for row in STATIONS.values() if row.station == station)
    return config.latitude, config.longitude


def idx_body(fxx: int) -> bytes:
    return (
        f"1:0:d=2025011500:TMP:2 m above ground:{fxx} hour fcst:\n"
        f"2:{len(MESSAGE)}:d=2025011500:DPT:2 m above ground:{fxx} hour fcst:\n"
    ).encode()


def bucket_handler(seen: list[str]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if url.endswith(".idx"):
            fxx = int(url.split("wrfsfcf")[1].split(".")[0])
            return httpx.Response(200, content=idx_body(fxx))
        return httpx.Response(206, content=MESSAGE)

    return handler


def test_the_gefs_decoder_cannot_read_an_hrrr_message() -> None:
    with pytest.raises(StopIteration):
        decode_point(MESSAGE, latitude=40.7790, longitude=-73.9692)


def test_the_two_dimensional_nearest_neighbour_lands_on_the_measured_cells(
    grid: HrrrGrid,
) -> None:
    cells = {station: grid.nearest(*coordinates(station)) for station in EXPECTED}

    for station, (cell, _) in EXPECTED.items():
        assert (cells[station].y, cells[station].x) == cell
        assert cells[station].latitude == pytest.approx(coordinates(station)[0], abs=0.05)


def test_the_decoded_temperatures_are_the_measured_fahrenheit(grid: HrrrGrid) -> None:
    points = [grid.nearest(*coordinates(station)) for station in EXPECTED]

    values = decode_points(MESSAGE, points)

    assert values == [value for _, value in EXPECTED.values()]
    assert all(isinstance(value, Decimal) for value in values)


def test_skipping_the_longitude_fold_lands_every_station_on_the_same_corner(
    grid: HrrrGrid,
) -> None:
    unfolded = HrrrGrid(grid.latitudes, (grid.longitudes + 360.0) % 360.0)

    cells = {}
    for station in EXPECTED:
        point = unfolded.nearest(*coordinates(station))
        cells[station] = (point.y, point.x)

    assert set(cells.values()) == {(1058, 0)}
    assert cells != {station: cell for station, (cell, _) in EXPECTED.items()}


def test_fold_longitudes_maps_the_bucket_range_onto_the_western_hemisphere(
    grid: HrrrGrid,
) -> None:
    raw = (grid.longitudes + 360.0) % 360.0

    folded = fold_longitudes(raw)

    assert raw.min() >= 0.0
    assert raw.max() > 180.0
    assert folded.max() < 0.0
    assert np.allclose(folded, grid.longitudes)


def test_build_url_matches_the_bucket_layout() -> None:
    assert build_hrrr_url(date(2025, 6, 15), cycle=0, fxx=24) == (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250615/conus/hrrr.t00z.wrfsfcf24.grib2"
    )
    assert CLASS_C_MEMBER == "hrrr"


@pytest.mark.parametrize(
    ("as_of", "run"),
    [
        (datetime(2025, 6, 15, 4, 0, tzinfo=UTC), datetime(2025, 6, 15, 0, tzinfo=UTC)),
        (datetime(2025, 6, 15, 3, 59, tzinfo=UTC), datetime(2025, 6, 14, 18, tzinfo=UTC)),
        (datetime(2025, 6, 15, 23, 59, tzinfo=UTC), datetime(2025, 6, 15, 18, tzinfo=UTC)),
        (datetime(2025, 6, 15, 21, 59, tzinfo=UTC), datetime(2025, 6, 15, 12, tzinfo=UTC)),
        (datetime(2025, 6, 15, 10, 0, tzinfo=UTC), datetime(2025, 6, 15, 6, tzinfo=UTC)),
    ],
)
def test_the_run_is_the_latest_extended_cycle_behind_the_publication_allowance(
    as_of: datetime, run: datetime
) -> None:
    chosen = choose_run(as_of)

    assert chosen == run
    assert chosen.hour in EXTENDED_CYCLES
    assert chosen <= as_of - PUBLICATION_ALLOWANCE
    assert PUBLICATION_ALLOWANCE == timedelta(hours=4)


@needs_tape
def test_every_run_the_freeze_needs_is_an_extended_cycle(legs: list[SampleLeg]) -> None:
    runs = {choose_run(leg.as_of) for leg in legs}

    assert {run.hour for run in runs} == set(EXTENDED_CYCLES)
    assert all(choose_run(leg.as_of) <= leg.as_of - PUBLICATION_ALLOWANCE for leg in legs)


@needs_tape
def test_the_freeze_needs_disjoint_field_sets_inside_the_published_horizon(
    legs: list[SampleLeg],
) -> None:
    fields: dict[int, set[tuple[datetime, int]]] = {24: set(), 36: set()}
    city_day_fields = {24: 0, 36: 0}
    for leg in {(leg.station, leg.event_date, leg.lead_hours): leg for leg in legs}.values():
        plan = leg_fields(leg)
        fields[leg.lead_hours].update((plan.run, fxx) for fxx in plan.fxx)
        city_day_fields[leg.lead_hours] += len(plan.fxx)

    fxx24 = {fxx for _, fxx in fields[24]}
    fxx36 = {fxx for _, fxx in fields[36]}
    assert (min(fxx24), max(fxx24)) == (5, 35)
    assert (min(fxx36), max(fxx36)) == (17, 47)
    assert len(fields[24]) == 13421
    assert len(fields[36]) == 13232
    assert fields[24] & fields[36] == set()
    assert len(fields[24] | fields[36]) == 26653
    assert city_day_fields == {24: 55128, 36: 52992}


async def test_fetch_reads_the_idx_then_the_record_range() -> None:
    seen: list[str] = []

    async with httpx.AsyncClient(transport=httpx.MockTransport(bucket_handler(seen))) as client:
        body = await fetch_tmp2m_field(datetime(2025, 1, 15, tzinfo=UTC), 24, client)

    assert body == MESSAGE
    assert len(seen) == 2
    assert seen[0].endswith("hrrr.t00z.wrfsfcf24.grib2.idx")
    assert seen[1].endswith("hrrr.t00z.wrfsfcf24.grib2")


async def test_a_transient_reset_is_retried_against_the_same_url() -> None:
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        attempts.append(url)
        if url.endswith(".idx") and attempts.count(url) == 1:
            raise httpx.ConnectError("connection reset by peer")
        if url.endswith(".idx"):
            return httpx.Response(200, content=idx_body(24))
        return httpx.Response(206, content=MESSAGE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        body = await fetch_tmp2m_field(datetime(2025, 1, 15, tzinfo=UTC), 24, client)

    idx_attempts = [url for url in attempts if url.endswith(".idx")]
    assert body == MESSAGE
    assert len(idx_attempts) == 2
    assert len(set(idx_attempts)) == 1


@needs_tape
async def test_the_cache_key_downloads_each_field_once_for_all_seven_cities(
    legs: list[SampleLeg], tmp_path: Path
) -> None:
    day = [
        leg
        for leg in {(leg.station, leg.event_date, leg.lead_hours): leg for leg in legs}.values()
        if leg.event_date == COUNTED_DAY and leg.lead_hours == 24
    ]
    requests = sorted({(plan.run, fxx) for plan in map(leg_fields, day) for fxx in plan.fxx})
    per_city = sum(len(leg_fields(leg).fxx) for leg in day)
    stations = {leg.station: coordinates(leg.station) for leg in day}
    seen: list[str] = []

    async with httpx.AsyncClient(transport=httpx.MockTransport(bucket_handler(seen))) as client:
        report = await pull_fields(
            requests,
            stations,
            DecodedFieldCache(tmp_path / "fields.jsonl"),
            client,
            concurrency=4,
        )

    assert len(day) == 7
    assert per_city == 168
    assert len(requests) == 27
    assert report.downloaded == 27
    assert report.downloaded < per_city
    assert report.cache_hits == 0
    assert len([url for url in seen if not url.endswith(".idx")]) == 27
    assert report.missing == ()
    assert set(report.values[requests[0]]) == set(stations)


async def test_the_cache_resumes_an_interrupted_pull(tmp_path: Path) -> None:
    run = datetime(2025, 1, 15, tzinfo=UTC)
    requests = [(run, 5), (run, 6)]
    stations = {"KNYC": coordinates("KNYC")}
    path = tmp_path / "fields.jsonl"
    seen: list[str] = []

    async with httpx.AsyncClient(transport=httpx.MockTransport(bucket_handler(seen))) as client:
        first = await pull_fields(
            requests[:1], stations, DecodedFieldCache(path), client, concurrency=2
        )
        second = await pull_fields(
            requests, stations, DecodedFieldCache(path), client, concurrency=2
        )

    assert first.downloaded == 1
    assert second.downloaded == 1
    assert second.cache_hits == 1
    assert second.values[requests[0]] == first.values[requests[0]]


async def test_a_missing_field_is_counted_rather_than_substituted() -> None:
    run = datetime(2025, 1, 15, tzinfo=UTC)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if "wrfsfcf06" in url:
            return httpx.Response(404)
        if url.endswith(".idx"):
            return httpx.Response(200, content=idx_body(5))
        return httpx.Response(206, content=MESSAGE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await pull_fields(
            [(run, 5), (run, 6)],
            {"KNYC": coordinates("KNYC")},
            DecodedFieldCache(None),
            client,
            concurrency=2,
        )

    assert report.downloaded == 1
    assert len(report.missing) == 1
    assert "f006" in report.missing[0]
    assert (run, 6) not in report.values
    assert {url.split("/conus/")[1].split(".grib2")[0] for url in seen} == {
        "hrrr.t00z.wrfsfcf05",
        "hrrr.t00z.wrfsfcf06",
    }


def test_the_recorded_message_is_the_whole_tmp_record() -> None:
    assert len(MESSAGE) == RECORD_BYTES


@LIVE_ONLY
def test_the_live_idx_and_record_sizes_are_the_measured_bytes() -> None:
    with httpx.Client(timeout=60.0) as client:
        idx = client.get(IDX_URL)

    start, end = tmp2m_byte_range(idx.text, 24)
    assert idx.status_code == 200
    assert len(idx.content) == 10551
    assert end - start + 1 == RECORD_BYTES
