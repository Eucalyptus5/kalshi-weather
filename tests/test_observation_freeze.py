from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from bot.lag.observation_freeze import (
    F2_STATIONS,
    INDEX_NAME,
    MAX,
    READING_SOURCE,
    ObservationSidecar,
    StationDay,
    index_payload,
    pull_observations,
    read_observation_index,
    read_observation_sidecar,
    sidecar_payload,
    write_observation_index,
)
from bot.lag.r0_universe import freeze_digest
from bot.main import STATIONS
from bot.markets.observation_window import observation_window
from bot.observations.basis_check import IEM_1MIN_URL, fetch_iem_1min_asos_archive
from bot.observations.metar import StationObservation
from bot.validation.reconcile import ACIS_URL


UTC = timezone.utc
IEM_HOST = "mesonet.agron.iastate.edu"
ACIS_HOST = "data.rcc-acis.org"
IEM_HEADER = "station,station_name,valid(UTC),tmpf"

DEN = "KDEN"
NYC = "KNYC"
PHX = "KPHX"

DAY_ONE = date(2026, 8, 2)
DAY_TWO = date(2026, 8, 3)
DAY_THREE = date(2026, 8, 4)
DAYS = (DAY_ONE, DAY_TWO)
OBSERVED_AT = datetime(2026, 8, 20, 15, 30, tzinfo=UTC)
DEFAULT_HIGH = "95"

NO_ROWS = object()
SERVER_ERROR = object()


def stamp(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def iem_body(station: str, rows: Sequence[tuple[datetime, str]]) -> str:
    lines = [IEM_HEADER]
    for valid_time, tmpf in rows:
        lines.append(f"{station[1:]},{station} ASOS,{valid_time.strftime('%Y-%m-%d %H:%M')},{tmpf}")
    return "\n".join(lines) + "\n"


def two_station_bodies() -> dict[str, str]:
    return {
        DEN: iem_body(
            DEN,
            (
                (stamp(DAY_ONE, 6, 59), "70.0"),
                (stamp(DAY_ONE, 7), "71.0"),
                (stamp(DAY_ONE, 20), "88.0"),
                (stamp(DAY_TWO, 7), "72.0"),
                (stamp(DAY_TWO, 18), "90.0"),
            ),
        ),
        NYC: iem_body(
            NYC,
            (
                (stamp(DAY_ONE, 5), "60.0"),
                (stamp(DAY_ONE, 18), "84.0"),
                (stamp(DAY_TWO, 5), "61.0"),
            ),
        ),
    }


def handler_for(
    bodies: Mapping[str, str],
    highs: Mapping[tuple[str, date], object] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    values = dict(highs or {})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == IEM_HOST:
            station = "K" + request.url.params["station"]
            if station not in bodies:
                return httpx.Response(404, text="unknown station")
            return httpx.Response(200, text=bodies[station])
        station = "K" + request.url.params["sid"]
        event_date = date.fromisoformat(request.url.params["sdate"])
        value = values.get((station, event_date), DEFAULT_HIGH)
        if value is SERVER_ERROR:
            return httpx.Response(503, json={"error": "acis is down"})
        if value is NO_ROWS:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"data": [[event_date.isoformat(), value]]})

    return handler


def transport_for(
    bodies: Mapping[str, str],
    highs: Mapping[tuple[str, date], object] | None = None,
    seen: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    served = handler_for(bodies, highs)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return served(request)

    return httpx.MockTransport(handler)


def frozen_dir(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    return directory


def test_the_station_set_is_the_repos_own_map() -> None:
    assert F2_STATIONS == {config.station: config.timezone for config in STATIONS.values()}
    assert len(F2_STATIONS) == 20
    assert all(station.startswith("K") for station in F2_STATIONS)


def test_the_pull_freezes_one_file_per_station_over_every_event_day(tmp_path: Path) -> None:
    digests = pull_observations([DEN, NYC], DAYS, tmp_path, transport_for(two_station_bodies()))

    assert sorted(digests) == [DEN, NYC]
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"{DEN}.json", f"{NYC}.json"]
    for station in (DEN, NYC):
        sidecar = read_observation_sidecar(tmp_path / f"{station}.json")
        assert sidecar.station == station
        assert sidecar.timezone == F2_STATIONS[station]
        assert sidecar.sha256 == digests[station]
        assert sorted(sidecar.days) == list(DAYS)
        assert {day.extreme for day in sidecar.days.values()} == {MAX}
        assert {day.station for day in sidecar.days.values()} == {station}
        assert {day.acis_f for day in sidecar.days.values()} == {Decimal(DEFAULT_HIGH)}


def test_each_station_is_asked_once_for_its_span_and_once_a_day_for_its_extreme(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    pull_observations([DEN, NYC], DAYS, tmp_path, transport_for(two_station_bodies(), seen=seen))

    archive = [request for request in seen if request.url.host == IEM_HOST]
    extremes = [request for request in seen if request.url.host == ACIS_HOST]
    assert sorted(request.url.params["station"] for request in archive) == ["DEN", "NYC"]
    assert [request.url.params["sts"] for request in archive] == ["2026-08-02T00:00Z"] * 2
    assert len(extremes) == 4
    assert {request.url.params["sid"] for request in extremes} == {"DEN", "NYC"}
    assert {request.url.params["elems"] for request in extremes} == {"maxt"}


def test_the_readings_land_in_ascending_order_whatever_order_the_archive_sent(
    tmp_path: Path,
) -> None:
    rows = (
        (stamp(DAY_ONE, 20), "88.0"),
        (stamp(DAY_ONE, 7), "71.0"),
        (stamp(DAY_ONE, 12), "80.0"),
    )

    pull_observations([DEN], (DAY_ONE,), tmp_path, transport_for({DEN: iem_body(DEN, rows)}))

    readings = read_observation_sidecar(tmp_path / f"{DEN}.json").days[DAY_ONE].readings
    assert [row.valid_time for row in readings] == [
        stamp(DAY_ONE, 7),
        stamp(DAY_ONE, 12),
        stamp(DAY_ONE, 20),
    ]
    assert [row.temp_f for row in readings] == [Decimal("71.0"), Decimal("80.0"), Decimal("88.0")]


def test_the_window_is_half_open_at_both_ends(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))

    sidecar = read_observation_sidecar(tmp_path / f"{DEN}.json")
    first = sidecar.days[DAY_ONE]
    second = sidecar.days[DAY_TWO]

    assert first.window_end == second.window_start
    assert len(first.readings) == 2
    assert len(second.readings) == 2
    assert [row.valid_time for row in first.readings] == [stamp(DAY_ONE, 7), stamp(DAY_ONE, 20)]
    assert [row.valid_time for row in second.readings] == [stamp(DAY_TWO, 7), stamp(DAY_TWO, 18)]
    frozen = {row.valid_time for day in sidecar.days.values() for row in day.readings}
    assert stamp(DAY_ONE, 6, 59) not in frozen


def test_the_window_is_the_stations_own_standard_time_not_utc_midnight(tmp_path: Path) -> None:
    bodies = {
        PHX: iem_body(PHX, ((stamp(DAY_ONE, 15), "105.0"),)),
        NYC: iem_body(NYC, ((stamp(DAY_ONE, 15), "85.0"),)),
    }

    pull_observations([PHX, NYC], (DAY_ONE,), tmp_path, transport_for(bodies))

    phoenix = read_observation_sidecar(tmp_path / f"{PHX}.json").days[DAY_ONE]
    newyork = read_observation_sidecar(tmp_path / f"{NYC}.json").days[DAY_ONE]

    assert phoenix.window_start == datetime(2026, 8, 2, 7, tzinfo=UTC)
    assert phoenix.window_end == datetime(2026, 8, 3, 7, tzinfo=UTC)
    assert newyork.window_start == datetime(2026, 8, 2, 5, tzinfo=UTC)
    assert newyork.window_end == datetime(2026, 8, 3, 5, tzinfo=UTC)
    assert newyork.window_start != datetime(2026, 8, 2, 4, tzinfo=UTC)
    assert phoenix.window_start != datetime(2026, 8, 2, tzinfo=UTC)
    assert len(phoenix.readings) == 1
    assert len(newyork.readings) == 1


def test_a_reading_survives_the_round_trip_as_the_fetcher_built_it(tmp_path: Path) -> None:
    bodies = two_station_bodies()

    async def straight_from_the_archive() -> list[StationObservation]:
        async with httpx.AsyncClient(transport=transport_for(bodies)) as client:
            return await fetch_iem_1min_asos_archive(DEN, DAY_ONE, DAY_TWO, client)

    pull_observations([DEN], DAYS, tmp_path, transport_for(bodies))
    fetched = asyncio.run(straight_from_the_archive())

    sidecar = read_observation_sidecar(tmp_path / f"{DEN}.json")
    start, _ = observation_window(F2_STATIONS[DEN], DAY_ONE)
    _, end = observation_window(F2_STATIONS[DEN], DAY_TWO)
    carried = [row for day in sorted(sidecar.days) for row in sidecar.days[day].readings]

    assert carried == [row for row in fetched if start <= row.valid_time < end]
    assert {row.source for row in carried} == {READING_SOURCE}
    for row in carried:
        assert isinstance(row.temp_f, Decimal)
        assert row.publication_time == row.valid_time
        assert row.valid_time.utcoffset() == timedelta(0)


def test_the_frozen_file_carries_strings_where_the_numbers_are(tmp_path: Path) -> None:
    highs = {(DEN, DAY_TWO): "M"}

    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies(), highs))

    payload = json.loads((tmp_path / f"{DEN}.json").read_text())
    assert payload["source"] == READING_SOURCE
    assert payload["days"][0]["acis_f"] == DEFAULT_HIGH
    assert payload["days"][1]["acis_f"] is None
    assert {type(row["temp_f"]) for day in payload["days"] for row in day["readings"]} == {str}


def test_the_sidecar_reproduces_its_digest_on_a_second_read(tmp_path: Path) -> None:
    written = pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))

    path = tmp_path / f"{DEN}.json"
    assert read_observation_sidecar(path).sha256 == written[DEN]
    assert read_observation_sidecar(path).sha256 == read_observation_sidecar(path).sha256


def test_a_tampered_sidecar_is_refused(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))
    path = tmp_path / f"{DEN}.json"
    path.write_text(path.read_text().replace("88.0", "98.0"))

    with pytest.raises(ValueError, match="sha256"):
        read_observation_sidecar(path)


def test_a_sidecar_carrying_no_sha256_is_refused(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))
    path = tmp_path / f"{DEN}.json"
    payload = json.loads(path.read_text())
    payload.pop("sha256")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="sha256"):
        read_observation_sidecar(path)


def test_the_pull_refuses_to_overwrite_a_frozen_station_file(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))

    with pytest.raises(FileExistsError):
        pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))


def test_the_order_the_stations_and_days_were_asked_for_does_not_move_the_digest(
    tmp_path: Path,
) -> None:
    bodies = two_station_bodies()
    forward = frozen_dir(tmp_path, "forward")
    reverse = frozen_dir(tmp_path, "reverse")

    ahead = pull_observations([DEN, NYC], DAYS, forward, transport_for(bodies))
    behind = pull_observations([NYC, DEN], DAYS[::-1], reverse, transport_for(bodies))

    assert ahead == behind
    assert write_observation_index(forward, OBSERVED_AT) == write_observation_index(
        reverse, OBSERVED_AT
    )
    for name in (f"{DEN}.json", f"{NYC}.json", INDEX_NAME):
        assert (forward / name).read_text() == (reverse / name).read_text()


@pytest.mark.parametrize("value", ("M", NO_ROWS), ids=("missing_token", "no_rows"))
def test_a_day_with_no_official_extreme_is_frozen_as_none_and_named(
    tmp_path: Path, value: object
) -> None:
    highs = {(DEN, DAY_TWO): value}

    pull_observations([DEN, NYC], DAYS, tmp_path, transport_for(two_station_bodies(), highs))
    write_observation_index(tmp_path, OBSERVED_AT)

    sidecar = read_observation_sidecar(tmp_path / f"{DEN}.json")
    index = read_observation_index(tmp_path)
    assert sorted(sidecar.days) == list(DAYS)
    assert sidecar.days[DAY_TWO].acis_f is None
    assert sidecar.days[DAY_ONE].acis_f == Decimal(DEFAULT_HIGH)
    assert index.stations[DEN].missing_acis == (DAY_TWO,)
    assert index.stations[NYC].missing_acis == ()


def test_a_station_whose_archive_carries_no_rows_freezes_empty_and_is_still_named(
    tmp_path: Path,
) -> None:
    bodies = two_station_bodies() | {NYC: iem_body(NYC, ())}

    pull_observations([DEN, NYC], DAYS, tmp_path, transport_for(bodies))
    write_observation_index(tmp_path, OBSERVED_AT)

    sidecar = read_observation_sidecar(tmp_path / f"{NYC}.json")
    index = read_observation_index(tmp_path)
    assert sorted(sidecar.days) == list(DAYS)
    assert {day.readings for day in sidecar.days.values()} == {()}
    assert index.stations[NYC].event_days == 2
    assert index.stations[NYC].decoded_minutes == {DAY_ONE: 0, DAY_TWO: 0}


def test_a_missing_temperature_is_not_a_decoded_minute(tmp_path: Path) -> None:
    rows = (
        (stamp(DAY_ONE, 7), "71.0"),
        (stamp(DAY_ONE, 8), "M"),
        (stamp(DAY_ONE, 20), "88.0"),
    )

    pull_observations([DEN], (DAY_ONE,), tmp_path, transport_for({DEN: iem_body(DEN, rows)}))
    write_observation_index(tmp_path, OBSERVED_AT)

    readings = read_observation_sidecar(tmp_path / f"{DEN}.json").days[DAY_ONE].readings
    assert len(readings) == 2
    assert stamp(DAY_ONE, 8) not in {row.valid_time for row in readings}
    assert read_observation_index(tmp_path).stations[DEN].decoded_minutes == {DAY_ONE: 2}


def test_an_archive_the_host_refuses_freezes_nothing(tmp_path: Path) -> None:
    bodies = {DEN: two_station_bodies()[DEN]}

    with pytest.raises(httpx.HTTPStatusError):
        pull_observations([DEN, NYC], DAYS, tmp_path, transport_for(bodies))

    assert list(tmp_path.iterdir()) == []


def test_an_extreme_the_host_refuses_freezes_nothing(tmp_path: Path) -> None:
    highs = {(NYC, DAY_TWO): SERVER_ERROR}

    with pytest.raises(httpx.HTTPStatusError):
        pull_observations([DEN, NYC], DAYS, tmp_path, transport_for(two_station_bodies(), highs))

    assert list(tmp_path.iterdir()) == []


def test_the_index_pins_the_window_the_sources_and_every_station_file(tmp_path: Path) -> None:
    digests = pull_observations([DEN, NYC], DAYS, tmp_path, transport_for(two_station_bodies()))

    written = write_observation_index(tmp_path, OBSERVED_AT)

    index = read_observation_index(tmp_path)
    assert index.sha256 == written
    assert index.observed_at == OBSERVED_AT
    assert index.start_date == DAY_ONE
    assert index.end_date == DAY_TWO
    assert index.extreme == MAX
    assert index.source == READING_SOURCE
    assert index.iem_url == IEM_1MIN_URL
    assert index.acis_url == ACIS_URL
    assert {station: row.sha256 for station, row in index.stations.items()} == digests
    assert index.stations[DEN].timezone == "America/Denver"
    assert index.stations[NYC].timezone == "America/New_York"
    assert index.stations[DEN].decoded_minutes == {DAY_ONE: 2, DAY_TWO: 2}
    assert index.stations[NYC].decoded_minutes == {DAY_ONE: 2, DAY_TWO: 1}
    assert index.stations[DEN].event_days == 2


def test_the_index_refuses_to_overwrite_a_frozen_one(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))
    write_observation_index(tmp_path, OBSERVED_AT)

    with pytest.raises(FileExistsError):
        write_observation_index(tmp_path, OBSERVED_AT)


def test_a_tampered_index_is_refused(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))
    write_observation_index(tmp_path, OBSERVED_AT)
    path = tmp_path / INDEX_NAME
    payload = json.loads(path.read_text())
    payload["stations"][0]["event_days"] = 99
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="sha256"):
        read_observation_index(tmp_path)


def test_an_index_carrying_no_sha256_is_refused(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))
    write_observation_index(tmp_path, OBSERVED_AT)
    path = tmp_path / INDEX_NAME
    payload = json.loads(path.read_text())
    payload.pop("sha256")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="sha256"):
        read_observation_index(tmp_path)


def test_the_index_digest_moves_when_a_station_file_moves(tmp_path: Path) -> None:
    left = frozen_dir(tmp_path, "left")
    right = frozen_dir(tmp_path, "right")
    bodies = two_station_bodies()
    moved = bodies | {
        DEN: iem_body(DEN, ((stamp(DAY_ONE, 7), "71.0"), (stamp(DAY_ONE, 21), "89.0")))
    }

    pull_observations([DEN, NYC], DAYS, left, transport_for(bodies))
    pull_observations([DEN, NYC], DAYS, right, transport_for(moved))

    assert read_observation_sidecar(left / f"{NYC}.json").sha256 == (
        read_observation_sidecar(right / f"{NYC}.json").sha256
    )
    assert write_observation_index(left, OBSERVED_AT) != write_observation_index(right, OBSERVED_AT)


def test_the_index_names_every_station_the_map_carries(tmp_path: Path) -> None:
    bodies = {
        station: iem_body(station, ((stamp(DAY_ONE, 15), "90.0"),)) for station in F2_STATIONS
    }

    pull_observations(sorted(F2_STATIONS), (DAY_ONE,), tmp_path, transport_for(bodies))
    write_observation_index(tmp_path, OBSERVED_AT)

    index = read_observation_index(tmp_path)
    assert sorted(index.stations) == sorted(config.station for config in STATIONS.values())
    assert len(index.stations) == 20
    assert {row.decoded_minutes[DAY_ONE] for row in index.stations.values()} == {1}
    assert {row.timezone for row in index.stations.values()} == {
        config.timezone for config in STATIONS.values()
    }


def test_the_order_the_days_arrive_in_does_not_move_the_sidecar_payload(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))
    sidecar = read_observation_sidecar(tmp_path / f"{DEN}.json")
    days = [sidecar.days[DAY_ONE], sidecar.days[DAY_TWO]]

    ahead = sidecar_payload(DEN, sidecar.timezone, days)
    behind = sidecar_payload(DEN, sidecar.timezone, days[::-1])

    assert [row["event_date"] for row in behind["days"]] == [
        DAY_ONE.isoformat(),
        DAY_TWO.isoformat(),
    ]
    assert ahead == behind
    assert freeze_digest(ahead) == freeze_digest(behind)


def test_the_order_the_sidecars_arrive_in_does_not_move_the_index_payload(tmp_path: Path) -> None:
    pull_observations([DEN, NYC], DAYS, tmp_path, transport_for(two_station_bodies()))
    ahead = {
        station: read_observation_sidecar(tmp_path / f"{station}.json") for station in (DEN, NYC)
    }
    behind = {station: ahead[station] for station in (NYC, DEN)}

    assert list(ahead) != list(behind)
    assert [row["station"] for row in index_payload(OBSERVED_AT, behind)["stations"]] == [DEN, NYC]
    assert index_payload(OBSERVED_AT, ahead) == index_payload(OBSERVED_AT, behind)
    assert freeze_digest(index_payload(OBSERVED_AT, ahead)) == freeze_digest(
        index_payload(OBSERVED_AT, behind)
    )


def test_the_span_the_archive_is_asked_for_is_the_windows_bounds_not_the_ask_order(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    pull_observations([DEN], DAYS[::-1], tmp_path, transport_for(two_station_bodies(), seen=seen))

    archive = [request for request in seen if request.url.host == IEM_HOST]
    assert [request.url.params["sts"] for request in archive] == ["2026-08-02T00:00Z"]
    assert [request.url.params["ets"] for request in archive] == ["2026-08-05T00:00Z"]
    assert sorted(read_observation_sidecar(tmp_path / f"{DEN}.json").days) == list(DAYS)


def test_the_stations_are_swept_in_a_settled_order_whatever_order_they_were_asked_for(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []
    bodies = {
        station: iem_body(station, ((stamp(DAY_ONE, 15), "90.0"),)) for station in F2_STATIONS
    }

    digests = pull_observations(
        sorted(F2_STATIONS, reverse=True), (DAY_ONE,), tmp_path, transport_for(bodies, seen=seen)
    )

    assert list(digests) == sorted(F2_STATIONS)
    assert [
        "K" + request.url.params["station"] for request in seen if request.url.host == IEM_HOST
    ] == sorted(F2_STATIONS)


def test_the_frozen_files_carry_the_indent_the_repos_other_sidecars_carry(tmp_path: Path) -> None:
    pull_observations([DEN], DAYS, tmp_path, transport_for(two_station_bodies()))
    write_observation_index(tmp_path, OBSERVED_AT)

    for path in (tmp_path / f"{DEN}.json", tmp_path / INDEX_NAME):
        text = path.read_text()
        assert text == json.dumps(json.loads(text), indent=1)
        assert text.splitlines()[1].startswith(' "')


def hand_built_day(event_date: date, minutes: int, acis_f: Decimal | None) -> StationDay:
    start, end = observation_window(F2_STATIONS[DEN], event_date)
    return StationDay(
        station=DEN,
        event_date=event_date,
        extreme=MAX,
        window_start=start,
        window_end=end,
        readings=tuple(
            StationObservation(
                station=DEN,
                valid_time=start + timedelta(minutes=offset),
                publication_time=start + timedelta(minutes=offset),
                temp_f=Decimal("70.0"),
                is_special=False,
                raw="",
                source=READING_SOURCE,
            )
            for offset in range(minutes)
        ),
        acis_f=acis_f,
    )


def hand_built_sidecar(order: Sequence[date]) -> ObservationSidecar:
    minutes = {DAY_ONE: 1, DAY_TWO: 2, DAY_THREE: 3}
    extremes = {DAY_ONE: None, DAY_TWO: Decimal(DEFAULT_HIGH), DAY_THREE: None}
    return ObservationSidecar(
        station=DEN,
        timezone=F2_STATIONS[DEN],
        days={day: hand_built_day(day, minutes[day], extremes[day]) for day in order},
        sha256="0" * 64,
    )


def test_the_order_a_sidecars_days_arrive_in_does_not_move_the_index_payload() -> None:
    ascending = hand_built_sidecar((DAY_ONE, DAY_TWO, DAY_THREE))
    shuffled = hand_built_sidecar((DAY_THREE, DAY_ONE, DAY_TWO))

    assert list(ascending.days) != list(shuffled.days)

    ahead = index_payload(OBSERVED_AT, {DEN: ascending})
    behind = index_payload(OBSERVED_AT, {DEN: shuffled})

    assert behind["stations"][0]["missing_acis"] == [DAY_ONE.isoformat(), DAY_THREE.isoformat()]
    assert behind["stations"][0]["decoded_minutes"] == {
        DAY_ONE.isoformat(): 1,
        DAY_TWO.isoformat(): 2,
        DAY_THREE.isoformat(): 3,
    }
    assert ahead == behind
    assert freeze_digest(ahead) == freeze_digest(behind)
