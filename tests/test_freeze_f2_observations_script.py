from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path

import httpx
import pytest

from bot.lag.observation_freeze import F2_STATIONS, INDEX_NAME, read_observation_index
from scripts.freeze_f2_observations import REPO_ROOT, build_parser, main, run
from tests.test_observation_freeze import (
    DAY_ONE,
    DAY_TWO,
    NO_ROWS,
    handler_for,
    iem_body,
    stamp,
)
from tests.test_tape_studies import argument_flags


SCRIPT = REPO_ROOT / "scripts" / "freeze_f2_observations.py"

Offline = Callable[..., list[httpx.Request]]


def every_station_body() -> dict[str, str]:
    return {
        station: iem_body(station, ((stamp(DAY_ONE, 15), "90.0"), (stamp(DAY_TWO, 15), "91.0")))
        for station in F2_STATIONS
    }


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Offline:
    client = httpx.AsyncClient

    def install(
        bodies: Mapping[str, str],
        highs: Mapping[tuple[str, date], object] | None = None,
    ) -> list[httpx.Request]:
        seen: list[httpx.Request] = []
        served = handler_for(bodies, highs)

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return served(request)

        def factory(**kwargs: object) -> httpx.AsyncClient:
            return client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr("bot.lag.observation_freeze.httpx.AsyncClient", factory)
        return seen

    return install


def argv_parts(out: Path, start: date = DAY_ONE, end: date = DAY_TWO) -> dict[str, str]:
    return {"--start": start.isoformat(), "--end": end.isoformat(), "--out": str(out)}


def argv_for(out: Path, start: date = DAY_ONE, end: date = DAY_TWO) -> list[str]:
    return [item for pair in argv_parts(out, start, end).items() for item in pair]


def without(out: Path, flag: str) -> list[str]:
    parts = argv_parts(out)
    del parts[flag]
    return [item for pair in parts.items() for item in pair]


def frozen(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
    *,
    bodies: Mapping[str, str] | None = None,
    highs: Mapping[tuple[str, date], object] | None = None,
) -> dict:
    offline(every_station_body() if bodies is None else bodies, highs)
    assert main(argv_for(tmp_path / "observations")) == 0
    return json.loads(capsys.readouterr().out)


def test_the_freeze_writes_a_file_for_every_station_and_names_its_digest(
    tmp_path: Path, offline: Offline, capsys: pytest.CaptureFixture[str]
) -> None:
    printed = frozen(tmp_path, offline, capsys)

    out = tmp_path / "observations"
    assert sorted(printed["stations"]) == sorted(F2_STATIONS)
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [INDEX_NAME, *(f"{station}.json" for station in F2_STATIONS)]
    )
    index = read_observation_index(out)
    assert printed["sha256"] == index.sha256
    for station, row in printed["stations"].items():
        assert row["sha256"] == index.stations[station].sha256
        assert row["timezone"] == F2_STATIONS[station]


def test_the_printed_window_is_the_inclusive_span_it_was_asked_for(
    tmp_path: Path, offline: Offline, capsys: pytest.CaptureFixture[str]
) -> None:
    printed = frozen(tmp_path, offline, capsys)

    assert printed["start_date"] == DAY_ONE.isoformat()
    assert printed["end_date"] == DAY_TWO.isoformat()
    assert printed["extreme"] == "max"
    assert printed["source"] == "iem_1min_asos_archive"
    assert printed["event_days"] == 40
    assert {row["event_days"] for row in printed["stations"].values()} == {2}


def test_the_printed_minutes_are_the_decoded_ones(
    tmp_path: Path, offline: Offline, capsys: pytest.CaptureFixture[str]
) -> None:
    bodies = every_station_body() | {
        "KDEN": iem_body(
            "KDEN",
            (
                (stamp(DAY_ONE, 15), "90.0"),
                (stamp(DAY_ONE, 16), "M"),
                (stamp(DAY_TWO, 15), "91.0"),
            ),
        )
    }

    printed = frozen(tmp_path, offline, capsys, bodies=bodies)

    assert printed["stations"]["KDEN"]["decoded_minutes"] == 2
    assert printed["decoded_minutes"] == 40


def test_a_day_with_no_readings_and_a_day_with_no_extreme_are_both_named(
    tmp_path: Path, offline: Offline, capsys: pytest.CaptureFixture[str]
) -> None:
    bodies = every_station_body() | {"KNYC": iem_body("KNYC", ((stamp(DAY_ONE, 15), "80.0"),))}
    highs = {("KDEN", DAY_TWO): "M", ("KPHX", DAY_ONE): NO_ROWS}

    printed = frozen(tmp_path, offline, capsys, bodies=bodies, highs=highs)

    assert printed["stations"]["KNYC"]["empty_days"] == [DAY_TWO.isoformat()]
    assert printed["empty_days"] == 1
    assert printed["stations"]["KDEN"]["missing_acis"] == [DAY_TWO.isoformat()]
    assert printed["stations"]["KPHX"]["missing_acis"] == [DAY_ONE.isoformat()]
    assert printed["missing_acis"] == 2
    assert printed["stations"]["KNYC"]["missing_acis"] == []


def test_a_second_freeze_refuses_to_overwrite_the_first(
    tmp_path: Path, offline: Offline, capsys: pytest.CaptureFixture[str]
) -> None:
    frozen(tmp_path, offline, capsys)
    offline(every_station_body())

    with pytest.raises(FileExistsError):
        main(argv_for(tmp_path / "observations"))


def test_a_window_that_runs_backwards_is_refused(tmp_path: Path, offline: Offline) -> None:
    offline(every_station_body())
    out = tmp_path / "observations"

    with pytest.raises(ValueError, match="precedes"):
        run(build_parser().parse_args(argv_for(out, start=DAY_TWO, end=DAY_ONE)))

    assert not out.exists()


@pytest.mark.parametrize("flag", ("--start", "--end", "--out"))
def test_the_freeze_will_not_run_without_the_flags_that_pin_it(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(without(tmp_path / "observations", flag))


def test_the_freeze_names_no_output_path_of_its_own() -> None:
    source = SCRIPT.read_text()

    assert argument_flags(source) == {"start", "end", "out"}
    assert "data/" not in source
