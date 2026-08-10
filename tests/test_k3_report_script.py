from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from bot.lag.mechanism_rates import (
    DECODE_DEFECT,
    MISSING_OBSERVATION,
    ROUNDING_DIFFERENCE,
    WINDOW_DIFFERENCE,
)
from bot.replay.analysis_stations import HIGH
from scripts.k3_report import (
    build_parser,
    load_ladders,
    read_published_minutes,
    run,
)


UTC = timezone.utc
ACIS_HOST = "data.rcc-acis.org"
ONE_MINUTE_PATH = "/cgi-bin/request/asos1min.py"
METAR_PATH = "/cgi-bin/request/asos.py"

DAY_ONE = date(2026, 8, 10)
DAY_TWO = date(2026, 8, 11)
DENVER = "KXHIGHDEN"
PHOENIX = "KXHIGHTPHX"

DENVER_STRIKES = [86, 87, 88, 89, 90, 91, 92]
PHOENIX_STRIKES = [102, 103, 104, 105, 106, 107, 108]

# The third Denver row quotes a station name that carries a comma. The production reader splits on
# commas, so it lands on a field that is not a timestamp and drops the row without saying so.
DENVER_MINUTES = (
    ("DENVER INTL", "2026-08-10 06:30", "70.0"),
    ("DENVER INTL", "2026-08-10 12:00", "88.0"),
    ('"DENVER, CO"', "2026-08-10 20:00", "85.0"),
    ("DENVER INTL", "2026-08-11 06:30", "91.0"),
    ("DENVER INTL", "2026-08-11 12:00", "92.0"),
    ("DENVER INTL", "2026-08-12 03:00", "93.0"),
)
PHOENIX_MINUTES = (
    ("PHOENIX SKY HARBOR", "2026-08-10 12:00", "105.0"),
    ("PHOENIX SKY HARBOR", "2026-08-10 22:00", "106.0"),
    ("PHOENIX SKY HARBOR", "2026-08-11 12:00", "107.0"),
    ("PHOENIX SKY HARBOR", "2026-08-11 22:00", "108.0"),
)
ONE_MINUTE = {"DEN": DENVER_MINUTES, "PHX": PHOENIX_MINUTES}
METAR = {
    "KDEN": (("2026-08-10 12:00", "31.1"),),
    "KPHX": (("2026-08-10 12:00", "40.6"),),
}
SETTLES = {
    ("DEN", DAY_ONE): "88",
    ("DEN", DAY_TWO): "93",
    ("PHX", DAY_ONE): "106",
}


def one_minute_body(rows: tuple[tuple[str, str, str], ...], sid: str) -> str:
    header = "station,station_name,valid(UTC),tmpf\n"
    return header + "".join(f"{sid},{name},{stamp},{tmpf}\n" for name, stamp, tmpf in rows)


def metar_body(rows: tuple[tuple[str, str], ...], station: str) -> str:
    header = "station,valid,tmpc\n"
    return header + "".join(f"{station},{stamp},{tmpc}\n" for stamp, tmpc in rows)


def acis_body(sid: str, day: date, settles: dict[tuple[str, date], str]) -> dict:
    value = settles.get((sid, day))
    if value is None:
        return {"meta": {}, "data": []}
    return {"meta": {}, "data": [[day.isoformat(), value]]}


def transport(
    seen: list[httpx.Request],
    *,
    minutes: dict[str, tuple[tuple[str, str, str], ...]] | None = None,
    settles: dict[tuple[str, date], str] | None = None,
) -> httpx.MockTransport:
    rows = ONE_MINUTE if minutes is None else minutes
    published = SETTLES if settles is None else settles

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        query = parse_qs(request.url.query.decode())
        if request.url.path == ONE_MINUTE_PATH:
            sid = query["station"][0]
            return httpx.Response(200, text=one_minute_body(rows[sid], sid))
        if request.url.path == METAR_PATH:
            station = query["station"][0]
            return httpx.Response(200, text=metar_body(METAR[station], station))
        if request.url.host == ACIS_HOST:
            sid = query["sid"][0]
            day = date.fromisoformat(query["sdate"][0])
            return httpx.Response(200, json=acis_body(sid, day, published))
        raise AssertionError(f"unexpected request {request.url}")

    return httpx.MockTransport(handler)


def refusing(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise AssertionError(f"the cache should have answered {request.url}")

    return httpx.MockTransport(handler)


def write_universe(path: Path, roots: list[str]) -> Path:
    path.write_text(json.dumps({"passing": roots, "sha256": "0" * 64}))
    return path


def write_ladders(path: Path, ladders: dict[str, dict[str, list[int]]]) -> Path:
    path.write_text(
        json.dumps(
            {
                root: {
                    token: {
                        "listed_strikes": strikes,
                        "winner": f"{root}-{token}-T{strikes[0]}",
                        "strike_type": "greater",
                        "floor_strike": strikes[0],
                        "cap_strike": None,
                        "close_time": "2026-08-11T02:00:00Z",
                    }
                    for token, strikes in days.items()
                }
                for root, days in ladders.items()
            }
        )
    )
    return path


BOTH_LADDERS = {
    DENVER: {"26AUG10": DENVER_STRIKES, "26AUG11": DENVER_STRIKES},
    PHOENIX: {"26AUG10": PHOENIX_STRIKES, "26AUG11": PHOENIX_STRIKES},
}


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "universe": write_universe(tmp_path / "r0_universe.json", [DENVER, PHOENIX]),
        "ladders": write_ladders(tmp_path / "ladders.json", BOTH_LADDERS),
        "cache": tmp_path / "cache",
        "out": tmp_path / "k3.json",
    }


def args_for(paths: dict[str, Path]) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--start",
            DAY_ONE.isoformat(),
            "--end",
            DAY_TWO.isoformat(),
            "--universe",
            str(paths["universe"]),
            "--ladders",
            str(paths["ladders"]),
            "--cache",
            str(paths["cache"]),
            "--out",
            str(paths["out"]),
        ]
    )


def wire(monkeypatch: pytest.MonkeyPatch, mock: httpx.MockTransport) -> None:
    monkeypatch.setattr("scripts.k3_report.httpx.AsyncHTTPTransport", lambda *a, **k: mock)


def row_of(payload: dict, mechanism: str) -> dict:
    (row,) = [item for item in payload["rows"] if item["mechanism"] == mechanism]
    return row


def test_the_ladder_file_is_read_into_the_days_inside_the_window(paths: dict[str, Path]) -> None:
    ladders, notes = load_ladders(paths["ladders"], [DENVER, PHOENIX], DAY_ONE, DAY_ONE)

    assert [(day.root, day.event_date) for day in ladders] == [
        (DENVER, DAY_ONE),
        (PHOENIX, DAY_ONE),
    ]
    assert ladders[0].station == "KDEN"
    assert ladders[0].timezone == "America/Denver"
    assert ladders[0].ladder == HIGH
    assert ladders[0].extreme == "max"
    assert ladders[0].listed_strikes == tuple(DENVER_STRIKES)
    assert notes["event_days_outside_window"] == 2
    assert notes["ladder_rows"] == 2
    assert notes["roots_without_ladders"] == []


def test_a_root_the_station_map_does_not_carry_is_named_rather_than_dropped(
    paths: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="KXHIGHNOWHERE"):
        load_ladders(paths["ladders"], [DENVER, "KXHIGHNOWHERE"], DAY_ONE, DAY_TWO)


def test_a_ladder_root_outside_the_universe_is_counted_not_driven(
    paths: dict[str, Path],
) -> None:
    ladders, notes = load_ladders(paths["ladders"], [DENVER], DAY_ONE, DAY_TWO)

    assert {day.root for day in ladders} == {DENVER}
    assert notes["ladder_roots_outside_universe"] == [PHOENIX]


def test_the_second_reader_keeps_the_quoted_row_the_production_split_drops() -> None:
    published, unparsable = read_published_minutes(one_minute_body(DENVER_MINUTES, "DEN"))

    assert unparsable == 0
    assert len(published) == len(DENVER_MINUTES)
    assert published[datetime(2026, 8, 10, 20, tzinfo=UTC)] == Decimal("85.0")


def test_a_published_value_that_is_not_a_number_is_counted_not_swallowed() -> None:
    body = one_minute_body(
        (
            ("DENVER INTL", "2026-08-10 12:00", "88.0"),
            ("DENVER INTL", "2026-08-10 12:01", "M"),
            ("DENVER INTL", "2026-08-10 12:02", "warm"),
        ),
        "DEN",
    )

    published, unparsable = read_published_minutes(body)

    assert len(published) == 1
    assert unparsable == 1


async def test_the_four_rows_come_back_off_a_two_station_two_day_window(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx.Request] = []
    wire(monkeypatch, transport(seen))

    assert await run(args_for(paths)) == 0

    payload = json.loads(paths["out"].read_text())
    assert [row["mechanism"] for row in payload["rows"]] == [
        DECODE_DEFECT,
        ROUNDING_DIFFERENCE,
        WINDOW_DIFFERENCE,
        MISSING_OBSERVATION,
    ]
    assert payload["stations"] == ["KDEN", "KPHX"]

    decode = row_of(payload, DECODE_DEFECT)
    assert (decode["numerator"], decode["denominator"]) == (1, 10)
    assert decode["rate"] == "0.100000"
    assert decode["matched"] is True
    assert decode["detail"]["stations"][0] == {
        "station": "KDEN",
        "published_minutes": 6,
        "decoded_minutes": 5,
        "absent": 1,
        "differing": 0,
        "unparsable_published": 0,
        "decoded_not_published": 0,
    }
    assert decode["detail"]["metar_agreement"]["compared_minutes"] == 2
    assert decode["detail"]["metar_agreement"]["agree"] == 2
    assert decode["detail"]["metar_agreement"]["max_abs_delta_f"] == "0.08"

    rounding = row_of(payload, ROUNDING_DIFFERENCE)
    assert (rounding["numerator"], rounding["denominator"]) == (1, 3)
    assert rounding["detail"]["no_settle"] == 1
    assert rounding["detail"]["separations"] == [
        {
            "root": DENVER,
            "station": "KDEN",
            "event_date": DAY_ONE.isoformat(),
            "extreme": "max",
            "observed_f": "91.0",
            "acis_f": "88",
            "separating_strikes": [88, 89, 90],
        }
    ]

    window = row_of(payload, WINDOW_DIFFERENCE)
    assert (window["numerator"], window["denominator"]) == (1, 4)
    assert window["detail"]["cities_on_dst"] == ["KDEN"]
    assert window["detail"]["cities_off_dst"] == ["KPHX"]
    assert window["detail"]["by_city"] == [
        {"station": "KDEN", "station_days": 2, "separated": 1, "on_dst": True},
        {"station": "KPHX", "station_days": 2, "separated": 0, "on_dst": False},
    ]
    assert window["detail"]["separations"][0]["standard_f"] == "91.0"
    assert window["detail"]["separations"][0]["wall_clock_f"] == "88.0"

    missing = row_of(payload, MISSING_OBSERVATION)
    assert missing["measurable"] is False
    assert missing["threshold"] == "0.10"
    assert "recorder tape" in missing["reason"]

    assert payload["selection"]["selected"] == DECODE_DEFECT
    assert payload["selection"]["undetermined"] is False
    assert payload["selection"]["unmeasured"] == [MISSING_OBSERVATION]
    assert "unaffected" in payload["selection"]["note"]


async def test_nothing_matching_leaves_the_selection_pending_the_unmeasured_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        "universe": write_universe(tmp_path / "r0_universe.json", [PHOENIX]),
        "ladders": write_ladders(tmp_path / "ladders.json", {PHOENIX: BOTH_LADDERS[PHOENIX]}),
        "cache": tmp_path / "cache",
        "out": tmp_path / "k3.json",
    }
    settles = {("PHX", DAY_ONE): "106", ("PHX", DAY_TWO): "108"}
    wire(monkeypatch, transport([], settles=settles))

    assert await run(args_for(paths)) == 0

    payload = json.loads(paths["out"].read_text())
    assert [row["matched"] for row in payload["rows"]] == [False, False, False, False]
    assert row_of(payload, DECODE_DEFECT)["numerator"] == 0
    assert row_of(payload, ROUNDING_DIFFERENCE)["denominator"] == 2
    assert row_of(payload, WINDOW_DIFFERENCE)["denominator"] == 2
    assert payload["selection"] == {
        "selected": None,
        "undetermined": True,
        "unmeasured": [MISSING_OBSERVATION],
        "unmeasured_ahead": [MISSING_OBSERVATION],
        "note": (
            "no measured row reached its threshold and missing_observation went unmeasured, so "
            "the selection is undetermined pending those rows"
        ),
    }


async def test_a_populated_cache_answers_a_rerun_without_a_single_request(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    first: list[httpx.Request] = []
    wire(monkeypatch, transport(first))
    assert await run(args_for(paths)) == 0
    first_payload = json.loads(paths["out"].read_text())

    second: list[httpx.Request] = []
    wire(monkeypatch, refusing(second))
    assert await run(args_for(paths)) == 0

    assert first
    assert second == []
    assert first_payload["fetched_payloads"] == len(first)
    assert json.loads(paths["out"].read_text())["fetched_payloads"] == 0
    assert row_of(json.loads(paths["out"].read_text()), DECODE_DEFECT) == row_of(
        first_payload, DECODE_DEFECT
    )


async def test_a_failing_pull_stops_the_run_rather_than_shrinking_the_denominator(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == ONE_MINUTE_PATH:
            return httpx.Response(503, text="upstream is down")
        raise AssertionError(f"unexpected request {request.url}")

    wire(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        await run(args_for(paths))

    assert not paths["out"].exists()
