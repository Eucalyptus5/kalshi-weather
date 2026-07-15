import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.r0_universe import (
    AGREE,
    DISAGREE,
    LOCK_CARVE_OUT,
    Coverage,
    R0Universe,
    freeze_digest,
    freeze_universe,
    lock_dependent_series,
    read_recorded_coverage,
    universe_payload,
    write_universe,
)
from bot.replay.run_scope import EVENT_DAYS_SCHEMA
from scripts.freeze_r0_universe import build_parser, run
from scripts.lag_report import R0_FRACTION_INVALID_MAX, R0_PASSING_SERIES


UTC = timezone.utc

THRESHOLD = Decimal("0.5")
PASSING = ("KXHIGHDEN", "KXHIGHCHI", LOCK_CARVE_OUT, "KXHIGHNY")
LADDER = 6
CITY_DAYS = 56


def coverage(
    cities: tuple[str, ...] = PASSING,
    *,
    widths: tuple[int, ...] = (LADDER,),
    city_days: int = CITY_DAYS,
) -> Coverage:
    return Coverage(
        cities=tuple(sorted(cities)), ladder_widths=widths, in_scope_city_days=city_days
    )


def universe(
    *,
    threshold: Decimal = THRESHOLD,
    passing: tuple[str, ...] = PASSING,
    tape: Coverage | None = None,
) -> R0Universe:
    return freeze_universe(
        fraction_invalid_max=threshold,
        passing=passing,
        coverage=tape if tape is not None else coverage(),
    )


def event_day_row(series: str, event_date: date, *, tickers: int, in_scope: bool) -> dict:
    midnight = datetime(event_date.year, event_date.month, event_date.day, tzinfo=UTC)
    return {
        "series": series,
        "station": series.replace("KXHIGH", "K"),
        "timezone": "America/New_York",
        "event_date": event_date,
        "window_start": midnight,
        "window_end": midnight,
        "tickers": tickers,
        "ladder_rows": tickers * 3,
        "first_event_at": midnight,
        "last_event_at": midnight,
        "covered": True,
        "evaluable": in_scope,
        "in_scope": in_scope,
        "day_index": 1 if in_scope else 0,
        "split": "discovery" if in_scope else "",
        "excluded_us": 0,
        "span_us": 0,
    }


def test_the_lock_dependent_set_is_the_passing_set_less_miami() -> None:
    lock = lock_dependent_series(PASSING)

    assert LOCK_CARVE_OUT in PASSING
    assert set(lock) == set(PASSING) - {LOCK_CARVE_OUT}
    assert len(lock) == len(PASSING) - 1
    assert lock == tuple(sorted(lock))


def test_the_carve_out_never_reaches_the_unconditional_universe() -> None:
    payload = universe_payload(universe())

    assert LOCK_CARVE_OUT in payload["passing"]
    assert LOCK_CARVE_OUT in payload["recorded"]
    assert LOCK_CARVE_OUT not in payload["lock_dependent"]
    assert payload["lock_carve_out"] == LOCK_CARVE_OUT


def test_the_recorded_set_and_the_passing_set_stay_distinct_fields() -> None:
    tape = coverage(PASSING + ("KXHIGHTSEA",))

    payload = universe_payload(universe(tape=tape))

    assert payload["recorded"] != payload["passing"]
    assert "KXHIGHTSEA" in payload["recorded"]
    assert "KXHIGHTSEA" not in payload["passing"]
    assert "KXHIGHTSEA" not in payload["lock_dependent"]


def test_a_loosened_threshold_moves_the_digest() -> None:
    tight = universe_payload(universe())
    loose = universe_payload(universe(threshold=Decimal("0.6")))

    assert freeze_digest(tight) == freeze_digest(universe_payload(universe()))
    assert freeze_digest(tight) != freeze_digest(loose)
    assert len(freeze_digest(tight)) == 64


def test_the_digest_moves_when_a_station_joins_or_leaves_the_passing_set() -> None:
    frozen = freeze_digest(universe_payload(universe()))
    joined = freeze_digest(universe_payload(universe(passing=PASSING + ("KXHIGHTSEA",))))
    left = freeze_digest(universe_payload(universe(passing=PASSING[:-1])))

    assert frozen != joined
    assert frozen != left
    assert joined != left


def test_the_threshold_round_trips_as_an_exact_decimal(tmp_path: Path) -> None:
    path = tmp_path / "r0_universe.json"

    write_universe(path, universe())

    raw = json.loads(path.read_text())["fraction_invalid_max"]
    assert isinstance(raw, str)
    assert Decimal(raw) == THRESHOLD
    assert Decimal(raw).as_tuple() == THRESHOLD.as_tuple()


def test_reconciliation_agrees_when_the_tape_carries_every_passing_station() -> None:
    frozen = universe()

    assert frozen.reconciliation == AGREE
    assert frozen.recorded_not_passing == ()
    assert frozen.passing_not_recorded == ()


def test_a_recorded_station_the_gate_never_passed_is_reported_not_reconciled_away() -> None:
    frozen = universe(tape=coverage(PASSING + ("KXHIGHTSEA",)))

    assert frozen.reconciliation == DISAGREE
    assert frozen.recorded_not_passing == ("KXHIGHTSEA",)
    assert frozen.passing_not_recorded == ()
    assert frozen.passing == tuple(sorted(PASSING))
    assert "KXHIGHTSEA" in frozen.recorded


def test_a_passing_station_absent_from_the_tape_is_reported() -> None:
    frozen = universe(tape=coverage(PASSING[:-1]))

    assert frozen.reconciliation == DISAGREE
    assert frozen.passing_not_recorded == (PASSING[-1],)
    assert frozen.recorded_not_passing == ()
    assert PASSING[-1] in frozen.passing
    assert PASSING[-1] not in frozen.recorded


def test_the_frozen_file_carries_its_own_digest_and_refuses_a_second_write(tmp_path: Path) -> None:
    path = tmp_path / "r0_universe.json"
    frozen = universe()

    digest = write_universe(path, frozen)

    payload = json.loads(path.read_text())
    assert payload.pop("sha256") == digest
    assert payload == universe_payload(frozen)
    assert freeze_digest(payload) == digest

    with pytest.raises(FileExistsError, match=str(path)):
        write_universe(path, frozen)


def test_the_recorded_coverage_is_read_off_the_event_day_table(tmp_path: Path) -> None:
    path = tmp_path / "event_days.parquet"
    rows = [
        event_day_row("KXHIGHDEN", date(2026, 7, 17), tickers=4, in_scope=False),
        event_day_row("KXHIGHDEN", date(2026, 7, 19), tickers=LADDER, in_scope=True),
        event_day_row("KXHIGHNY", date(2026, 7, 19), tickers=LADDER, in_scope=True),
        event_day_row("KXHIGHTSEA", date(2026, 7, 19), tickers=LADDER, in_scope=True),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA), path)

    tape = read_recorded_coverage(path)

    assert tape.cities == ("KXHIGHDEN", "KXHIGHNY", "KXHIGHTSEA")
    assert tape.ladder_widths == (LADDER,)
    assert tape.in_scope_city_days == 3


def test_the_script_freezes_the_constants_the_gate_actually_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    days = tmp_path / "event_days.parquet"
    rows = [
        event_day_row(series, date(2026, 7, 19), tickers=LADDER, in_scope=True)
        for series in R0_PASSING_SERIES
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA), days)
    out = tmp_path / "run_scope"

    code = run(build_parser().parse_args(["--event-days", str(days), "--out", str(out)]))

    payload = json.loads((out / "r0_universe.json").read_text())
    assert code == 0
    assert payload["fraction_invalid_max"] == str(R0_FRACTION_INVALID_MAX)
    assert payload["passing"] == sorted(R0_PASSING_SERIES)
    assert len(payload["lock_dependent"]) == len(R0_PASSING_SERIES) - 1
    assert payload["reconciliation"] == AGREE
    assert payload["sha256"] in capsys.readouterr().out


def test_the_script_reports_a_disagreement_without_raising(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    days = tmp_path / "event_days.parquet"
    rows = [
        event_day_row(series, date(2026, 7, 19), tickers=LADDER, in_scope=True)
        for series in R0_PASSING_SERIES[:-1]
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA), days)
    out = tmp_path / "run_scope"

    code = run(build_parser().parse_args(["--event-days", str(days), "--out", str(out)]))

    payload = json.loads((out / "r0_universe.json").read_text())
    assert code == 1
    assert payload["reconciliation"] == DISAGREE
    assert payload["passing_not_recorded"] == [R0_PASSING_SERIES[-1]]
    assert payload["passing"] == sorted(R0_PASSING_SERIES)
    assert DISAGREE in capsys.readouterr().out
