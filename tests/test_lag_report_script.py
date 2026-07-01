from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from bot.main import STATIONS
from bot.storage.sqlite import Base, make_engine
from scripts.lag_report import (
    BOOK_SOURCES,
    DEFAULT_LATENCY_CURVE_S,
    DEFAULT_NOTIONAL_CAP,
    GATE_STACK_S,
    R0_FRACTION_INVALID_MAX,
    R0_PASSING_SERIES,
    R0_WINDOW_END_EXCLUSIVE,
    R0_WINDOW_START,
    build_parser,
    run,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
UTC = timezone.utc

TICKER = "KXHIGHDEN-26JUN15-T70"
EVENT_DATE = date(2026, 6, 15)
T0 = datetime(2026, 6, 15, 18, 0, tzinfo=UTC)

IEM_BODY = "station,station_name,valid(UTC),tmpf\nDEN,DENVER,2026-06-15 18:00,75.0\n"
ACIS_BODY = '{"data": [["2026-06-15", "78"]]}'


def _make_db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    Base.metadata.create_all(make_engine(path))
    return path


def _db_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def _insert_market(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO markets "
        "(ticker, series, event_date, is_monthly, is_tail, strike_low, status, "
        "last_seen_at, created_at) VALUES (?, 'KXHIGHDEN', ?, 0, 1, '70', 'active', ?, ?)",
        (TICKER, EVENT_DATE.isoformat(), _db_ts(T0), _db_ts(T0)),
    )


def _insert_ws_book(conn: sqlite3.Connection) -> None:
    snapshot_at = _db_ts(T0 - timedelta(minutes=5))
    rows = [
        (TICKER, snapshot_at, 10, "yes", "0.400000", "10", 1),
        (TICKER, snapshot_at, 10, "no", "0.600000", "10", 1),
        (TICKER, snapshot_at, 10, "no", "0.040000", "7", 1),
        (TICKER, _db_ts(T0 + timedelta(seconds=83)), 11, "yes", "0.950000", "8", 0),
        (TICKER, _db_ts(T0 + timedelta(seconds=83)), 12, "no", "0.600000", "-10", 0),
        (
            TICKER,
            _db_ts(T0 + timedelta(seconds=83, milliseconds=100)),
            13,
            "yes",
            "0.100000",
            "1",
            0,
        ),
        (
            TICKER,
            _db_ts(T0 + timedelta(seconds=83, milliseconds=200)),
            14,
            "yes",
            "0.100000",
            "1",
            0,
        ),
    ]
    conn.executemany(
        "INSERT INTO ws_book_events "
        "(ticker, received_at, seq, side, price, size, is_snapshot, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, '2026-06-15 00:00:00.000000')",
        rows,
    )


def _insert_rest_snapshots(conn: sqlite3.Connection) -> None:
    stale = _db_ts(T0 - timedelta(minutes=5))
    fillable = _db_ts(T0 + timedelta(seconds=130))
    conn.executemany(
        "INSERT INTO orderbook_snapshots "
        "(ticker, snapshot_at, yes_ask, yes_bid, no_ask, no_bid, "
        "yes_ask_depth, yes_bid_depth, no_ask_depth, no_bid_depth, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (TICKER, stale, "0.40", "0.38", "0.62", "0.60", 10, 10, 10, 10, stale),
            (TICKER, fillable, "0.40", "0.38", "0.62", "0.60", 3, 3, 3, 3, fillable),
        ],
    )


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def _transport(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "mesonet.agron.iastate.edu" in host:
            return httpx.Response(200, text=IEM_BODY)
        if "rcc-acis.org" in host:
            return httpx.Response(200, text=ACIS_BODY)
        raise AssertionError(f"unexpected host: {host}")

    real_async_client = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("timeout", None)
        return real_async_client(transport=httpx.MockTransport(_transport))

    monkeypatch.setattr("scripts.lag_report.httpx.AsyncClient", _factory)


@pytest.fixture
def empty_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def _transport(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="station,station_name,valid(UTC),tmpf\n")

    real_async_client = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("timeout", None)
        return real_async_client(transport=httpx.MockTransport(_transport))

    monkeypatch.setattr("scripts.lag_report.httpx.AsyncClient", _factory)


def _namespace(db_path: Path, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "db": db_path,
        "start": EVENT_DATE,
        "end_date": date(2026, 6, 16),
        "series": ("KXHIGHDEN",),
        "book_source": "rest",
        "latency_total_s": DEFAULT_LATENCY_CURVE_S,
        "notional_cap": DEFAULT_NOTIONAL_CAP,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "lag_report.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--db" in result.stdout
    assert "--series" in result.stdout
    assert "--book-source" in result.stdout


def test_default_arg_values() -> None:
    args = build_parser().parse_args([])
    assert args.db == REPO_ROOT / "data" / "state.db"
    assert args.start == R0_WINDOW_START
    assert args.end_date == R0_WINDOW_END_EXCLUSIVE
    assert tuple(args.series) == R0_PASSING_SERIES
    assert args.book_source == "rest"
    assert tuple(args.latency_total_s) == DEFAULT_LATENCY_CURVE_S
    assert args.notional_cap == DEFAULT_NOTIONAL_CAP


def test_default_universe_is_the_full_r0_passing_set() -> None:
    assert len(build_parser().parse_args([]).series) == 20


def test_latency_curve_default_covers_the_preregistered_stacks() -> None:
    assert DEFAULT_LATENCY_CURVE_S == (30, 60, 90, 120)
    assert GATE_STACK_S == 90
    assert GATE_STACK_S in DEFAULT_LATENCY_CURVE_S


def test_latency_total_s_accepts_a_comma_separated_curve() -> None:
    args = build_parser().parse_args(["--latency-total-s", "45,75"])
    assert tuple(args.latency_total_s) == (45, 75)


def test_book_source_choices() -> None:
    assert BOOK_SOURCES == ("rest", "ws")
    assert build_parser().parse_args(["--book-source", "ws"]).book_source == "ws"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--book-source", "grib"])


def test_kxhighmia_is_in_the_r1_universe() -> None:
    assert "KXHIGHMIA" in R0_PASSING_SERIES
    assert "KXHIGHMIA" in build_parser().parse_args([]).series


def test_r0_passing_series_is_a_frozen_twenty_station_subset_of_stations() -> None:
    assert len(R0_PASSING_SERIES) == 20
    assert set(R0_PASSING_SERIES) <= set(STATIONS)
    assert len(set(R0_PASSING_SERIES)) == len(R0_PASSING_SERIES)


def test_r0_window_spans_fifteen_days() -> None:
    assert R0_WINDOW_START == date(2026, 7, 18)
    assert (R0_WINDOW_END_EXCLUSIVE - R0_WINDOW_START).days == 15


def test_r0_fraction_invalid_max_is_decimal() -> None:
    assert R0_FRACTION_INVALID_MAX == Decimal("0.5")
    assert type(R0_FRACTION_INVALID_MAX) is Decimal


def test_june_anchor_stations_are_inside_the_passing_universe() -> None:
    assert "KXHIGHDEN" in R0_PASSING_SERIES
    assert "KXHIGHCHI" in R0_PASSING_SERIES


def test_decimal_notional_cap_passthrough() -> None:
    args = build_parser().parse_args(["--notional-cap", "50.5"])
    assert args.notional_cap == Decimal("50.5")
    assert type(args.notional_cap) is Decimal


async def test_run_with_no_series_short_circuits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = await run(_namespace(_make_db(tmp_path), series=()))

    assert rc == 0
    assert "no series to process" in capsys.readouterr().out


async def test_run_empty_window_short_circuits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = await run(
        _namespace(_make_db(tmp_path), start=date(2026, 6, 15), end_date=date(2026, 6, 13))
    )

    assert rc == 0
    assert "empty window" in capsys.readouterr().out


async def test_run_against_empty_db_prints_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], empty_http: None
) -> None:
    rc = await run(_namespace(_make_db(tmp_path)))

    assert rc == 0
    out = capsys.readouterr().out
    assert "RAW (all events)" in out
    assert "GATE INPUTS" in out
    assert "CAPTURE" in out


async def test_rest_path_still_finds_the_stale_quote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_http: None
) -> None:
    db_path = _make_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    _insert_market(conn)
    _insert_rest_snapshots(conn)
    conn.commit()
    conn.close()

    rc = await run(_namespace(db_path))

    assert rc == 0
    out = capsys.readouterr().out
    assert "book_source=rest" in out
    assert "KXHIGHDEN  n_filled=1" in out
    for stack in DEFAULT_LATENCY_CURVE_S:
        assert f"-- {stack}s stack" in out


async def test_ws_path_reads_the_book_from_the_delta_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_http: None
) -> None:
    db_path = _make_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    _insert_market(conn)
    _insert_ws_book(conn)
    conn.commit()
    conn.close()

    rc = await run(_namespace(db_path, book_source="ws"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "book_source=ws" in out
    assert "events_found=1  events_used=1  gap_excluded=0  no_coverage=0" in out
    assert "cadence_s=0" in out
    assert "[snapshot_floor]" not in out

    gate_block = out.split("== GATE INPUTS")[1].split("== RAW")[0]
    assert "median_lag_s=83" in gate_block
    assert "p25_lag_s=83" in gate_block


async def test_ws_capture_fills_only_while_the_stale_ask_survives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_http: None
) -> None:
    db_path = _make_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    _insert_market(conn)
    _insert_ws_book(conn)
    conn.commit()
    conn.close()

    await run(_namespace(db_path, book_source="ws"))

    out = capsys.readouterr().out
    raw_block = out.split("== CAPTURE (RAW)")[1].split("== CAPTURE (NET OF FLOOR)")[0]
    filled = {
        stack: "POOLED  n_filled=1" in raw_block.split(f"-- {stack}s stack")[1].split("--")[0]
        for stack in DEFAULT_LATENCY_CURVE_S
    }
    assert filled == {30: True, 60: True, 90: False, 120: False}


async def test_ws_path_excludes_and_counts_gapped_events(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_http: None
) -> None:
    db_path = _make_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    _insert_market(conn)
    _insert_ws_book(conn)
    conn.execute(
        "INSERT INTO ws_gaps (ticker, detected_at, last_seq, reason, created_at) "
        "VALUES (?, ?, 12, 'seq_skip', '2026-06-15 00:00:00.000000')",
        (TICKER, _db_ts(T0 + timedelta(seconds=100))),
    )
    conn.commit()
    conn.close()

    rc = await run(_namespace(db_path, book_source="ws"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "events_found=1  events_used=0  gap_excluded=1  no_coverage=0" in out


async def test_ws_path_counts_events_without_ws_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_http: None
) -> None:
    db_path = _make_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    _insert_market(conn)
    conn.commit()
    conn.close()

    rc = await run(_namespace(db_path, book_source="ws"))

    assert rc == 0
    assert "events_found=1  events_used=0  gap_excluded=0  no_coverage=1" in capsys.readouterr().out
