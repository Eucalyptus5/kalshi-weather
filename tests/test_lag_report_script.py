from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from bot.main import STATIONS
from scripts.lag_report import (
    DEFAULT_LATENCY_TOTAL_S,
    DEFAULT_NOTIONAL_CAP,
    DEFAULT_SERIES,
    DEFAULT_START,
    EXCLUDED_SERIES,
    R0_FRACTION_INVALID_MAX,
    R0_PASSING_SERIES,
    R0_WINDOW_END_EXCLUSIVE,
    R0_WINDOW_START,
    build_parser,
    filter_series,
    run,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


_SCHEMA = """
CREATE TABLE markets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(64) NOT NULL UNIQUE,
    series VARCHAR(32) NOT NULL,
    event_date DATE NOT NULL,
    is_monthly BOOLEAN NOT NULL,
    is_tail BOOLEAN NOT NULL,
    strike_low VARCHAR NOT NULL,
    strike_high VARCHAR,
    close_time DATETIME,
    status VARCHAR(16) NOT NULL,
    last_seen_at DATETIME NOT NULL,
    created_at DATETIME NOT NULL
);
CREATE TABLE orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(64) NOT NULL,
    snapshot_at DATETIME NOT NULL,
    yes_ask VARCHAR NOT NULL,
    yes_bid VARCHAR NOT NULL,
    no_ask VARCHAR NOT NULL,
    no_bid VARCHAR NOT NULL,
    yes_ask_depth INTEGER,
    yes_bid_depth INTEGER,
    no_ask_depth INTEGER,
    no_bid_depth INTEGER,
    created_at DATETIME NOT NULL
);
"""


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


def test_default_arg_values() -> None:
    args = build_parser().parse_args([])
    assert args.db == REPO_ROOT / "data" / "state.db"
    assert args.start == DEFAULT_START
    assert args.end_date is None
    assert tuple(args.series) == DEFAULT_SERIES
    assert args.latency_total_s == DEFAULT_LATENCY_TOTAL_S
    assert args.notional_cap == DEFAULT_NOTIONAL_CAP


def test_filter_series_drops_kxhighmia_with_warning() -> None:
    kept, warnings = filter_series(("KXHIGHDEN", "KXHIGHMIA", "KXHIGHCHI"))
    assert kept == ("KXHIGHDEN", "KXHIGHCHI")
    assert len(warnings) == 1
    assert "basis_excluded: KXHIGHMIA" in warnings[0]
    assert "L1.5" in warnings[0]


def test_filter_series_keeps_all_when_no_excluded() -> None:
    kept, warnings = filter_series(("KXHIGHDEN", "KXHIGHCHI"))
    assert kept == ("KXHIGHDEN", "KXHIGHCHI")
    assert warnings == []


async def test_run_emits_warning_for_kxhighmia_and_completes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()

    try:
        end_date = date(2026, 6, 14)
        namespace = argparse.Namespace(
            db=db_path,
            start=date(2026, 6, 13),
            end_date=end_date,
            series=("KXHIGHMIA",),
            latency_total_s=DEFAULT_LATENCY_TOTAL_S,
            notional_cap=DEFAULT_NOTIONAL_CAP,
        )
        rc = await run(namespace)
        assert rc == 0
        captured = capsys.readouterr()
        assert "basis_excluded: KXHIGHMIA" in captured.out
        assert "no series to process" in captured.out
    finally:
        db_path.unlink()


async def test_run_against_empty_db_prints_report(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()

    def _no_network_transport(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="station,station_name,valid(UTC),tmpf\n")

    real_async_client = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("timeout", None)
        return real_async_client(transport=httpx.MockTransport(_no_network_transport))

    monkeypatch.setattr("scripts.lag_report.httpx.AsyncClient", _factory)

    try:
        namespace = argparse.Namespace(
            db=db_path,
            start=date(2026, 6, 13),
            end_date=date(2026, 6, 14),
            series=("KXHIGHDEN",),
            latency_total_s=DEFAULT_LATENCY_TOTAL_S,
            notional_cap=DEFAULT_NOTIONAL_CAP,
        )
        rc = await run(namespace)
        assert rc == 0
        captured = capsys.readouterr()
        assert "RAW (all events)" in captured.out
        assert "CAPTURE" in captured.out
    finally:
        db_path.unlink()


async def test_run_empty_window_short_circuits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()

    try:
        namespace = argparse.Namespace(
            db=db_path,
            start=date(2026, 6, 15),
            end_date=date(2026, 6, 13),
            series=("KXHIGHDEN",),
            latency_total_s=DEFAULT_LATENCY_TOTAL_S,
            notional_cap=DEFAULT_NOTIONAL_CAP,
        )
        rc = await run(namespace)
        assert rc == 0
        captured = capsys.readouterr()
        assert "empty window" in captured.out
    finally:
        db_path.unlink()


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


def test_anchor_series_are_inside_the_passing_universe() -> None:
    assert all(s in R0_PASSING_SERIES for s in DEFAULT_SERIES)


def test_kxhighmia_passes_r0_but_remains_in_excluded_series() -> None:
    assert "KXHIGHMIA" in R0_PASSING_SERIES
    assert "KXHIGHMIA" in EXCLUDED_SERIES


def test_decimal_notional_cap_passthrough() -> None:
    args = build_parser().parse_args(["--notional-cap", "50.5"])
    assert args.notional_cap == Decimal("50.5")
    assert type(args.notional_cap) is Decimal


async def test_capture_finds_stale_quote_for_single_event_ticker(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticker = "KXHIGHDEN-26JUN15-T70"
    event_date = date(2026, 6, 15)
    t0_iso = "2026-06-15 18:00:00.000000"
    stale_iso = "2026-06-15 17:55:00.000000"
    fillable_iso = "2026-06-15 18:02:10.000000"

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO markets "
        "(ticker, series, event_date, is_monthly, is_tail, strike_low, status, "
        "last_seen_at, created_at) VALUES (?, 'KXHIGHDEN', ?, 0, 1, '70', 'active', ?, ?)",
        (ticker, event_date.isoformat(), t0_iso, t0_iso),
    )
    conn.executemany(
        "INSERT INTO orderbook_snapshots "
        "(ticker, snapshot_at, yes_ask, yes_bid, no_ask, no_bid, "
        "yes_ask_depth, yes_bid_depth, no_ask_depth, no_bid_depth, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (ticker, stale_iso, "0.40", "0.38", "0.62", "0.60", 10, 10, 10, 10, stale_iso),
            (ticker, fillable_iso, "0.40", "0.38", "0.62", "0.60", 3, 3, 3, 3, fillable_iso),
        ],
    )
    conn.commit()
    conn.close()

    iem_body = "station,station_name,valid(UTC),tmpf\nDEN,DENVER,2026-06-15 18:00,75.0\n"
    acis_body = '{"data": [["2026-06-15", "78"]]}'

    def _transport(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "mesonet.agron.iastate.edu" in host:
            return httpx.Response(200, text=iem_body)
        if "rcc-acis.org" in host:
            return httpx.Response(200, text=acis_body)
        raise AssertionError(f"unexpected host: {host}")

    real_async_client = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("timeout", None)
        return real_async_client(transport=httpx.MockTransport(_transport))

    monkeypatch.setattr("scripts.lag_report.httpx.AsyncClient", _factory)

    try:
        namespace = argparse.Namespace(
            db=db_path,
            start=event_date,
            end_date=date(2026, 6, 16),
            series=("KXHIGHDEN",),
            latency_total_s=DEFAULT_LATENCY_TOTAL_S,
            notional_cap=DEFAULT_NOTIONAL_CAP,
        )
        rc = await run(namespace)
        assert rc == 0
        captured = capsys.readouterr()
        assert "KXHIGHDEN  n_filled=1" in captured.out
    finally:
        db_path.unlink()
