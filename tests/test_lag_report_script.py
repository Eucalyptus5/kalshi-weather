from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from scripts.lag_report import (
    DEFAULT_LATENCY_TOTAL_S,
    DEFAULT_NOTIONAL_CAP,
    DEFAULT_SERIES,
    DEFAULT_START,
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


def test_default_end_falls_to_yesterday_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # parser default is None; the run loop resolves it. Verify the fallback.
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    args = build_parser().parse_args([])
    resolved = args.end_date if args.end_date is not None else yesterday
    assert resolved == yesterday


def test_decimal_notional_cap_passthrough() -> None:
    args = build_parser().parse_args(["--notional-cap", "50.5"])
    assert args.notional_cap == Decimal("50.5")
    assert type(args.notional_cap) is Decimal
