from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.snapshot_loader import PROD_ERA_START, load_snapshots


UTC = timezone.utc


_SCHEMA = """
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
CREATE INDEX ix_orderbook_snapshots_ticker_snapshot_at
    ON orderbook_snapshots(ticker, snapshot_at);
"""


def _build_db(rows: list[tuple]) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    if rows:
        conn.executemany(
            "INSERT INTO orderbook_snapshots "
            "(ticker, snapshot_at, yes_ask, yes_bid, no_ask, no_bid, "
            "yes_ask_depth, yes_bid_depth, no_ask_depth, no_bid_depth, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    conn.commit()
    conn.close()
    return path


def _row(
    ticker: str,
    snapshot_at: datetime,
    *,
    yes_ask: str = "0.500000",
    yes_bid: str = "0.400000",
    no_ask: str = "0.600000",
    no_bid: str = "0.500000",
    yes_ask_depth: int | None = 10,
    yes_bid_depth: int | None = 10,
    no_ask_depth: int | None = 10,
    no_bid_depth: int | None = 10,
) -> tuple:
    return (
        ticker,
        snapshot_at.isoformat(),
        yes_ask,
        yes_bid,
        no_ask,
        no_bid,
        yes_ask_depth,
        yes_bid_depth,
        no_ask_depth,
        no_bid_depth,
        snapshot_at.isoformat(),
    )


def test_clamps_to_prod_era_cutover() -> None:
    pre1 = PROD_ERA_START - timedelta(hours=2)
    pre2 = PROD_ERA_START - timedelta(seconds=1)
    post = PROD_ERA_START + timedelta(seconds=10)
    db = _build_db([_row("TEST", pre1), _row("TEST", pre2), _row("TEST", post)])
    try:
        rows = load_snapshots(
            db,
            "TEST",
            start=pre1 - timedelta(days=1),
            end=PROD_ERA_START + timedelta(days=1),
        )
        assert len(rows) == 1
        assert rows[0].snapshot_at == post
    finally:
        db.unlink()


def test_window_filters_both_ends() -> None:
    base = PROD_ERA_START + timedelta(days=1)
    t1 = base - timedelta(hours=1)
    t2 = base + timedelta(minutes=30)
    t3 = base + timedelta(hours=2)
    db = _build_db([_row("TEST", t1), _row("TEST", t2), _row("TEST", t3)])
    try:
        rows = load_snapshots(db, "TEST", start=base, end=base + timedelta(hours=1))
        assert [r.snapshot_at for r in rows] == [t2]
    finally:
        db.unlink()


def test_ticker_filter() -> None:
    t = PROD_ERA_START + timedelta(days=1)
    db = _build_db([_row("A", t), _row("B", t + timedelta(seconds=10))])
    try:
        rows = load_snapshots(db, "A", start=t - timedelta(hours=1), end=t + timedelta(hours=1))
        assert len(rows) == 1
        assert rows[0].ticker == "A"
    finally:
        db.unlink()


def test_all_fields_roundtrip_with_decimal_and_none_depth() -> None:
    t = PROD_ERA_START + timedelta(days=2)
    db = _build_db(
        [
            _row(
                "TEST",
                t,
                yes_ask="0.420000",
                yes_bid="0.410000",
                no_ask="0.590000",
                no_bid="0.580000",
                yes_ask_depth=None,
                yes_bid_depth=7,
                no_ask_depth=None,
                no_bid_depth=3,
            )
        ]
    )
    try:
        rows = load_snapshots(db, "TEST", start=t - timedelta(hours=1), end=t + timedelta(hours=1))
        assert len(rows) == 1
        row = rows[0]
        assert type(row.yes_ask) is Decimal
        assert type(row.yes_bid) is Decimal
        assert row.yes_ask == Decimal("0.420000")
        assert row.yes_bid == Decimal("0.410000")
        assert row.no_ask == Decimal("0.590000")
        assert row.no_bid == Decimal("0.580000")
        assert row.yes_ask_depth is None
        assert row.yes_bid_depth == 7
        assert row.no_ask_depth is None
        assert row.no_bid_depth == 3
    finally:
        db.unlink()


def test_empty_db_returns_empty() -> None:
    db = _build_db([])
    try:
        rows = load_snapshots(
            db,
            "TEST",
            start=PROD_ERA_START,
            end=PROD_ERA_START + timedelta(days=1),
        )
        assert rows == []
    finally:
        db.unlink()


def test_ordering_ascending_by_snapshot_at() -> None:
    base = PROD_ERA_START + timedelta(days=1)
    later = base + timedelta(minutes=30)
    earlier = base + timedelta(minutes=10)
    middle = base + timedelta(minutes=20)
    db = _build_db([_row("TEST", later), _row("TEST", earlier), _row("TEST", middle)])
    try:
        rows = load_snapshots(db, "TEST", start=base, end=base + timedelta(hours=1))
        assert [r.snapshot_at for r in rows] == [earlier, middle, later]
    finally:
        db.unlink()


def test_loader_does_not_hold_write_lock() -> None:
    t = PROD_ERA_START + timedelta(days=1)
    db = _build_db([_row("TEST", t)])
    try:
        load_snapshots(db, "TEST", start=t - timedelta(hours=1), end=t + timedelta(hours=1))
        conn = sqlite3.connect(str(db), timeout=2)
        conn.execute(
            "INSERT INTO orderbook_snapshots "
            "(ticker, snapshot_at, yes_ask, yes_bid, no_ask, no_bid, created_at) "
            "VALUES ('X', '2026-06-14T00:00:00+00:00', '0.5', '0.4', '0.6', '0.5', "
            "'2026-06-14T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()
    finally:
        db.unlink()


def test_read_only_uri_rejects_writes_via_loader() -> None:
    t = PROD_ERA_START + timedelta(days=1)
    db = _build_db([_row("TEST", t)])
    try:
        conn = sqlite3.connect(f"file:{db.absolute()}?mode=ro", uri=True)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO orderbook_snapshots "
                "(ticker, snapshot_at, yes_ask, yes_bid, no_ask, no_bid, created_at) "
                "VALUES ('X', '2026-06-14T00:00:00+00:00', '0.5', '0.4', '0.6', '0.5', "
                "'2026-06-14T00:00:00+00:00')"
            )
        conn.close()
    finally:
        db.unlink()
