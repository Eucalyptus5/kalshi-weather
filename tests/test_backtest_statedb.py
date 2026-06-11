import csv
import json
import sqlite3
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from bot.backtest.forecast_replay import StationSpec
from bot.backtest.normalize import CanonicalSnapshot
from bot.backtest.statedb import (
    StateDbForecastReplay,
    fetch_decision_snapshots,
    run_survival,
    settlement_rows,
)
from bot.forecast.cdf import EnsembleCDF

_LEAD = timedelta(hours=25)
_STALENESS = timedelta(minutes=30)
_START = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)
_CLOSE = datetime(2026, 1, 16, 6, 0, tzinfo=timezone.utc)
_DECISION = _CLOSE - _LEAD
_TAIL_TICKER = "KXHIGHDEN-26JAN15-T85"
_MEMBERS = np.array([59.5, 61.5, 63.5])
_STATION = StationSpec(
    name="KDEN", latitude=39.8466, longitude=-104.6562, timezone="America/Denver"
)

_SCHEMA = """
CREATE TABLE markets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(64) UNIQUE,
    series VARCHAR(32),
    close_time DATETIME,
    status VARCHAR(16)
);
CREATE TABLE orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(64),
    snapshot_at DATETIME,
    yes_ask NUMERIC(10, 6),
    yes_bid NUMERIC(10, 6),
    no_ask NUMERIC(10, 6),
    no_bid NUMERIC(10, 6),
    yes_ask_depth INTEGER,
    yes_bid_depth INTEGER,
    no_ask_depth INTEGER,
    no_bid_depth INTEGER
);
CREATE INDEX ix_orderbook_snapshots_ticker_snapshot_at
    ON orderbook_snapshots (ticker, snapshot_at);
CREATE TABLE forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station VARCHAR(16),
    run_time DATETIME,
    valid_date DATE,
    members_json VARCHAR
);
"""


def stored(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S.%f")


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(_SCHEMA)
    yield db
    db.close()


def add_market(
    db: sqlite3.Connection, ticker: str, close: datetime | None, status: str = "active"
) -> None:
    db.execute(
        "INSERT INTO markets (ticker, series, close_time, status) VALUES (?, ?, ?, ?)",
        (ticker, ticker.split("-", 1)[0], None if close is None else stored(close), status),
    )


def add_book(
    db: sqlite3.Connection,
    ticker: str,
    at: datetime,
    *,
    yes_ask: float = 0.15,
    yes_bid: float = 0.12,
    depth: int = 100,
) -> None:
    db.execute(
        """
        INSERT INTO orderbook_snapshots
            (ticker, snapshot_at, yes_ask, yes_bid, no_ask, no_bid,
             yes_ask_depth, yes_bid_depth, no_ask_depth, no_bid_depth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            stored(at),
            yes_ask,
            yes_bid,
            1 - yes_bid,
            1 - yes_ask,
            depth,
            depth,
            depth,
            depth,
        ),
    )


def add_forecast(
    db: sqlite3.Connection,
    station: str,
    run_time: datetime,
    valid_date: date,
    members: list[float],
) -> None:
    db.execute(
        "INSERT INTO forecasts (station, run_time, valid_date, members_json) VALUES (?, ?, ?, ?)",
        (station, stored(run_time), valid_date.isoformat(), json.dumps(members)),
    )


def make_settled(ticker: str, close: datetime | None, result: str = "no") -> CanonicalSnapshot:
    return CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-", 1)[0],
        status="finalized",
        result=result,
        yes_ask=Decimal("0.15"),
        yes_bid=Decimal("0.12"),
        no_ask=Decimal("0.88"),
        no_bid=Decimal("0.85"),
        last_price=Decimal("0.12"),
        volume=Decimal(0),
        volume_24h=Decimal(0),
        open_interest=Decimal(0),
        close_time=close,
    )


class FixedReplay:
    def __init__(self) -> None:
        self._cdf = EnsembleCDF.from_members(_MEMBERS, smoothing=0.65)

    async def replay(self, station: StationSpec, valid_date: date, as_of: datetime) -> EnsembleCDF:
        return self._cdf


def test_fetch_decision_snapshots_picks_latest_qualifying_row(conn: sqlite3.Connection) -> None:
    add_market(conn, _TAIL_TICKER, _CLOSE)
    add_book(conn, _TAIL_TICKER, _DECISION - timedelta(minutes=8))
    add_book(conn, _TAIL_TICKER, _DECISION - timedelta(minutes=5), yes_ask=0.16)
    add_book(conn, _TAIL_TICKER, _DECISION + timedelta(minutes=30), yes_ask=0.99)

    stale = "KXHIGHDEN-26JAN15-T80"
    add_market(conn, stale, _CLOSE)
    add_book(conn, stale, _DECISION - timedelta(hours=2))

    closeless = "KXHIGHDEN-26JAN15-T75"
    add_market(conn, closeless, None)
    add_book(conn, closeless, _DECISION - timedelta(minutes=5))

    rows = fetch_decision_snapshots(conn, _LEAD, _START, _STALENESS)

    assert [r.snap.ticker for r in rows] == [_TAIL_TICKER]
    assert rows[0].snapshot_at == _DECISION - timedelta(minutes=5)
    assert rows[0].snap.yes_ask == Decimal("0.16")
    assert rows[0].snap.event_ticker == "KXHIGHDEN-26JAN15"
    assert rows[0].snap.series_ticker == "KXHIGHDEN"
    assert rows[0].snap.status == "active"
    assert rows[0].snap.result == ""


def test_fetch_decision_snapshots_excludes_rows_before_start(conn: sqlite3.Connection) -> None:
    add_market(conn, _TAIL_TICKER, _CLOSE)
    add_book(conn, _TAIL_TICKER, _DECISION - timedelta(minutes=20))

    late_start = _DECISION - timedelta(minutes=10)
    assert fetch_decision_snapshots(conn, _LEAD, late_start, _STALENESS) == []
    assert len(fetch_decision_snapshots(conn, _LEAD, _START, _STALENESS)) == 1


def test_decision_snapshots_are_aware_and_keep_zero_depths(conn: sqlite3.Connection) -> None:
    add_market(conn, _TAIL_TICKER, _CLOSE)
    add_book(conn, _TAIL_TICKER, _DECISION - timedelta(minutes=5), depth=0)

    [row] = fetch_decision_snapshots(conn, _LEAD, _START, _STALENESS)

    assert row.snapshot_at.tzinfo == timezone.utc
    assert row.snap.close_time == _CLOSE
    assert row.snap.close_time.tzinfo == timezone.utc
    assert row.snap.yes_bid_size == Decimal("0")
    assert row.snap.no_bid_size == Decimal("0")


async def test_forecast_replay_picks_latest_run_before_as_of(conn: sqlite3.Connection) -> None:
    valid = date(2026, 1, 15)
    runs = [
        (datetime(2026, 1, 14, 0, 0, tzinfo=timezone.utc), [50.0, 52.0, 54.0]),
        (datetime(2026, 1, 14, 6, 0, tzinfo=timezone.utc), [60.0, 62.0, 64.0]),
        (datetime(2026, 1, 14, 18, 0, tzinfo=timezone.utc), [70.0, 72.0, 74.0]),
    ]
    for run_time, members in runs:
        add_forecast(conn, "KDEN", run_time, valid, members)

    cdf = await StateDbForecastReplay(conn).replay(
        _STATION, valid, datetime(2026, 1, 14, 12, 0, tzinfo=timezone.utc)
    )

    assert list(cdf.members) == [60.0, 62.0, 64.0]


async def test_forecast_replay_missing_raises_lookup_error(conn: sqlite3.Connection) -> None:
    add_forecast(
        conn, "KDEN", datetime(2026, 1, 14, 0, 0, tzinfo=timezone.utc), date(2026, 1, 14), [50.0]
    )

    with pytest.raises(LookupError, match="KDEN"):
        await StateDbForecastReplay(conn).replay(
            _STATION, date(2026, 1, 15), datetime(2026, 1, 14, 12, 0, tzinfo=timezone.utc)
        )


def test_settlement_rows_pin_snapshot_at_to_close() -> None:
    snap = make_settled(_TAIL_TICKER, _CLOSE)

    [row] = settlement_rows([snap])

    assert row.snapshot_at == _CLOSE
    assert row.snap is snap


def test_settlement_rows_reject_missing_close() -> None:
    with pytest.raises(ValueError, match=_TAIL_TICKER):
        settlement_rows([make_settled(_TAIL_TICKER, None)])


async def test_run_survival_end_to_end(conn: sqlite3.Connection, tmp_path: Path) -> None:
    add_market(conn, _TAIL_TICKER, _CLOSE)
    add_book(conn, _TAIL_TICKER, _DECISION - timedelta(minutes=1))
    snapshots = fetch_decision_snapshots(conn, _LEAD, _START, _STALENESS)
    snapshots += settlement_rows([make_settled(_TAIL_TICKER, _CLOSE, result="no")])
    out_dir = tmp_path / "report"

    verdict, summary, runs, logs = await run_survival(snapshots, FixedReplay(), _LEAD, out_dir)

    assert summary == out_dir / "summary.md"
    assert summary.exists()
    assert (out_dir / "orders.csv").exists()
    assert (out_dir / "orders.parquet").exists()
    assert len(runs) == 4
    assert all(r.window == "out_of_sample" for r in runs)
    assert {(r.strategy, r.bankroll) for r in runs} == {
        ("tails", Decimal("20000")),
        ("tails", Decimal("500")),
        ("edge", Decimal("20000")),
        ("edge", Decimal("500")),
    }
    assert set(logs) == {"tails:20000", "tails:500", "edge:20000", "edge:500"}
    text = summary.read_text()
    assert "$20000" in text
    assert "$500" in text
    with (out_dir / "orders.csv").open(newline="") as f:
        order_rows = list(csv.DictReader(f))
    assert order_rows
    assert any(r["depth_source"] == "snapshot" for r in order_rows)
    assert verdict.final == "survive"


async def test_run_survival_zero_orders_is_insufficient_data(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    add_market(conn, _TAIL_TICKER, _CLOSE)
    add_book(conn, _TAIL_TICKER, _DECISION - timedelta(minutes=1), depth=0)
    snapshots = fetch_decision_snapshots(conn, _LEAD, _START, _STALENESS)
    snapshots += settlement_rows([make_settled(_TAIL_TICKER, _CLOSE)])

    verdict, summary, runs, logs = await run_survival(
        snapshots, FixedReplay(), _LEAD, tmp_path / "report"
    )

    assert verdict.final == "insufficient_data"
    assert summary.exists()
    assert all(score.n_orders == 0 for r in runs for score in r.report.cities)
