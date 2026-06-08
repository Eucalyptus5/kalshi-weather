from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bot.backtest.engine import (
    BacktestConfig,
    ReplayLog,
    ReplaySnapshot,
    iter_canonical_snapshots,
    run_replay,
)
from bot.backtest.forecast_replay import StationSpec
from bot.backtest.ingest import CANONICAL_SCHEMA
from bot.backtest.normalize import CanonicalSnapshot
from bot.forecast.cdf import EnsembleCDF

_LEAD = timedelta(hours=25)
_BANKROLL = Decimal("20000")
_MEMBERS = np.array([59.5, 61.5, 63.5])
_OVERLAP_START = datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc)

_PARITY_SCHEMA = pa.schema(
    [pa.field("snapshot_at", pa.timestamp("us", tz="UTC"))] + list(CANONICAL_SCHEMA)
)


class FixedReplay:
    def __init__(self) -> None:
        self._cdf = EnsembleCDF.from_members(_MEMBERS, smoothing=0.65)

    async def replay(self, station: StationSpec, valid_date: date, as_of: datetime) -> EnsembleCDF:
        return self._cdf


def _market_specs() -> list[tuple[str, datetime, Decimal, Decimal, int]]:
    base_close = _OVERLAP_START + timedelta(days=1, hours=6)
    specs: list[tuple[str, datetime, Decimal, Decimal, int]] = []
    for i in range(10):
        ticker = f"KXHIGHDEN-26MAR{30 + i // 5}-B{59 + (i % 5)}.5"
        close = base_close + timedelta(days=i)
        yes_ask = Decimal("0.10") + Decimal(i % 3) * Decimal("0.05")
        yes_bid = yes_ask - Decimal("0.02")
        depth = 3000 + (i % 4) * 1000
        specs.append((ticker, close, yes_ask, yes_bid, depth))
    return specs


def _snapshot_for(
    ticker: str,
    close: datetime,
    yes_ask: Decimal,
    yes_bid: Decimal,
    depth: int,
) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-", 1)[0],
        status="active",
        result="",
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=Decimal("1") - yes_bid,
        no_bid=Decimal("1") - yes_ask,
        last_price=yes_bid,
        volume=Decimal(0),
        volume_24h=Decimal(0),
        open_interest=Decimal(0),
        close_time=close,
        yes_bid_size=Decimal(depth),
        no_bid_size=Decimal(depth),
    )


def _write_parity_parquet(path: Path) -> None:
    rows = []
    for ticker, close, yes_ask, yes_bid, depth in _market_specs():
        snap = _snapshot_for(ticker, close, yes_ask, yes_bid, depth)
        row = {name: getattr(snap, name) for name in CANONICAL_SCHEMA.names}
        row["snapshot_at"] = close - _LEAD - timedelta(minutes=1)
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows, schema=_PARITY_SCHEMA), path)


def _write_state_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE orderbook_snapshots "
        "(ticker TEXT, event_ticker TEXT, series_ticker TEXT, status TEXT, result TEXT, "
        "snapshot_at TEXT, yes_ask TEXT, yes_bid TEXT, no_ask TEXT, no_bid TEXT, last_price TEXT, "
        "volume TEXT, volume_24h TEXT, open_interest TEXT, close_time TEXT, "
        "yes_bid_size TEXT, no_bid_size TEXT)"
    )
    for ticker, close, yes_ask, yes_bid, depth in _market_specs():
        snap = _snapshot_for(ticker, close, yes_ask, yes_bid, depth)
        snapshot_at = close - _LEAD - timedelta(minutes=1)
        conn.execute(
            "INSERT INTO orderbook_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                snap.ticker,
                snap.event_ticker,
                snap.series_ticker,
                snap.status,
                snap.result,
                snapshot_at.isoformat(sep=" "),
                str(snap.yes_ask),
                str(snap.yes_bid),
                str(snap.no_ask),
                str(snap.no_bid),
                str(snap.last_price),
                str(snap.volume),
                str(snap.volume_24h),
                str(snap.open_interest),
                close.isoformat(sep=" "),
                str(depth),
                str(depth),
            ),
        )
    conn.commit()
    conn.close()


def make_overlap_parity_fixture(tmp_path: Path) -> tuple[Path, Path]:
    db_path = tmp_path / "state.db"
    parquet_path = tmp_path / "canonical.parquet"
    _write_state_db(db_path)
    _write_parity_parquet(parquet_path)
    return db_path, parquet_path


def _iter_state_db_snapshots(db_path: Path) -> list[ReplaySnapshot]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM orderbook_snapshots").fetchall()
    conn.close()
    out: list[ReplaySnapshot] = []
    for r in rows:
        snap = CanonicalSnapshot(
            ticker=r["ticker"],
            event_ticker=r["event_ticker"],
            series_ticker=r["series_ticker"],
            status=r["status"],
            result=r["result"],
            yes_ask=Decimal(r["yes_ask"]),
            yes_bid=Decimal(r["yes_bid"]),
            no_ask=Decimal(r["no_ask"]),
            no_bid=Decimal(r["no_bid"]),
            last_price=Decimal(r["last_price"]),
            volume=Decimal(r["volume"]),
            volume_24h=Decimal(r["volume_24h"]),
            open_interest=Decimal(r["open_interest"]),
            close_time=datetime.fromisoformat(r["close_time"]),
            yes_bid_size=Decimal(r["yes_bid_size"]),
            no_bid_size=Decimal(r["no_bid_size"]),
        )
        out.append(
            ReplaySnapshot(
                snapshot_at=datetime.fromisoformat(r["snapshot_at"]),
                snap=snap,
            )
        )
    return out


def _decision_id(snapshot_at: datetime, ticker: str, close_time: datetime) -> str:
    return f"{ticker}@{close_time.isoformat()}#{snapshot_at.isoformat()}"


async def _decision_tuples(
    snapshots: list[ReplaySnapshot],
    config: BacktestConfig,
) -> dict[str, tuple[str, str | None, int | None, str, str]]:
    log = ReplayLog()
    orders = await run_replay(snapshots, FixedReplay(), "edge", config, log=log)
    by_key: dict[tuple[str, datetime], ReplaySnapshot] = {}
    for row in snapshots:
        if row.snap.close_time is None:
            continue
        key = (row.snap.ticker, row.snap.close_time)
        best = by_key.get(key)
        if best is None or row.snapshot_at > best.snapshot_at:
            by_key[key] = row

    filled = {o.market_ticker: o for o in orders}
    failed: dict[str, str] = {f.market_ticker: f.name for f in log.failures}

    tuples: dict[str, tuple[str, str | None, int | None, str, str]] = {}
    for (ticker, close_time), row in by_key.items():
        sid = _decision_id(row.snapshot_at, ticker, close_time)
        order = filled.get(ticker)
        if order is not None:
            tuples[sid] = (ticker, order.action.value, order.contracts, sid, "filled")
        else:
            tuples[sid] = (ticker, None, None, sid, failed.get(ticker, "unknown"))
    return tuples


async def test_state_db_and_parquet_adapters_agree_on_decision_tuples(tmp_path: Path) -> None:
    db_path, parquet_path = make_overlap_parity_fixture(tmp_path)
    config = BacktestConfig(lead=_LEAD, bankroll=_BANKROLL)

    state_db_rows = _iter_state_db_snapshots(db_path)
    parquet_rows = list(iter_canonical_snapshots(parquet_path))

    state_db_tuples = await _decision_tuples(state_db_rows, config)
    parquet_tuples = await _decision_tuples(parquet_rows, config)

    assert state_db_tuples, "state.db adapter produced no decisions"
    assert set(state_db_tuples) == set(parquet_tuples)

    total = len(state_db_tuples)
    mismatches = 0
    first_divergence: tuple[object, object] | None = None
    for sid in sorted(state_db_tuples):
        if state_db_tuples[sid] != parquet_tuples[sid]:
            mismatches += 1
            if first_divergence is None:
                first_divergence = (state_db_tuples[sid], parquet_tuples[sid])

    match_rate = (total - mismatches) / total
    assert match_rate >= 0.99, (
        f"adapter parity {match_rate:.4f} below 0.99 over {total} decisions; "
        f"first divergence state_db={first_divergence[0]} parquet={first_divergence[1]}"
        if first_divergence is not None
        else f"adapter parity {match_rate:.4f} below 0.99 over {total} decisions"
    )


async def test_parquet_adapter_yields_replay_snapshots_field_identical_to_state_db(
    tmp_path: Path,
) -> None:
    db_path, parquet_path = make_overlap_parity_fixture(tmp_path)

    state_db_rows = {(r.snap.ticker, r.snapshot_at): r for r in _iter_state_db_snapshots(db_path)}
    parquet_rows = {
        (r.snap.ticker, r.snapshot_at): r for r in iter_canonical_snapshots(parquet_path)
    }

    assert set(state_db_rows) == set(parquet_rows)
    for key, parquet_row in parquet_rows.items():
        state_row = state_db_rows[key]
        assert parquet_row.snapshot_at == state_row.snapshot_at
        assert parquet_row.snap.ticker == state_row.snap.ticker
        assert parquet_row.snap.yes_ask == state_row.snap.yes_ask
        assert parquet_row.snap.yes_bid == state_row.snap.yes_bid
        assert parquet_row.snap.no_ask == state_row.snap.no_ask
        assert parquet_row.snap.no_bid == state_row.snap.no_bid
        assert parquet_row.snap.close_time == state_row.snap.close_time
        assert parquet_row.snap.yes_bid_size == state_row.snap.yes_bid_size
        assert parquet_row.snap.no_bid_size == state_row.snap.no_bid_size
