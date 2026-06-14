from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from bot.backtest.engine import ReplaySnapshot
from bot.backtest.normalize import CanonicalSnapshot, _cents_to_dollars


_TICK_SCHEMA = pa.schema(
    [
        pa.field("trade_id", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("count", pa.int64()),
        pa.field("yes_price", pa.decimal128(10, 2)),
        pa.field("no_price", pa.decimal128(10, 2)),
        pa.field("taker_side", pa.string()),
        pa.field("created_time", pa.timestamp("us", tz="UTC")),
    ]
)


def ingest_ticks(shard_paths: list[Path], out_path: Path, series: list[str]) -> int:
    if not series:
        raise ValueError("series must contain at least one ticker prefix")

    dataset = ds.dataset([str(p) for p in shard_paths], format="parquet")

    ticker_field = pc.field("ticker")
    filter_expr = pc.starts_with(ticker_field, f"{series[0]}-")
    for prefix in series[1:]:
        filter_expr = filter_expr | pc.starts_with(ticker_field, f"{prefix}-")

    raw = dataset.to_table(filter=filter_expr)

    trade_ids = raw.column("trade_id").to_pylist()
    seen: set[str] = set()
    keep: list[int] = []
    for i, tid in enumerate(trade_ids):
        if tid not in seen:
            seen.add(tid)
            keep.append(i)

    deduped = raw.take(keep)

    sorted_indices = pc.sort_indices(
        deduped,
        sort_keys=[("ticker", "ascending"), ("created_time", "ascending")],
    )
    deduped = deduped.take(sorted_indices)

    rows = deduped.to_pylist()
    table = pa.table(
        {
            "trade_id": [r["trade_id"] for r in rows],
            "ticker": [r["ticker"] for r in rows],
            "count": [r["count"] for r in rows],
            "yes_price": [_cents_to_dollars(r["yes_price"]) for r in rows],
            "no_price": [_cents_to_dollars(r["no_price"]) for r in rows],
            "taker_side": [r["taker_side"] for r in rows],
            "created_time": [r["created_time"] for r in rows],
        },
        schema=_TICK_SCHEMA,
    )

    pq.write_table(table, out_path)
    return table.num_rows


def tick_decision_snapshots(
    tick_path: Path,
    market_state: Sequence[CanonicalSnapshot],
    lead: timedelta,
    staleness: timedelta,
    depth_window: timedelta,
) -> tuple[list[ReplaySnapshot], int]:
    table = pq.read_table(tick_path)
    rows = table.to_pylist()

    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)

    out: list[ReplaySnapshot] = []
    omitted = 0

    for snap in market_state:
        if snap.close_time is None:
            continue
        as_of = snap.close_time - lead
        staleness_floor = as_of - staleness
        depth_floor = as_of - depth_window

        ticker_rows = by_ticker.get(snap.ticker, [])
        candidate: dict | None = None
        for row in ticker_rows:
            t: datetime = row["created_time"]
            if t <= as_of and t >= staleness_floor:
                if candidate is None or t > candidate["created_time"]:
                    candidate = row

        if candidate is None:
            omitted += 1
            out.append(ReplaySnapshot(snapshot_at=snap.close_time, snap=snap))
            continue

        last_price = candidate["yes_price"]
        no_price = Decimal("1") - last_price

        trailing_count = sum(1 for row in ticker_rows if depth_floor < row["created_time"] <= as_of)

        decision_snap = snap.model_copy(
            update={
                "result": "",
                "yes_ask": last_price,
                "yes_bid": last_price,
                "no_ask": no_price,
                "no_bid": no_price,
                "last_price": last_price,
                "yes_bid_size": Decimal(trailing_count),
                "no_bid_size": Decimal(trailing_count),
            }
        )
        out.append(ReplaySnapshot(snapshot_at=as_of, snap=decision_snap))
        out.append(ReplaySnapshot(snapshot_at=snap.close_time, snap=snap))

    return out, omitted
