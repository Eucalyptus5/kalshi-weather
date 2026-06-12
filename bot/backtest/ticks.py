from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from bot.backtest.normalize import _cents_to_dollars


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
