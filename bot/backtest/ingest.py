from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from bot.backtest.normalize import CanonicalSnapshot, from_trevorjs


CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string()),
        pa.field("event_ticker", pa.string()),
        pa.field("series_ticker", pa.string()),
        pa.field("status", pa.string()),
        pa.field("result", pa.string()),
        pa.field("yes_ask", pa.decimal128(18, 6)),
        pa.field("yes_bid", pa.decimal128(18, 6)),
        pa.field("no_ask", pa.decimal128(18, 6)),
        pa.field("no_bid", pa.decimal128(18, 6)),
        pa.field("last_price", pa.decimal128(18, 6)),
        pa.field("volume", pa.decimal128(28, 6)),
        pa.field("volume_24h", pa.decimal128(28, 6)),
        pa.field("open_interest", pa.decimal128(28, 6)),
        pa.field("open_time", pa.timestamp("us", tz="UTC")),
        pa.field("close_time", pa.timestamp("us", tz="UTC")),
        pa.field("created_time", pa.timestamp("us", tz="UTC")),
        pa.field("floor_strike", pa.int32()),
        pa.field("strike_type", pa.string()),
        pa.field("observed_value", pa.decimal128(18, 6)),
        pa.field("yes_bid_size", pa.decimal128(28, 6)),
        pa.field("no_bid_size", pa.decimal128(28, 6)),
    ]
)


def ingest_weather_snapshots(
    shard_paths: list[Path],
    out_path: Path,
    series: list[str],
) -> int:
    if not series:
        raise ValueError("series must contain at least one ticker prefix")

    dataset = ds.dataset([str(p) for p in shard_paths], format="parquet")

    ticker_field = pc.field("ticker")
    filter_expr = pc.starts_with(ticker_field, f"{series[0]}-")
    for prefix in series[1:]:
        filter_expr = filter_expr | pc.starts_with(ticker_field, f"{prefix}-")

    filtered = dataset.to_table(filter=filter_expr)

    snapshots = [from_trevorjs(row) for row in filtered.to_pylist()]
    table = _snapshots_to_table(snapshots)
    pq.write_table(table, out_path)
    return table.num_rows


def _snapshots_to_table(snapshots: list[CanonicalSnapshot]) -> pa.Table:
    columns: dict[str, list] = {field.name: [] for field in CANONICAL_SCHEMA}
    for snap in snapshots:
        for name in columns:
            columns[name].append(getattr(snap, name))
    return pa.table(columns, schema=CANONICAL_SCHEMA)
