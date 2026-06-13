from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bot.backtest.ticks import ingest_ticks


_TRADES_SCHEMA = pa.schema(
    [
        pa.field("trade_id", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("count", pa.int64()),
        pa.field("yes_price", pa.int64()),
        pa.field("no_price", pa.int64()),
        pa.field("taker_side", pa.string()),
        pa.field("created_time", pa.timestamp("us", tz="UTC")),
    ]
)

_BASE_TIME = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)


def _trade(
    trade_id: str,
    ticker: str,
    *,
    count: int = 1,
    yes_price: int = 50,
    no_price: int = 50,
    taker_side: str = "yes",
    created_time: datetime | None = None,
) -> dict:
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "count": count,
        "yes_price": yes_price,
        "no_price": no_price,
        "taker_side": taker_side,
        "created_time": created_time or _BASE_TIME,
    }


def _write_shard(rows: list[dict], path: Path) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=_TRADES_SCHEMA), path)
    return path


def _make_acceptance_fixture(path: Path) -> Path:
    rows = [
        _trade("t01", "KXHIGHNY-25MAR01-T50", yes_price=60, no_price=40),
        _trade("t02", "KXHIGHNY-25MAR01-T55", yes_price=30, no_price=70),
        _trade("t03", "KXHIGHCHI-25MAR01-T50", yes_price=45, no_price=55),
        _trade("t01", "KXHIGHNY-25MAR01-T50", yes_price=60, no_price=40),
        _trade("t04", "KXHIGHMOV-25MAR01-T50"),
        _trade("t05", "KXHIGHHOU-25MAR01-T55"),
        _trade("t06", "KXHIGHMOVFOO-25MAR01-T50"),
        _trade("t07", "KXPRES-26-DEM"),
        _trade("t08", "KXNFL-26W14-SF"),
        _trade("t09", "KXBTCD-26JUN09-T70000"),
    ]
    return _write_shard(rows, path)


def test_acceptance_fixture_writes_3_rows(tmp_path: Path) -> None:
    shard = _make_acceptance_fixture(tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    n = ingest_ticks(
        [shard],
        out,
        series=[
            "KXHIGHNY",
            "KXHIGHCHI",
            "KXHIGHAUS",
            "KXHIGHMIA",
            "KXHIGHDEN",
            "KXHIGHPHIL",
            "KXHIGHLAX",
        ],
    )

    assert n == 3
    written = pq.read_table(out)
    assert written.num_rows == 3


def test_returns_row_count(tmp_path: Path) -> None:
    shard = _make_acceptance_fixture(tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    n = ingest_ticks([shard], out, series=["KXHIGHNY"])
    assert n == 2


def test_output_sorted_by_ticker_then_created_time(tmp_path: Path) -> None:
    t1 = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 3, 1, 11, 0, tzinfo=timezone.utc)
    t3 = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
    t4 = datetime(2025, 3, 1, 9, 0, tzinfo=timezone.utc)
    rows = [
        _trade("a3", "KXHIGHNY-25MAR01-T55", created_time=t3),
        _trade("a1", "KXHIGHCHI-25MAR01-T50", created_time=t1),
        _trade("a2", "KXHIGHNY-25MAR01-T50", created_time=t2),
        _trade("a4", "KXHIGHNY-25MAR01-T55", created_time=t4),
    ]
    shard = _write_shard(rows, tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    ingest_ticks([shard], out, series=["KXHIGHNY", "KXHIGHCHI"])

    written = pq.read_table(out).to_pylist()
    result = [(r["ticker"], r["created_time"]) for r in written]
    assert result == [
        ("KXHIGHCHI-25MAR01-T50", t1),
        ("KXHIGHNY-25MAR01-T50", t2),
        ("KXHIGHNY-25MAR01-T55", t4),
        ("KXHIGHNY-25MAR01-T55", t3),
    ]


def test_price_normalization(tmp_path: Path) -> None:
    rows = [_trade("x1", "KXHIGHNY-25MAR01-T50", yes_price=12, no_price=88)]
    shard = _write_shard(rows, tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    ingest_ticks([shard], out, series=["KXHIGHNY"])

    written = pq.read_table(out).to_pylist()
    assert len(written) == 1
    assert written[0]["yes_price"] == Decimal("0.12")
    assert written[0]["no_price"] == Decimal("0.88")


def test_dedup_across_two_shards(tmp_path: Path) -> None:
    shard1 = _write_shard(
        [_trade("dup01", "KXHIGHNY-25MAR01-T50", yes_price=40, no_price=60)],
        tmp_path / "s1.parquet",
    )
    shard2 = _write_shard(
        [
            _trade("dup01", "KXHIGHNY-25MAR01-T50", yes_price=40, no_price=60),
            _trade("uniq02", "KXHIGHNY-25MAR01-T55", yes_price=55, no_price=45),
        ],
        tmp_path / "s2.parquet",
    )
    out = tmp_path / "ticks.parquet"

    n = ingest_ticks([shard1, shard2], out, series=["KXHIGHNY"])

    assert n == 2
    written = pq.read_table(out)
    assert written.num_rows == 2


def test_overmatch_prefixes_excluded(tmp_path: Path) -> None:
    rows = [
        _trade("m1", "KXHIGHMOV-25MAR01-T50"),
        _trade("m2", "KXHIGHHOU-25MAR01-T50"),
        _trade("m3", "KXHIGHT-25MAR01-T50"),
        _trade("m4", "KXHIGHNY-25MAR01-T50"),
        _trade("m5", "KXHIGHNYFOO-25MAR01-T50"),
    ]
    shard = _write_shard(rows, tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    n = ingest_ticks([shard], out, series=["KXHIGHNY"])

    assert n == 1


def test_output_schema_columns(tmp_path: Path) -> None:
    rows = [_trade("s1", "KXHIGHNY-25MAR01-T50", yes_price=50, no_price=50)]
    shard = _write_shard(rows, tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    ingest_ticks([shard], out, series=["KXHIGHNY"])

    schema = pq.read_table(out).schema
    assert set(schema.names) == {
        "trade_id",
        "ticker",
        "count",
        "yes_price",
        "no_price",
        "taker_side",
        "created_time",
    }
