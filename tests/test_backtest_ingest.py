from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bot.backtest.ingest import ingest_weather_snapshots
from tests.fixtures.trevorjs_market_state_sample import make_market_state_table


def _write_shard(table: pa.Table, path: Path) -> Path:
    pq.write_table(table, path)
    return path


def test_filters_to_kxhighden_only(tmp_path: Path) -> None:
    shard = _write_shard(make_market_state_table(), tmp_path / "0000.parquet")
    out = tmp_path / "weather.parquet"

    rows = ingest_weather_snapshots([shard], out, series=["KXHIGHDEN"])

    assert rows == 4
    written = pq.read_table(out)
    assert written.num_rows == 4
    tickers = written.column("ticker").to_pylist()
    assert all(t.startswith("KXHIGHDEN-") for t in tickers)


def test_written_schema_has_canonical_columns(tmp_path: Path) -> None:
    shard = _write_shard(make_market_state_table(), tmp_path / "0000.parquet")
    out = tmp_path / "weather.parquet"

    ingest_weather_snapshots([shard], out, series=["KXHIGHDEN"])

    schema = pq.read_table(out).schema
    expected = {
        "ticker",
        "event_ticker",
        "series_ticker",
        "status",
        "result",
        "yes_ask",
        "yes_bid",
        "no_ask",
        "no_bid",
        "last_price",
        "volume",
        "volume_24h",
        "open_interest",
        "open_time",
        "close_time",
        "created_time",
        "floor_strike",
        "strike_type",
        "observed_value",
        "yes_bid_size",
        "no_bid_size",
    }
    assert set(schema.names) == expected


def test_open_row_survives_ingest(tmp_path: Path) -> None:
    shard = _write_shard(make_market_state_table(), tmp_path / "0000.parquet")
    out = tmp_path / "weather.parquet"

    ingest_weather_snapshots([shard], out, series=["KXHIGHDEN"])

    written = pq.read_table(out)
    results = written.column("result").to_pylist()
    assert results.count("") == 1
    assert results.count("yes") + results.count("no") == 3


def test_cent_prices_normalize_to_decimal_dollars(tmp_path: Path) -> None:
    shard = _write_shard(make_market_state_table(), tmp_path / "0000.parquet")
    out = tmp_path / "weather.parquet"

    ingest_weather_snapshots([shard], out, series=["KXHIGHDEN"])

    written = pq.read_table(out).to_pylist()
    by_ticker = {r["ticker"]: r for r in written}
    row = by_ticker["KXHIGHDEN-26APR03-T58"]
    assert row["yes_ask"] == Decimal("0.87")
    assert row["yes_bid"] == Decimal("0.85")
    assert row["no_ask"] == Decimal("0.15")
    assert row["no_bid"] == Decimal("0.13")
    assert row["last_price"] == Decimal("0.86")
    assert row["series_ticker"] == "KXHIGHDEN"


def test_multi_series_filter(tmp_path: Path) -> None:
    shard = _write_shard(make_market_state_table(), tmp_path / "0000.parquet")
    out = tmp_path / "weather.parquet"

    rows = ingest_weather_snapshots([shard], out, series=["KXHIGHDEN", "KXPRES"])

    assert rows == 5
    tickers = pq.read_table(out).column("ticker").to_pylist()
    assert sum(1 for t in tickers if t.startswith("KXHIGHDEN-")) == 4
    assert sum(1 for t in tickers if t.startswith("KXPRES-")) == 1


def test_non_matching_series_yields_empty_parquet(tmp_path: Path) -> None:
    shard = _write_shard(make_market_state_table(), tmp_path / "0000.parquet")
    out = tmp_path / "weather.parquet"

    rows = ingest_weather_snapshots([shard], out, series=["KXHIGHLAX"])

    assert rows == 0
    written = pq.read_table(out)
    assert written.num_rows == 0
    assert "ticker" in written.schema.names
