from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import bot.backtest


def test_pyarrow_parquet_roundtrip(tmp_path: Path) -> None:
    assert bot.backtest is not None

    table = pa.table({"ticker": ["A", "B", "C"], "qty": [1, 2, 3]})
    path = tmp_path / "smoke.parquet"
    pq.write_table(table, path)

    read = pq.read_table(path)

    assert read.equals(table)
