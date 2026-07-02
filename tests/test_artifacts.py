import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.markets.parser import series_id
from bot.replay.artifacts import (
    MIN_DISK_FREE_FRACTION,
    BudgetExceeded,
    BudgetGuard,
    ByteBudget,
    LadderEmitter,
    TouchEmitter,
    byte_budget,
    bytes_written,
    measure_byte_budget,
    partition_key,
    project_artifact_bytes,
    write_inventory,
    write_trades,
    ws_raw_daily_bytes,
)
from bot.replay.forward_pass import run_forward_pass
from bot.replay.inventory import (
    ExclusionInventory,
    SeqBoundaryDetector,
    TickerInventory,
    build_inventory,
    read_gap_rows,
)
from scripts import ws_watchdog


UTC = timezone.utc
T0 = datetime(2026, 7, 19, 23, 59, 0, 155692, tzinfo=UTC)
DB_TS = "%Y-%m-%d %H:%M:%S.%f"
DEN = "KXHIGHDEN-26JUL19-B85.5"
CHI = "KXHIGHCHI-26JUL19-T75"
RAIN = "KXRAINCHIM-26JUL-1"

BOOK_SQL = """
CREATE TABLE ws_book_events (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, received_at DATETIME NOT NULL,
    seq INTEGER NOT NULL, side VARCHAR(8) NOT NULL, price VARCHAR NOT NULL,
    size VARCHAR NOT NULL, is_snapshot BOOLEAN NOT NULL, created_at DATETIME NOT NULL,
    ts_ms INTEGER, PRIMARY KEY (id))
"""
TRADES_SQL = """
CREATE TABLE ws_trades (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, received_at DATETIME NOT NULL,
    yes_price VARCHAR NOT NULL, count VARCHAR NOT NULL, taker_side VARCHAR(8) NOT NULL,
    created_at DATETIME NOT NULL, trade_id VARCHAR NOT NULL, ts_ms INTEGER NOT NULL,
    PRIMARY KEY (id))
"""
GAPS_SQL = """
CREATE TABLE ws_gaps (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, detected_at DATETIME NOT NULL,
    last_seq INTEGER NOT NULL, reason VARCHAR(64) NOT NULL, created_at DATETIME NOT NULL,
    PRIMARY KEY (id))
"""


def _at(offset_s: int) -> datetime:
    return T0 + timedelta(seconds=offset_s)


def _ts(offset_s: int) -> str:
    return _at(offset_s).strftime(DB_TS)


def _book(
    row_id: int,
    ticker: str,
    offset_s: int,
    seq: int,
    side: str,
    price: str,
    size: str,
    is_snapshot: bool,
) -> tuple[object, ...]:
    at = _ts(offset_s)
    ts_ms = None if is_snapshot else 1_753_000_000_000 + row_id
    return (row_id, ticker, at, seq, side, price, size, int(is_snapshot), at, ts_ms)


def _trade(
    row_id: int, ticker: str, offset_s: int, yes_price: str, count: str, taker_side: str
) -> tuple[object, ...]:
    at = _ts(offset_s)
    return (
        row_id,
        ticker,
        at,
        yes_price,
        count,
        taker_side,
        at,
        f"t{row_id}",
        1_753_000_000_000 + row_id,
    )


BOOK_ROWS = [
    _book(1, DEN, 0, 1, "yes", "0.4000", "10.00", True),
    _book(2, DEN, 0, 1, "no", "0.5500", "7.00", True),
    _book(3, CHI, 1, 2, "yes", "0.1", "5", True),
    _book(4, CHI, 1, 2, "no", "0.8500", "3.00", True),
    _book(5, DEN, 2, 3, "yes", "0.4100", "2.00", False),
    _book(6, RAIN, 61, 4, "yes", "0.2000", "4.00", True),
    _book(7, RAIN, 61, 4, "no", "0.7500", "6.00", True),
    _book(8, DEN, 62, 5, "yes", "0.4200", "1.00", False),
    _book(9, RAIN, 63, 6, "yes", "0.2100", "2.00", False),
    _book(10, CHI, 64, 7, "yes", "0.1", "-5", False),
]

TRADE_ROWS = [
    _trade(1, DEN, 0, "0.07", "3", "yes"),
    _trade(2, RAIN, 61, "0.2000", "12", "no"),
]

GAP_ROWS = [(1, "", _ts(62), 5, "connection_reset", _ts(62))]

FULL_BUDGET = byte_budget(
    f_bavail=22_000_000,
    f_frsize=4096,
    f_blocks=50_000_000,
    ws_raw_daily_bytes=387_000_000,
    pass_hours=Decimal("12"),
)


def build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(BOOK_SQL)
    conn.execute(TRADES_SQL)
    conn.execute(GAPS_SQL)
    conn.executemany("INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", BOOK_ROWS)
    conn.executemany("INSERT INTO ws_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", TRADE_ROWS)
    conn.executemany("INSERT INTO ws_gaps VALUES (?, ?, ?, ?, ?, ?)", GAP_ROWS)
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    build_db(path)
    return path


def artifact(out_dir: Path) -> dict[str, list[dict[str, object]]]:
    return {
        str(path.relative_to(out_dir)): pq.read_table(path).to_pylist()
        for path in sorted(out_dir.rglob("*.parquet"))
    }


def touch_pass(db_path: Path, out_dir: Path) -> dict[str, list[dict[str, object]]]:
    run_forward_pass(db_path, out_dir, [TouchEmitter()], barrier_rows=10_000)
    return artifact(out_dir)


def test_series_root_is_an_opaque_split_that_survives_the_recorded_kxrain_form() -> None:
    with pytest.raises(ValueError):
        series_id(RAIN)

    assert partition_key(RAIN, _at(61)) == "KXRAINCHIM-2026-07-20"
    assert partition_key(DEN, _at(0)) == "KXHIGHDEN-2026-07-19"


def test_partition_is_the_received_at_date_not_a_date_parsed_from_the_ticker(
    tmp_path: Path, db_path: Path
) -> None:
    files = touch_pass(db_path, tmp_path / "out")

    assert sorted(files) == [
        "touch/KXHIGHCHI-2026-07-19-b000001.parquet",
        "touch/KXHIGHCHI-2026-07-20-b000001.parquet",
        "touch/KXHIGHDEN-2026-07-19-b000001.parquet",
        "touch/KXHIGHDEN-2026-07-20-b000001.parquet",
        "touch/KXRAINCHIM-2026-07-20-b000001.parquet",
    ]
    assert [r["id"] for r in files["touch/KXHIGHDEN-2026-07-20-b000001.parquet"]] == [8]
    assert [r["id"] for r in files["touch/KXHIGHDEN-2026-07-19-b000001.parquet"]] == [1, 2, 5]


def test_price_and_size_carry_the_quantized_scale(tmp_path: Path, db_path: Path) -> None:
    files = touch_pass(db_path, tmp_path / "out")

    chi = files["touch/KXHIGHCHI-2026-07-19-b000001.parquet"][0]
    assert chi["yes_bid"] == "0.1000"
    assert chi["yes_bid_depth"] == "5.00"
    assert str(Decimal("0.1")) == "0.1"
    empty = files["touch/KXHIGHCHI-2026-07-20-b000001.parquet"][0]
    assert empty["yes_bid"] == "0.0000"
    assert empty["yes_bid_depth"] == "0.00"
    assert empty["no_ask"] == "1.0000"
    assert empty["no_ask_depth"] == "0.00"


def test_the_touch_series_carries_one_row_per_book_event(tmp_path: Path, db_path: Path) -> None:
    files = touch_pass(db_path, tmp_path / "out")

    assert sum(len(rows) for rows in files.values()) == len(BOOK_ROWS)
    den = files["touch/KXHIGHDEN-2026-07-19-b000001.parquet"]
    assert list(den[0]) == [
        "id",
        "ticker",
        "received_at",
        "ts_ms",
        "yes_bid",
        "yes_bid_depth",
        "yes_ask",
        "yes_ask_depth",
        "no_bid",
        "no_bid_depth",
        "no_ask",
        "no_ask_depth",
    ]
    assert den[0] == {
        "id": 1,
        "ticker": DEN,
        "received_at": _at(0),
        "ts_ms": None,
        "yes_bid": "0.4000",
        "yes_bid_depth": "10.00",
        "yes_ask": "1.0000",
        "yes_ask_depth": "0.00",
        "no_bid": "0.0000",
        "no_bid_depth": "0.00",
        "no_ask": "0.6000",
        "no_ask_depth": "10.00",
    }
    assert den[2] == {
        "id": 5,
        "ticker": DEN,
        "received_at": _at(2),
        "ts_ms": 1_753_000_000_005,
        "yes_bid": "0.4100",
        "yes_bid_depth": "2.00",
        "yes_ask": "0.4500",
        "yes_ask_depth": "7.00",
        "no_bid": "0.5500",
        "no_bid_depth": "7.00",
        "no_ask": "0.5900",
        "no_ask_depth": "2.00",
    }


def test_the_ladder_artifact_covers_only_the_selected_tickers_and_windows(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    run_forward_pass(
        db_path, out_dir, [LadderEmitter([DEN], [(_at(0), _at(60))])], barrier_rows=10_000
    )
    files = artifact(out_dir)

    assert sorted(files) == ["ladder/KXHIGHDEN-2026-07-19-b000001.parquet"]
    rows = files["ladder/KXHIGHDEN-2026-07-19-b000001.parquet"]
    assert [r["id"] for r in rows] == [1, 2, 5]
    assert rows[2]["yes_bid"] == "0.4100"
    assert rows[2]["yes_prices"] == ["0.4100", "0.4000"]
    assert rows[2]["yes_sizes"] == ["2.00", "10.00"]
    assert rows[2]["yes_levels"] == 2
    assert rows[2]["no_prices"] == ["0.5500"]
    assert rows[2]["no_sizes"] == ["7.00"]
    assert rows[2]["no_levels"] == 1


def test_ladder_depth_truncates_the_levels_and_still_counts_them(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    run_forward_pass(
        db_path,
        out_dir,
        [LadderEmitter([DEN], [(_at(0), _at(60))], depth=1)],
        barrier_rows=10_000,
    )

    rows = artifact(out_dir)["ladder/KXHIGHDEN-2026-07-19-b000001.parquet"]
    assert rows[2]["yes_prices"] == ["0.4100"]
    assert rows[2]["yes_sizes"] == ["2.00"]
    assert rows[2]["yes_levels"] == 2
    assert LadderEmitter([DEN], [(_at(0), _at(60))]).depth is None


def test_trades_are_their_own_artifact_with_no_price_derived_in_decimal(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    rows = write_trades(db_path, out_dir, FULL_BUDGET)
    files = artifact(out_dir)

    assert rows == len(TRADE_ROWS)
    assert sorted(files) == [
        "trades/KXHIGHDEN-2026-07-19-b000000.parquet",
        "trades/KXRAINCHIM-2026-07-20-b000000.parquet",
    ]
    assert files["trades/KXHIGHDEN-2026-07-19-b000000.parquet"][0] == {
        "id": 1,
        "ticker": DEN,
        "received_at": _at(0),
        "ts_ms": 1_753_000_000_001,
        "yes_price": "0.0700",
        "no_price": "0.9300",
        "count": "3.00",
        "taker_side": "yes",
        "trade_id": "t1",
    }
    assert str(1 - float("0.07")) == "0.9299999999999999"


def test_the_inventory_carries_the_three_tables_the_scalars_and_the_budget(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    exclusions = ExclusionInventory(read_gap_rows(db_path))
    seq = SeqBoundaryDetector()
    tickers = TickerInventory()
    run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[exclusions, seq, tickers],
        barrier_rows=10_000,
    )
    paths = write_inventory(out_dir, build_inventory(exclusions, seq, tickers), FULL_BUDGET)
    files = artifact(out_dir)

    assert [p.name for p in paths] == [
        "windows-b000000.parquet",
        "boundaries-b000000.parquet",
        "coverage-b000000.parquet",
        "scalars-b000000.parquet",
    ]
    windows = files["inventory/windows-b000000.parquet"]
    assert len(windows) == 1
    assert windows[0]["gap_id"] == 1
    assert windows[0]["ticker"] == ""
    assert windows[0]["reason"] == "connection_reset"
    assert windows[0]["start"] == _at(61)
    assert windows[0]["end"] == _at(62)
    coverage = {r["ticker"]: r for r in files["inventory/coverage-b000000.parquet"]}
    assert coverage[DEN]["rows"] == 4
    assert coverage[DEN]["first_received_at"] == _at(0)
    assert coverage[DEN]["last_received_at"] == _at(62)
    assert files["inventory/boundaries-b000000.parquet"] == []
    scalars = {r["name"]: r["value"] for r in files["inventory/scalars-b000000.parquet"]}
    assert scalars["gap_rows"] == "1"
    assert scalars["seq_skips"] == "0"
    assert scalars["seq_resubscribes"] == "0"
    assert scalars["tickers"] == "3"
    assert scalars["budget_bytes"] == "41115000000"
    assert scalars["budget_floor_bytes"] == "40960000000"
    assert scalars["ladder_depth"] == "none"
    assert int(scalars["bytes_written"]) > 0


def test_the_inventory_records_a_truncated_ladder_depth(tmp_path: Path, db_path: Path) -> None:
    out_dir = tmp_path / "out"
    exclusions = ExclusionInventory(read_gap_rows(db_path))
    seq = SeqBoundaryDetector()
    tickers = TickerInventory()
    run_forward_pass(db_path, out_dir, [], accumulators=[exclusions, seq, tickers])
    write_inventory(out_dir, build_inventory(exclusions, seq, tickers), FULL_BUDGET, ladder_depth=4)

    rows = artifact(out_dir)["inventory/scalars-b000000.parquet"]
    assert {r["name"]: r["value"] for r in rows}["ladder_depth"] == "4"


def test_the_byte_budget_deducts_the_floor_from_total_not_from_free() -> None:
    assert FULL_BUDGET.free == 90_112_000_000
    assert FULL_BUDGET.floor == 40_960_000_000
    assert FULL_BUDGET.floor != int(MIN_DISK_FREE_FRACTION * FULL_BUDGET.free)
    assert FULL_BUDGET.recorder_day == 5_100_000_000
    assert FULL_BUDGET.ws_raw_day == 387_000_000
    assert FULL_BUDGET.wal == 2_550_000_000
    assert FULL_BUDGET.budget_bytes == 41_115_000_000
    assert float(MIN_DISK_FREE_FRACTION) == ws_watchdog.MIN_DISK_FREE_FRACTION


def test_a_full_filesystem_yields_a_negative_budget() -> None:
    budget = byte_budget(
        f_bavail=10_000_000,
        f_frsize=4096,
        f_blocks=50_000_000,
        ws_raw_daily_bytes=387_000_000,
        pass_hours=Decimal("24"),
    )

    assert budget.free == 40_960_000_000
    assert budget.wal == 5_100_000_000
    assert budget.budget_bytes == -10_587_000_000


def test_the_ws_raw_deduction_is_the_widest_gzipped_day(tmp_path: Path) -> None:
    raw = tmp_path / "ws_raw"
    raw.mkdir()
    (raw / "2026-07-30.jsonl.gz").write_bytes(b"a" * 263)
    (raw / "2026-07-31.jsonl.gz").write_bytes(b"b" * 387)
    (raw / "2026-08-01.jsonl").write_bytes(b"c" * 9999)

    assert ws_raw_daily_bytes(raw) == 387
    with pytest.raises(ValueError):
        ws_raw_daily_bytes(tmp_path)


def test_measure_byte_budget_reads_the_live_filesystem(tmp_path: Path) -> None:
    raw = tmp_path / "ws_raw"
    raw.mkdir()
    (raw / "2026-07-31.jsonl.gz").write_bytes(b"b" * 387)
    budget = measure_byte_budget(tmp_path, raw, Decimal("6"))

    assert budget.free > 0
    assert budget.floor > 0
    assert budget.ws_raw_day == 387
    assert budget.wal == 1_275_000_000
    assert budget.budget_bytes == (
        budget.free - budget.floor - budget.recorder_day - budget.ws_raw_day - budget.wal
    )


def test_a_negative_budget_stops_before_the_first_write(tmp_path: Path, db_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    broke = ByteBudget(
        free=1_000, floor=2_000, recorder_day=0, ws_raw_day=0, wal=0, budget_bytes=-1_000
    )
    inventory = build_inventory(
        ExclusionInventory(read_gap_rows(db_path)), SeqBoundaryDetector(), TickerInventory()
    )

    with pytest.raises(BudgetExceeded):
        BudgetGuard(out_dir, broke)
    with pytest.raises(BudgetExceeded):
        write_trades(db_path, out_dir, broke)
    with pytest.raises(BudgetExceeded):
        write_inventory(out_dir, inventory, broke)
    assert list(out_dir.rglob("*")) == []


def test_the_pass_aborts_at_the_budget_instead_of_filling_the_disk(
    tmp_path: Path, db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="bot.replay.forward_pass")
    out_dir = tmp_path / "out"
    tight = ByteBudget(free=1_000, floor=0, recorder_day=0, ws_raw_day=0, wal=0, budget_bytes=1_000)

    with pytest.raises(BudgetExceeded) as caught:
        run_forward_pass(
            db_path,
            out_dir,
            [TouchEmitter()],
            accumulators=[BudgetGuard(out_dir, tight, check_rows=1)],
            barrier_rows=3,
        )

    assert "budget=1000" in str(caught.value)
    assert not any("forward_pass complete" in r.getMessage() for r in caplog.records)
    written = sorted(r["id"] for rows in artifact(out_dir).values() for r in rows)
    assert written == [1, 2, 3]
    assert bytes_written(out_dir) >= tight.budget_bytes


def test_the_projection_states_both_denominators_and_names_the_binding_one() -> None:
    projection = project_artifact_bytes(
        measured_bytes=2_000_000_000,
        measured_rows=20_000_000,
        measured_source_bytes=4_000_000_000,
        total_rows=458_900_000,
        total_bytes=95_800_000_000,
        budget=FULL_BUDGET,
    )

    assert projection.by_rows == 45_890_000_000
    assert projection.by_bytes == 47_900_000_000
    assert projection.basis == "bytes"
    assert projection.projected_bytes == 47_900_000_000
    assert projection.bytes_per_row == Decimal("100")
    assert projection.budget_bytes == 41_115_000_000
    assert projection.fits is False


def test_the_projection_binds_on_rows_when_rows_project_larger() -> None:
    projection = project_artifact_bytes(
        measured_bytes=2_000_000_000,
        measured_rows=20_000_000,
        measured_source_bytes=8_000_000_000,
        total_rows=458_900_000,
        total_bytes=95_800_000_000,
        budget=FULL_BUDGET,
    )

    assert projection.by_rows == 45_890_000_000
    assert projection.by_bytes == 23_950_000_000
    assert projection.basis == "rows"
    assert projection.projected_bytes == 45_890_000_000
    assert projection.fits is False


def test_a_projection_under_the_budget_fits() -> None:
    projection = project_artifact_bytes(
        measured_bytes=1_000_000_000,
        measured_rows=20_000_000,
        measured_source_bytes=4_000_000_000,
        total_rows=458_900_000,
        total_bytes=95_800_000_000,
        budget=FULL_BUDGET,
    )

    assert projection.by_rows == 22_945_000_000
    assert projection.by_bytes == 23_950_000_000
    assert projection.projected_bytes == 23_950_000_000
    assert projection.bytes_per_row == Decimal("50")
    assert projection.fits is True


def test_every_artifact_lands_where_the_consumer_expects_and_reads_back(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    exclusions = ExclusionInventory(read_gap_rows(db_path))
    seq = SeqBoundaryDetector()
    tickers = TickerInventory()
    guard = BudgetGuard(out_dir, FULL_BUDGET, check_rows=1)
    result = run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter(), LadderEmitter([DEN, RAIN], [(_at(0), _at(120))])],
        accumulators=[exclusions, seq, tickers, guard],
        barrier_rows=10_000,
    )
    write_trades(db_path, out_dir, FULL_BUDGET)
    write_inventory(out_dir, build_inventory(exclusions, seq, tickers), FULL_BUDGET)

    assert result.rows == len(BOOK_ROWS)
    assert sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*.parquet")) == [
        "inventory/boundaries-b000000.parquet",
        "inventory/coverage-b000000.parquet",
        "inventory/scalars-b000000.parquet",
        "inventory/windows-b000000.parquet",
        "ladder/KXHIGHDEN-2026-07-19-b000001.parquet",
        "ladder/KXHIGHDEN-2026-07-20-b000001.parquet",
        "ladder/KXRAINCHIM-2026-07-20-b000001.parquet",
        "touch/KXHIGHCHI-2026-07-19-b000001.parquet",
        "touch/KXHIGHCHI-2026-07-20-b000001.parquet",
        "touch/KXHIGHDEN-2026-07-19-b000001.parquet",
        "touch/KXHIGHDEN-2026-07-20-b000001.parquet",
        "touch/KXRAINCHIM-2026-07-20-b000001.parquet",
        "trades/KXHIGHDEN-2026-07-19-b000000.parquet",
        "trades/KXRAINCHIM-2026-07-20-b000000.parquet",
    ]
    for path in out_dir.rglob("*.parquet"):
        assert pq.read_table(path).num_columns > 0
    assert 0 < bytes_written(out_dir) < guard.budget.budget_bytes
