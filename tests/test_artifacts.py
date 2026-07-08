import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.markets.parser import series_id
from bot.replay.artifacts import (
    LADDER_DEPTH,
    MIN_DISK_FREE_FRACTION,
    TOUCH_SCHEMA,
    BudgetExceeded,
    BudgetGuard,
    ByteBudget,
    LadderEmitter,
    TouchEmitter,
    byte_budget,
    bytes_written,
    directory_roots,
    manifest_roots,
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
LOW = "KXLOWTCHI-26JUL19-T55"
PHX = "KXHIGHTPHX-26JUL17-B93.5"
BWI = "KXHIGHTBWI-26AUG05-B93.5"

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

SCOPE_ROWS = [
    _book(1, CHI, 0, 1, "yes", "0.1000", "5.00", True),
    _book(2, LOW, 0, 1, "yes", "0.2000", "5.00", True),
    _book(3, RAIN, 0, 1, "yes", "0.3000", "5.00", True),
    _book(4, PHX, 0, 1, "yes", "0.4000", "5.00", True),
    _book(5, BWI, 0, 1, "yes", "0.5000", "5.00", True),
]

LIFE_ROWS = [
    _book(1, DEN, 0, 1, "yes", "0.4000", "10.00", True),
    _book(2, DEN, 2 * 86_400, 2, "yes", "0.4100", "9.00", True),
]

DEPTH_ROWS = [
    _book(i + 1, DEN, 0, 1, "yes", f"0.{10 + i}00", f"{i + 1}.00", True) for i in range(8)
] + [_book(i + 9, DEN, 0, 1, "no", f"0.{20 + i}00", f"{i + 1}.00", True) for i in range(8)]

RESUME_ROWS = [
    _book(1, CHI, 0, 1, "yes", "0.1000", "5.00", True),
    _book(2, RAIN, 0, 1, "yes", "0.3000", "5.00", True),
    _book(3, LOW, 0, 1, "yes", "0.2000", "5.00", True),
    _book(4, RAIN, 0, 2, "yes", "0.3100", "4.00", False),
]

TRADE_ROWS = [
    _trade(1, DEN, 0, "0.07", "3", "yes"),
    _trade(2, RAIN, 61, "0.2000", "12", "no"),
]

# Gapped on purpose: under contiguous ids a ceiling and a row limit select the same set.
TRADE_PAGE_IDS = [1, 2, 3, 5, 8, 13, 14, 15, 21, 22, 30, 31]
TRADE_PAGE_ROWS = [_trade(row_id, DEN, 0, "0.4000", "2.00", "yes") for row_id in TRADE_PAGE_IDS]
TRADE_PAGE_FILE = "trades/KXHIGHDEN-2026-07-19-b000000.parquet"

GAP_ROWS = [(1, "", _ts(62), 5, "connection_reset", _ts(62))]

FULL_BUDGET = byte_budget(
    f_bavail=22_000_000,
    f_frsize=4096,
    f_blocks=50_000_000,
    ws_raw_daily_bytes=387_000_000,
    pass_hours=Decimal("12"),
)


def drain_record(name: str, kind: str) -> dict[str, object]:
    return {
        "object": f"forward-pass/f5b-20260808/{kind}/{name}",
        "name": name,
        "kind": kind,
        "size": 174888,
        "md5": "tEAAXk2CohCKDaUi5Mma+w==",
        "rows": 5051,
        "generation": "1786157919502358",
    }


def write_manifest(path: Path, records: list[dict[str, object] | None]) -> Path:
    lines = ["" if record is None else json.dumps(record) for record in records]
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def build_book_db(path: Path, rows: list[tuple[object, ...]]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(BOOK_SQL)
    conn.executemany("INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def build_trades_db(path: Path, rows: list[tuple[object, ...]]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(TRADES_SQL)
    conn.executemany("INSERT INTO ws_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


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


@pytest.fixture
def trades_db(tmp_path: Path) -> Path:
    return build_trades_db(tmp_path / "trades.db", TRADE_PAGE_ROWS)


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


def test_the_ladder_scope_is_a_prefix_on_the_series_root_not_a_list_of_known_roots(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    emitter = LadderEmitter()
    run_forward_pass(
        build_book_db(tmp_path / "scope.db", SCOPE_ROWS), out_dir, [emitter], barrier_rows=10_000
    )

    assert sorted(artifact(out_dir)) == [
        "ladder/KXHIGHCHI-2026-07-19-b000001.parquet",
        "ladder/KXHIGHTBWI-2026-07-19-b000001.parquet",
        "ladder/KXHIGHTPHX-2026-07-19-b000001.parquet",
        "ladder/KXLOWTCHI-2026-07-19-b000001.parquet",
    ]
    assert emitter.roots == {"KXHIGHCHI", "KXHIGHTBWI", "KXHIGHTPHX", "KXLOWTCHI"}


def test_the_ladder_scope_drops_kxrain_in_its_recorded_form(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    emitter = LadderEmitter()
    run_forward_pass(
        build_book_db(tmp_path / "rain.db", [_book(1, RAIN, 0, 1, "yes", "0.3000", "5.00", True)]),
        out_dir,
        [emitter],
        barrier_rows=10_000,
    )

    with pytest.raises(ValueError):
        series_id(RAIN)
    assert emitter.roots == set()
    assert artifact(out_dir) == {}


def test_the_ladder_covers_the_whole_recorded_life_of_a_root(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    run_forward_pass(
        build_book_db(tmp_path / "life.db", LIFE_ROWS),
        out_dir,
        [LadderEmitter()],
        barrier_rows=10_000,
    )
    files = artifact(out_dir)

    assert sorted(files) == [
        "ladder/KXHIGHDEN-2026-07-19-b000001.parquet",
        "ladder/KXHIGHDEN-2026-07-21-b000001.parquet",
    ]
    assert [r["id"] for r in files["ladder/KXHIGHDEN-2026-07-19-b000001.parquet"]] == [1]
    assert [r["id"] for r in files["ladder/KXHIGHDEN-2026-07-21-b000001.parquet"]] == [2]


def test_the_ladder_carries_six_levels_a_side_and_still_counts_them_all(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    run_forward_pass(
        build_book_db(tmp_path / "depth.db", DEPTH_ROWS),
        out_dir,
        [LadderEmitter()],
        barrier_rows=10_000,
    )

    row = artifact(out_dir)["ladder/KXHIGHDEN-2026-07-19-b000001.parquet"][-1]
    assert row["yes_prices"] == ["0.1700", "0.1600", "0.1500", "0.1400", "0.1300", "0.1200"]
    assert row["yes_sizes"] == ["8.00", "7.00", "6.00", "5.00", "4.00", "3.00"]
    assert row["yes_levels"] == 8
    assert row["no_prices"] == ["0.2700", "0.2600", "0.2500", "0.2400", "0.2300", "0.2200"]
    assert row["no_sizes"] == ["8.00", "7.00", "6.00", "5.00", "4.00", "3.00"]
    assert row["no_levels"] == 8


def test_the_depth_slice_does_not_shorten_the_level_count(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    run_forward_pass(
        build_book_db(tmp_path / "count.db", DEPTH_ROWS),
        out_dir,
        [LadderEmitter()],
        barrier_rows=10_000,
    )

    row = artifact(out_dir)["ladder/KXHIGHDEN-2026-07-19-b000001.parquet"][-1]
    for side in ("yes", "no"):
        prices = [Decimal(price) for price in row[f"{side}_prices"]]
        assert len(prices) == LADDER_DEPTH
        assert len(row[f"{side}_sizes"]) == LADDER_DEPTH
        assert prices == sorted(prices, reverse=True)
        assert row[f"{side}_levels"] == 8


def test_the_ladder_row_carries_the_touch_row_verbatim(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    run_forward_pass(
        build_book_db(tmp_path / "both.db", DEPTH_ROWS),
        out_dir,
        [TouchEmitter(), LadderEmitter()],
        barrier_rows=10_000,
    )
    files = artifact(out_dir)

    ladder_rows = files["ladder/KXHIGHDEN-2026-07-19-b000001.parquet"]
    touch_rows = files["touch/KXHIGHDEN-2026-07-19-b000001.parquet"]
    assert len(ladder_rows) == len(DEPTH_ROWS)
    for mine, theirs in zip(ladder_rows, touch_rows, strict=True):
        assert {name: mine[name] for name in TOUCH_SCHEMA.names} == theirs


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


def test_the_trades_ceiling_keeps_its_own_id_and_drops_every_id_above(
    tmp_path: Path, trades_db: Path
) -> None:
    out_dir = tmp_path / "out"

    rows = write_trades(trades_db, out_dir, FULL_BUDGET, max_id=14)

    assert rows == 7
    assert [row["id"] for row in artifact(out_dir)[TRADE_PAGE_FILE]] == [1, 2, 3, 5, 8, 13, 14]


def test_the_trades_ceiling_holds_where_it_falls_mid_batch(tmp_path: Path, trades_db: Path) -> None:
    out_dir = tmp_path / "out"

    rows = write_trades(trades_db, out_dir, FULL_BUDGET, batch_rows=3, max_id=14)

    assert rows == 7
    assert [row["id"] for row in artifact(out_dir)[TRADE_PAGE_FILE]] == [1, 2, 3, 5, 8, 13, 14]


def test_a_trades_ceiling_under_every_id_writes_nothing(tmp_path: Path, trades_db: Path) -> None:
    out_dir = tmp_path / "out"

    assert write_trades(trades_db, out_dir, FULL_BUDGET, max_id=0) == 0
    assert list(out_dir.rglob("*.parquet")) == []


def test_no_trades_ceiling_reads_the_table_to_its_end(tmp_path: Path, trades_db: Path) -> None:
    unbounded = tmp_path / "unbounded"
    ceiling = tmp_path / "ceiling"

    rows = write_trades(trades_db, unbounded, FULL_BUDGET)

    assert rows == len(TRADE_PAGE_ROWS)
    assert [row["id"] for row in artifact(unbounded)[TRADE_PAGE_FILE]] == TRADE_PAGE_IDS
    assert write_trades(trades_db, ceiling, FULL_BUDGET, max_id=TRADE_PAGE_IDS[-1]) == len(
        TRADE_PAGE_ROWS
    )
    assert (unbounded / TRADE_PAGE_FILE).read_bytes() == (ceiling / TRADE_PAGE_FILE).read_bytes()


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
    assert scalars["ladder_scope"] == "none"
    assert scalars["ladder_roots"] == "none"
    assert scalars["ladder_depth"] == "none"
    assert int(scalars["bytes_written"]) > 0


def test_the_inventory_records_the_ladder_scope_the_roots_it_hit_and_the_depth(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    exclusions = ExclusionInventory(read_gap_rows(db_path))
    seq = SeqBoundaryDetector()
    tickers = TickerInventory()
    emitter = LadderEmitter()
    run_forward_pass(
        db_path,
        out_dir,
        [emitter],
        accumulators=[exclusions, seq, tickers],
        barrier_rows=10_000,
    )
    write_inventory(out_dir, build_inventory(exclusions, seq, tickers), FULL_BUDGET, ladder=emitter)

    rows = artifact(out_dir)["inventory/scalars-b000000.parquet"]
    scalars = {r["name"]: r["value"] for r in rows}
    assert scalars["ladder_scope"] == "KXHIGH,KXLOW"
    assert scalars["ladder_roots"] == "KXHIGHCHI,KXHIGHDEN"
    assert scalars["ladder_depth"] == "6"
    assert scalars["tickers"] == "3"


def test_the_inventory_names_every_root_the_pass_wrote_not_only_the_resumed_segment(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    db = build_book_db(tmp_path / "resume.db", RESUME_ROWS)
    run_forward_pass(db, out_dir, [LadderEmitter()], accumulators=[TickerInventory()], max_id=2)
    emitter = LadderEmitter()
    tickers = TickerInventory()
    run_forward_pass(db, out_dir, [emitter], accumulators=[tickers])
    inventory = build_inventory(ExclusionInventory([]), SeqBoundaryDetector(), tickers)
    write_inventory(out_dir, inventory, FULL_BUDGET, ladder=emitter)

    assert sorted(name for name in artifact(out_dir) if name.startswith("ladder/")) == [
        "ladder/KXHIGHCHI-2026-07-19-b000001.parquet",
        "ladder/KXLOWTCHI-2026-07-19-b000002.parquet",
    ]
    assert emitter.roots == {"KXHIGHCHI", "KXLOWTCHI"}
    scalars = {
        r["name"]: r["value"] for r in artifact(out_dir)["inventory/scalars-b000000.parquet"]
    }
    assert scalars["ladder_roots"] == "KXHIGHCHI,KXLOWTCHI"
    assert scalars["tickers"] == "3"


def test_the_inventory_names_every_root_after_the_drain_unlinks_the_files(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    tickers = TickerInventory()
    emitter = LadderEmitter()
    run_forward_pass(db_path, out_dir, [emitter], accumulators=[tickers], barrier_rows=10_000)
    for path in (out_dir / "ladder").glob("*.parquet"):
        path.unlink()
    inventory = build_inventory(ExclusionInventory([]), SeqBoundaryDetector(), tickers)
    write_inventory(out_dir, inventory, FULL_BUDGET, ladder=emitter)

    assert list((out_dir / "ladder").glob("*.parquet")) == []
    scalars = {
        r["name"]: r["value"] for r in artifact(out_dir)["inventory/scalars-b000000.parquet"]
    }
    assert scalars["ladder_roots"] == "KXHIGHCHI,KXHIGHDEN"


def test_the_inventory_names_every_root_after_a_resume_and_a_drain(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    db = build_book_db(tmp_path / "resume.db", RESUME_ROWS)
    run_forward_pass(db, out_dir, [LadderEmitter()], accumulators=[TickerInventory()], max_id=2)
    emitter = LadderEmitter()
    tickers = TickerInventory()
    run_forward_pass(db, out_dir, [emitter], accumulators=[tickers])
    for path in (out_dir / "ladder").glob("*.parquet"):
        path.unlink()
    inventory = build_inventory(ExclusionInventory([]), SeqBoundaryDetector(), tickers)
    write_inventory(out_dir, inventory, FULL_BUDGET, ladder=emitter)

    assert emitter.roots == {"KXHIGHCHI", "KXLOWTCHI"}
    scalars = {
        r["name"]: r["value"] for r in artifact(out_dir)["inventory/scalars-b000000.parquet"]
    }
    assert scalars["ladder_roots"] == "KXHIGHCHI,KXLOWTCHI"


def test_the_manifest_yields_the_roots_of_the_kind_asked_for_and_nothing_else(
    tmp_path: Path,
) -> None:
    path = write_manifest(
        tmp_path / "manifest.jsonl",
        [
            drain_record("KXHIGHAUS-2026-07-17-b000001.parquet", "ladder"),
            drain_record("KXHIGHAUS-2026-07-18-b000002.parquet", "ladder"),
            None,
            drain_record("KXLOWTCHI-2026-07-17-b000001.parquet", "ladder"),
            drain_record("KXRAINCHIM-2026-07-17-b000001.parquet", "touch"),
            drain_record("KXHIGHDEN", "ladder"),
        ],
    )

    assert manifest_roots(path, "ladder") == {"KXHIGHAUS", "KXLOWTCHI", "KXHIGHDEN"}
    assert manifest_roots(path, "touch") == {"KXRAINCHIM"}
    assert manifest_roots(path, "trades") == set()


def test_the_reconstructed_roots_are_the_manifest_unioned_with_what_is_still_on_disk(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    run_forward_pass(db_path, out_dir, [LadderEmitter()], barrier_rows=10_000)
    shipped = write_manifest(
        tmp_path / "manifest.jsonl",
        [drain_record("KXHIGHAUS-2026-07-17-b000001.parquet", "ladder")],
    )
    on_disk = directory_roots(out_dir / "ladder")

    assert on_disk == {"KXHIGHCHI", "KXHIGHDEN"}
    assert manifest_roots(shipped, "ladder") | on_disk == {"KXHIGHAUS", "KXHIGHCHI", "KXHIGHDEN"}


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
    emitter = LadderEmitter()
    result = run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter(), emitter],
        accumulators=[exclusions, seq, tickers, guard],
        barrier_rows=10_000,
    )
    write_trades(db_path, out_dir, FULL_BUDGET)
    write_inventory(out_dir, build_inventory(exclusions, seq, tickers), FULL_BUDGET, ladder=emitter)

    assert result.rows == len(BOOK_ROWS)
    assert sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*.parquet")) == [
        "inventory/boundaries-b000000.parquet",
        "inventory/coverage-b000000.parquet",
        "inventory/scalars-b000000.parquet",
        "inventory/windows-b000000.parquet",
        "ladder/KXHIGHCHI-2026-07-19-b000001.parquet",
        "ladder/KXHIGHCHI-2026-07-20-b000001.parquet",
        "ladder/KXHIGHDEN-2026-07-19-b000001.parquet",
        "ladder/KXHIGHDEN-2026-07-20-b000001.parquet",
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
