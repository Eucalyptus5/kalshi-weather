import json
import sqlite3
from collections.abc import Callable
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.lag.ws_book import WsGapError
from bot.replay.forward_pass import SourceRow, run_forward_pass
from bot.replay.inventory import (
    ExclusionInventory,
    ExclusionWindow,
    PassInventory,
    SeqBoundary,
    SeqBoundaryDetector,
    TickerCoverage,
    TickerInventory,
    build_inventory,
    check_excluded,
    read_gap_rows,
)


UTC = timezone.utc
T0 = datetime(2026, 7, 19, 4, 59, 0, 155692, tzinfo=UTC)
DB_TS = "%Y-%m-%d %H:%M:%S.%f"
DEN = "KXHIGHDEN-26JUL19-B85"
CHI = "KXHIGHCHI-26JUL19-B75"
BOS = "KXHIGHBOS-26JUL19-B70"

BOOK_SCHEMA = """
CREATE TABLE ws_book_events (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, received_at DATETIME NOT NULL,
    seq INTEGER NOT NULL, side VARCHAR(8) NOT NULL, price VARCHAR NOT NULL,
    size VARCHAR NOT NULL, is_snapshot BOOLEAN NOT NULL, created_at DATETIME NOT NULL,
    ts_ms INTEGER, PRIMARY KEY (id))
"""

GAP_SCHEMA = """
CREATE TABLE ws_gaps (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, detected_at DATETIME NOT NULL,
    last_seq INTEGER NOT NULL, reason VARCHAR(64) NOT NULL, created_at DATETIME NOT NULL,
    PRIMARY KEY (id))
"""


def at(offset_s: float) -> datetime:
    return T0 + timedelta(seconds=offset_s)


def book(
    row_id: int,
    ticker: str,
    offset_s: float,
    seq: int,
    side: str = "yes",
    price: str = "0.4000",
    size: str = "1.00",
    is_snapshot: bool = False,
) -> tuple[object, ...]:
    stamp = at(offset_s).strftime(DB_TS)
    ts_ms = None if is_snapshot else 1_753_000_000_000 + row_id
    return (row_id, ticker, stamp, seq, side, price, size, int(is_snapshot), stamp, ts_ms)


def gap(
    row_id: int,
    ticker: str,
    offset_s: float,
    last_seq: int = 0,
    reason: str = "connection_reset",
) -> tuple[object, ...]:
    stamp = at(offset_s).strftime(DB_TS)
    return (row_id, ticker, stamp, last_seq, reason, stamp)


def build_db(path: Path, rows: list[tuple[object, ...]], gaps: list[tuple[object, ...]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(BOOK_SCHEMA)
    conn.execute(GAP_SCHEMA)
    conn.executemany("INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.executemany("INSERT INTO ws_gaps VALUES (?, ?, ?, ?, ?, ?)", gaps)
    conn.commit()
    conn.close()


def source_rows(rows: list[tuple[object, ...]]) -> list[SourceRow]:
    return [
        SourceRow(
            id=row[0],
            ticker=row[1],
            received_at=datetime.strptime(row[2], DB_TS).replace(tzinfo=UTC),
            seq=row[3],
            side=row[4],
            price=row[5],
            size=row[6],
            is_snapshot=bool(row[7]),
            ts_ms=row[9],
        )
        for row in rows
    ]


BOOK_ROWS = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
    book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
    book(3, CHI, 1.0, 2, "yes", "0.2000", "4.00", True),
    book(4, DEN, 2.0, 3, "yes", "0.4100", "2.00"),
    book(5, CHI, 4.0, 4, "yes", "0.2100", "1.00"),
    book(6, DEN, 12.0, 1, "yes", "0.4000", "9.00", True),
    book(7, CHI, 12.5, 2, "yes", "0.2000", "3.00", True),
]

WIDE_GAP = [gap(1, "", 11.0, last_seq=4)]


def inventory_window() -> ExclusionWindow:
    return ExclusionWindow(
        gap_id=1,
        ticker="",
        start=at(4.0),
        end=at(12.0),
        detected_at=at(11.0),
        last_seq=4,
        reason="connection_reset",
    )


def inventory_over(rows: list[tuple[object, ...]], db_path: Path) -> ExclusionInventory:
    inventory = ExclusionInventory(read_gap_rows(db_path))
    for row in source_rows(rows):
        inventory.observe(row)
    return inventory


def test_connection_wide_gap_is_one_window_that_covers_every_ticker(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS, WIDE_GAP)

    windows = inventory_over(BOOK_ROWS, db_path).windows()

    assert len(windows) == 1
    assert windows[0].ticker == ""
    for ticker in (DEN, CHI, BOS):
        with pytest.raises(WsGapError):
            check_excluded(windows, ticker, at(8.0))


def test_ticker_scoped_gap_covers_only_that_ticker(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS, [gap(1, DEN, 11.0, last_seq=4, reason="seq_skip")])

    windows = inventory_over(BOOK_ROWS, db_path).windows()

    assert len(windows) == 1
    assert windows[0].ticker == DEN
    with pytest.raises(WsGapError):
        check_excluded(windows, DEN, at(8.0))
    check_excluded(windows, CHI, at(8.0))


def test_left_edge_is_the_last_received_at_before_the_gap_not_detected_at(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS, WIDE_GAP)

    window = inventory_over(BOOK_ROWS, db_path).windows()[0]

    assert window.start == at(4.0)
    assert window.detected_at == at(11.0)
    assert window.start != window.detected_at
    assert window.end == at(12.0)
    assert window.last_seq == 4
    assert window.reason == "connection_reset"


def test_the_window_edges_are_live_data_and_are_not_excluded(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS, WIDE_GAP)

    windows = inventory_over(BOOK_ROWS, db_path).windows()

    check_excluded(windows, DEN, at(4.0))
    check_excluded(windows, DEN, at(12.0))
    with pytest.raises(WsGapError):
        check_excluded(windows, DEN, at(4.000001))
    with pytest.raises(WsGapError):
        check_excluded(windows, DEN, at(11.999999))


def test_gap_error_names_the_window_it_came_from(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS, WIDE_GAP)

    windows = inventory_over(BOOK_ROWS, db_path).windows()

    with pytest.raises(WsGapError) as excinfo:
        check_excluded(windows, DEN, at(8.0))
    message = str(excinfo.value)
    assert "gap 1" in message
    assert "connection_reset" in message
    assert str(at(4.0)) in message
    assert str(at(12.0)) in message
    assert str(at(11.0)) in message


def test_a_gap_after_the_last_row_leaves_the_window_open_on_the_right(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS[:5], WIDE_GAP)

    window = inventory_over(BOOK_ROWS[:5], db_path).windows()[0]

    assert window.start == at(4.0)
    assert window.end is None
    with pytest.raises(WsGapError):
        check_excluded([window], DEN, at(900.0))


def detect(rows: list[tuple[object, ...]]) -> SeqBoundaryDetector:
    detector = SeqBoundaryDetector()
    for row in source_rows(rows):
        detector.observe(row)
    return detector


def test_a_snapshot_batch_sharing_received_at_and_seq_is_one_message() -> None:
    rows = [
        book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
        book(2, DEN, 0.0, 1, "yes", "0.3900", "5.00", True),
        book(3, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
        book(4, DEN, 1.0, 2),
    ]

    detector = detect(rows)

    assert detector.skips() == 0
    assert detector.boundaries() == []


def test_a_backwards_seq_is_a_resubscribe_and_not_a_skip() -> None:
    rows = [
        book(1, DEN, 0.0, 1),
        book(2, CHI, 1.0, 2),
        book(3, DEN, 2.0, 3),
        book(4, DEN, 20.0, 1, "yes", "0.4000", "9.00", True),
        book(5, CHI, 21.0, 2),
    ]

    detector = detect(rows)

    assert detector.skips() == 0
    assert detector.resubscribes() == 1
    boundary = detector.boundaries()[0]
    assert (boundary.kind, boundary.id, boundary.prev_seq, boundary.seq) == ("resubscribe", 4, 3, 1)
    assert boundary.received_at == at(20.0)


def test_a_forward_jump_of_more_than_one_is_a_skip() -> None:
    rows = [
        book(1, DEN, 0.0, 1),
        book(2, CHI, 1.0, 2),
        book(3, DEN, 2.0, 5),
        book(4, CHI, 3.0, 6),
    ]

    detector = detect(rows)

    assert detector.skips() == 1
    assert detector.resubscribes() == 0
    boundary = detector.boundaries()[0]
    assert (boundary.kind, boundary.id, boundary.prev_seq, boundary.seq) == ("skip", 3, 2, 5)
    assert boundary.received_at == at(2.0)


def test_the_global_stream_is_clean_where_a_per_ticker_stream_would_read_as_skips() -> None:
    rows = [
        book(1, DEN, 0.0, 1),
        book(2, CHI, 1.0, 2),
        book(3, DEN, 2.0, 3),
        book(4, CHI, 3.0, 4),
        book(5, DEN, 4.0, 5),
        book(6, CHI, 5.0, 6),
    ]

    per_ticker = {DEN: [1, 3, 5], CHI: [2, 4, 6]}
    assert [b - a for seqs in per_ticker.values() for a, b in zip(seqs, seqs[1:])] == [2, 2, 2, 2]

    detector = detect(rows)

    assert detector.skips() == 0
    assert detector.boundaries() == []


def test_per_ticker_row_counts_and_edges() -> None:
    rows = [
        book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
        book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
        book(3, CHI, 1.0, 2, "yes", "0.2000", "4.00", True),
        book(4, BOS, 2.0, 3, "yes", "0.1000", "3.00", True),
        book(5, DEN, 3.5, 4),
        book(6, CHI, 9.0, 5),
        book(7, DEN, 11.25, 6),
    ]

    inventory = TickerInventory()
    for row in source_rows(rows):
        inventory.observe(row)

    assert inventory.coverage() == [
        TickerCoverage(ticker=BOS, rows=1, first_received_at=at(2.0), last_received_at=at(2.0)),
        TickerCoverage(ticker=CHI, rows=2, first_received_at=at(1.0), last_received_at=at(9.0)),
        TickerCoverage(ticker=DEN, rows=4, first_received_at=at(0.0), last_received_at=at(11.25)),
    ]


def test_seq_skips_and_gap_rows_are_separate_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    gaps = [gap(1, "", 11.0, last_seq=4), gap(2, "", 60.0, last_seq=9)]
    rows = [
        book(1, DEN, 0.0, 1),
        book(2, CHI, 4.0, 2),
        book(3, DEN, 12.0, 5),
    ]
    build_db(db_path, rows, gaps)

    inventory = build_inventory(inventory_over(rows, db_path), detect(rows), TickerInventory())

    assert {
        field.name: getattr(inventory, field.name)
        for field in fields(inventory)
        if isinstance(getattr(inventory, field.name), int)
    } == {"gap_rows": 2, "seq_skips": 1, "seq_resubscribes": 0}


def resumed_through_json[A: (ExclusionInventory, SeqBoundaryDetector, TickerInventory)](
    make: Callable[[], A],
    rows: list[tuple[object, ...]],
    split: int,
) -> tuple[A, A]:
    whole = make()
    for row in source_rows(rows):
        whole.observe(row)
    first = make()
    for row in source_rows(rows[:split]):
        first.observe(row)
    second = make()
    second.restore(json.loads(json.dumps(first.state())))
    for row in source_rows(rows[split:]):
        second.observe(row)
    return whole, second


def test_exclusion_inventory_survives_a_json_checkpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS, WIDE_GAP)
    gaps = read_gap_rows(db_path)

    for split in range(len(BOOK_ROWS) + 1):
        whole, resumed = resumed_through_json(lambda: ExclusionInventory(gaps), BOOK_ROWS, split)
        assert resumed.windows() == whole.windows()
        assert whole.windows()[0].start == at(4.0)


def test_seq_detector_survives_a_json_checkpoint() -> None:
    rows = [
        book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
        book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
        book(3, CHI, 1.0, 2),
        book(4, DEN, 2.0, 5),
        book(5, CHI, 20.0, 1),
        book(6, DEN, 21.0, 2),
    ]

    for split in range(len(rows) + 1):
        whole, resumed = resumed_through_json(SeqBoundaryDetector, rows, split)
        assert resumed.boundaries() == whole.boundaries()
        assert (whole.skips(), whole.resubscribes()) == (1, 1)


def test_ticker_inventory_survives_a_json_checkpoint() -> None:
    for split in range(len(BOOK_ROWS) + 1):
        whole, resumed = resumed_through_json(TickerInventory, BOOK_ROWS, split)
        assert resumed.coverage() == whole.coverage()


def expected_inventory(gap_rows: int) -> PassInventory:
    return PassInventory(
        gap_rows=gap_rows,
        seq_skips=0,
        seq_resubscribes=1,
        windows=(inventory_window(),),
        boundaries=(
            SeqBoundary(id=6, received_at=at(12.0), prev_seq=4, seq=1, kind="resubscribe"),
        ),
        coverage=(
            TickerCoverage(
                ticker=CHI, rows=3, first_received_at=at(1.0), last_received_at=at(12.5)
            ),
            TickerCoverage(
                ticker=DEN, rows=4, first_received_at=at(0.0), last_received_at=at(12.0)
            ),
        ),
    )


def test_the_driver_drives_the_inventory_accumulators(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS, WIDE_GAP)
    exclusions = ExclusionInventory(read_gap_rows(db_path))
    seq = SeqBoundaryDetector()
    tickers = TickerInventory()

    result = run_forward_pass(
        db_path,
        tmp_path / "out",
        [],
        accumulators=[exclusions, seq, tickers],
        read_batch_rows=3,
        barrier_rows=4,
    )

    assert result.rows == len(BOOK_ROWS)
    assert build_inventory(exclusions, seq, tickers) == expected_inventory(1)


def test_the_inventory_survives_a_driver_resume(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    build_db(db_path, BOOK_ROWS[:5], WIDE_GAP)
    out_dir = tmp_path / "out"
    run_forward_pass(
        db_path,
        out_dir,
        [],
        accumulators=[ExclusionInventory(read_gap_rows(db_path)), SeqBoundaryDetector()],
        read_batch_rows=3,
        barrier_rows=4,
    )
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", BOOK_ROWS[5:]
    )
    conn.commit()
    conn.close()

    exclusions = ExclusionInventory(read_gap_rows(db_path))
    seq = SeqBoundaryDetector()
    result = run_forward_pass(
        db_path,
        out_dir,
        [],
        accumulators=[exclusions, seq],
        read_batch_rows=3,
        barrier_rows=4,
    )

    assert result.rows == 2
    assert exclusions.windows() == [inventory_window()]
    assert seq.boundaries() == list(expected_inventory(1).boundaries)
