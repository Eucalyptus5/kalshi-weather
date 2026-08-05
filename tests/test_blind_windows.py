import hashlib
import logging
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.replay.artifacts import COVERAGE_SCHEMA
from bot.replay.blind_windows import (
    BLIND_WINDOWS_SCHEMA,
    FROZEN_END,
    FROZEN_START,
    BlindScan,
    BlindWindow,
    BlindWindowDetector,
    attach_gap_rows,
    build_summary,
    scan_blind_windows,
    write_blind_windows,
    write_coverage,
)
from bot.replay.forward_pass import SourceRow
from bot.replay.inventory import GapRow, SeqBoundaryDetector, TickerCoverage


UTC = timezone.utc
T0 = datetime(2026, 7, 30, 3, 1, 0, tzinfo=UTC)
DB_TS = "%Y-%m-%d %H:%M:%S.%f"
SECOND = 1_000_000
TICK = timedelta(microseconds=1)
TICKER = "KXHIGHDEN-26JUL30-B85"
NY = "KXHIGHNY-26JUL30-B70"
CHI = "KXHIGHCHI-26JUL30-B90"

BOOK_SCHEMA = """
CREATE TABLE ws_book_events (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, received_at DATETIME NOT NULL,
    seq INTEGER NOT NULL, is_snapshot BOOLEAN NOT NULL, PRIMARY KEY (id))
"""
GAP_SCHEMA = """
CREATE TABLE ws_gaps (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, detected_at DATETIME NOT NULL,
    last_seq INTEGER NOT NULL, reason VARCHAR(64) NOT NULL, PRIMARY KEY (id))
"""


def at(offset_us: int = 0) -> datetime:
    return T0 + timedelta(microseconds=offset_us)


def book(
    row_id: int, offset_us: int, seq: int, is_snapshot: bool = False, ticker: str = TICKER
) -> tuple[object, ...]:
    return (row_id, ticker, at(offset_us).strftime(DB_TS), seq, is_snapshot)


def gap(row_id: int, offset_us: int, reason: str = "connection_reset") -> tuple[object, ...]:
    return (row_id, "", at(offset_us).strftime(DB_TS), 0, reason)


def build_db(
    path: Path, rows: Sequence[tuple[object, ...]], gaps: Sequence[tuple[object, ...]] = ()
) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(BOOK_SCHEMA)
    conn.execute(GAP_SCHEMA)
    conn.executemany("INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?)", list(rows))
    conn.executemany("INSERT INTO ws_gaps VALUES (?, ?, ?, ?, ?)", list(gaps))
    conn.commit()
    conn.close()
    return path


def window(
    boundary_id: int,
    prev_id: int,
    start: datetime,
    end: datetime,
    prev_seq: int = 2,
    seq: int = 1,
    end_id: int | None = None,
    burst_messages: int = 1,
) -> BlindWindow:
    return BlindWindow(
        boundary_id=boundary_id,
        prev_id=prev_id,
        end_id=boundary_id if end_id is None else end_id,
        burst_messages=burst_messages,
        ticker="",
        start=start,
        end=end,
        prev_seq=prev_seq,
        seq=seq,
        gap_id=None,
        gap_reason=None,
        gap_detected_at=None,
    )


def coverage(rows: int, first_us: int, last_us: int, ticker: str = TICKER) -> TickerCoverage:
    return TickerCoverage(
        ticker=ticker, rows=rows, first_received_at=at(first_us), last_received_at=at(last_us)
    )


def gap_row(row_id: int, detected_at: datetime, reason: str = "connection_reset") -> GapRow:
    return GapRow(id=row_id, ticker="", detected_at=detected_at, last_seq=0, reason=reason)


def source_row(row_id: int, offset_us: int, seq: int) -> SourceRow:
    return SourceRow(
        id=row_id,
        ticker=TICKER,
        received_at=at(offset_us),
        seq=seq,
        side="yes",
        price="0.4000",
        size="1.00",
        is_snapshot=False,
        ts_ms=None,
    )


SCAN_END = at(30 * SECOND)

MIXED_ROWS = [
    book(1, 0, 1),
    book(2, 0, 1),
    book(3, 1 * SECOND, 2),
    book(4, 4 * SECOND, 1),
    book(5, 4 * SECOND, 1),
    book(6, 5 * SECOND, 2),
    book(7, 6 * SECOND, 3),
    book(8, 20 * SECOND, 1),
    book(9, 21 * SECOND, 2),
]

BURST_ROWS = [
    book(1, 0, 1),
    book(2, SECOND, 2),
    book(3, 4 * SECOND, 1, True),
    book(4, 4 * SECOND, 1, True),
    book(5, 5 * SECOND, 2, True),
    book(6, 6 * SECOND, 3),
    book(7, 20 * SECOND, 1, True),
    book(8, 21 * SECOND, 2, True),
]

INTERLEAVED_ROWS = [
    book(1, 0, 1),
    book(2, SECOND, 2, ticker=NY),
    book(3, 2 * SECOND, 3),
    book(4, 3 * SECOND, 4, ticker=CHI),
    book(5, 4 * SECOND, 5, ticker=NY),
    book(6, 5 * SECOND, 6),
]


def test_a_monotone_stream_holds_no_boundary(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db", [book(1, 0, 1), book(2, SECOND, 2), book(3, 2 * SECOND, 3)]
    )

    assert scan_blind_windows(db_path) == BlindScan(
        (), 3, at(2 * SECOND), (coverage(3, 0, 2 * SECOND),)
    )


def test_a_backwards_seq_is_one_window_bounded_by_the_messages_either_side(
    tmp_path: Path,
) -> None:
    rows = [book(1, 0, 1), book(2, SECOND, 2), book(3, 4 * SECOND, 1), book(4, 5 * SECOND, 2)]
    db_path = build_db(tmp_path / "state.db", rows)

    scan = scan_blind_windows(db_path)

    assert scan.rows == 4
    assert scan.windows == (window(3, 2, at(SECOND), at(4 * SECOND)),)


def test_rows_of_one_message_collapse_and_the_left_edge_is_the_message_before(
    tmp_path: Path,
) -> None:
    rows = [
        book(1, 0, 1),
        book(2, SECOND, 2),
        book(3, SECOND, 2),
        book(4, 4 * SECOND, 1),
        book(5, 4 * SECOND, 1),
        book(6, 4 * SECOND, 1),
        book(7, 5 * SECOND, 2),
    ]
    db_path = build_db(tmp_path / "state.db", rows)

    scan = scan_blind_windows(db_path)

    assert scan.rows == 7
    assert scan.windows == (window(4, 3, at(SECOND), at(4 * SECOND)),)


def test_the_window_runs_to_the_last_snapshot_of_the_redelivered_burst(tmp_path: Path) -> None:
    rows = [
        book(1, 0, 1),
        book(2, SECOND, 2),
        book(3, 4 * SECOND, 1, True),
        book(4, 5 * SECOND, 2, True),
        book(5, 6 * SECOND, 3),
    ]
    db_path = build_db(tmp_path / "state.db", rows)

    scan = scan_blind_windows(db_path)

    assert scan.windows == (window(3, 2, at(SECOND), at(5 * SECOND), end_id=4, burst_messages=2),)


def test_a_longer_burst_counts_every_snapshot_message_it_covers(tmp_path: Path) -> None:
    rows = [
        book(1, 0, 1),
        book(2, SECOND, 2),
        book(3, 4 * SECOND, 1, True),
        book(4, 5 * SECOND, 2, True),
        book(5, 6 * SECOND, 3, True),
        book(6, 7 * SECOND, 4),
    ]
    db_path = build_db(tmp_path / "state.db", rows)

    scan = scan_blind_windows(db_path)

    assert scan.windows == (window(3, 2, at(SECOND), at(6 * SECOND), end_id=5, burst_messages=3),)


def test_a_boundary_that_is_a_delta_closes_where_it_lands(tmp_path: Path) -> None:
    rows = [
        book(1, 0, 1),
        book(2, SECOND, 2),
        book(3, 4 * SECOND, 1),
        book(4, 5 * SECOND, 2, True),
        book(5, 6 * SECOND, 3),
    ]
    db_path = build_db(tmp_path / "state.db", rows)

    scan = scan_blind_windows(db_path)

    assert scan.windows == (window(3, 2, at(SECOND), at(4 * SECOND)),)
    assert (scan.windows[0].end_id, scan.windows[0].burst_messages) == (3, 1)


def test_rows_of_one_snapshot_message_in_the_burst_collapse_to_that_message(
    tmp_path: Path,
) -> None:
    rows = [
        book(1, 0, 1),
        book(2, SECOND, 2),
        book(3, 4 * SECOND, 1, True),
        book(4, 5 * SECOND, 2, True),
        book(5, 5 * SECOND, 2, True),
        book(6, 5 * SECOND, 2, True),
        book(7, 6 * SECOND, 3),
    ]
    db_path = build_db(tmp_path / "state.db", rows)

    scan = scan_blind_windows(db_path)

    assert scan.windows == (window(3, 2, at(SECOND), at(5 * SECOND), end_id=6, burst_messages=2),)


def test_a_boundary_inside_an_open_burst_closes_it_and_opens_its_own(tmp_path: Path) -> None:
    rows = [
        book(1, 0, 1),
        book(2, SECOND, 2),
        book(3, 4 * SECOND, 1, True),
        book(4, 5 * SECOND, 2, True),
        book(5, 6 * SECOND, 1, True),
        book(6, 7 * SECOND, 2),
    ]
    db_path = build_db(tmp_path / "state.db", rows)

    windows = scan_blind_windows(db_path).windows

    assert windows == (
        window(3, 2, at(SECOND), at(5 * SECOND), end_id=4, burst_messages=2),
        window(5, 4, at(5 * SECOND), at(6 * SECOND), end_id=5),
    )
    assert windows[0].end <= windows[1].start


def test_a_burst_still_open_when_the_scan_ends_is_still_a_window(tmp_path: Path) -> None:
    rows = [
        book(1, 0, 1),
        book(2, SECOND, 2),
        book(3, 4 * SECOND, 1, True),
        book(4, 5 * SECOND, 2, True),
    ]
    db_path = build_db(tmp_path / "state.db", rows)

    scan = scan_blind_windows(db_path)

    assert scan.rows == 4
    assert scan.windows == (window(3, 2, at(SECOND), at(5 * SECOND), end_id=4, burst_messages=2),)


def test_batching_does_not_split_a_burst(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", BURST_ROWS)

    one = scan_blind_windows(db_path, batch_rows=1)
    many = scan_blind_windows(db_path, batch_rows=10_000)

    assert one == many
    assert one.windows == (
        window(3, 2, at(SECOND), at(5 * SECOND), end_id=5, burst_messages=2),
        window(
            7,
            6,
            at(6 * SECOND),
            at(21 * SECOND),
            prev_seq=3,
            seq=1,
            end_id=8,
            burst_messages=2,
        ),
    )


def test_a_ceiling_inside_a_burst_still_yields_the_window(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", BURST_ROWS)

    scan = scan_blind_windows(db_path, max_id=4)

    assert scan.rows == 4
    assert scan.windows == (window(3, 2, at(SECOND), at(4 * SECOND), end_id=4),)


def test_a_forward_seq_skip_is_not_a_boundary(tmp_path: Path) -> None:
    rows = [book(1, 0, 1), book(2, SECOND, 2), book(3, 2 * SECOND, 9), book(4, 3 * SECOND, 10)]
    db_path = build_db(tmp_path / "state.db", rows)

    assert scan_blind_windows(db_path) == BlindScan(
        (), 4, at(3 * SECOND), (coverage(4, 0, 3 * SECOND),)
    )


def test_the_detector_needs_no_database() -> None:
    detector = BlindWindowDetector()
    for row_id, offset_us, seq in ((1, 0, 4), (2, SECOND, 5), (3, 3 * SECOND, 1)):
        detector.observe(row_id, at(offset_us).strftime(DB_TS), seq, False)

    assert detector.windows() == [window(3, 2, at(SECOND), at(3 * SECOND), prev_seq=5, seq=1)]


def test_a_seq_that_repeats_is_a_boundary_the_shipped_detector_also_counts() -> None:
    stream = ((1, 0, 4), (2, SECOND, 5), (3, 3 * SECOND, 5))
    detector = BlindWindowDetector()
    shipped = SeqBoundaryDetector()
    for row_id, offset_us, seq in stream:
        detector.observe(row_id, at(offset_us).strftime(DB_TS), seq, False)
        shipped.observe(source_row(row_id, offset_us, seq))

    assert detector.windows() == [window(3, 2, at(SECOND), at(3 * SECOND), prev_seq=5, seq=5)]
    assert (shipped.resubscribes(), shipped.skips()) == (1, 0)


def test_blind_us_is_a_whole_microsecond_count(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", [book(1, 0, 5), book(2, 845_123, 1)])

    blind = scan_blind_windows(db_path).windows[0].blind_us

    assert blind == 845_123
    assert isinstance(blind, int)


def test_batching_does_not_change_what_the_scan_finds(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", MIXED_ROWS)

    one = scan_blind_windows(db_path, batch_rows=1)
    many = scan_blind_windows(db_path, batch_rows=10_000)

    assert one == many
    assert one.rows == len(MIXED_ROWS)
    assert one.windows == (
        window(4, 3, at(SECOND), at(4 * SECOND)),
        window(8, 7, at(6 * SECOND), at(20 * SECOND), prev_seq=3, seq=1),
    )


def test_max_id_drops_a_boundary_above_the_ceiling(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", MIXED_ROWS)

    scan = scan_blind_windows(db_path, max_id=7)

    assert scan.rows == 7
    assert scan.windows == (window(4, 3, at(SECOND), at(4 * SECOND)),)


def test_the_scan_reports_the_received_at_of_the_last_row_it_read(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", MIXED_ROWS)

    assert scan_blind_windows(db_path).last_received_at == at(21 * SECOND)
    assert scan_blind_windows(db_path, max_id=7).last_received_at == at(6 * SECOND)


def test_a_scan_that_reads_nothing_has_no_last_row(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", [])

    assert scan_blind_windows(db_path) == BlindScan((), 0, None, ())


def test_each_read_is_its_own_statement_and_the_source_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot.replay import blind_windows

    db_path = build_db(tmp_path / "state.db", MIXED_ROWS)
    statements: list[str] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        conn = real_connect(target, *args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(blind_windows.sqlite3, "connect", spy)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    scan_blind_windows(db_path, batch_rows=2)

    selects = [line for line in statements if "ws_book_events" in line]
    assert len(selects) == -(-len(MIXED_ROWS) // 2) + 1
    assert all("id > " in line for line in selects)
    assert all("SELECT id, ticker, received_at, seq, is_snapshot FROM" in line for line in selects)
    assert "PRAGMA query_only=ON" in statements
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert not (tmp_path / "state.db-wal").exists()


def test_progress_is_logged_and_nothing_reaches_stdout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    caplog.set_level(logging.INFO, logger="bot.replay.blind_windows")
    db_path = build_db(tmp_path / "state.db", MIXED_ROWS)

    scan_blind_windows(db_path, batch_rows=3, progress_rows=3)

    messages = [r.getMessage() for r in caplog.records if r.name == "bot.replay.blind_windows"]
    assert any("rows=3" in m and "elapsed_s=" in m and "rows_per_s=" in m for m in messages), (
        messages
    )
    assert any("rows=9" in m for m in messages), messages
    assert capsys.readouterr().out == ""


def test_a_gap_row_attaches_to_the_window_whose_start_it_follows() -> None:
    windows = [
        window(3, 2, at(SECOND), at(4 * SECOND)),
        window(9, 8, at(20 * SECOND), at(24 * SECOND)),
    ]
    row = gap_row(7, at(5 * SECOND), reason="seq_skip")

    attached, unmatched = attach_gap_rows(windows, [row], scanned_through=SCAN_END)

    assert unmatched == []
    assert (attached[0].gap_id, attached[0].gap_reason) == (7, "seq_skip")
    assert attached[0].gap_detected_at == at(5 * SECOND)
    assert (attached[1].gap_id, attached[1].gap_reason, attached[1].gap_detected_at) == (
        None,
        None,
        None,
    )


def test_a_gap_row_earlier_than_every_window_matches_nothing() -> None:
    windows = [
        window(3, 2, at(SECOND), at(4 * SECOND)),
        window(9, 8, at(20 * SECOND), at(24 * SECOND)),
    ]
    row = gap_row(7, at(0))

    attached, unmatched = attach_gap_rows(windows, [row], scanned_through=SCAN_END)

    assert unmatched == [row]
    assert [w.gap_id for w in attached] == [None, None]


def test_a_gap_row_later_than_the_scanned_range_matches_nothing() -> None:
    windows = [
        window(3, 2, at(SECOND), at(4 * SECOND)),
        window(9, 8, at(20 * SECOND), at(24 * SECOND)),
    ]
    row = gap_row(7, at(120 * SECOND))

    attached, unmatched = attach_gap_rows(windows, [row], scanned_through=SCAN_END)

    assert unmatched == [row]
    assert [w.gap_id for w in attached] == [None, None]


@pytest.mark.parametrize(("detected_at", "reached"), [(SCAN_END - TICK, True), (SCAN_END, False)])
def test_the_last_window_runs_to_where_the_scan_stopped_reading(
    detected_at: datetime, reached: bool
) -> None:
    windows = [
        window(3, 2, at(SECOND), at(4 * SECOND)),
        window(9, 8, at(20 * SECOND), at(24 * SECOND)),
    ]
    row = gap_row(7, detected_at)

    attached, unmatched = attach_gap_rows(windows, [row], scanned_through=SCAN_END)

    assert (attached[1].gap_id == 7) is reached
    assert unmatched == ([] if reached else [row])


@pytest.mark.parametrize(
    ("detected_at", "reached"), [(at(20 * SECOND) - TICK, 3), (at(20 * SECOND), 9)]
)
def test_a_gap_row_stops_at_the_next_windows_start(detected_at: datetime, reached: int) -> None:
    windows = [
        window(3, 2, at(SECOND), at(4 * SECOND)),
        window(9, 8, at(20 * SECOND), at(24 * SECOND)),
    ]

    attached, unmatched = attach_gap_rows(
        windows, [gap_row(7, detected_at)], scanned_through=SCAN_END
    )

    assert unmatched == []
    assert [w.boundary_id for w in attached if w.gap_id == 7] == [reached]


def test_a_gap_row_stamped_after_the_window_closed_still_belongs_to_it() -> None:
    windows = [
        window(3, 2, at(SECOND), at(4 * SECOND)),
        window(9, 8, at(20 * SECOND), at(24 * SECOND)),
    ]
    row = gap_row(7, at(19 * SECOND))

    attached, unmatched = attach_gap_rows(windows, [row], scanned_through=SCAN_END)

    assert unmatched == []
    assert attached[0].gap_id == 7
    assert attached[0].gap_detected_at > attached[0].end


def test_the_earliest_gap_row_wins_a_window_and_the_rest_go_unmatched() -> None:
    windows = [window(3, 2, at(SECOND), at(4 * SECOND))]
    first = gap_row(7, at(5 * SECOND))
    second = gap_row(8, at(6 * SECOND))

    attached, unmatched = attach_gap_rows(windows, [second, first], scanned_through=SCAN_END)

    assert attached[0].gap_id == 7
    assert unmatched == [second]


@pytest.mark.parametrize(
    ("widths", "p50", "p90", "total"),
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, 9, 55),
        ([1, 2, 3, 4, 5, 6, 7], 4, 7, 28),
        ([9], 9, 9, 9),
    ],
)
def test_the_percentiles_are_nearest_rank(
    widths: list[int], p50: int, p90: int, total: int
) -> None:
    windows = [
        window(i, i - 1, at(i * SECOND), at(i * SECOND + width))
        for i, width in enumerate(reversed(widths), start=1)
    ]

    summary = build_summary(windows)

    assert summary.percentiles == "nearest_rank"
    assert (summary.p50_blind_us, summary.p90_blind_us) == (p50, p90)
    assert summary.max_blind_us == max(widths)
    assert summary.total_blind_us == total
    assert summary.windows == len(widths)


def test_the_split_counts_the_windows_a_gap_row_reached() -> None:
    windows = [
        window(3, 2, at(SECOND), at(4 * SECOND)),
        window(9, 8, at(20 * SECOND), at(24 * SECOND)),
        window(15, 14, at(25 * SECOND), at(26 * SECOND)),
    ]
    attached, _ = attach_gap_rows(windows, [gap_row(7, at(5 * SECOND))], scanned_through=SCAN_END)

    summary = build_summary(attached)

    assert (summary.with_gap_row, summary.without_gap_row) == (1, 2)


def test_the_frozen_subset_takes_the_start_bound_and_drops_the_microsecond_before_it() -> None:
    early = window(1, 0, FROZEN_START - TICK, FROZEN_START)
    bound = window(2, 1, FROZEN_START, FROZEN_START + 2 * TICK)
    last = window(3, 2, FROZEN_END - TICK, FROZEN_END)
    after = window(4, 3, FROZEN_END, FROZEN_END + TICK)

    summary = build_summary([early, bound, last, after])

    assert summary.windows == 4
    assert summary.frozen_windows == 2
    assert summary.frozen_total_blind_us == 3
    assert (summary.frozen_max_blind_us, summary.frozen_p50_blind_us) == (2, 1)
    assert (summary.frozen_start, summary.frozen_end) == (FROZEN_START, FROZEN_END)


def test_the_frozen_window_is_the_fifteen_days_from_the_eighteenth() -> None:
    assert FROZEN_START == datetime(2026, 7, 18, tzinfo=UTC)
    assert FROZEN_END == datetime(2026, 8, 2, tzinfo=UTC)
    assert (FROZEN_END - FROZEN_START).days == 15


def test_the_parquet_round_trips_every_column(tmp_path: Path) -> None:
    path = tmp_path / "blind.parquet"
    windows = [
        window(3, 2, at(SECOND), at(2 * SECOND + 398_909), end_id=5, burst_messages=3),
        window(9, 8, FROZEN_START - TICK, FROZEN_START),
    ]
    attached, _ = attach_gap_rows(
        windows, [gap_row(7, at(5 * SECOND), reason="seq_skip")], scanned_through=SCAN_END
    )

    write_blind_windows(path, attached)
    table = pq.read_table(path)

    assert table.schema.equals(BLIND_WINDOWS_SCHEMA)
    assert table.schema.names == [
        "boundary_id",
        "prev_id",
        "end_id",
        "burst_messages",
        "ticker",
        "start",
        "end",
        "blind_us",
        "prev_seq",
        "seq",
        "has_gap_row",
        "gap_id",
        "gap_reason",
        "gap_detected_at",
        "in_frozen_window",
    ]
    assert table.to_pylist() == [
        {
            "boundary_id": 9,
            "prev_id": 8,
            "end_id": 9,
            "burst_messages": 1,
            "ticker": "",
            "start": FROZEN_START - TICK,
            "end": FROZEN_START,
            "blind_us": 1,
            "prev_seq": 2,
            "seq": 1,
            "has_gap_row": False,
            "gap_id": None,
            "gap_reason": None,
            "gap_detected_at": None,
            "in_frozen_window": False,
        },
        {
            "boundary_id": 3,
            "prev_id": 2,
            "end_id": 5,
            "burst_messages": 3,
            "ticker": "",
            "start": at(SECOND),
            "end": at(2 * SECOND + 398_909),
            "blind_us": 1_398_909,
            "prev_seq": 2,
            "seq": 1,
            "has_gap_row": True,
            "gap_id": 7,
            "gap_reason": "seq_skip",
            "gap_detected_at": at(5 * SECOND),
            "in_frozen_window": True,
        },
    ]


def test_nothing_already_written_is_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "blind.parquet"
    path.write_bytes(b"PAR1")

    with pytest.raises(FileExistsError, match=str(path)):
        write_blind_windows(path, [window(3, 2, at(SECOND), at(4 * SECOND))])

    assert path.read_bytes() == b"PAR1"


def test_coverage_follows_each_ticker_across_the_rows_that_interleave_it(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", INTERLEAVED_ROWS)

    scan = scan_blind_windows(db_path)

    assert scan.coverage == (
        coverage(1, 3 * SECOND, 3 * SECOND, ticker=CHI),
        coverage(3, 0, 5 * SECOND),
        coverage(2, SECOND, 4 * SECOND, ticker=NY),
    )
    assert [row.ticker for row in scan.coverage] == sorted(row.ticker for row in scan.coverage)


def test_batching_does_not_change_the_coverage_the_scan_accumulates(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", INTERLEAVED_ROWS)

    assert scan_blind_windows(db_path, batch_rows=1) == scan_blind_windows(
        db_path, batch_rows=10_000
    )


def test_the_rows_of_a_repeated_message_are_each_counted(tmp_path: Path) -> None:
    rows = [book(1, 0, 1), book(2, SECOND, 2), book(3, SECOND, 2), book(4, SECOND, 2)]
    db_path = build_db(tmp_path / "state.db", rows)

    scan = scan_blind_windows(db_path)

    assert scan.coverage == (coverage(4, 0, SECOND),)
    assert scan.windows == ()


def test_the_coverage_counts_add_up_to_the_rows_the_scan_read(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", MIXED_ROWS)

    scan = scan_blind_windows(db_path)

    assert sum(row.rows for row in scan.coverage) == scan.rows == len(MIXED_ROWS)


def test_a_ceiling_bounds_the_coverage_too(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", INTERLEAVED_ROWS)

    scan = scan_blind_windows(db_path, max_id=4)

    assert scan.coverage == (
        coverage(1, 3 * SECOND, 3 * SECOND, ticker=CHI),
        coverage(2, 0, 2 * SECOND),
        coverage(1, SECOND, SECOND, ticker=NY),
    )


def test_the_coverage_parquet_round_trips_every_column(tmp_path: Path) -> None:
    path = tmp_path / "coverage.parquet"
    rows = [coverage(3, 0, 5 * SECOND), coverage(2, SECOND, 4 * SECOND, ticker=NY)]

    write_coverage(path, rows)
    table = pq.read_table(path)

    assert table.schema.equals(COVERAGE_SCHEMA)
    assert table.to_pylist() == [
        {
            "ticker": TICKER,
            "rows": 3,
            "first_received_at": at(0),
            "last_received_at": at(5 * SECOND),
        },
        {
            "ticker": NY,
            "rows": 2,
            "first_received_at": at(SECOND),
            "last_received_at": at(4 * SECOND),
        },
    ]


def test_no_coverage_already_written_is_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "coverage.parquet"
    path.write_bytes(b"PAR1")

    with pytest.raises(FileExistsError, match=str(path)):
        write_coverage(path, [coverage(1, 0, 0)])

    assert path.read_bytes() == b"PAR1"


def test_an_injected_frozen_window_decides_the_column(tmp_path: Path) -> None:
    path = tmp_path / "blind.parquet"
    start = at(2 * SECOND)
    end = at(5 * SECOND)
    windows = [
        window(1, 0, start - TICK, start),
        window(2, 1, start, start + TICK),
        window(3, 2, end - TICK, end),
        window(4, 3, end, end + TICK),
    ]

    write_blind_windows(path, windows, frozen_start=start, frozen_end=end)

    rows = pq.read_table(path).to_pylist()
    assert [row["in_frozen_window"] for row in rows] == [False, True, True, False]


def test_the_module_window_is_what_the_defaults_write(tmp_path: Path) -> None:
    path = tmp_path / "blind.parquet"
    windows = [
        window(1, 0, FROZEN_START - TICK, FROZEN_START),
        window(2, 1, FROZEN_START, FROZEN_START + TICK),
        window(3, 2, FROZEN_END - TICK, FROZEN_END),
        window(4, 3, FROZEN_END, FROZEN_END + TICK),
    ]

    write_blind_windows(path, windows)

    rows = pq.read_table(path).to_pylist()
    assert [row["in_frozen_window"] for row in rows] == [False, True, True, False]
