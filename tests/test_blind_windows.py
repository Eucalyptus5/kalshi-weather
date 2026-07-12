import hashlib
import logging
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.replay.blind_windows import (
    BLIND_WINDOWS_SCHEMA,
    FROZEN_END,
    FROZEN_START,
    BlindWindow,
    BlindWindowDetector,
    attach_gap_rows,
    build_summary,
    scan_blind_windows,
    write_blind_windows,
)
from bot.replay.forward_pass import SourceRow
from bot.replay.inventory import GapRow, SeqBoundaryDetector


UTC = timezone.utc
T0 = datetime(2026, 7, 30, 3, 1, 0, tzinfo=UTC)
DB_TS = "%Y-%m-%d %H:%M:%S.%f"
SECOND = 1_000_000
TICK = timedelta(microseconds=1)

BOOK_SCHEMA = """
CREATE TABLE ws_book_events (
    id INTEGER NOT NULL, received_at DATETIME NOT NULL, seq INTEGER NOT NULL, PRIMARY KEY (id))
"""
GAP_SCHEMA = """
CREATE TABLE ws_gaps (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, detected_at DATETIME NOT NULL,
    last_seq INTEGER NOT NULL, reason VARCHAR(64) NOT NULL, PRIMARY KEY (id))
"""


def at(offset_us: int = 0) -> datetime:
    return T0 + timedelta(microseconds=offset_us)


def book(row_id: int, offset_us: int, seq: int) -> tuple[object, ...]:
    return (row_id, at(offset_us).strftime(DB_TS), seq)


def gap(row_id: int, offset_us: int, reason: str = "connection_reset") -> tuple[object, ...]:
    return (row_id, "", at(offset_us).strftime(DB_TS), 0, reason)


def build_db(
    path: Path, rows: Sequence[tuple[object, ...]], gaps: Sequence[tuple[object, ...]] = ()
) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(BOOK_SCHEMA)
    conn.execute(GAP_SCHEMA)
    conn.executemany("INSERT INTO ws_book_events VALUES (?, ?, ?)", list(rows))
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
) -> BlindWindow:
    return BlindWindow(
        boundary_id=boundary_id,
        prev_id=prev_id,
        ticker="",
        start=start,
        end=end,
        prev_seq=prev_seq,
        seq=seq,
        gap_id=None,
        gap_reason=None,
        gap_detected_at=None,
    )


def gap_row(row_id: int, detected_at: datetime, reason: str = "connection_reset") -> GapRow:
    return GapRow(id=row_id, ticker="", detected_at=detected_at, last_seq=0, reason=reason)


def source_row(row_id: int, offset_us: int, seq: int) -> SourceRow:
    return SourceRow(
        id=row_id,
        ticker="KXHIGHDEN-26JUL30-B85",
        received_at=at(offset_us),
        seq=seq,
        side="yes",
        price="0.4000",
        size="1.00",
        is_snapshot=False,
        ts_ms=None,
    )


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


def test_a_monotone_stream_holds_no_boundary(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db", [book(1, 0, 1), book(2, SECOND, 2), book(3, 2 * SECOND, 3)]
    )

    assert scan_blind_windows(db_path) == ([], 3)


def test_a_backwards_seq_is_one_window_bounded_by_the_messages_either_side(
    tmp_path: Path,
) -> None:
    rows = [book(1, 0, 1), book(2, SECOND, 2), book(3, 4 * SECOND, 1), book(4, 5 * SECOND, 2)]
    db_path = build_db(tmp_path / "state.db", rows)

    windows, scanned = scan_blind_windows(db_path)

    assert scanned == 4
    assert windows == [window(3, 2, at(SECOND), at(4 * SECOND))]


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

    windows, scanned = scan_blind_windows(db_path)

    assert scanned == 7
    assert windows == [window(4, 3, at(SECOND), at(4 * SECOND))]


def test_a_forward_seq_skip_is_not_a_boundary(tmp_path: Path) -> None:
    rows = [book(1, 0, 1), book(2, SECOND, 2), book(3, 2 * SECOND, 9), book(4, 3 * SECOND, 10)]
    db_path = build_db(tmp_path / "state.db", rows)

    assert scan_blind_windows(db_path) == ([], 4)


def test_the_detector_needs_no_database() -> None:
    detector = BlindWindowDetector()
    for row_id, offset_us, seq in ((1, 0, 4), (2, SECOND, 5), (3, 3 * SECOND, 1)):
        detector.observe(row_id, at(offset_us).strftime(DB_TS), seq)

    assert detector.windows() == [window(3, 2, at(SECOND), at(3 * SECOND), prev_seq=5, seq=1)]


def test_a_seq_that_repeats_is_a_boundary_the_shipped_detector_also_counts() -> None:
    stream = ((1, 0, 4), (2, SECOND, 5), (3, 3 * SECOND, 5))
    detector = BlindWindowDetector()
    shipped = SeqBoundaryDetector()
    for row_id, offset_us, seq in stream:
        detector.observe(row_id, at(offset_us).strftime(DB_TS), seq)
        shipped.observe(source_row(row_id, offset_us, seq))

    assert detector.windows() == [window(3, 2, at(SECOND), at(3 * SECOND), prev_seq=5, seq=5)]
    assert (shipped.resubscribes(), shipped.skips()) == (1, 0)


def test_blind_us_is_a_whole_microsecond_count(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", [book(1, 0, 5), book(2, 845_123, 1)])

    blind = scan_blind_windows(db_path)[0][0].blind_us

    assert blind == 845_123
    assert isinstance(blind, int)


def test_batching_does_not_change_what_the_scan_finds(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", MIXED_ROWS)

    one = scan_blind_windows(db_path, batch_rows=1)
    many = scan_blind_windows(db_path, batch_rows=10_000)

    assert one == many
    assert one[1] == len(MIXED_ROWS)
    assert one[0] == [
        window(4, 3, at(SECOND), at(4 * SECOND)),
        window(8, 7, at(6 * SECOND), at(20 * SECOND), prev_seq=3, seq=1),
    ]


def test_max_id_drops_a_boundary_above_the_ceiling(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", MIXED_ROWS)

    windows, scanned = scan_blind_windows(db_path, max_id=7)

    assert scanned == 7
    assert windows == [window(4, 3, at(SECOND), at(4 * SECOND))]


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

    attached, unmatched = attach_gap_rows(windows, [row])

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

    attached, unmatched = attach_gap_rows(windows, [row])

    assert unmatched == [row]
    assert [w.gap_id for w in attached] == [None, None]


def test_the_earliest_gap_row_wins_a_window_and_the_rest_go_unmatched() -> None:
    windows = [window(3, 2, at(SECOND), at(4 * SECOND))]
    first = gap_row(7, at(5 * SECOND))
    second = gap_row(8, at(6 * SECOND))

    attached, unmatched = attach_gap_rows(windows, [second, first])

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
        window(15, 14, at(40 * SECOND), at(41 * SECOND)),
    ]
    attached, _ = attach_gap_rows(windows, [gap_row(7, at(5 * SECOND))])

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
        window(3, 2, at(SECOND), at(2 * SECOND + 398_909)),
        window(9, 8, FROZEN_START - TICK, FROZEN_START),
    ]
    attached, _ = attach_gap_rows(windows, [gap_row(7, at(5 * SECOND), reason="seq_skip")])

    write_blind_windows(path, attached)
    table = pq.read_table(path)

    assert table.schema.equals(BLIND_WINDOWS_SCHEMA)
    assert table.to_pylist() == [
        {
            "boundary_id": 9,
            "prev_id": 8,
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
