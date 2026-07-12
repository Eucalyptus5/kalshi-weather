import logging
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.main import WsRawTape
from bot.replay.blind_windows import FROZEN_END, FROZEN_START
from bot.replay.clears import (
    MID_LIFE_CLEARS_SCHEMA,
    MidLifeClear,
    build_summary,
    scan_clears,
    write_mid_life_clears,
)
from tests.test_raw_tape import AUS, DEN, frame


UTC = timezone.utc
T0 = datetime(2026, 7, 30, 3, 0, 0, tzinfo=UTC)
SECOND = 1_000_000
TICK = timedelta(microseconds=1)
DB_TS = "%Y-%m-%d %H:%M:%S.%f"

BOOK_SCHEMA = """
CREATE TABLE ws_book_events (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, received_at DATETIME NOT NULL,
    seq INTEGER NOT NULL, side VARCHAR(8) NOT NULL, price TEXT NOT NULL, size TEXT NOT NULL,
    is_snapshot BOOLEAN NOT NULL, ts_ms INTEGER, PRIMARY KEY (id))
"""


def at(offset_us: int = 0) -> datetime:
    return T0 + timedelta(microseconds=offset_us)


def row(
    row_id: int,
    stamp: datetime,
    side: str,
    price: str,
    size: str,
    *,
    seq: int = 1,
    snapshot: bool = False,
    ticker: str = AUS,
) -> tuple[object, ...]:
    return (row_id, ticker, stamp.strftime(DB_TS), seq, side, price, size, int(snapshot), None)


def build_db(path: Path, rows: Sequence[tuple[object, ...]]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(BOOK_SCHEMA)
    conn.executemany("INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", list(rows))
    conn.commit()
    conn.close()
    return path


def write_clears(directory: Path, stamps: Sequence[datetime], ticker: str = AUS) -> list[Path]:
    tape = WsRawTape(directory)
    for seq, stamp in enumerate(stamps, start=1):
        tape.write(frame("orderbook_snapshot", seq, ticker), stamp)
    tape.close()
    return sorted(directory.glob("*.jsonl.gz"))


def mid_life_row(**overrides: object) -> MidLifeClear:
    fields: dict[str, object] = {
        "ticker": AUS,
        "clear_at": at(10 * SECOND),
        "anchor_at": at(0),
        "yes_levels": 2,
        "no_levels": 1,
        "total_levels": 3,
        "next_row_at": at(11 * SECOND),
        "next_snapshot_at": at(20 * SECOND),
        "stale_end": at(20 * SECOND),
        "stale_rows": 2,
        "unresolved": False,
        "clear_since_anchor": False,
    }
    return MidLifeClear(**(fields | overrides))


ANCHORED = [
    row(1, at(0), "yes", "0.5000", "10.00", snapshot=True),
    row(2, at(0), "yes", "0.4900", "5.00", snapshot=True),
    row(3, at(0), "no", "0.4000", "7.00", snapshot=True),
]


def test_a_clear_with_nothing_after_it_is_terminal_and_never_folds_the_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot.replay import clears

    db_path = build_db(tmp_path / "state.db", ANCHORED)
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND)])
    statements: list[str] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        conn = real_connect(target, *args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(clears.sqlite3, "connect", spy)
    scan = scan_clears(db_path, paths)

    assert (scan.clears, scan.terminal, scan.no_anchor, scan.empty_book) == (1, 1, 0, 0)
    assert scan.mid_life == ()
    reads = [line for line in statements if "ws_book_events" in line]
    assert len(reads) == 1
    assert "is_snapshot" not in reads[0]
    assert "PRAGMA query_only=ON" in statements


def test_later_activity_without_a_snapshot_to_anchor_on_is_counted(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            row(1, at(0), "yes", "0.5000", "10.00"),
            row(2, at(20 * SECOND), "yes", "0.5000", "-10.00"),
            row(3, at(30 * SECOND), "yes", "0.5000", "4.00", seq=2, snapshot=True),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND)])

    scan = scan_clears(db_path, paths)

    assert (scan.clears, scan.terminal, scan.no_anchor, scan.empty_book) == (1, 0, 1, 0)
    assert scan.mid_life == ()


def test_a_fold_the_deltas_emptied_holds_nothing_the_clear_could_wipe(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            *ANCHORED,
            row(4, at(1 * SECOND), "yes", "0.5000", "-10.00"),
            row(5, at(2 * SECOND), "yes", "0.4900", "-5.00"),
            row(6, at(3 * SECOND), "no", "0.4000", "-7.00"),
            row(7, at(20 * SECOND), "yes", "0.6000", "3.00"),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND)])

    scan = scan_clears(db_path, paths)

    assert (scan.clears, scan.terminal, scan.no_anchor, scan.empty_book) == (1, 0, 0, 1)
    assert scan.mid_life == ()


def test_a_clear_over_a_live_book_records_what_the_fold_believed(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            *ANCHORED,
            row(4, at(11 * SECOND), "yes", "0.5000", "-1.00"),
            row(5, at(12 * SECOND), "yes", "0.5000", "-1.00"),
            row(6, at(20 * SECOND), "yes", "0.6000", "4.00", seq=2, snapshot=True),
            row(7, at(21 * SECOND), "yes", "0.6000", "-1.00", seq=3),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND)])

    scan = scan_clears(db_path, paths)

    assert (scan.clears, scan.terminal, scan.no_anchor, scan.empty_book) == (1, 0, 0, 0)
    assert scan.mid_life == (mid_life_row(),)
    assert scan.mid_life[0].stale_us == 10 * SECOND
    assert isinstance(scan.mid_life[0].stale_us, int)


def test_the_stale_span_takes_neither_edge_and_both_microseconds_inside_them(
    tmp_path: Path,
) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            row(1, at(0), "yes", "0.5000", "10.00", snapshot=True),
            row(2, at(10 * SECOND), "yes", "0.4000", "2.00"),
            row(3, at(10 * SECOND) + TICK, "yes", "0.4000", "1.00"),
            row(4, at(20 * SECOND) - TICK, "yes", "0.4000", "1.00"),
            row(5, at(20 * SECOND), "yes", "0.6000", "4.00", seq=2, snapshot=True),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND)])

    scan = scan_clears(db_path, paths)

    assert scan.mid_life == (
        mid_life_row(
            yes_levels=2,
            no_levels=0,
            total_levels=2,
            next_row_at=at(10 * SECOND) + TICK,
            stale_rows=2,
        ),
    )


def test_a_clear_no_snapshot_ever_corrects_is_stale_to_the_last_row(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            *ANCHORED,
            row(4, at(11 * SECOND), "yes", "0.5000", "-1.00"),
            row(5, at(12 * SECOND), "yes", "0.5000", "-1.00"),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND)])

    scan = scan_clears(db_path, paths)
    summary = build_summary(scan)

    assert scan.mid_life == (
        mid_life_row(next_snapshot_at=None, stale_end=at(12 * SECOND), unresolved=True),
    )
    assert scan.mid_life[0].stale_us == 2 * SECOND
    assert (summary.unresolved, summary.mid_life) == (1, 1)


def test_only_a_clear_after_the_anchor_marks_the_fold_as_unreliable(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            row(1, at(-20 * SECOND), "yes", "0.5000", "10.00", snapshot=True),
            row(2, at(0), "yes", "0.5000", "10.00", seq=2, snapshot=True),
            row(3, at(11 * SECOND), "yes", "0.5000", "-1.00", seq=3),
            row(4, at(20 * SECOND), "yes", "0.6000", "4.00", seq=4, snapshot=True),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(-10 * SECOND), at(5 * SECOND), at(10 * SECOND)])

    scan = scan_clears(db_path, paths)
    summary = build_summary(scan)

    assert [clear.clear_at for clear in scan.mid_life] == [
        at(-10 * SECOND),
        at(5 * SECOND),
        at(10 * SECOND),
    ]
    assert [clear.anchor_at for clear in scan.mid_life] == [at(-20 * SECOND), at(0), at(0)]
    assert [clear.clear_since_anchor for clear in scan.mid_life] == [False, False, True]
    assert [clear.stale_rows for clear in scan.mid_life] == [0, 1, 1]
    assert (summary.mid_life, summary.stale_rows, summary.with_stale_rows) == (3, 2, 2)


def test_a_clear_below_the_frozen_bound_is_not_enumerated(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            row(1, FROZEN_START - timedelta(seconds=1), "yes", "0.5000", "10.00", snapshot=True),
            row(2, FROZEN_START + timedelta(seconds=1), "yes", "0.5000", "-1.00"),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [FROZEN_START - TICK, FROZEN_START])

    scan = scan_clears(db_path, paths)

    assert (len(paths), scan.days, scan.clears) == (2, 2, 1)
    assert [clear.clear_at for clear in scan.mid_life] == [FROZEN_START]


def test_a_clear_above_the_frozen_bound_is_not_enumerated(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            row(1, FROZEN_END - timedelta(seconds=1), "yes", "0.5000", "10.00", snapshot=True),
            row(2, FROZEN_END + timedelta(seconds=1), "yes", "0.5000", "-1.00"),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [FROZEN_END])

    scan = scan_clears(db_path, paths)

    assert (scan.clears, scan.mid_life) == (0, ())


def test_a_clear_on_one_ticker_says_nothing_about_another(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            *ANCHORED,
            row(4, at(11 * SECOND), "yes", "0.5000", "-1.00"),
            row(5, at(20 * SECOND), "yes", "0.6000", "4.00", seq=2, snapshot=True),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND)], ticker=DEN)

    scan = scan_clears(db_path, paths)

    assert (scan.clears, scan.terminal, scan.mid_life) == (1, 1, ())


def test_a_negative_running_level_stops_the_scan(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            row(1, at(0), "yes", "0.5000", "10.00", snapshot=True),
            row(2, at(1 * SECOND), "yes", "0.5000", "-11.00"),
            row(3, at(20 * SECOND), "yes", "0.6000", "4.00", seq=2, snapshot=True),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND)])

    with pytest.raises(ValueError, match="negative level"):
        scan_clears(db_path, paths)


def test_each_day_reports_its_own_progress(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    caplog.set_level(logging.INFO, logger="bot.replay.clears")
    db_path = build_db(tmp_path / "state.db", ANCHORED)
    paths = write_clears(
        tmp_path / "ws_raw", [at(10 * SECOND), at(10 * SECOND) + timedelta(days=1)]
    )

    scan_clears(db_path, paths)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert all("2026-07-3" in message and "clears=" in message for message in messages), messages
    assert capsys.readouterr().out == ""


def test_the_summary_counts_every_class_and_the_mid_life_totals(tmp_path: Path) -> None:
    db_path = build_db(
        tmp_path / "state.db",
        [
            *ANCHORED,
            row(4, at(11 * SECOND), "yes", "0.5000", "-1.00"),
            row(5, at(20 * SECOND), "yes", "0.6000", "4.00", seq=2, snapshot=True),
            row(6, at(30 * SECOND), "yes", "0.6000", "-1.00", seq=3),
        ],
    )
    paths = write_clears(tmp_path / "ws_raw", [at(10 * SECOND), at(19 * SECOND), at(40 * SECOND)])

    summary = build_summary(scan_clears(db_path, paths))

    assert (summary.days, summary.clears) == (1, 3)
    assert (summary.terminal, summary.no_anchor, summary.empty_book) == (1, 0, 0)
    assert (summary.mid_life, summary.unresolved, summary.with_stale_rows) == (2, 0, 1)
    assert (summary.stale_rows, summary.stale_us) == (1, 11 * SECOND)


def test_the_parquet_round_trips_every_column(tmp_path: Path) -> None:
    path = tmp_path / "clears.parquet"
    rows = [
        mid_life_row(),
        mid_life_row(
            ticker=DEN,
            next_snapshot_at=None,
            stale_end=at(12 * SECOND),
            stale_rows=0,
            unresolved=True,
            clear_since_anchor=True,
        ),
    ]

    write_mid_life_clears(path, rows)
    table = pq.read_table(path)

    assert table.schema.equals(MID_LIFE_CLEARS_SCHEMA)
    assert table.to_pylist() == [
        {
            "ticker": AUS,
            "clear_at": at(10 * SECOND),
            "anchor_at": at(0),
            "yes_levels": 2,
            "no_levels": 1,
            "total_levels": 3,
            "next_row_at": at(11 * SECOND),
            "next_snapshot_at": at(20 * SECOND),
            "stale_end": at(20 * SECOND),
            "stale_us": 10 * SECOND,
            "stale_rows": 2,
            "unresolved": False,
            "clear_since_anchor": False,
        },
        {
            "ticker": DEN,
            "clear_at": at(10 * SECOND),
            "anchor_at": at(0),
            "yes_levels": 2,
            "no_levels": 1,
            "total_levels": 3,
            "next_row_at": at(11 * SECOND),
            "next_snapshot_at": None,
            "stale_end": at(12 * SECOND),
            "stale_us": 2 * SECOND,
            "stale_rows": 0,
            "unresolved": True,
            "clear_since_anchor": True,
        },
    ]


def test_nothing_already_written_is_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "clears.parquet"
    path.write_bytes(b"PAR1")

    with pytest.raises(FileExistsError, match=str(path)):
        write_mid_life_clears(path, [mid_life_row()])

    assert path.read_bytes() == b"PAR1"
