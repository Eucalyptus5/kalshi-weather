import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bot.replay.blind_windows import FROZEN_END, FROZEN_START
from bot.replay.forward_pass import _decode_ts
from bot.replay.parity import _db, _oracle_levels
from bot.replay.raw_tape import ClearFrame, read_clear_frames


logger = logging.getLogger(__name__)

TERMINAL = "terminal"
NO_ANCHOR = "no_anchor"
EMPTY_BOOK = "empty_book"
MID_LIFE = "mid_life"

_ZERO = Decimal("0")
_MICROSECOND = timedelta(microseconds=1)

_FIRST_AFTER = (
    "SELECT received_at FROM ws_book_events "
    "WHERE ticker = ? AND received_at > ? ORDER BY received_at LIMIT 1"
)
_NEXT_SNAPSHOT = (
    "SELECT received_at FROM ws_book_events "
    "WHERE ticker = ? AND is_snapshot = 1 AND received_at > ? ORDER BY received_at LIMIT 1"
)
_LAST_ROW = (
    "SELECT received_at FROM ws_book_events WHERE ticker = ? ORDER BY received_at DESC LIMIT 1"
)
_ROWS_AFTER = "SELECT COUNT(*) FROM ws_book_events WHERE ticker = ? AND received_at > ?"
_ROWS_BETWEEN = f"{_ROWS_AFTER} AND received_at < ?"

MID_LIFE_CLEARS_SCHEMA = pa.schema(
    [
        ("ticker", pa.string()),
        ("clear_at", pa.timestamp("us", tz="UTC")),
        ("anchor_at", pa.timestamp("us", tz="UTC")),
        ("yes_levels", pa.int64()),
        ("no_levels", pa.int64()),
        ("total_levels", pa.int64()),
        ("next_row_at", pa.timestamp("us", tz="UTC")),
        ("next_snapshot_at", pa.timestamp("us", tz="UTC")),
        ("stale_end", pa.timestamp("us", tz="UTC")),
        ("stale_us", pa.int64()),
        ("stale_rows", pa.int64()),
        ("unresolved", pa.bool_()),
        ("clear_since_anchor", pa.bool_()),
    ]
)


@dataclass(frozen=True, slots=True)
class MidLifeClear:
    ticker: str
    clear_at: datetime
    anchor_at: datetime
    yes_levels: int
    no_levels: int
    total_levels: int
    next_row_at: datetime
    # A both-sides-empty snapshot writes no rows, so the staleness can only end at a populated
    # one; when none arrives the fold is wrong for the rest of the ticker's recorded life.
    next_snapshot_at: datetime | None
    stale_end: datetime
    stale_rows: int
    unresolved: bool
    clear_since_anchor: bool

    @property
    def stale_us(self) -> int:
        return (self.stale_end - self.clear_at) // _MICROSECOND


@dataclass(frozen=True, slots=True)
class ClearScan:
    days: int
    clears: int
    terminal: int
    no_anchor: int
    empty_book: int
    mid_life: tuple[MidLifeClear, ...]


@dataclass(frozen=True, slots=True)
class ClearSummary:
    days: int
    clears: int
    terminal: int
    no_anchor: int
    empty_book: int
    mid_life: int
    stale_rows: int
    stale_us: int
    unresolved: int
    with_stale_rows: int


def scan_clears(db_path: Path, paths: Sequence[Path]) -> ClearScan:
    counts = {TERMINAL: 0, NO_ANCHOR: 0, EMPTY_BOOK: 0}
    mid_life: list[MidLifeClear] = []
    seen: dict[str, list[datetime]] = {}
    clears = 0
    conn = _connect(db_path)
    try:
        for path in paths:
            size = path.stat().st_size
            logger.info("clears reading path=%s bytes=%d", path.name, size)
            started = time.monotonic()
            found = 0
            for frame in read_clear_frames(path):
                if not FROZEN_START <= frame.received_at < FROZEN_END:
                    continue
                found += 1
                earlier = seen.setdefault(frame.ticker, [])
                status, clear = _classify(conn, frame, earlier)
                earlier.append(frame.received_at)
                if clear is None:
                    counts[status] += 1
                else:
                    mid_life.append(clear)
            clears += found
            elapsed = time.monotonic() - started
            logger.info(
                "clears path=%s found=%d clears=%d mid_life=%d elapsed_s=%.3f bytes_per_s=%.0f",
                path.name,
                found,
                clears,
                len(mid_life),
                elapsed,
                size / elapsed,
            )
    finally:
        conn.close()
    return ClearScan(
        days=len(paths),
        clears=clears,
        terminal=counts[TERMINAL],
        no_anchor=counts[NO_ANCHOR],
        empty_book=counts[EMPTY_BOOK],
        mid_life=tuple(mid_life),
    )


# Terminality is one indexed seek and holds for the overwhelming majority of clears; the fold
# behind it reads every row back to the anchor, so it only runs on what survives.
def _classify(
    conn: sqlite3.Connection, frame: ClearFrame, earlier: Sequence[datetime]
) -> tuple[str, MidLifeClear | None]:
    clear_db = _db(frame.received_at)
    after = conn.execute(_FIRST_AFTER, (frame.ticker, clear_db)).fetchone()
    if after is None:
        return TERMINAL, None
    governing = _oracle_levels(conn, frame.ticker, clear_db)
    if governing is None:
        return NO_ANCHOR, None
    anchor_db, levels = governing
    yes = sum(1 for size in levels["yes"].values() if size > _ZERO)
    no = sum(1 for size in levels["no"].values() if size > _ZERO)
    if yes + no == 0:
        return EMPTY_BOOK, None
    anchor_at = _decode_ts(anchor_db)
    snapshot = conn.execute(_NEXT_SNAPSHOT, (frame.ticker, clear_db)).fetchone()
    if snapshot is None:
        stale_end_db = conn.execute(_LAST_ROW, (frame.ticker,)).fetchone()[0]
        counting = (_ROWS_AFTER, (frame.ticker, clear_db))
    else:
        stale_end_db = snapshot[0]
        counting = (_ROWS_BETWEEN, (frame.ticker, clear_db, stale_end_db))
    stale_end = _decode_ts(stale_end_db)
    return MID_LIFE, MidLifeClear(
        ticker=frame.ticker,
        clear_at=frame.received_at,
        anchor_at=anchor_at,
        yes_levels=yes,
        no_levels=no,
        total_levels=yes + no,
        next_row_at=_decode_ts(after[0]),
        next_snapshot_at=None if snapshot is None else stale_end,
        stale_end=stale_end,
        stale_rows=conn.execute(*counting).fetchone()[0],
        unresolved=snapshot is None,
        clear_since_anchor=any(anchor_at < other < frame.received_at for other in earlier),
    )


def build_summary(scan: ClearScan) -> ClearSummary:
    return ClearSummary(
        days=scan.days,
        clears=scan.clears,
        terminal=scan.terminal,
        no_anchor=scan.no_anchor,
        empty_book=scan.empty_book,
        mid_life=len(scan.mid_life),
        stale_rows=sum(clear.stale_rows for clear in scan.mid_life),
        stale_us=sum(clear.stale_us for clear in scan.mid_life),
        unresolved=sum(1 for clear in scan.mid_life if clear.unresolved),
        with_stale_rows=sum(1 for clear in scan.mid_life if clear.stale_rows > 0),
    )


def write_mid_life_clears(path: Path, clears: Sequence[MidLifeClear]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    rows = [
        {
            "ticker": clear.ticker,
            "clear_at": clear.clear_at,
            "anchor_at": clear.anchor_at,
            "yes_levels": clear.yes_levels,
            "no_levels": clear.no_levels,
            "total_levels": clear.total_levels,
            "next_row_at": clear.next_row_at,
            "next_snapshot_at": clear.next_snapshot_at,
            "stale_end": clear.stale_end,
            "stale_us": clear.stale_us,
            "stale_rows": clear.stale_rows,
            "unresolved": clear.unresolved,
            "clear_since_anchor": clear.clear_since_anchor,
        }
        for clear in clears
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=MID_LIFE_CLEARS_SCHEMA), path)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn
