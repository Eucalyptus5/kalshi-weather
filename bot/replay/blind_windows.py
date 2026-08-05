import logging
import sqlite3
import time
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bot.replay.artifacts import COVERAGE_SCHEMA
from bot.replay.forward_pass import _decode_ts
from bot.replay.inventory import GapRow, TickerCoverage


logger = logging.getLogger(__name__)

SCAN_BATCH_ROWS = 200_000
PROGRESS_ROWS = 10_000_000
NEAREST_RANK = "nearest_rank"

FROZEN_START = datetime(2026, 7, 18, tzinfo=timezone.utc)
FROZEN_END = datetime(2026, 8, 2, tzinfo=timezone.utc)

_MICROSECOND = timedelta(microseconds=1)
_COLUMNS = "id, ticker, received_at, seq, is_snapshot"
_SELECT = f"SELECT {_COLUMNS} FROM ws_book_events WHERE id > ? ORDER BY id LIMIT ?"
_SELECT_BOUNDED = (
    f"SELECT {_COLUMNS} FROM ws_book_events WHERE id > ? AND id <= ? ORDER BY id LIMIT ?"
)

BLIND_WINDOWS_SCHEMA = pa.schema(
    [
        ("boundary_id", pa.int64()),
        ("prev_id", pa.int64()),
        ("end_id", pa.int64()),
        ("burst_messages", pa.int64()),
        ("ticker", pa.string()),
        ("start", pa.timestamp("us", tz="UTC")),
        ("end", pa.timestamp("us", tz="UTC")),
        ("blind_us", pa.int64()),
        ("prev_seq", pa.int64()),
        ("seq", pa.int64()),
        ("has_gap_row", pa.bool_()),
        ("gap_id", pa.int64()),
        ("gap_reason", pa.string()),
        ("gap_detected_at", pa.timestamp("us", tz="UTC")),
        ("in_frozen_window", pa.bool_()),
    ]
)


@dataclass(frozen=True, slots=True)
class BlindWindow:
    boundary_id: int
    prev_id: int
    end_id: int
    burst_messages: int
    # A boundary tears down the whole connection, so the window belongs to every subscribed
    # ticker at once and names none of them.
    ticker: str
    start: datetime
    end: datetime
    prev_seq: int
    seq: int
    gap_id: int | None
    gap_reason: str | None
    gap_detected_at: datetime | None

    @property
    def blind_us(self) -> int:
        return (self.end - self.start) // _MICROSECOND


class BlindWindowDetector:
    def __init__(self) -> None:
        self._message: tuple[str, int] | None = None
        self._last_id = 0
        self._windows: list[BlindWindow] = []
        self._open: BlindWindow | None = None

    def observe(self, row_id: int, received_at: str, seq: int, is_snapshot: bool) -> None:
        message = (received_at, seq)
        if message == self._message:
            self._last_id = row_id
            return
        previous, prev_id = self._message, self._last_id
        self._message = message
        self._last_id = row_id
        boundary = previous is not None and seq <= previous[1]
        # The re-delivered burst runs to the last snapshot before the new series resumes its
        # deltas, so an open window closes on the message before whatever ends the burst.
        if self._open is not None:
            if boundary or not is_snapshot:
                self._windows.append(
                    replace(self._open, end=_decode_ts(previous[0]), end_id=prev_id)
                )
                self._open = None
            else:
                self._open = replace(self._open, burst_messages=self._open.burst_messages + 1)
        if not boundary:
            return
        window = BlindWindow(
            boundary_id=row_id,
            prev_id=prev_id,
            end_id=row_id,
            burst_messages=1,
            ticker="",
            start=_decode_ts(previous[0]),
            end=_decode_ts(received_at),
            prev_seq=previous[1],
            seq=seq,
            gap_id=None,
            gap_reason=None,
            gap_detected_at=None,
        )
        if is_snapshot:
            self._open = window
        else:
            self._windows.append(window)

    def windows(self) -> list[BlindWindow]:
        if self._open is None:
            return list(self._windows)
        closed = replace(self._open, end=_decode_ts(self._message[0]), end_id=self._last_id)
        return [*self._windows, closed]


@dataclass(frozen=True, slots=True)
class BlindScan:
    windows: tuple[BlindWindow, ...]
    rows: int
    last_received_at: datetime | None
    coverage: tuple[TickerCoverage, ...]


def scan_blind_windows(
    db_path: Path,
    *,
    max_id: int | None = None,
    batch_rows: int = SCAN_BATCH_ROWS,
    progress_rows: int = PROGRESS_ROWS,
) -> BlindScan:
    detector = BlindWindowDetector()
    select = _SELECT if max_id is None else _SELECT_BOUNDED
    bound = () if max_id is None else (max_id,)
    started = time.monotonic()
    rows = 0
    since_log = 0
    last_id = 0
    last_received_at: datetime | None = None
    counts: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    last_seen: dict[str, str] = {}
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        # A held ORDER BY id cursor pins the recorder's WAL until it shuts down, so every batch
        # is its own statement carrying the last id forward.
        while True:
            batch = conn.execute(select, (last_id, *bound, batch_rows)).fetchall()
            if not batch:
                break
            for row_id, ticker, received_at, seq, is_snapshot in batch:
                seen = counts.get(ticker, 0)
                counts[ticker] = seen + 1
                if seen == 0:
                    first_seen[ticker] = received_at
                last_seen[ticker] = received_at
                detector.observe(row_id, received_at, seq, bool(is_snapshot))
            last_id = batch[-1][0]
            last_received_at = _decode_ts(batch[-1][2])
            rows += len(batch)
            since_log += len(batch)
            if since_log >= progress_rows:
                since_log = 0
                _log_progress(rows, last_id, len(detector.windows()), started)
    finally:
        conn.close()
    _log_progress(rows, last_id, len(detector.windows()), started)
    coverage = tuple(
        TickerCoverage(
            ticker=ticker,
            rows=counts[ticker],
            first_received_at=_decode_ts(first_seen[ticker]),
            last_received_at=_decode_ts(last_seen[ticker]),
        )
        for ticker in sorted(counts)
    )
    return BlindScan(tuple(detector.windows()), rows, last_received_at, coverage)


def _log_progress(rows: int, last_id: int, windows: int, started: float) -> None:
    elapsed = time.monotonic() - started
    logger.info(
        "blind_windows rows=%d id=%d windows=%d elapsed_s=%.3f rows_per_s=%.0f",
        rows,
        last_id,
        windows,
        elapsed,
        rows / elapsed,
    )


def attach_gap_rows(
    windows: Sequence[BlindWindow], gaps: Sequence[GapRow], *, scanned_through: datetime | None
) -> tuple[list[BlindWindow], list[GapRow]]:
    ordered = sorted(windows, key=lambda window: window.start)
    starts = [window.start for window in ordered]
    bounds = [*starts[1:], scanned_through]
    matched: dict[int, GapRow] = {}
    unmatched: list[GapRow] = []
    # detected_at is stamped at persist, after the reconnect slept its backoff, so a gap row
    # always falls at or after the start of the boundary that produced it and can fall after its
    # end. The window it belongs to runs to where the next one starts, and the last one runs to
    # where the scan stopped reading: past that lie boundaries the scan never saw.
    for row in sorted(gaps, key=lambda row: (row.detected_at, row.id)):
        index = bisect_right(starts, row.detected_at) - 1
        if index < 0 or index in matched or row.detected_at >= bounds[index]:
            unmatched.append(row)
        else:
            matched[index] = row
    attached = [
        window
        if index not in matched
        else replace(
            window,
            gap_id=matched[index].id,
            gap_reason=matched[index].reason,
            gap_detected_at=matched[index].detected_at,
        )
        for index, window in enumerate(ordered)
    ]
    return attached, unmatched


@dataclass(frozen=True, slots=True)
class BlindSummary:
    windows: int
    with_gap_row: int
    without_gap_row: int
    total_blind_us: int
    percentiles: str
    p50_blind_us: int
    p90_blind_us: int
    max_blind_us: int
    frozen_start: datetime
    frozen_end: datetime
    frozen_windows: int
    frozen_with_gap_row: int
    frozen_without_gap_row: int
    frozen_total_blind_us: int
    frozen_p50_blind_us: int
    frozen_p90_blind_us: int
    frozen_max_blind_us: int


def build_summary(
    windows: Sequence[BlindWindow],
    *,
    frozen_start: datetime = FROZEN_START,
    frozen_end: datetime = FROZEN_END,
) -> BlindSummary:
    frozen = [window for window in windows if frozen_start <= window.start < frozen_end]
    widths = sorted(window.blind_us for window in windows)
    frozen_widths = sorted(window.blind_us for window in frozen)
    return BlindSummary(
        windows=len(windows),
        with_gap_row=sum(1 for window in windows if window.gap_id is not None),
        without_gap_row=sum(1 for window in windows if window.gap_id is None),
        total_blind_us=sum(widths),
        percentiles=NEAREST_RANK,
        p50_blind_us=_percentile(widths, 50),
        p90_blind_us=_percentile(widths, 90),
        max_blind_us=max(widths, default=0),
        frozen_start=frozen_start,
        frozen_end=frozen_end,
        frozen_windows=len(frozen),
        frozen_with_gap_row=sum(1 for window in frozen if window.gap_id is not None),
        frozen_without_gap_row=sum(1 for window in frozen if window.gap_id is None),
        frozen_total_blind_us=sum(frozen_widths),
        frozen_p50_blind_us=_percentile(frozen_widths, 50),
        frozen_p90_blind_us=_percentile(frozen_widths, 90),
        frozen_max_blind_us=max(frozen_widths, default=0),
    )


def _percentile(widths: list[int], percent: int) -> int:
    if not widths:
        return 0
    return widths[-(-percent * len(widths) // 100) - 1]


def write_blind_windows(
    path: Path,
    windows: Sequence[BlindWindow],
    *,
    frozen_start: datetime = FROZEN_START,
    frozen_end: datetime = FROZEN_END,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    rows = [
        {
            "boundary_id": window.boundary_id,
            "prev_id": window.prev_id,
            "end_id": window.end_id,
            "burst_messages": window.burst_messages,
            "ticker": window.ticker,
            "start": window.start,
            "end": window.end,
            "blind_us": window.blind_us,
            "prev_seq": window.prev_seq,
            "seq": window.seq,
            "has_gap_row": window.gap_id is not None,
            "gap_id": window.gap_id,
            "gap_reason": window.gap_reason,
            "gap_detected_at": window.gap_detected_at,
            "in_frozen_window": frozen_start <= window.start < frozen_end,
        }
        for window in windows
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=BLIND_WINDOWS_SCHEMA), path)


def write_coverage(path: Path, coverage: Sequence[TickerCoverage]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    rows = [
        {
            "ticker": row.ticker,
            "rows": row.rows,
            "first_received_at": row.first_received_at,
            "last_received_at": row.last_received_at,
        }
        for row in coverage
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=COVERAGE_SCHEMA), path)
