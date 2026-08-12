from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

from bot.lag.event_study import NO_BAND2, YES_BAND2, OrderbookSnapshotRow, median_int
from bot.lag.mid import mid2, ticks, two_sided


class WsGapError(ValueError):
    """A recorded ws gap covers part of the window the caller asked about."""


@dataclass(frozen=True, slots=True)
class EventBookProbe:
    at_t0: OrderbookSnapshotRow
    at_decision: dict[int, OrderbookSnapshotRow]
    lag_s: int | None
    cadence_s: int | None


def _format_db_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def _parse_db_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


_LATEST_SNAPSHOT = (
    "SELECT received_at, seq FROM ws_book_events "
    "WHERE ticker = ? AND is_snapshot = 1 AND received_at <= ? "
    "ORDER BY received_at DESC, id DESC LIMIT 1"
)

_SNAPSHOT_BATCH = (
    "SELECT side, price, size FROM ws_book_events "
    "WHERE ticker = ? AND is_snapshot = 1 AND received_at = ? AND seq = ? "
    "ORDER BY id"
)

_GAP = (
    "SELECT detected_at, reason FROM ws_gaps "
    "WHERE (ticker = ? OR ticker = '') AND detected_at > ? AND detected_at <= ? "
    "ORDER BY detected_at, id LIMIT 1"
)

_DELTAS = (
    "SELECT side, price, size FROM ws_book_events "
    "WHERE ticker = ? AND is_snapshot = 0 AND received_at > ? AND received_at <= ? "
    "ORDER BY received_at, id"
)

_FORWARD = (
    "SELECT received_at, seq, side, price, size, is_snapshot FROM ws_book_events "
    "WHERE ticker = ? AND received_at > ? AND received_at <= ? "
    "ORDER BY received_at, id"
)


@contextmanager
def open_book_db(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(f"file:{db_path.absolute()}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size = -65536")
    try:
        yield conn
    finally:
        conn.close()


def book_state_at(db_path: Path, ticker: str, t: datetime) -> OrderbookSnapshotRow | None:
    t_db = _format_db_ts(t)
    with open_book_db(db_path) as conn:
        latest = conn.execute(_LATEST_SNAPSHOT, (ticker, t_db)).fetchone()
        if latest is None:
            return None
        snapshot_at_db, seq = latest
        gap = conn.execute(_GAP, (ticker, snapshot_at_db, t_db)).fetchone()
        if gap is not None:
            raise WsGapError(
                f"ws gap invalidates {ticker} book at {t.isoformat()}: "
                f"reason={gap[1]} detected_at={gap[0]}"
            )
        book = _book_from_snapshot(conn, ticker, snapshot_at_db, seq)
        _apply(book, conn.execute(_DELTAS, (ticker, snapshot_at_db, t_db)), ticker)
    return _row_from_book(ticker, t, book)


def probe_event(
    conn: sqlite3.Connection,
    ticker: str,
    t0: datetime,
    side_locked: Literal["yes", "no"],
    *,
    decision_offsets_s: Sequence[int],
    window_s: int = 24 * 3600,
    cadence_window_s: int = 600,
) -> EventBookProbe | None:
    t0_db = _format_db_ts(t0)
    latest = conn.execute(_LATEST_SNAPSHOT, (ticker, t0_db)).fetchone()
    if latest is None:
        return None
    snapshot_at_db, seq = latest

    window_end = t0 + timedelta(seconds=window_s)
    gap = conn.execute(_GAP, (ticker, snapshot_at_db, _format_db_ts(window_end))).fetchone()
    gap_at = _parse_db_ts(gap[0]) if gap is not None else None
    if gap_at is not None and gap_at <= t0:
        raise _gap_error(ticker, t0, gap)

    book = _book_from_snapshot(conn, ticker, snapshot_at_db, seq)
    _apply(book, conn.execute(_DELTAS, (ticker, snapshot_at_db, t0_db)), ticker)
    at_t0 = _row_from_book(ticker, t0, book)

    offsets = sorted(decision_offsets_s)
    pending = list(offsets)
    at_decision: dict[int, OrderbookSnapshotRow] = {}
    cadence_end = t0 + timedelta(seconds=cadence_window_s)
    arrivals: list[datetime] = []
    band_cross_at: datetime | None = None
    snapshot_key: tuple[str, int] | None = None
    settled_snapshot_at: datetime | None = None

    cursor = conn.execute(_FORWARD, (ticker, t0_db, _format_db_ts(window_end)))
    for received_at_db, row_seq, side, price, size, is_snapshot in cursor:
        received_at = _parse_db_ts(received_at_db)
        if gap_at is not None and received_at >= gap_at:
            break
        while pending and t0 + timedelta(seconds=pending[0]) < received_at:
            offset = pending.pop(0)
            at_decision[offset] = _row_from_book(ticker, t0 + timedelta(seconds=offset), book)
        if settled_snapshot_at is not None and received_at != settled_snapshot_at:
            if band_cross_at is None and _in_band(book, side_locked):
                band_cross_at = settled_snapshot_at
            settled_snapshot_at = None

        if is_snapshot:
            if snapshot_key != (received_at_db, row_seq):
                snapshot_key = (received_at_db, row_seq)
                book["yes"].clear()
                book["no"].clear()
            book[side][Decimal(price)] = Decimal(size)
            settled_snapshot_at = received_at
        else:
            _apply_one(book, ticker, side, price, size)
            if band_cross_at is None and _in_band(book, side_locked):
                band_cross_at = received_at

        if not arrivals or arrivals[-1] != received_at:
            if received_at <= cadence_end:
                arrivals.append(received_at)
        if band_cross_at is not None and not pending and received_at > cadence_end:
            break
    cursor.close()

    if settled_snapshot_at is not None and band_cross_at is None and _in_band(book, side_locked):
        band_cross_at = settled_snapshot_at
    for offset in pending:
        at_decision[offset] = _row_from_book(ticker, t0 + timedelta(seconds=offset), book)

    needed_until = max(t0 + timedelta(seconds=offsets[-1]), band_cross_at or window_end)
    if gap_at is not None and gap_at <= needed_until:
        raise _gap_error(ticker, t0, gap)

    return EventBookProbe(
        at_t0=at_t0,
        at_decision=at_decision,
        lag_s=None if band_cross_at is None else int((band_cross_at - t0).total_seconds()),
        cadence_s=_cadence(arrivals),
    )


def _gap_error(ticker: str, t0: datetime, gap: tuple[str, str]) -> WsGapError:
    return WsGapError(
        f"ws gap invalidates {ticker} event at {t0.isoformat()}: "
        f"reason={gap[1]} detected_at={gap[0]}"
    )


def _book_from_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    snapshot_at_db: str,
    seq: int,
) -> dict[str, dict[Decimal, Decimal]]:
    book: dict[str, dict[Decimal, Decimal]] = {"yes": {}, "no": {}}
    for side, price, size in conn.execute(_SNAPSHOT_BATCH, (ticker, snapshot_at_db, seq)):
        book[side][Decimal(price)] = Decimal(size)
    return book


def _apply(
    book: dict[str, dict[Decimal, Decimal]],
    deltas: Iterable[tuple[str, str, str]],
    ticker: str,
) -> None:
    for side, price, size in deltas:
        _apply_one(book, ticker, side, price, size)


def _apply_one(
    book: dict[str, dict[Decimal, Decimal]],
    ticker: str,
    side: str,
    price: str,
    delta: str,
) -> None:
    levels = book[side]
    key = Decimal(price)
    size = levels.get(key, Decimal("0")) + Decimal(delta)
    if size < 0:
        raise ValueError(f"negative level for {ticker} side={side} price={price} size={size}")
    if size == 0:
        levels.pop(key, None)
    else:
        levels[key] = size


def _in_band(book: dict[str, dict[Decimal, Decimal]], side_locked: str) -> bool:
    yes_bid, yes_depth = _best(book["yes"])
    no_bid, no_depth = _best(book["no"])
    yes_ticks = ticks(yes_bid)
    no_ticks = ticks(no_bid)
    if not two_sided(yes_ticks, no_ticks, yes_depth=yes_depth, no_depth=no_depth):
        return False
    mid = mid2(yes_ticks, no_ticks)
    if side_locked == "yes":
        return mid >= YES_BAND2
    return mid <= NO_BAND2


def _cadence(arrivals: list[datetime]) -> int | None:
    if len(arrivals) < 3:
        return None
    return median_int(
        [int((arrivals[i + 1] - arrivals[i]).total_seconds()) for i in range(len(arrivals) - 1)]
    )


def _row_from_book(
    ticker: str,
    t: datetime,
    book: dict[str, dict[Decimal, Decimal]],
) -> OrderbookSnapshotRow:
    yes_bid, yes_bid_depth = _best(book["yes"])
    no_bid, no_bid_depth = _best(book["no"])
    return OrderbookSnapshotRow(
        ticker=ticker,
        snapshot_at=t,
        yes_bid=yes_bid,
        yes_ask=Decimal("1") - no_bid,
        no_bid=no_bid,
        no_ask=Decimal("1") - yes_bid,
        yes_ask_depth=no_bid_depth,
        yes_bid_depth=yes_bid_depth,
        no_ask_depth=yes_bid_depth,
        no_bid_depth=no_bid_depth,
    )


def _best(levels: dict[Decimal, Decimal]) -> tuple[Decimal, int]:
    live = [(price, size) for price, size in levels.items() if size > 0]
    if not live:
        return Decimal("0"), 0
    price, size = max(live, key=lambda level: level[0])
    return price, int(size)
