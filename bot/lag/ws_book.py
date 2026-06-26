from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from bot.lag.event_study import OrderbookSnapshotRow


def _format_db_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


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


def book_state_at(db_path: Path, ticker: str, t: datetime) -> OrderbookSnapshotRow | None:
    t_db = _format_db_ts(t)
    conn = sqlite3.connect(f"file:{db_path.absolute()}?mode=ro", uri=True)
    try:
        latest = conn.execute(_LATEST_SNAPSHOT, (ticker, t_db)).fetchone()
        if latest is None:
            return None
        snapshot_at_db, seq = latest
        gap = conn.execute(_GAP, (ticker, snapshot_at_db, t_db)).fetchone()
        if gap is not None:
            raise ValueError(
                f"ws gap invalidates {ticker} book at {t.isoformat()}: "
                f"reason={gap[1]} detected_at={gap[0]}"
            )
        batch = conn.execute(_SNAPSHOT_BATCH, (ticker, snapshot_at_db, seq)).fetchall()
        deltas = conn.execute(_DELTAS, (ticker, snapshot_at_db, t_db)).fetchall()
    finally:
        conn.close()

    book: dict[str, dict[Decimal, Decimal]] = {"yes": {}, "no": {}}
    for side, price, size in batch:
        book[side][Decimal(price)] = Decimal(size)
    for side, price, delta in deltas:
        levels = book[side]
        key = Decimal(price)
        size = levels.get(key, Decimal("0")) + Decimal(delta)
        if size < 0:
            raise ValueError(f"negative level for {ticker} side={side} price={price} size={size}")
        if size == 0:
            levels.pop(key, None)
        else:
            levels[key] = size

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
