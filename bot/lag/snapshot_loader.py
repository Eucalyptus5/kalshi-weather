from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from bot.lag.event_study import OrderbookSnapshotRow


PROD_ERA_START = datetime.fromisoformat("2026-06-13T00:00:18+00:00")


_QUERY = (
    "SELECT ticker, snapshot_at, yes_bid, yes_ask, no_bid, no_ask, "
    "yes_ask_depth, yes_bid_depth, no_ask_depth, no_bid_depth "
    "FROM orderbook_snapshots "
    "WHERE ticker = ? AND snapshot_at >= ? AND snapshot_at < ? "
    "ORDER BY snapshot_at ASC"
)


def load_snapshots(
    db_path: Path,
    ticker: str,
    start: datetime,
    end: datetime,
) -> list[OrderbookSnapshotRow]:
    effective_start = max(start, PROD_ERA_START)
    if effective_start >= end:
        return []

    conn = sqlite3.connect(f"file:{db_path.absolute()}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            _QUERY,
            (ticker, effective_start.isoformat(), end.isoformat()),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    out: list[OrderbookSnapshotRow] = []
    for row in rows:
        snapshot_at = _parse_ts(row[1])
        out.append(
            OrderbookSnapshotRow(
                ticker=row[0],
                snapshot_at=snapshot_at,
                yes_bid=Decimal(str(row[2])),
                yes_ask=Decimal(str(row[3])),
                no_bid=Decimal(str(row[4])) if row[4] is not None else None,
                no_ask=Decimal(str(row[5])) if row[5] is not None else None,
                yes_ask_depth=row[6],
                yes_bid_depth=row[7],
                no_ask_depth=row[8],
                no_bid_depth=row[9],
            )
        )
    return out


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
