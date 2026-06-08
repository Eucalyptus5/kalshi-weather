from __future__ import annotations

import sqlite3
import statistics
from datetime import datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path


ALL_SERIES = "*"
LEAD_BUCKETS: tuple[str, ...] = ("<=2h", "2-8h", "8-24h", "24-72h", ">72h")
MIN_BUCKET_ROWS = 200


def lead_bucket_for(lead: timedelta) -> str:
    hours = lead.total_seconds() / 3600
    if hours <= 2:
        return "<=2h"
    if hours <= 8:
        return "2-8h"
    if hours <= 24:
        return "8-24h"
    if hours <= 72:
        return "24-72h"
    return ">72h"


def parse_dt(raw: str) -> datetime:
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_timezone.utc)
    return dt


def load_depth_table(db_path: Path) -> dict[tuple[str, str], Decimal]:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"state db not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    samples: dict[tuple[str, str], list[int]] = {}
    try:
        cursor = conn.execute(
            """
            SELECT m.series, m.close_time, o.snapshot_at, o.yes_bid_depth, o.no_bid_depth
              FROM orderbook_snapshots o
              JOIN markets m ON m.ticker = o.ticker
             WHERE m.close_time IS NOT NULL
               AND o.yes_bid_depth IS NOT NULL
               AND o.no_bid_depth IS NOT NULL
            """
        )
        for series, close_raw, snapshot_raw, yes_depth, no_depth in cursor:
            lead = parse_dt(close_raw) - parse_dt(snapshot_raw)
            if lead.total_seconds() < 0:
                continue
            bucket = lead_bucket_for(lead)
            samples.setdefault((series, bucket), []).append(min(int(yes_depth), int(no_depth)))
    finally:
        conn.close()

    by_bucket: dict[str, list[int]] = {}
    for (_, bucket), values in samples.items():
        by_bucket.setdefault(bucket, []).extend(values)

    table: dict[tuple[str, str], Decimal] = {}
    for bucket, values in by_bucket.items():
        table[(ALL_SERIES, bucket)] = Decimal(str(statistics.median(values)))
    for (series, bucket), values in samples.items():
        if len(values) >= MIN_BUCKET_ROWS:
            table[(series, bucket)] = Decimal(str(statistics.median(values)))
        else:
            table[(series, bucket)] = table[(ALL_SERIES, bucket)]
    return table
