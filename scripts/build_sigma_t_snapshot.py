from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "state.db"
OUTPUT_PATH = REPO_ROOT / "tests" / "data" / "sigma_t_snapshot.json"


def _bucket_24h(lead_hours: float) -> int:
    if lead_hours < 0:
        return 0
    return int((lead_hours // 24) * 24)


def _population_std(members: list[float]) -> float:
    if not members:
        return 0.0
    mean = sum(members) / len(members)
    return math.sqrt(sum((x - mean) ** 2 for x in members) / len(members))


def build() -> dict[str, dict[str, float | int]]:
    """Walk forecasts in state.db; bucket by floor-to-24h lead time (close = valid_date+1d); return median+n per bucket."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT run_time, valid_date, members_json FROM forecasts").fetchall()
    conn.close()

    by_bucket: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        run_time = datetime.fromisoformat(row["run_time"])
        valid_date = datetime.fromisoformat(row["valid_date"]).date()
        close_time = datetime(valid_date.year, valid_date.month, valid_date.day) + timedelta(days=1)
        lead_hours = (close_time - run_time).total_seconds() / 3600
        if lead_hours < 0:
            continue
        members = json.loads(row["members_json"])
        sigma = _population_std([float(x) for x in members])
        by_bucket[_bucket_24h(lead_hours)].append(sigma)

    snapshot: dict[str, dict[str, float | int]] = {}
    for bucket, sigmas in sorted(by_bucket.items()):
        snapshot[str(bucket)] = {
            "median": float(median(sigmas)),
            "n": len(sigmas),
        }
    return snapshot


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"data/state.db not found at {DB_PATH}")
    snapshot = build()
    if not snapshot:
        raise SystemExit("no forecasts in data/state.db; cannot build snapshot")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT_PATH} buckets={len(snapshot)}")


if __name__ == "__main__":
    main()
