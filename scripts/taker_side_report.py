from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.taker_side import (  # noqa: E402
    DISCOVERY,
    HOLDOUT,
    TRADES_QUERY,
    DecodeTally,
    ScopedTicker,
    Split,
    TradeCounts,
    count_trades,
    read_exclusions,
    read_scope_days,
    read_split,
    read_trade_frames,
    read_trades,
    scoped_tickers,
    tally_frames,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
MARKETS_QUERY = "SELECT ticker FROM markets WHERE ticker LIKE 'KXHIGH%'"
UTC = timezone.utc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="settle the taker side wire key and count the scoped trade tape"
    )
    parser.add_argument(
        "--run-scope", type=Path, required=True, help="the frozen run scope directory"
    )
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "state.db")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "ws_raw")
    parser.add_argument("--mode", choices=("frame", "tape", "trades"), default="tape")
    return parser


def tape_days(split: Split) -> list[tuple[date, datetime, datetime]]:
    days = []
    day = split.scope_start.date()
    while day <= (split.scope_end - timedelta(microseconds=1)).date():
        opens = datetime.combine(day, time.min, UTC)
        days.append(
            (
                day,
                max(opens, split.scope_start),
                min(opens + timedelta(days=1), split.scope_end),
            )
        )
        day += timedelta(days=1)
    return days


def format_tally(label: str, tally: DecodeTally) -> str:
    return (
        f"{label} frames={tally.frames} kxhigh={tally.high_frames} "
        f"outcome={dict(tally.outcome_state)} legacy={dict(tally.legacy_state)} "
        f"both_present={tally.both_present} disagreed={tally.disagreed} "
        f"outcome_values={dict(tally.outcome_values)} legacy_values={dict(tally.legacy_values)}"
    )


def format_counts(label: str, tickers: int, counts: TradeCounts) -> str:
    return (
        f"{label} tickers={tickers} rows={counts.rows} "
        f"dropped_excluded={counts.dropped_excluded} empty_side={counts.empty_side} "
        f"side_values={counts.side_values} distinct_trade_ids={counts.distinct_trade_ids} "
        f"duplicated_values={counts.duplicated_values} surplus_rows={counts.surplus_rows}"
    )


def run_frame(args: argparse.Namespace, split: Split) -> int:
    day, start, end = tape_days(split)[0]
    frame = next(read_trade_frames(args.raw_dir / f"{day}.jsonl.gz", start=start, end=end))
    print(f"day {day}")
    print(json.dumps(frame, separators=(",", ":")))
    print(f"frame_keys {sorted(frame)}")
    print(f"msg_keys {sorted(frame['msg'])}")
    return 0


def run_tape(args: argparse.Namespace, split: Split) -> int:
    pooled = DecodeTally()
    for day, start, end in tape_days(split):
        tally = tally_frames(
            read_trade_frames(args.raw_dir / f"{day}.jsonl.gz", start=start, end=end)
        )
        print(format_tally(str(day), tally), flush=True)
        pooled = pooled + tally
    print(format_tally("pooled", pooled))
    for keys, n in pooled.key_sets.most_common():
        print(f"key_set n={n} keys={keys}")
    return 0


def run_trades(args: argparse.Namespace, split: Split) -> int:
    days = read_scope_days(args.run_scope / "event_days.parquet")
    excluded = read_exclusions(args.run_scope / "exclusions.parquet")
    conn = sqlite3.connect(f"file:{args.db.absolute()}?mode=ro", uri=True)
    scoped = scoped_tickers([row[0] for row in conn.execute(MARKETS_QUERY)], days, split)

    print(f"scope {split.scope_start.isoformat()} {split.scope_end.isoformat()}")
    print(f"cities={len({day.series for day in days})} event_days={len(days)}")
    print(f"tickers={len(scoped)} exclusions={len(excluded.intervals)}")
    plan = conn.execute("EXPLAIN QUERY PLAN " + TRADES_QUERY, ("x", "a", "b")).fetchall()
    print(f"plan {[row[3] for row in plan]}")

    subsets: list[tuple[str, tuple[ScopedTicker, ...]]] = [
        ("pooled", scoped),
        (DISCOVERY, tuple(t for t in scoped if t.split == DISCOVERY)),
        (HOLDOUT, tuple(t for t in scoped if t.split == HOLDOUT)),
    ]
    for label, subset in subsets:
        rows = chain.from_iterable(read_trades(conn, ticker) for ticker in subset)
        print(format_counts(label, len(subset), count_trades(rows, excluded)), flush=True)
    conn.close()
    return 0


def run(args: argparse.Namespace) -> int:
    split = read_split(args.run_scope / "split.json")
    if args.mode == "frame":
        return run_frame(args, split)
    if args.mode == "tape":
        return run_tape(args, split)
    return run_trades(args, split)


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
