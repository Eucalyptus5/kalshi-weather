from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import date, datetime
from datetime import timezone as _timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.backtest.engine import Bucket, replay  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "state.db"
REPORT_PATH = Path("/tmp/replay_backtest_report.txt")
DEFAULT_START_DATE = date(2026, 5, 18)
STRATEGY_CHOICES: tuple[str, ...] = ("edge", "tails", "all")


def _open_readonly() -> sqlite3.Connection:
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "n/a"
    return f"{(num / denom) * 100:.2f}%"


def render_report(
    buckets: dict[str, Bucket],
    n_total: int,
    n_missing: int,
    cutoff: datetime,
    strategy_filter: str,
) -> str:
    lines: list[str] = []
    now = datetime.now(tz=_timezone.utc).isoformat(timespec="seconds")
    lines.append(f"# strategy replay backtest report\tgenerated_at={now}")
    lines.append(f"# regime_filter\tintended_at >= {cutoff.isoformat()}")
    lines.append(f"# strategy_filter\t{strategy_filter}")
    lines.append(f"# rows_considered\t{n_total}")
    lines.append(f"# missing_inputs\t{n_missing}")
    lines.append("# regime_label\tpost_floor (single bucket; pre-floor excluded)")
    lines.append(
        "# caveat\thistorical realized_pnl dollar amounts not used; "
        "only simulated_pnl.outcome win/loss"
    )
    lines.append(
        "# caveat\toverlays seeded from simulated fills within the replay window; "
        "pre-window position carry not modeled"
    )
    lines.append(
        "# caveat\tcalibration brief 07 Commit B not in bet path; raw_q is the model output"
    )
    lines.append("")
    for strategy in sorted(buckets.keys()):
        b = buckets[strategy]
        lines.append(f"== bucket\tstrategy={strategy}\tregime=post_floor")
        lines.append(f"n_replayed\t{b.n_replayed}")
        lines.append(f"n_current_would_place\t{b.n_current_would_place}")
        lines.append(f"n_current_would_skip\t{b.n_current_would_skip}")
        lines.append(
            "win_rate_would_place\t"
            f"{_pct(b.placed_wins, b.placed_total_outcome)}\t"
            f"(n={b.placed_total_outcome})"
        )
        lines.append(
            "win_rate_would_skip\t"
            f"{_pct(b.skipped_wins, b.skipped_total_outcome)}\t"
            f"(n={b.skipped_total_outcome})"
        )
        avg_place = (b.placed_size_sum / b.n_current_would_place) if b.n_current_would_place else 0
        avg_hist = (b.historical_size_sum / b.n_replayed) if b.n_replayed else 0
        lines.append(f"avg_size_would_place_contracts\t{avg_place:.2f}")
        lines.append(f"avg_size_historical_contracts\t{avg_hist:.2f}")
        top3 = b.skip_reasons.most_common(3)
        for i, (reason, count) in enumerate(top3, start=1):
            lines.append(f"top_skip_reason_{i}\t{reason}\t{count}")
        if not top3:
            lines.append("top_skip_reason_1\t(none)\t0")
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="replay paper trades through current strategy")
    parser.add_argument(
        "--start-date",
        type=lambda s: date.fromisoformat(s),
        default=DEFAULT_START_DATE,
        help="inclusive lower bound on intended_at (YYYY-MM-DD), default 2026-05-18",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default="all",
        help="restrict replay to one strategy",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cutoff = datetime.combine(args.start_date, datetime.min.time(), tzinfo=_timezone.utc)
    logging.disable(logging.CRITICAL)
    if not DB_PATH.exists():
        sys.stderr.write(f"db not found: {DB_PATH}\n")
        return 2
    conn = _open_readonly()
    try:
        buckets, n_total, n_missing = replay(conn, cutoff, args.strategy)
    finally:
        conn.close()
    report = render_report(buckets, n_total, n_missing, cutoff, args.strategy)
    REPORT_PATH.write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
