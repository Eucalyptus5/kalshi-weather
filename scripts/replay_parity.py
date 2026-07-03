from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.replay.parity import (  # noqa: E402
    ParityResult,
    compare_points,
    gap_windows,
    sample_points,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "state.db"
DEFAULT_POINTS = 200


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="parity gate for the forward pass against book_state_at"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--points",
        type=int,
        default=DEFAULT_POINTS,
        help="point budget; the realized sample is at least this large",
    )
    return parser


def format_result(result: ParityResult, elapsed_s: float) -> str:
    counts = result.counts()
    lines = [
        "== PARITY vs book_state_at",
        f"points={len(result.results)}  compared={counts['agreed'] + counts['disagreed']}  "
        f"agreed={counts['agreed']}  agreed_on_raise={counts['agreed_on_raise']}  "
        f"excluded_tied_delta={counts['excluded_tied_delta']}  disagreed={counts['disagreed']}",
        f"blind={sum(1 for r in result.results if r.blind)}  rows_read={result.rows_read}  "
        f"elapsed_s={elapsed_s:.1f}",
        "-- by kind",
        "  " + "  ".join(f"{k}={n}" for k, n in sorted(result.kinds().items())),
        "-- by cohort",
        "  " + "  ".join(f"{k}={n}" for k, n in sorted(result.cohorts().items())),
        "-- inventory scalars",
    ]
    lines.extend(f"  {name}={value}" for name, value in sorted(result.scalars().items()))
    lines.append("-- disagreements")
    if not result.disagreements():
        lines.append("  none")
    lines.extend(f"  {outcome.detail}" for outcome in result.disagreements())
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    windows = gap_windows(args.db)
    points = sample_points(args.db, args.points, windows)
    result = compare_points(args.db, points, windows)
    print(format_result(result, time.monotonic() - started))
    return 1 if result.disagreements() else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
