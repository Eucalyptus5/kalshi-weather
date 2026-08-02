from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.r0_universe import (  # noqa: E402
    DISAGREE,
    freeze_universe,
    read_recorded_coverage,
    write_universe,
)
from bot.replay.analysis_stations import LOW_TO_HIGH  # noqa: E402
from scripts.lag_report import R0_FRACTION_INVALID_MAX, R0_PASSING_SERIES  # noqa: E402


R0_LOW_SERIES: tuple[str, ...] = tuple(sorted(LOW_TO_HIGH))
R0_UNION_SERIES: tuple[str, ...] = tuple(sorted(R0_PASSING_SERIES + R0_LOW_SERIES))

# Only the high twenty have been through an R0 basis check, so the low and union sets are claims
# about what was recorded, not about basis quality, and neither is the default.
PASSING_SETS: dict[str, tuple[str, ...]] = {
    "high": R0_PASSING_SERIES,
    "low": R0_LOW_SERIES,
    "union": R0_UNION_SERIES,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="freeze R0's threshold and passing station set against the recorded tape"
    )
    parser.add_argument(
        "--event-days", type=Path, required=True, help="the frozen event-day parquet"
    )
    parser.add_argument("--out", type=Path, required=True, help="directory to write the freeze to")
    parser.add_argument("--passing", choices=sorted(PASSING_SETS), default="high")
    return parser


def run(args: argparse.Namespace) -> int:
    coverage = read_recorded_coverage(args.event_days)
    universe = freeze_universe(
        fraction_invalid_max=R0_FRACTION_INVALID_MAX,
        passing=PASSING_SETS[args.passing],
        coverage=coverage,
    )

    payload: dict[str, object] = {
        "event_days": str(args.event_days),
        "out": str(args.out),
        "passing_set": args.passing,
        "fraction_invalid_max": str(universe.fraction_invalid_max),
        "passing": len(universe.passing),
        "lock_carve_out": len(universe.passing) - len(universe.lock_dependent),
        "lock_dependent": len(universe.lock_dependent),
        "recorded": len(universe.recorded),
        "ladder_widths": ",".join(str(width) for width in universe.ladder_widths),
        "in_scope_city_days": universe.in_scope_city_days,
        "reconciliation": universe.reconciliation,
        "recorded_not_passing": ",".join(universe.recorded_not_passing),
        "passing_not_recorded": ",".join(universe.passing_not_recorded),
    }
    if universe.reconciliation == DISAGREE:
        print(format_report(payload))
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    payload["sha256"] = write_universe(args.out / "r0_universe.json", universe)
    print(format_report(payload))
    return 0


def format_report(payload: Mapping[str, object]) -> str:
    return "\n".join(["== R0 UNIVERSE"] + [f"{name}={value}" for name, value in payload.items()])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
