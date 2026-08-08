from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.main import STATIONS, StationConfig  # noqa: E402
from bot.replay.analysis_stations import ANALYSIS_STATIONS, LOW_STATIONS  # noqa: E402
from bot.replay.run_scope import (  # noqa: E402
    D_EVAL,
    QUIET_BAND,
    RECORDED_GAP,
    RESUBSCRIBE_BLIND,
    SUBSCRIPTION_WIDE,
    apply_spans,
    blind_exclusions,
    event_day_inventory,
    freeze_split,
    quiet_band_exclusions,
    read_blind_windows,
    read_coverage,
    union_overlap_us,
    write_event_days,
    write_exclusions,
    write_split,
)
from scripts.replay_blind_windows import utc_stamp  # noqa: E402


CLASSES = (RECORDED_GAP, SUBSCRIPTION_WIDE, RESUBSCRIBE_BLIND, QUIET_BAND)

STATION_SETS: dict[str, dict[str, StationConfig]] = {
    "high": STATIONS,
    "low": LOW_STATIONS,
    "both": ANALYSIS_STATIONS,
}

# Concurrent with venue-wide REST 503s, so the recorded bracket understates how long the venue
# was unreadable; the pad is a per-incident judgment, not a standing rule.
PADDED_INCIDENT = datetime(2026, 7, 23, 7, 43, 55, tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="freeze the run scope the tape studies read their universe from"
    )
    parser.add_argument(
        "--blind-windows", type=Path, required=True, help="the corrected blind window parquet"
    )
    parser.add_argument(
        "--coverage", type=Path, required=True, help="the forward-pass ticker coverage parquet"
    )
    parser.add_argument("--out", type=Path, required=True, help="directory to write the scope to")
    parser.add_argument(
        "--start",
        type=utc_stamp,
        required=True,
        help="the earliest instant an event-day window may open",
    )
    parser.add_argument(
        "--cohort",
        choices=sorted(STATION_SETS),
        default="high",
        help="which ladder the inventory is built over",
    )
    parser.add_argument("--d-eval", type=int, default=D_EVAL, help="event-days to freeze")
    parser.add_argument(
        "--incident",
        type=utc_stamp,
        default=PADDED_INCIDENT,
        help="the gap whose recorded bracket is widened at both ends",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    blind = read_blind_windows(args.blind_windows)
    coverage = read_coverage(args.coverage)
    tape_first = min(row["first_received_at"] for row in coverage)
    tape_last = max(row["last_received_at"] for row in coverage)

    over_tape = blind_exclusions(
        blind, scope_start=tape_first, scope_end=tape_last, incident=args.incident
    )
    inventory = event_day_inventory(
        coverage,
        stations=STATION_SETS[args.cohort],
        exclusions=over_tape,
        tape_first=tape_first,
        tape_last=tape_last,
        scope_open=args.start,
        d_eval=args.d_eval,
    )
    split = freeze_split(inventory.days, d_eval=args.d_eval)
    exclusions = blind_exclusions(
        blind, scope_start=split.scope_start, scope_end=split.scope_end, incident=args.incident
    ) + quiet_band_exclusions(split.scope_start, split.scope_end)
    padded = [item for item in exclusions if item.padded]
    if split.scope_start <= args.incident <= split.scope_end and len(padded) != 1:
        raise ValueError(
            f"the named incident is not in the blind window table: padded={len(padded)}"
        )

    days = apply_spans(inventory.days, exclusions)
    args.out.mkdir(parents=True, exist_ok=True)
    write_exclusions(args.out / "exclusions.parquet", exclusions)
    write_event_days(args.out / "event_days.parquet", days)
    digest = write_split(args.out / "split.json", split)

    payload: dict[str, object] = {
        "blind_windows": str(args.blind_windows),
        "coverage": str(args.coverage),
        "out": str(args.out),
        "cohort": args.cohort,
        "d_eval": args.d_eval,
        "scope_open": args.start.isoformat(),
        "tape_first": tape_first.isoformat(),
        "tape_last": tape_last.isoformat(),
        "scope_start": split.scope_start.isoformat(),
        "scope_end": split.scope_end.isoformat(),
        "cities": len(split.cities),
        "event_days": len(days),
        "covered_days": sum(1 for day in days if day.covered),
        "evaluable_days": sum(1 for day in days if day.evaluable),
        "in_scope_days": sum(1 for day in days if day.in_scope),
        "over_tolerance_days": inventory.over_tolerance,
        "worst_outage_us": max(
            inventory.outage_us[(day.series, day.event_date)] for day in days if day.in_scope
        ),
        "first_evaluable_event_day": split.discovery_days[0].isoformat(),
        "last_evaluable_event_day": split.holdout_days[-1].isoformat(),
        "boundary_event_day": split.boundary_event_day.isoformat(),
        "exclusions": len(exclusions),
    }
    for name in CLASSES:
        members = [item for item in exclusions if item.exclusion_class == name]
        payload[f"{name}_intervals"] = len(members)
        payload[f"{name}_us"] = sum(item.duration_us for item in members)
    payload |= {
        "padded_boundary_id": padded[0].boundary_id if padded else None,
        "padded_gap_id": padded[0].gap_id if padded else None,
        "union_excluded_us": union_overlap_us(exclusions, split.scope_start, split.scope_end),
        "split_sha256": digest,
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=1))
    print(format_report(payload))
    return 0


def format_report(payload: Mapping[str, object]) -> str:
    return "\n".join(["== RUN SCOPE"] + [f"{name}={value}" for name, value in payload.items()])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
