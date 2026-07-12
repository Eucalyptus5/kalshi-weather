from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.replay.blind_windows import (  # noqa: E402
    BlindWindow,
    attach_gap_rows,
    build_summary,
    scan_blind_windows,
    write_blind_windows,
)
from bot.replay.inventory import read_gap_rows  # noqa: E402
from bot.replay.raw_tape import check_tape_counts, count_frames_in_windows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="inventory the blind window every subscription boundary leaves behind"
    )
    parser.add_argument("--db", type=Path, required=True, help="the recorder database, read only")
    parser.add_argument("--out", type=Path, required=True, help="parquet table to write")
    parser.add_argument("--max-id", type=int, default=None, help="bound the last id scanned")
    parser.add_argument("--summary", type=Path, default=None, help="write the summary as json")
    parser.add_argument("--raw-dir", type=Path, default=None, help="directory of gzipped days")
    parser.add_argument(
        "--validate-day",
        type=date.fromisoformat,
        action="append",
        default=[],
        metavar="YYYY-MM-DD",
        help="check every window starting on this UTC day against the raw tape; repeatable",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.validate_day and args.raw_dir is None:
        raise ValueError("--validate-day needs --raw-dir")
    started = time.monotonic()
    scan = scan_blind_windows(args.db, max_id=args.max_id)
    windows, unmatched = attach_gap_rows(
        scan.windows, read_gap_rows(args.db), scanned_through=scan.last_received_at
    )
    write_blind_windows(args.out, windows)
    payload: dict[str, object] = {
        "db": str(args.db),
        "out": str(args.out),
        "max_id": args.max_id,
        "rows_scanned": scan.rows,
        **{
            name: value.isoformat() if isinstance(value, datetime) else value
            for name, value in asdict(build_summary(windows)).items()
        },
        "unmatched_gap_rows": len(unmatched),
    }

    check = None
    if args.validate_day:
        # A day named twice would otherwise be scanned twice and count every frame twice, which
        # reads as the database window being wider than the tape.
        days = sorted(set(args.validate_day))
        sample = [window for window in windows if window.start.date() in days]
        paths = [args.raw_dir / f"{day.isoformat()}.jsonl.gz" for day in days]
        check = check_tape_counts(sample, count_frames_in_windows(paths, sample))
        payload |= {
            "validated_days": ",".join(day.isoformat() for day in days),
            "validated_windows": check.sampled,
            "tape_agreed": check.agreed,
            "tape_wider": check.wider,
            "tape_extra_frames": check.extra_frames,
            "tape_falsified": check.falsified,
        }
    payload["elapsed_s"] = round(time.monotonic() - started, 1)

    if args.summary is not None:
        args.summary.write_text(json.dumps(payload, indent=1))
    print(format_report(payload, () if check is None else check.falsifying))
    return 1 if check is not None and check.falsified else 0


def format_report(payload: Mapping[str, object], falsifying: Sequence[BlindWindow]) -> str:
    lines = ["== BLIND WINDOWS"]
    lines.extend(f"{name}={value}" for name, value in payload.items())
    lines.extend(
        f"falsifying={window.boundary_id} {window.start.isoformat()} {window.end.isoformat()} "
        f"blind_us={window.blind_us}"
        for window in falsifying
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
