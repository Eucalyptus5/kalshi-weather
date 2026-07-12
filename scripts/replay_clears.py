from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.replay.blind_windows import FROZEN_END, FROZEN_START  # noqa: E402
from bot.replay.clears import (  # noqa: E402
    MidLifeClear,
    build_summary,
    scan_clears,
    write_mid_life_clears,
)


FROZEN_DAYS = [
    (FROZEN_START + timedelta(days=offset)).date()
    for offset in range((FROZEN_END - FROZEN_START).days)
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="enumerate the both-sides-empty snapshots the database never recorded"
    )
    parser.add_argument("--db", type=Path, required=True, help="the recorder database, read only")
    parser.add_argument("--raw-dir", type=Path, required=True, help="directory of gzipped days")
    parser.add_argument("--out", type=Path, required=True, help="parquet table to write")
    parser.add_argument("--summary", type=Path, default=None, help="write the summary as json")
    parser.add_argument(
        "--day",
        type=date.fromisoformat,
        action="append",
        default=[],
        metavar="YYYY-MM-DD",
        help="scan this UTC day; repeatable; defaults to the frozen window",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    # A day named twice would otherwise be scanned twice and count every clear on it twice.
    days = sorted(set(args.day)) if args.day else list(FROZEN_DAYS)
    paths = [args.raw_dir / f"{day.isoformat()}.jsonl.gz" for day in days]
    # Fifteen days of tape take hours, and both of these otherwise surface at the end of them.
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"no tape at {' '.join(missing)}")

    started = time.monotonic()
    scan = scan_clears(args.db, paths)
    write_mid_life_clears(args.out, scan.mid_life)
    payload: dict[str, object] = {
        "db": str(args.db),
        "raw_dir": str(args.raw_dir),
        "out": str(args.out),
        "scanned_days": ",".join(day.isoformat() for day in days),
        **asdict(build_summary(scan)),
        "elapsed_s": round(time.monotonic() - started, 1),
    }

    if args.summary is not None:
        args.summary.write_text(json.dumps(payload, indent=1))
    print(format_report(payload, scan.mid_life))
    return 0


def format_report(payload: Mapping[str, object], clears: Sequence[MidLifeClear]) -> str:
    lines = ["== MID LIFE CLEARS"]
    lines.extend(f"{name}={value}" for name, value in payload.items())
    for clear in clears:
        snapshot = "none" if clear.next_snapshot_at is None else clear.next_snapshot_at.isoformat()
        lines.append(
            f"clear={clear.ticker} {clear.clear_at.isoformat()} next_snapshot={snapshot} "
            f"stale_us={clear.stale_us} stale_rows={clear.stale_rows}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
