from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.observation_freeze import (  # noqa: E402
    F2_STATIONS,
    pull_observations,
    read_observation_index,
    write_observation_index,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "freeze the decoded one-minute observations and the official daily extreme "
            "behind the settlement study"
        )
    )
    parser.add_argument(
        "--start", type=date.fromisoformat, required=True, help="first event day of the window"
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        required=True,
        help="last event day of the window, inclusive",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="the directory the sidecars are frozen into"
    )
    return parser


def run(args: argparse.Namespace) -> int:
    span = (args.end - args.start).days
    if span < 0:
        raise ValueError(f"--end {args.end.isoformat()} precedes --start {args.start.isoformat()}")
    event_dates = [args.start + timedelta(days=offset) for offset in range(span + 1)]

    args.out.mkdir(parents=True, exist_ok=True)
    pull_observations(sorted(F2_STATIONS), event_dates, args.out)
    digest = write_observation_index(args.out, datetime.now(tz=timezone.utc))
    index = read_observation_index(args.out)

    frozen = {}
    for station, row in sorted(index.stations.items()):
        frozen[station] = {
            "sha256": row.sha256,
            "timezone": row.timezone,
            "event_days": row.event_days,
            "decoded_minutes": sum(row.decoded_minutes.values()),
            "empty_days": [
                day.isoformat()
                for day, minutes in sorted(row.decoded_minutes.items())
                if minutes == 0
            ],
            "missing_acis": [day.isoformat() for day in row.missing_acis],
        }

    print(
        json.dumps(
            {
                "out": str(args.out),
                "observed_at": index.observed_at.isoformat(),
                "start_date": index.start_date.isoformat(),
                "end_date": index.end_date.isoformat(),
                "extreme": index.extreme,
                "source": index.source,
                "sha256": digest,
                "event_days": sum(item["event_days"] for item in frozen.values()),
                "decoded_minutes": sum(item["decoded_minutes"] for item in frozen.values()),
                "empty_days": sum(len(item["empty_days"]) for item in frozen.values()),
                "missing_acis": sum(len(item["missing_acis"]) for item in frozen.values()),
                "stations": frozen,
            },
            indent=1,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
