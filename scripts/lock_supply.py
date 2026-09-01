from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.lock_convergence import ROUNDING_MARGIN_F  # noqa: E402
from bot.lag.lock_supply import (  # noqa: E402
    DISCOVERY_STATION_DAY_MIN,
    HOLDOUT_STATION_DAY_MIN,
    POOLED,
    LockSupply,
    SplitSupply,
    StationDaySupply,
    read_ladders,
    read_observations,
    summarize,
    survey_lock_supply,
)
from bot.lag.observation_freeze import read_observation_index  # noqa: E402
from bot.lag.tape_studies import load_run_scope  # noqa: E402
from bot.replay.run_scope import DISCOVERY, HOLDOUT  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="count the clean lock events the frozen listed ladders supply"
    )
    parser.add_argument(
        "--run-scope", type=Path, required=True, help="the frozen run-scope directory"
    )
    parser.add_argument(
        "--closes", type=Path, required=True, help="the directory the close sidecars are frozen in"
    )
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="the directory the observation freeze is written to",
    )
    return parser


def split_payload(item: SplitSupply) -> dict:
    return {
        "split": item.split,
        "station_days": item.station_days,
        "station_days_with_readings": item.station_days_with_readings,
        "markets": item.markets,
        "clean": item.clean,
        "ambiguous": item.ambiguous,
        "no_lock": item.no_lock,
        "window_open": item.window_open,
        "mid_day": item.mid_day,
        "window_open_station_days": item.window_open_station_days,
        "mid_day_station_days": item.mid_day_station_days,
        "ambiguous_to_clean": (
            None if item.ambiguous_to_clean is None else str(item.ambiguous_to_clean)
        ),
        "ambiguous_to_mid_day": (
            None if item.ambiguous_to_mid_day is None else str(item.ambiguous_to_mid_day)
        ),
    }


def station_day_payload(row: StationDaySupply) -> dict:
    return {
        "series": row.series,
        "station": row.station,
        "event_date": row.event_date.isoformat(),
        "split": row.split,
        "readings": row.readings,
        "markets": row.markets,
        "clean": row.clean,
        "ambiguous": row.ambiguous,
        "no_lock": row.no_lock,
        "window_open": row.window_open,
        "mid_day": row.mid_day,
    }


def root_supply(supply: LockSupply, root: str) -> LockSupply:
    rows = tuple(row for row in supply.station_days if row.series == root)
    return LockSupply(roots=(root,), markets=sum(row.markets for row in rows), station_days=rows)


def run(args: argparse.Namespace) -> int:
    scope = load_run_scope(args.run_scope)
    index = read_observation_index(args.observations)
    supply = survey_lock_supply(
        scope,
        read_ladders(args.closes),
        read_observations(args.observations, index),
    )
    discovery = summarize(supply, DISCOVERY)
    holdout = summarize(supply, HOLDOUT)

    print(
        json.dumps(
            {
                "run_scope": str(args.run_scope),
                "closes": str(args.closes),
                "observations": str(args.observations),
                "observations_sha256": index.sha256,
                "roots": list(supply.roots),
                "markets": supply.markets,
                "rounding_margin_f": str(ROUNDING_MARGIN_F),
                "discovery_station_day_min": DISCOVERY_STATION_DAY_MIN,
                "discovery_mid_day_station_days": discovery.mid_day_station_days,
                "holdout_station_day_min": HOLDOUT_STATION_DAY_MIN,
                "holdout_mid_day_station_days": holdout.mid_day_station_days,
                "splits": {
                    DISCOVERY: split_payload(discovery),
                    HOLDOUT: split_payload(holdout),
                    POOLED: split_payload(summarize(supply, None)),
                },
                "by_root": {
                    root: split_payload(summarize(root_supply(supply, root), None))
                    for root in supply.roots
                },
                "station_days": [station_day_payload(row) for row in supply.station_days],
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
