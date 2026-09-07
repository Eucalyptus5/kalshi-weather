from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.forecast_sample import (  # noqa: E402
    F4_LEADS,
    SPLIT_BOUNDARY,
    build_sample,
    era_caps,
    era_index,
    read_sample_plan,
    read_tick_tape,
    sidecar_path,
    write_sample_freeze,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
MARKETS = REPO_ROOT / "data" / "backtest" / "weather_markets.parquet"
TICKS = REPO_ROOT / "data" / "backtest" / "ticks_kxhigh.parquet"
SAMPLE_PLAN = REPO_ROOT / "data" / "backtest" / "results" / "sample_plan.json"
ERA_REPORT = REPO_ROOT / "data" / "backtest" / "results" / "era_report.json"
INGEST_REPORT = REPO_ROOT / "data" / "backtest" / "results" / "ingest_report.json"
DEFAULT_OUT = REPO_ROOT / "data" / "tape_studies" / "f4_inputs"
FREEZE_NAME = "sample.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="freeze the (ticker, lead) legs and entry prices the forecast study scores"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="the directory the frozen sample and its sidecar are written to",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    plan = read_sample_plan(SAMPLE_PLAN)
    caps = era_caps(ERA_REPORT)
    eras = era_index(INGEST_REPORT)
    tape = read_tick_tape(TICKS)

    by_lead = {lead: build_sample(MARKETS, tape, plan, caps, eras, lead) for lead in F4_LEADS}
    legs = [leg for lead in F4_LEADS for leg in by_lead[lead]]

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / FREEZE_NAME
    digest = write_sample_freeze(legs, path)

    leads = {}
    for lead, rows in by_lead.items():
        days = {leg.event_date for leg in rows}
        discovery = {day for day in days if day < SPLIT_BOUNDARY}
        leads[lead] = {
            "legs": len(rows),
            "event_days": len(days),
            "discovery_days": len(discovery),
            "holdout_days": len(days - discovery),
        }

    print(
        json.dumps(
            {
                "out": str(path),
                "sidecar": str(sidecar_path(path)),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "legs": len(legs),
                "unread_days": len(plan.unread_days),
                "discovery_days": len(plan.discovery_days),
                "holdout_days": len(plan.holdout_days),
                "leads": leads,
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
