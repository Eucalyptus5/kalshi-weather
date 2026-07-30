from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.lead_lag_run import (  # noqa: E402
    FORWARD,
    RESULTS_NAME,
    REVERSE,
    execute,
    result_payload,
)
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)
READINGS = (FORWARD, REVERSE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read the cross-city lead-lag off the frozen run scope, reported only"
    )
    parser.add_argument("--run-id", required=True, help="names the run directory under --run-root")
    parser.add_argument(
        "--preregistration", type=Path, required=True, help="the file the manifest hashes"
    )
    parser.add_argument(
        "--run-scope", type=Path, required=True, help="the frozen run-scope directory"
    )
    parser.add_argument(
        "--artifacts", type=Path, required=True, help="the forward-pass artifact root"
    )
    parser.add_argument("--rtt-samples", type=Path, required=True, help="the read-RTT sample file")
    parser.add_argument(
        "--floor-source",
        required=True,
        choices=FLOOR_SOURCES,
        help="which round trip supplied the latency floor",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="the bootstrap seed every resample runs under"
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    return parser


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    try:
        result = execute(
            run_id=args.run_id,
            preregistration=args.preregistration,
            repo=args.repo,
            run_scope=args.run_scope,
            artifacts=args.artifacts,
            rtt_samples=args.rtt_samples,
            floor_source=FloorSource(args.floor_source),
            seed=args.seed,
            run_root=args.run_root,
        )
    except ManifestIncomplete as exc:
        print(exc, file=sys.stderr)
        return 1

    payload = result_payload(result) | {"elapsed_s": round(time.monotonic() - started, 1)}
    (args.run_root / args.run_id / RESULTS_NAME).write_text(json.dumps(payload, indent=1))
    print(format_report(payload))
    return 0


def format_report(payload: dict) -> str:
    ceiling = payload["corridor_day_ceiling"]
    reporting_floor = payload["corridor_day_min_report"]
    exclusions = payload["exclusions"]
    lines = [
        f"== Q2 CROSS-CITY LEAD-LAG  run_id={payload['run_id']}  reported only, no gate",
        "  this reading carries no verdict and evaluates no threshold",
        "",
        "== RUN",
        f"  manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"  seed={payload['bootstrap_seed']}  resamples={payload['bootstrap_resamples']}  "
        f"block_days={payload['block_days']}  ci_level={payload['ci_level']}",
        f"  move_bar_cents={payload['move_bar_cents']}  "
        f"round_trip_fee_cents_at_mid={payload['round_trip_fee_cents_at_mid']}  "
        f"window_s={payload['window_s']}",
        f"  latency_floor_source={payload['latency_floor_source']}",
        f"  split={payload['split']}  discovery_days={len(payload['discovery_days'])}  "
        f"corridor_day_ceiling={ceiling}  corridor_day_min_report={reporting_floor}",
        "",
    ]
    for reading in READINGS:
        lines.extend(
            _format_reading(
                payload[reading], ceiling, reporting_floor, payload["corridor_day_unit"]
            )
        )
        lines.append("")
    lines.extend(
        [
            "== EPISODES",
            f"  offered={payload['episodes']['offered']}  kept={payload['episodes']['kept']}  "
            f"{FORWARD}={payload['episodes'][FORWARD]}  {REVERSE}={payload['episodes'][REVERSE]}",
            "",
            "== EXCLUSIONS",
            f"  candidates={exclusions['candidates']} excluded={exclusions['excluded']} "
            f"fraction={exclusions['excluded_fraction']} "
            f"out_of_window={exclusions['out_of_window']} "
            f"out_of_scope={exclusions['out_of_scope']}",
            "  " + " ".join(f"{name}={count}" for name, count in exclusions["by_class"].items()),
            "",
            "== COVERAGE",
            f"  rows={payload['rows']}  city_event_days={payload['city_event_days']}  "
            f"no_atm_series={len(payload['no_atm_series'])}",
            *(f"  no series {key}" for key in payload["no_atm_series"]),
            *(
                f"  {key} atm={ticker}"
                for key, ticker in payload["atm_ticker_per_city_day"].items()
            ),
            "",
            "== PAIRS",
        ]
    )
    lines.extend(
        f"  {name} {block['corridor']} {block['upstream_series']} -> {block['downstream_series']}"
        for name, block in payload["pairs"].items()
    )
    lines.append(f"  unpaired={','.join(payload['unpaired_roots'])}")
    return "\n".join(lines)


def _format_reading(item: dict, ceiling: int, reporting_floor: int, unit: str) -> list[str]:
    lines = [
        f"== {item['reading'].upper()}",
        f"  episodes={item['episodes']}  corridor_days={item['corridor_days']}/{ceiling} {unit}  "
        f"n_blocks={item['n_blocks']}",
        *_format_estimate(item, ceiling, reporting_floor, unit),
        "  -- PAIRS",
        *(
            f"    {name} {block['corridor']} episodes={block['episodes']} "
            f"median_lead_s={block['median_lead_s']}"
            for name, block in item["per_pair"].items()
        ),
        "  -- CORRIDORS",
        *(
            f"    {name} episodes={block['episodes']} corridor_days={block['corridor_days']} "
            f"median_lead_s={block['median_lead_s']}"
            for name, block in item["per_corridor"].items()
        ),
        "  -- EPISODES PER CORRIDOR-DAY",
    ]
    lines.extend(f"    {name} {count}" for name, count in item["episodes_per_corridor_day"].items())
    return lines


def _format_estimate(item: dict, ceiling: int, reporting_floor: int, unit: str) -> list[str]:
    if item["median_lead_s"] is None:
        return [
            f"  corridor_days={item['corridor_days']} of a ceiling of {ceiling} {unit}, under "
            f"the reporting floor of {reporting_floor}: the estimate is declined"
        ]
    interval = item["interval"]
    return [
        f"  median_lead_s={item['median_lead_s']}  "
        f"ci{interval['ci_level']}=[{_bound(interval['low'])}, {_bound(interval['high'])}]  "
        f"tail={interval['tail']:.4f}  tested={interval['tested']}"
    ]


def _bound(value: str | None) -> str:
    return "none admitted" if value is None else value


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
