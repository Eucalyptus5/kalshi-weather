from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.ladder_run import (  # noqa: E402
    ECONOMIC_BAR_PRICE,
    ECONOMIC_BAR_PRICE_SOURCE,
    ECONOMIC_BAR_SIZE,
    RESULTS_NAME,
    execute,
    result_payload,
)
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402
from bot.replay.analysis_stations import HIGH, LOW  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read the ladder consistency gate and its holdout off the frozen run scope"
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
    parser.add_argument(
        "--cohort",
        choices=(HIGH, LOW),
        default=None,
        help="which ladder of the frozen scope the run reads",
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
            economic_bar_size=ECONOMIC_BAR_SIZE,
            economic_bar_price=ECONOMIC_BAR_PRICE,
            economic_bar_price_source=ECONOMIC_BAR_PRICE_SOURCE,
            seed=args.seed,
            run_root=args.run_root,
            cohort=args.cohort,
        )
    except ManifestIncomplete as exc:
        print(exc, file=sys.stderr)
        return 1

    payload = result_payload(result) | {"elapsed_s": round(time.monotonic() - started, 1)}
    (args.run_root / args.run_id / RESULTS_NAME).write_text(json.dumps(payload, indent=1))
    print(format_report(payload))
    return 0


def format_report(payload: dict) -> str:
    discovery = payload["discovery"]
    exclusions = payload["exclusions"]
    ladders = payload["ladders"]
    universe = payload["universe"]
    lines = [
        f"== Q1 LADDER CONSISTENCY  run_id={payload['run_id']}  verdict={payload['verdict']}",
        f"median_excess_cents={discovery['median_excess_cents']}  "
        f"ci{discovery['ci_level']}=[{_fixed(discovery['ci_low'], 4)}, "
        f"{_fixed(discovery['ci_high'], 4)}]  p_value={_fixed(discovery['p_value'], 5)}  "
        f"n={discovery['n_city_days']} city event-days",
        "",
        "== GATE",
        *_format_gate(payload["gate"], payload["replication_skipped"]),
        "",
        "== REPLICATION",
        *_format_replication(payload["replication"], payload["replication_skipped"]),
        "",
        "== RUN",
        f"  manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"  seed={payload['bootstrap_seed']}  resamples={payload['bootstrap_resamples']}  "
        f"excess_bar={payload['excess_bar']}  depth_min={payload['depth_min']}",
        f"  t_persist_s={payload['t_persist_s']}  "
        f"latency_floor_source={payload['latency_floor_source']}",
        f"  universe={universe['read']}  series={','.join(universe['series'])}",
        "",
        "== DISCOVERY",
        *_format_split(discovery),
        "",
        "== HOLDOUT",
        *_format_split(payload["holdout"]),
        "",
        "== EPISODES",
        *_format_episodes(payload["episodes"]),
        "",
    ]
    for name, block in payload["distributions"].items():
        lines.extend(_format_distributions(name, block))
        lines.append("")
    lines.extend(
        [
            "== EXCLUSIONS",
            f"  candidates={exclusions['candidates']} excluded={exclusions['excluded']} "
            f"fraction={exclusions['excluded_fraction']} "
            f"out_of_window={exclusions['out_of_window']} "
            f"out_of_scope={exclusions['out_of_scope']}",
            "  " + " ".join(f"{name}={count}" for name, count in exclusions["by_class"].items()),
            "",
            "== LADDERS",
            f"  in_scope={ladders['in_scope']} complete={ladders['complete']} "
            f"incomplete={ladders['incomplete']}",
            *(f"  incomplete {key}" for key in ladders["incomplete_keys"]),
            "",
            "== COVERAGE",
            f"  rows={payload['rows']}  cities={','.join(payload['cities'])}",
        ]
    )
    lines.extend(
        f"  {name} tickers={count}" for name, count in payload["tickers_per_city_day"].items()
    )
    return "\n".join(lines)


def _fixed(value: float | None, places: int) -> str:
    return "None" if value is None else f"{value:.{places}f}"


def _format_split(item: dict) -> list[str]:
    return [
        f"  split={item['split']}  median_excess_cents={item['median_excess_cents']}  "
        f"ci{item['ci_level']}=[{_fixed(item['ci_low'], 4)}, {_fixed(item['ci_high'], 4)}]  "
        f"p_value={_fixed(item['p_value'], 5)}",
        f"  n_city_days={item['n_city_days']}  population={item['population']}  "
        f"episodes={item['episodes']}  clusters={item['clusters']}",
    ]


def _format_episodes(block: dict) -> list[str]:
    return [
        f"  {name} found={counts['found']} tradeable={counts['tradeable']} "
        f"kept={counts['kept']} censored={counts['censored']} "
        f"incomplete_states={counts['incomplete_states']}"
        for name, counts in block.items()
    ]


def _format_distributions(title: str, block: dict) -> list[str]:
    lines = [f"== DISTRIBUTIONS ({title})"]
    lines.extend(
        f"  {group} {metric} count={summary['count']} min={summary['min']} "
        f"p25={summary['p25']} median={summary['median']} p75={summary['p75']} "
        f"p90={summary['p90']} max={summary['max']}"
        for group, metrics in block.items()
        for metric, summary in metrics.items()
    )
    return lines


def _format_gate(gate: dict | None, skipped: str) -> list[str]:
    if gate is None:
        return [f"  not evaluated: {skipped}"]
    return [
        f"  estimate={gate['estimate']}  threshold={gate['threshold']}  "
        f"direction={gate['direction']}  p_value={gate['p_value']:.5f}  alpha={gate['alpha']}",
        f"  n={gate['n']} {gate['n_unit']}  n_min={gate['n_min']}  "
        f"economic={gate['economic']}  significant={gate['significant']}  "
        f"powered={gate['powered']}  undecidable={gate['undecidable']}  "
        f"passed={gate['passed']}",
    ]


def _format_replication(replication: dict | None, skipped: str) -> list[str]:
    if replication is None:
        return [f"  not evaluated: {skipped}"]
    return [
        f"  holdout_estimate={replication['holdout_estimate']}  "
        f"discovery_estimate={replication['discovery_estimate']}  "
        f"p_value={replication['holdout_p_value']:.5f}  alpha={replication['alpha']}",
        f"  holdout_n={replication['holdout_n']} {replication['n_unit']}  "
        f"holdout_n_min={replication['holdout_n_min']}  "
        f"same_sign={replication['same_sign']}  magnitude={replication['magnitude']}  "
        f"significant={replication['significant']}  powered={replication['powered']}  "
        f"undecidable={replication['undecidable']}  replicated={replication['replicated']}",
    ]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
