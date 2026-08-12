from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402
from bot.lag.taker_flow_run import (  # noqa: E402
    ECONOMIC_BAR_PRICE,
    ECONOMIC_BAR_PRICE_SOURCE,
    ECONOMIC_BAR_SIZE,
    MAKER_RATE,
    MAKER_RATE_SOURCE,
    RESULTS_NAME,
    execute,
    result_payload,
)
from bot.replay.analysis_stations import HIGH, LOW  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read the taker flow gate and its holdout off the frozen run scope"
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
            maker_rate=MAKER_RATE,
            maker_rate_source=MAKER_RATE_SOURCE,
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
    prints = payload["prints"]
    exclusions = payload["exclusions"]
    drops = payload["kernel_drops"]
    lines = [
        f"== Q3 TAKER FLOW  run_id={payload['run_id']}  verdict={payload['verdict']}",
        f"manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"seed={payload['bootstrap_seed']}  resamples={payload['bootstrap_resamples']}  "
        f"cent_bar={payload['cent_bar']}  primary_horizon_s={payload['primary_horizon_s']}",
        "",
        "== PRINTS",
        f"  in_scope pooled={prints['in_scope_pooled']} "
        f"discovery={prints['in_scope_discovery']} holdout={prints['in_scope_holdout']}",
        f"  screened pooled={prints['screened_pooled']} "
        f"discovery={prints['screened_discovery']} holdout={prints['screened_holdout']}",
        f"  empty_side={prints['empty_side']} duplicate_trade_id={prints['duplicate_trade_id']} "
        f"out_of_scope={prints['out_of_scope']} out_of_window={prints['out_of_window']}",
        "",
        "== DISCOVERY (primary)",
        *_format_readout(payload["discovery"]),
        "",
        "== HOLDOUT (primary)",
        *_format_readout(payload["holdout"]),
        "",
        "== GATE",
        *_format_gate(payload["gate"], payload["replication_skipped"]),
        "",
        "== REPLICATION",
        *_format_replication(payload["replication"], payload["replication_skipped"]),
        "",
        "== HORIZON CURVE (discovery)",
    ]
    for item in payload["horizon_curve"]:
        lines.append(f"  -- {item['horizon_s']}s")
        lines.extend(_format_readout(item))
    lines.extend(
        [
            "",
            "== EXCLUSIONS",
            f"  candidates={exclusions['candidates']} excluded={exclusions['excluded']} "
            f"fraction={exclusions['excluded_fraction']}",
            "  " + " ".join(f"{name}={count}" for name, count in exclusions["by_class"].items()),
            "",
            "== KERNEL DROPS",
            f"  unresolved={drops['unresolved']} uncovered={drops['uncovered']} "
            f"one_sided={drops['one_sided']} host_clock={drops['host_clock']} "
            f"read_ts_violations={drops['read_ts_violations']} "
            f"fractional_size_prints={drops['fractional_size_prints']}",
            "",
            "== COVERAGE",
            f"  cities={','.join(payload['cities'])}",
        ]
    )
    lines.extend(
        f"  {name} tickers={count}" for name, count in payload["tickers_per_city_day"].items()
    )
    return "\n".join(lines)


def _fixed(value: float | None, places: int) -> str:
    return "None" if value is None else f"{value:.{places}f}"


def _format_readout(item: dict) -> list[str]:
    return [
        f"  mean_net_cents={item['mean_net_cents']}  "
        f"ci{item['ci_level']}=[{_fixed(item['ci_low'], 4)}, {_fixed(item['ci_high'], 4)}]  "
        f"n_prints={item['n_prints']}  contracts={item['contracts']}  "
        f"clusters={item['clusters']}  p_value={_fixed(item['p_value'], 5)}",
        f"  candidates={item['candidates']}  excluded={item['excluded']}  "
        f"excluded_fraction={item['excluded_fraction']}  out_of_window={item['out_of_window']}  "
        f"unresolved={item['unresolved']}  uncovered={item['uncovered']}  "
        f"one_sided={item['one_sided']}  host_clock={item['host_clock']}",
    ]


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
