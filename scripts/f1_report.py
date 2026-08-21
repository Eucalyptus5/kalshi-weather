from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.maker_edge_run import (  # noqa: E402
    COHORT,
    MAKER_RATE,
    MAKER_RATE_SOURCE,
    RESULTS_NAME,
    execute,
    result_payload,
)
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402
from bot.lag.tape_studies import SELF_CHARGED_BAR, SELF_CHARGED_BAR_SOURCE  # noqa: E402
from bot.replay.analysis_stations import HIGH, LOW  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read the maker edge gate and its holdout off the frozen run scope"
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
    parser.add_argument(
        "--closes", type=Path, required=True, help="the directory of frozen close sidecars"
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
        default=COHORT,
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
            closes=args.closes,
            rtt_samples=args.rtt_samples,
            floor_source=FloorSource(args.floor_source),
            maker_rate=MAKER_RATE,
            maker_rate_source=MAKER_RATE_SOURCE,
            economic_bar_size=SELF_CHARGED_BAR,
            economic_bar_price=SELF_CHARGED_BAR,
            economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
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
    fills = payload["fills"]
    exclusions = payload["exclusions"]
    sensitivity = payload["published_rate_sensitivity"]
    lines = [
        f"== F1 MAKER EDGE  run_id={payload['run_id']}  verdict={payload['verdict']}",
        f"manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"seed={payload['bootstrap_seed']}  resamples={payload['bootstrap_resamples']}  "
        f"primary_horizon_s={payload['primary_horizon_s']}  alpha={payload['alpha']:.6f}",
        f"bar={payload['bar']} strict={payload['bar_is_strict']} source={payload['bar_source']}  "
        f"maker_rate={payload['maker_rate']}",
        "",
        "== FILLS",
        f"  offered={fills['offered']} yes_fills={fills['yes_fills']} "
        f"no_fills={fills['no_fills']} yes_empty={fills['yes_empty']} "
        f"no_empty={fills['no_empty']} unnamed_markets={fills['unnamed_markets']}",
        f"  scored discovery={fills['scored_discovery']} holdout={fills['scored_holdout']}",
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
        lines.append(f"  -- {item['horizon_s']}s  window_cap_s={item['window_cap_s']}")
        lines.extend(_format_readout(item))
    lines.extend(
        [
            "",
            "== PUBLISHED-RATE SENSITIVITY (reported, not gating)",
            f"  rate={sensitivity['rate']} source={sensitivity['rate_source']} "
            f"horizon_s={sensitivity['horizon_s']}",
            f"  discovery={sensitivity['discovery_edge_cents_per_contract']} "
            f"market_days={sensitivity['discovery_market_days']}  "
            f"holdout={sensitivity['holdout_edge_cents_per_contract']} "
            f"market_days={sensitivity['holdout_market_days']}",
            "",
            "== EXCLUSIONS",
            f"  candidates={exclusions['candidates']} excluded={exclusions['excluded']} "
            f"fraction={exclusions['excluded_fraction']}",
            "  " + " ".join(f"{name}={count}" for name, count in exclusions["by_class"].items()),
            "",
            "== COVERAGE",
            f"  cities={','.join(payload['cities'])}  "
            f"market_day_min_discovery={payload['market_day_min_discovery']}",
        ]
    )
    lines.extend(
        f"  {name} markets={count}" for name, count in payload["markets_per_city_day"].items()
    )
    return "\n".join(lines)


def _fixed(value: float | None, places: int) -> str:
    return "None" if value is None else f"{value:.{places}f}"


def _format_readout(item: dict) -> list[str]:
    return [
        f"  edge_cents_per_contract={item['edge_cents_per_contract']}  "
        f"ci{item['ci_level']}=[{_fixed(item['ci_low'], 4)}, {_fixed(item['ci_high'], 4)}]  "
        f"p_value={_fixed(item['p_value'], 5)}  market_days={item['market_days']}  "
        f"n_fills={item['n_fills']}  contracts={item['contracts']}",
        f"  degenerate={item['degenerate']}  "
        f"replicate_spread={_fixed(item['replicate_spread'], 6)}  "
        f"modelled={item['modelled']}  no_mid_drops={item['no_mid_drops']}  "
        f"no_mid_fraction={item['no_mid_fraction']}",
        f"  candidates={item['candidates']}  excluded={item['excluded']}  "
        f"excluded_fraction={item['excluded_fraction']}  "
        f"out_of_window={item['out_of_window']}  out_of_scope={item['out_of_scope']}",
    ]


def _format_gate(gate: dict | None, skipped: str) -> list[str]:
    if gate is None:
        return [f"  not evaluated: {skipped}"]
    return [
        f"  estimate={gate['estimate']}  threshold={gate['threshold']}  "
        f"direction={gate['direction']}  p_value={gate['p_value']:.5f}  alpha={gate['alpha']:.6f}",
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
