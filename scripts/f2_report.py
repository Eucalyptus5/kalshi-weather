from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.near_lock import read_observations  # noqa: E402
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402
from bot.lag.settlement_run import (  # noqa: E402
    BOOTSTRAP_SEED,
    COHORT,
    NO_GATE_ESTIMATE,
    RESULTS_NAME,
    execute,
    result_payload,
)
from bot.lag.tape_studies import SELF_CHARGED_BAR, SELF_CHARGED_BAR_SOURCE  # noqa: E402
from bot.replay.analysis_stations import HIGH, LOW  # noqa: E402
from scripts.q4_report import read_settles_cache  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)
UNIDENTIFIABLE = "the settling row rests on an extreme the window has not finished producing"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read the settlement lag gate and its holdout off the frozen run scope"
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
    parser.add_argument(
        "--settlement-sources",
        type=Path,
        required=True,
        help="the frozen series settlement-source sidecar the run is read under",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="the arrivals export the readings come from",
    )
    parser.add_argument(
        "--settles",
        type=Path,
        required=True,
        help="the official daily readings the run compares to",
    )
    parser.add_argument("--rtt-samples", type=Path, required=True, help="the read-RTT sample file")
    parser.add_argument(
        "--floor-source",
        required=True,
        choices=FLOOR_SOURCES,
        help="which round trip supplied the latency floor",
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
            settlement_sources=args.settlement_sources,
            observations=read_observations(args.observations),
            settles=read_settles_cache(args.settles),
            rtt_samples=args.rtt_samples,
            floor_source=FloorSource(args.floor_source),
            economic_bar_size=SELF_CHARGED_BAR,
            economic_bar_price=SELF_CHARGED_BAR,
            economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
            seed=BOOTSTRAP_SEED,
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
    straddles = payload["straddles"]
    entries = payload["entries"]
    deltas = payload["deltas"]
    source = payload["settlement_source"]
    screen = payload["screen"]
    pre_boundary = payload["holdout_pre_boundary"]
    lines = [
        f"== F2 SETTLEMENT LAG  run_id={payload['run_id']}  verdict={payload['verdict']}",
        f"manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"seed={payload['bootstrap_seed']}  resamples={payload['bootstrap_resamples']}  "
        f"alpha={payload['alpha']:.6f}  cohort={payload['cohort']}",
        f"bar={payload['bar']} strict={payload['bar_is_strict']} source={payload['bar_source']}  "
        f"size={payload['size']}",
        "",
        "== STRADDLES",
        f"  city_event_days={straddles['city_event_days']} straddles={straddles['straddles']} "
        f"no_readings={straddles['no_readings']} unsettled={straddles['unsettled']} "
        f"no_straddle={straddles['no_straddle']}",
        f"  identifiable_ex_ante={payload['identifiable_ex_ante']}: {UNIDENTIFIABLE}",
        "",
        "== ENTRIES",
        f"  n={entries['n']} entry_at_close_n={entries['entry_at_close_n']} "
        f"open_at_first_reading_n={entries['open_at_first_reading_n']} "
        f"at_or_before_close_n={entries['at_or_before_close_n']}",
        "",
        "== DELTAS",
        f"  abs_delta_gt_1={deltas['abs_delta_gt_1']} abs_delta_le_1={deltas['abs_delta_le_1']} "
        f"reading={deltas['supported_reading']}",
        "  " + " ".join(f"{name}={count}" for name, count in deltas["delta_histogram"].items()),
        "",
        "== SETTLEMENT SOURCE",
        f"  observed_at={source['observed_at']} observation_source={source['observation_source']} "
        f"sha256={source['sha256']}",
        f"  boundary_date={source['boundary_date']} "
        f"days_before_boundary={source['days_before_boundary']} "
        f"days_on_or_after_boundary={source['days_on_or_after_boundary']}",
        "  " + " ".join(f"{name}={count}" for name, count in source["roots_per_source"].items()),
        "",
        "== DISCOVERY",
        *_format_readout(payload["discovery"]),
        "",
        "== HOLDOUT",
        *_format_readout(payload["holdout"]),
        "",
        "== HOLDOUT BEFORE THE BOUNDARY (reported only, gates nothing)",
        f"  {pre_boundary['reported_only']}",
        *_format_readout(pre_boundary),
        "",
        "== GATE",
        *_format_gate(payload["gate"]),
        "",
        "== REPLICATION",
        *_format_replication(payload["replication"], payload["replication_skipped"]),
        "",
        "== SCREEN",
        f"  candidates={screen['candidates']} excluded={screen['excluded']} "
        f"dropped={screen['dropped']} city_event_days_kept={screen['city_event_days_kept']} "
        f"city_event_days_lost={screen['city_event_days_lost']}",
        "",
        "== COVERAGE",
        f"  cities={','.join(payload['cities'])}  "
        f"city_event_day_min_discovery={payload['city_event_day_min_discovery']}",
    ]
    lines.extend(
        f"  {name} ladder_rows={rows}" for name, rows in payload["ladder_rows_per_city_day"].items()
    )
    return "\n".join(lines)


def _fixed(value: float | None, places: int) -> str:
    return "None" if value is None else f"{value:.{places}f}"


def _format_readout(item: dict) -> list[str]:
    return [
        f"  net_profit_cents_per_contract={item['net_profit_cents_per_contract']}  "
        f"ci{item['ci_level']}=[{_fixed(item['ci_low'], 4)}, {_fixed(item['ci_high'], 4)}]  "
        f"p_value={_fixed(item['p_value'], 5)}  city_event_days={item['city_event_days']}",
        f"  degenerate={item['degenerate']}  "
        f"replicate_spread={_fixed(item['replicate_spread'], 6)}  "
        f"straddles={item['straddles']}  priced={item['priced']}  "
        f"one_sided={item['one_sided']}  no_row={item['no_row']}  censored={item['censored']}",
    ]


def _format_gate(gate: dict | None) -> list[str]:
    if gate is None:
        return [f"  not evaluated: {NO_GATE_ESTIMATE}"]
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
