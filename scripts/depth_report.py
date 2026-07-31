from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.depth_map_run import (  # noqa: E402
    ALL_LEGS,
    ATM,
    RESULTS_NAME,
    UNIVERSES,
    execute,
    result_payload,
)
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)
TITLES = {ALL_LEGS: "ALL LEGS", ATM: "AT THE MONEY"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="map touch depth, book-walk capacity and resilience off the frozen run scope"
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
        "--db", type=Path, required=True, help="the recorder state db, read for close times only"
    )
    parser.add_argument("--rtt-samples", type=Path, required=True, help="the read-RTT sample file")
    parser.add_argument(
        "--floor-source",
        required=True,
        choices=FLOOR_SOURCES,
        help="which round trip supplied the latency floor",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="the seed the manifest records; nothing resamples"
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
            db=args.db,
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
    bars = payload["bars"]
    headline = payload["headline"]
    depth = headline["median_touch_depth_contracts"]
    prints = headline["resilience"]
    rows = payload["rows"]
    lines = [
        f"== D2 LIQUIDITY AND DEPTH MAP  run_id={payload['run_id']}  reported only, no gate",
        "  this reading carries no gate, no alpha and no multiplicity count",
        "",
        "== VERDICT",
        f"  the small-capacity presumption is {payload['verdict'].upper()}",
        f"  {headline['universe']} leg, final {bars['final_hours']}h, pooled: "
        f"median touch depth yes={depth['yes']} no={depth['no']} contracts "
        f"against a bar of {bars['touch_depth_bar_contracts']}",
        f"  {headline['universe']} leg, final {bars['final_hours']}h: "
        f"events={prints['events']} replenished={prints['replenished']} "
        f"fraction={prints['replenished_fraction']} median_s={prints['median_replenish_s']}",
        "",
        "== RUN",
        f"  manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"  latency_floor_source={payload['latency_floor_source']}",
        f"  scope={payload['scope_start']} .. {payload['scope_end']}  "
        f"event_days={len(payload['event_days'])}  cities={len(payload['cities'])}",
        f"  slippage_bar_cents={bars['slippage_bar_cents']}  "
        f"cumulative_ticks={bars['cumulative_ticks']}  "
        f"replenish={bars['replenish_fraction']} within {bars['replenish_window_s']}s  "
        f"bucket_hours={bars['bucket_hours']}",
        f"  ladder_rows={rows['ladder']}  cells={rows['cells']}  "
        f"resilience_states={rows['resilience_states']}  trades={rows['trades']}",
        "",
    ]
    for name in UNIVERSES:
        lines.extend(_format_universe(name, payload[name]))
        lines.append("")
    lines.extend(_format_exclusions(payload))
    lines.append("")
    lines.extend(_format_coverage(payload))
    return "\n".join(lines)


def _format_universe(name: str, block: dict) -> list[str]:
    lines = [f"== {TITLES[name]}  cube cells={len(block['cube'])} in results.json"]
    for axis, title in (
        ("by_city", "CITY"),
        ("by_bucket", "HOURS TO CLOSE"),
        ("by_hour", "UTC HOUR"),
    ):
        lines.append(f"  -- BY {title}")
        lines.extend(f"    {key} {_format_cell(cell)}" for key, cell in block[axis].items())
    lines.append("  -- CENSORING BY HOURS TO CLOSE")
    lines.extend(
        f"    {key} censored_s={cell['censored_s']} p50={cell['censored_p50_contracts']}"
        for key, cell in block["by_bucket"].items()
    )
    lines.append("  -- RESILIENCE")
    lines.append(
        f"    prints={block['resilience']['prints']} "
        f"no_pre_state={block['resilience']['no_pre_state']} "
        f"no_match={block['resilience']['no_match']} "
        f"no_post_state={block['resilience']['no_post_state']} "
        f"past_close={block['resilience']['past_close']} "
        f"out_of_window={block['resilience']['out_of_window']} "
        f"excluded={block['resilience']['excluded']}"
    )
    lines.append(f"    pooled {_format_prints(block['resilience']['pooled'])}")
    lines.extend(
        f"    {key} {_format_prints(tally)}"
        for key, tally in block["resilience"]["by_city_bucket"].items()
    )
    return lines


def _format_cell(cell: dict) -> str:
    touch = cell["touch_contracts"]
    cum5 = cell["cum5_contracts"]
    capacity = cell["capacity_contracts"]
    return (
        f"kept_s={cell['kept_s']} "
        f"touch={touch['p25']}/{touch['p50']}/{touch['p75']} "
        f"cum5={cum5['p25']}/{cum5['p50']}/{cum5['p75']} "
        f"capacity={capacity['p25']}/{capacity['p50']}/{capacity['p75']} "
        f"spread_p50={cell['spread_p50_cents']} spread_min={cell['spread_min_cents']} "
        f"two_sided={cell['two_sided_fraction']}"
    )


def _format_prints(tally: dict) -> str:
    return (
        f"matched={tally['matched']} events={tally['events']} "
        f"replenished={tally['replenished']} fraction={tally['replenished_fraction']} "
        f"median_s={tally['median_replenish_s']}"
    )


def _format_exclusions(payload: dict) -> list[str]:
    block = payload["exclusions"]
    return [
        "== EXCLUSIONS",
        f"  candidates={block['candidates']} excluded={block['excluded']} "
        f"fraction={block['excluded_fraction']} out_of_window={block['out_of_window']} "
        f"past_close={block['past_close']}",
        f"  kept_span_s={block['kept_span_s']} nominal_span_s={block['nominal_span_s']} "
        f"retained={block['retained_fraction']}",
        "  " + " ".join(f"{name}={count}" for name, count in block["by_class"].items()),
    ]


def _format_coverage(payload: dict) -> list[str]:
    return [
        "== COVERAGE",
        f"  cities={','.join(payload['cities'])}",
        *(f"  {key} atm={ticker}" for key, ticker in payload["atm_ticker_per_city_day"].items()),
        *(f"  no atm leg {key}" for key in payload["no_atm_leg"]),
    ]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
