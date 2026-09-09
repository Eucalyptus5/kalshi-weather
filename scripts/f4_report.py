from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.backtest.historical_open_meteo import CALIBRATION_PATH  # noqa: E402
from bot.lag.forecast_class_run import (  # noqa: E402
    BLEND_LABEL,
    BOOTSTRAP_SEED,
    COHORT,
    NO_GATE_ESTIMATE,
    RESULTS_NAME,
    execute,
    result_payload,
)
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402
from bot.lag.tape_studies import SELF_CHARGED_BAR, SELF_CHARGED_BAR_SOURCE  # noqa: E402
from bot.replay.analysis_stations import HIGH, LOW  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
DEFAULT_INPUTS = DEFAULT_RUN_ROOT / "f4_inputs"
DEFAULT_SAMPLE = DEFAULT_INPUTS / "sample.jsonl"
DEFAULT_MARKETS = REPO_ROOT / "data" / "backtest" / "weather_markets.parquet"
# Relative on purpose: the manifest hashes the file but records the path it was given, and an
# absolute default would make every digest a function of where the tree is checked out.
DEFAULT_PREREGISTRATION = Path("improvements/active/f4_preregistration.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read the forecast class gate at 24h and the 36h lead beside it"
    )
    parser.add_argument("--run-id", required=True, help="names the run directory under --run-root")
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
        help="the file the manifest hashes",
    )
    parser.add_argument(
        "--sample", type=Path, default=DEFAULT_SAMPLE, help="the frozen market-leg sample"
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_INPUTS,
        help="the directory holding the three frozen forecast classes",
    )
    parser.add_argument(
        "--markets",
        type=Path,
        default=DEFAULT_MARKETS,
        help="the market table the full event ladders are read from",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=CALIBRATION_PATH,
        help="the spread calibration the external sigmas are read from",
    )
    parser.add_argument(
        "--cohort",
        choices=(HIGH, LOW),
        default=COHORT,
        help="which ladder of the frozen sample the run reads",
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
            sample=args.sample,
            classes=args.classes,
            markets=args.markets,
            calibration=args.calibration,
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
    blend = payload["blend"]
    screen = payload["screen"]
    entries = payload["entries"]
    depth = payload["depth"]
    walked = payload["walked_tick"]
    band = payload["sigma_band"]
    reported = payload["lead_36h"]
    ladders = payload["ladder_sums"]
    lines = [
        f"== F4 FORECAST CLASS  run_id={payload['run_id']}  verdict={payload['verdict']}",
        f"manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"seed={payload['bootstrap_seed']}  resamples={payload['bootstrap_resamples']}  "
        f"alpha={payload['alpha']:.6f}  cohort={payload['cohort']}",
        f"bar={payload['bar']} strict={payload['bar_is_strict']} source={payload['bar_source']}  "
        f"size={payload['size']}",
        f"gating_lead_hours={payload['gating_lead_hours']}  "
        f"screen_rule={payload['screen_rule']}  tick_rule={payload['tick_rule']}",
        "",
        "== BLEND",
        *_format_blend(blend, payload["briers"][BLEND_LABEL]),
        "",
        "== DISCOVERY",
        *_format_readout(payload["discovery"]),
        "",
        "== HOLDOUT",
        *_format_readout(payload["holdout"]),
        "",
        "== GATE",
        *_format_gate(payload["gate"]),
        "",
        "== REPLICATION",
        *_format_replication(payload["replication"], payload["replication_skipped"]),
        "",
        "== CLASSES (reported only, gates nothing)",
        *_format_classes(payload["classes"], payload["briers"]),
        "",
        "== LEAD 36H (reported only, gates nothing)",
        f"  {reported['reported_only']}",
        *_format_blend(reported["blend"], reported["briers"][BLEND_LABEL]),
        *_format_readout(reported["discovery"]),
        *_format_readout(reported["holdout"]),
        *_format_classes(reported["classes"], reported["briers"]),
        "",
        "== WALKED TICK (reported only, gates nothing)",
        f"  {walked['reported_only']}",
        f"  tick_rule={walked['tick_rule']}",
        *_format_readout(walked["discovery"]),
        *_format_readout(walked["holdout"]),
        "",
        "== SIGMA BAND (reported only, gates nothing)",
        f"  {band['reported_only']}",
        *(_format_band(item) for item in band["bands"]),
        "",
        "== DEPTH",
        f"  candidates {_format_depth(depth['candidates'])}",
        f"  kept {_format_depth(depth['kept'])}",
        "",
        "== SCREEN",
        f"  rule={screen['rule']} candidates={screen['candidates']} kept={screen['kept']} "
        f"dropped={screen['dropped']} event_days_kept={screen['event_days_kept']} "
        f"event_days_lost={screen['event_days_lost']}",
        "",
        "== ENTRIES",
        f"  n={entries['n']} traded={entries['traded']} untraded={entries['untraded']} "
        f"not_blendable_legs={entries['not_blendable_legs']} "
        f"not_blendable_city_days={entries['not_blendable_city_days']} "
        f"not_blendable_event_days_lost={entries['not_blendable_event_days_lost']}",
        "  "
        + " ".join(
            f"{item['forecast_class']}:native={item['native_xnd']},"
            f"external={item['external_calibration']}"
            for item in payload["sigma_source_tally"]
        ),
        "",
        "== LADDER SUMS",
        f"  checked={ladders['checked']} failed={ladders['failed']} skipped={ladders['skipped']}",
        *(f"  {failure}" for failure in ladders["failures"]),
        "",
        "== COVERAGE",
        f"  event_day_min_discovery={payload['event_day_min_discovery']}",
        "  " + " ".join(f"{name}={count}" for name, count in payload["row_counts"].items()),
    ]
    return "\n".join(lines)


def _fixed(value: float | None, places: int) -> str:
    return "None" if value is None else f"{value:.{places}f}"


def _format_blend(blend: dict, brier: dict) -> list[str]:
    return [
        f"  members={','.join(blend['members'])}",
        "  " + " ".join(f"{member}={weight}" for member, weight in blend["weights"].items()),
        f"  blend_fitted_on_event_days={blend['fitted_on_event_days']}  "
        f"fitted_on_split={blend['fitted_on_split']}",
        f"  blend_weights_sha256={blend['sha256']}",
        f"  brier={brier['brier']}  baseline_brier={brier['baseline_brier']}  "
        f"skill={brier['skill']}  n={brier['n']}",
    ]


def _format_readout(item: dict) -> list[str]:
    return [
        f"  {item['split']} net_profit_cents_per_contract="
        f"{item['net_profit_cents_per_contract']}  "
        f"ci{item['ci_level']}=[{_fixed(item['ci_low'], 4)}, {_fixed(item['ci_high'], 4)}]  "
        f"p_value={_fixed(item['p_value'], 5)}  event_days={item['event_days']}",
        f"  {item['split']} degenerate={item['degenerate']}  "
        f"replicate_spread={_fixed(item['replicate_spread'], 6)}  legs={item['legs']}  "
        f"traded={item['traded']}  untraded={item['untraded']}",
    ]


def _format_classes(classes: list[dict], briers: dict) -> list[str]:
    lines = []
    for item in classes:
        brier = briers[item["member"]]
        lines.append(
            f"  {item['member']} class={item['forecast_class']} gating={item['gating']}: "
            f"{item['reported_only']}"
        )
        lines.append(
            f"  {item['member']} discovery="
            f"{item['discovery']['net_profit_cents_per_contract']}  "
            f"holdout={item['holdout']['net_profit_cents_per_contract']}  "
            f"brier={brier['brier']}  baseline_brier={brier['baseline_brier']}  "
            f"skill={brier['skill']}"
        )
    return lines


def _format_band(band: dict) -> str:
    points = " ".join(
        f"{point['multiplier']}x={point['net_profit_cents_per_contract']}/traded={point['traded']}"
        for point in band["points"]
    )
    return f"  {band['label']} lead={band['lead_hours']}h {band['split']} {points}"


def _format_depth(item: dict) -> str:
    return (
        f"prints_p10={item['prints_p10']} prints_p50={item['prints_p50']} "
        f"prints_p90={item['prints_p90']} prints_max={item['prints_max']} "
        f"contracts_p10={item['contracts_p10']} contracts_p50={item['contracts_p50']} "
        f"contracts_p90={item['contracts_p90']} contracts_max={item['contracts_max']} "
        f"contracts_below_size={item['contracts_below_size']}"
    )


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
