from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.depth_continuity import RESULTS_NAME, execute, result_payload  # noqa: E402
from bot.lag.depth_map_run import SIDES  # noqa: E402
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="check the ws-derived depths against the prod-era REST snapshot series"
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
        "--db", type=Path, required=True, help="the recorder state db, read for snapshots only"
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
    rows = payload["rows"]
    disagreements = payload["disagreements"]
    lines = [
        f"== D3 REST CONTINUITY CHECK  run_id={payload['run_id']}  reported only, no gate",
        "  this reading carries no verdict and evaluates no threshold",
        "",
        "== ROWS",
        f"  returned={rows['returned']} era_dropped={rows['era_dropped']} "
        f"out_of_window={rows['out_of_window']} excluded={rows['excluded']} "
        f"no_coverage={rows['no_coverage']} null_depth={rows['null_depth']} "
        f"compared={rows['compared']}",
        "  " + " ".join(f"{name}={count}" for name, count in rows["by_class"].items()),
        "",
        "== RUN",
        f"  manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"  latency_floor_source={payload['latency_floor_source']}",
        f"  scope={payload['scope_start']} .. {payload['scope_end']}  "
        f"event_days={len(payload['event_days'])}  cities={len(payload['cities'])}",
        f"  era_start={payload['era_start']}  sample_max={payload['sample_max']}",
        "",
        "== POOLED",
        *_format_pooled(payload["pooled"]),
        "",
        "== BY CITY",
        *(f"  {series} {_format_city(block)}" for series, block in payload["by_city"].items()),
        "",
        f"== DISAGREEMENTS  {len(disagreements)} of at most {payload['sample_max']}",
        *(f"  {_format_disagreement(item)}" for item in disagreements),
    ]
    return "\n".join(lines)


def _format_pooled(block: dict) -> list[str]:
    difference = block["depth_difference_contracts"]
    return [
        f"  compared={block['compared']}  depth_compared={block['depth_compared']}",
        f"  price_agree={block['price_agree']} fraction={block['price_agree_fraction']}",
        f"  depth_agree_exact={block['depth_agree']} fraction={block['depth_agree_fraction']}",
        f"  depth_agree_truncated={block['depth_agree_truncated']} "
        f"fraction={block['depth_agree_truncated_fraction']}",
        f"  fractional_ws_depth={block['fractional_ws_depth']} "
        f"fraction={block['fractional_ws_depth_fraction']}",
        *(f"  rest-ws {side} contracts {_format_side(difference[side])}" for side in SIDES),
    ]


def _format_side(side: dict) -> str:
    return (
        f"p25={side['p25']} p50={side['p50']} p75={side['p75']} "
        f"within_one={side['within_one_contract']} "
        f"fraction={side['within_one_contract_fraction']}"
    )


def _format_city(block: dict) -> str:
    difference = block["depth_difference_contracts"]
    return (
        f"compared={block['compared']} price={block['price_agree_fraction']} "
        f"depth={block['depth_agree_fraction']} "
        f"truncated={block['depth_agree_truncated_fraction']} "
        f"fractional={block['fractional_ws_depth_fraction']} "
        + " ".join(f"{side}_p50={difference[side]['p50']}" for side in SIDES)
    )


def _format_disagreement(item: dict) -> str:
    rest = item["rest"]
    ws = item["ws"]
    return (
        f"{item['ticker']} {item['snapshot_at']} "
        f"rest yes={rest['yes_bid']}/{rest['yes_bid_depth']} "
        f"no={rest['no_bid']}/{rest['no_bid_depth']}  "
        f"ws yes={ws['yes_bid']}/{ws['yes_bid_depth']} no={ws['no_bid']}/{ws['no_bid_depth']}"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
