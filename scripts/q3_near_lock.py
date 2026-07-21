from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.near_lock import execute, result_payload  # noqa: E402
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402
from bot.lag.taker_flow_run import RESULTS_NAME  # noqa: E402


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)
ARRIVALS_QUERY = (
    "SELECT station, source, obs_time, tmpf, received_at FROM ws_obs_arrivals "
    "ORDER BY station, obs_time"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read the taker flow inside the lock window off the frozen run scope"
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
        "--state-db", type=Path, required=True, help="the recorder state db, opened read-only"
    )
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="the arrivals export the locks are detected from, written if it is not there yet",
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


# The recorder writes this database while the export reads it, so the read is one indexed pass and
# nothing else: no pragma that touches the WAL, no cursor held open past the fetch.
def export_observations(state_db: Path, path: Path) -> int:
    conn = sqlite3.connect(f"file:{state_db.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rows = conn.execute(ARRIVALS_QUERY).fetchall()
    finally:
        conn.close()
    path.write_text(
        "".join(
            json.dumps(
                {
                    "station": station,
                    "source": source,
                    "obs_time": _utc(obs_time),
                    "tmpf": tmpf,
                    "received_at": _utc(received_at),
                }
            )
            + "\n"
            for station, source, obs_time, tmpf, received_at in rows
        )
    )
    return len(rows)


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    if not args.observations.exists():
        logger.info(
            "near_lock export rows=%d path=%s",
            export_observations(args.state_db, args.observations),
            args.observations,
        )
    try:
        result = execute(
            run_id=args.run_id,
            preregistration=args.preregistration,
            repo=args.repo,
            run_scope=args.run_scope,
            artifacts=args.artifacts,
            observations=args.observations,
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
    locks = payload["locks"]
    prints = payload["prints"]
    exclusions = payload["exclusions"]
    drops = payload["kernel_drops"]
    lines = [
        f"== Q3 NEAR-LOCK STRATUM  run_id={payload['run_id']}  "
        f"stratum_status={payload['stratum_status']}",
        f"{payload['reported_only']}: nothing here decides Q3",
        f"manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
        f"seed={payload['bootstrap_seed']}  resamples={payload['bootstrap_resamples']}  "
        f"primary_horizon_s={payload['primary_horizon_s']}  "
        f"lock_half_width_s={payload['lock_half_width_s']}  n_min={payload['stratum_n_min']}",
        "",
        "== LOCKS",
        f"  markets={locks['markets']} locked={locks['locked']} "
        f"ambiguous={locks['ambiguous']} ambiguous_fraction={locks['ambiguous_fraction']} "
        f"no_lock={locks['no_lock']}",
        "",
        "== PRINTS",
        f"  in_window pooled={prints['in_window_pooled']} "
        f"discovery={prints['in_window_discovery']} holdout={prints['in_window_holdout']}",
        f"  screened pooled={prints['screened_pooled']} "
        f"discovery={prints['screened_discovery']} holdout={prints['screened_holdout']}",
        f"  empty_side={prints['empty_side']} duplicate_trade_id={prints['duplicate_trade_id']} "
        f"out_of_scope={prints['out_of_scope']} out_of_window={prints['out_of_window']}",
        f"  outside_lock_window={prints['outside_lock_window']} "
        f"on_a_market_that_never_locked={prints['on_a_market_that_never_locked']}",
        "",
        "== DISCOVERY (reported, not gating)",
        *_format_reading(payload["discovery"]),
        "",
        "== HOLDOUT (reported, not gating)",
        *_format_reading(payload["holdout"]),
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
    lines.extend(
        f"  {name} tickers={count}" for name, count in payload["tickers_per_city_day"].items()
    )
    return "\n".join(lines)


def _utc(value: str) -> str:
    return datetime.fromisoformat(value + "+00:00").isoformat()


def _fixed(value: float | None, places: int) -> str:
    return "None" if value is None else f"{value:.{places}f}"


def _format_reading(item: dict) -> list[str]:
    return [
        f"  status={item['status']}  mean_net_cents={item['mean_net_cents']}  "
        f"ci{item['ci_level']}=[{_fixed(item['ci_low'], 4)}, {_fixed(item['ci_high'], 4)}]  "
        f"n_prints={item['n_prints']}  n_min={item['n_min']}  "
        f"contracts={item['contracts']}  clusters={item['clusters']}  "
        f"p_value={_fixed(item['p_value'], 5)}",
        f"  candidates={item['candidates']}  excluded={item['excluded']}  "
        f"excluded_fraction={item['excluded_fraction']}  out_of_window={item['out_of_window']}  "
        f"unresolved={item['unresolved']}  uncovered={item['uncovered']}  "
        f"one_sided={item['one_sided']}  host_clock={item['host_clock']}",
    ]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
