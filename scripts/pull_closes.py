from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.placement_grid import pull_closes, read_sidecar  # noqa: E402
from bot.lag.tape_studies import load_run_scope  # noqa: E402
from bot.replay.analysis_stations import HIGH, LOW, in_cohort  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="freeze the close of every settled market the frozen run scope reaches"
    )
    parser.add_argument(
        "--run-scope", type=Path, required=True, help="the frozen run-scope directory"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="the directory the sidecars are frozen into"
    )
    parser.add_argument(
        "--cohort",
        choices=(HIGH, LOW),
        default=None,
        help="which ladder of the frozen scope the pull sweeps",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    scope = load_run_scope(args.run_scope)
    roots = in_cohort({series for series, _ in scope.event_days}, args.cohort)
    min_ts = int(scope.scope_start.timestamp())
    max_ts = int(scope.scope_end.timestamp())

    args.out.mkdir(parents=True, exist_ok=True)
    digests = pull_closes(roots, min_ts, max_ts, args.out)

    frozen = {}
    for root, digest in sorted(digests.items()):
        sidecar = read_sidecar(args.out / f"{root}.json")
        closes = sorted(market.close_time for market in sidecar.markets.values())
        frozen[root] = {
            "sha256": digest,
            "settled_markets": len(sidecar.markets),
            "voided": len(sidecar.voided),
            "first_close": closes[0].isoformat() if closes else None,
            "last_close": closes[-1].isoformat() if closes else None,
        }

    print(
        json.dumps(
            {
                "run_scope": str(args.run_scope),
                "out": str(args.out),
                "cohort": args.cohort,
                "min_ts": min_ts,
                "max_ts": max_ts,
                "min_ts_iso": datetime.fromtimestamp(min_ts, tz=timezone.utc).isoformat(),
                "max_ts_iso": datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat(),
                "settled_markets": sum(item["settled_markets"] for item in frozen.values()),
                "voided": sum(item["voided"] for item in frozen.values()),
                "roots": frozen,
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
