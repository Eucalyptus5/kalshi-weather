from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.fee_regime import pull_fee_regime, read_fee_regime  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="freeze the fee_type the series api carries for every root a run will sweep"
    )
    parser.add_argument(
        "--roots", nargs="+", required=True, help="the series roots the run reads the tape of"
    )
    parser.add_argument("--out", type=Path, required=True, help="where the sidecar is frozen")
    parser.add_argument(
        "--observed-at",
        type=datetime.fromisoformat,
        required=True,
        help="the instant the sweep is stamped with, ISO-8601",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    digest = pull_fee_regime(args.roots, args.observed_at, args.out)
    regime = read_fee_regime(args.out)
    print(
        json.dumps(
            {
                "path": str(args.out),
                "sha256": digest,
                "observed_at": regime.observed_at.isoformat(),
                "series": {
                    root: {"fee_type": item.fee_type, "fee_multiplier": item.fee_multiplier}
                    for root, item in sorted(regime.series.items())
                },
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
