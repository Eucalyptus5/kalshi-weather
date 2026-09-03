from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.fee_floor import MAKER_RATE_SOURCE, PUBLISHED_MAKER_RATE  # noqa: E402
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import (  # noqa: E402
    MANIFEST_NAME,
    ManifestIncomplete,
    write_manifest,
)
from bot.lag.tape_studies import (  # noqa: E402
    KIND_SCHEMAS,
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    NoRowsConsumed,
    assemble_run_inputs,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)
KINDS = tuple(KIND_SCHEMAS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="write the manifest a tape study run is read under"
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
        "--kind",
        dest="kinds",
        action="append",
        required=True,
        choices=KINDS,
        help="an artifact kind the run reads, repeated once per kind",
    )
    parser.add_argument(
        "--floor-source",
        required=True,
        choices=FLOOR_SOURCES,
        help="which round trip supplied the latency floor",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="the bootstrap seed the manifest records"
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        inputs = assemble_run_inputs(
            run_id=args.run_id,
            preregistration=args.preregistration,
            repo=args.repo,
            run_scope=args.run_scope,
            artifacts=args.artifacts,
            kinds=tuple(args.kinds),
            rtt_samples=args.rtt_samples,
            floor_source=FloorSource(args.floor_source),
            maker_rate=PUBLISHED_MAKER_RATE,
            maker_rate_source=MAKER_RATE_SOURCE,
            economic_bar_size=SELF_CHARGED_BAR,
            economic_bar_price=SELF_CHARGED_BAR,
            economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
            bootstrap_seed=args.seed,
        )
        write_manifest(args.run_root, inputs)
    except (ManifestIncomplete, NoRowsConsumed) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(format_report(json.loads((args.run_root / args.run_id / MANIFEST_NAME).read_text())))
    return 0


def format_report(payload: Mapping[str, object]) -> str:
    return "\n".join(["== TAPE STUDY RUN"] + [f"{name}={value}" for name, value in payload.items()])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
