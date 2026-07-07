from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.replay.artifacts import directory_roots, manifest_roots  # noqa: E402


DEFAULT_KIND = "ladder"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="rebuild the root list of a forward pass from its drain manifest"
    )
    parser.add_argument("manifest", type=Path, help="append-only jsonl the drain writes")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="the pass output subdirectory for this kind; names the files the drain still owes",
    )
    parser.add_argument("--kind", default=DEFAULT_KIND)
    return parser


def run(args: argparse.Namespace) -> int:
    shipped = manifest_roots(args.manifest, args.kind)
    backlog = set() if args.artifact_dir is None else directory_roots(args.artifact_dir)
    print(format_report(args, shipped, backlog))
    return 0


def format_report(args: argparse.Namespace, shipped: set[str], backlog: set[str]) -> str:
    roots = shipped | backlog
    return "\n".join(
        [
            "== MANIFEST ROOTS",
            f"manifest={args.manifest}",
            f"kind={args.kind}",
            f"artifact_dir={args.artifact_dir}",
            f"shipped={len(shipped)}",
            f"backlog_only={len(backlog - shipped)}",
            f"roots={','.join(sorted(roots))}",
            f"count={len(roots)}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
