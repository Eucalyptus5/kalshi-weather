from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.replay.artifacts import SCALARS_SCHEMA  # noqa: E402
from bot.replay.blind_windows import (  # noqa: E402
    FROZEN_END,
    FROZEN_START,
    NEAREST_RANK,
    _percentile,
)


CORRECTION_TABLES = ("blind_windows", "mid_life_clears")
DEFECT = (
    "the shipped windows table models ws_gaps rows alone, so it holds only the subscription "
    "boundaries that persisted one and misses the rest"
)

_MICROSECOND = timedelta(microseconds=1)
_FRACTION = Decimal("0.000001")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="state the coverage the shipped forward-pass inventory understates"
    )
    parser.add_argument(
        "--blind-windows", type=Path, required=True, help="the blind window parquet"
    )
    parser.add_argument(
        "--blind-summary", type=Path, required=True, help="the blind window summary json"
    )
    parser.add_argument("--clears", type=Path, required=True, help="the mid-life clear parquet")
    parser.add_argument(
        "--clears-summary", type=Path, required=True, help="the mid-life clear summary json"
    )
    parser.add_argument(
        "--shipped-windows", type=Path, required=True, help="the shipped ws_gaps windows parquet"
    )
    parser.add_argument("--out", type=Path, required=True, help="corrected scalars parquet")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    scalars = summarise(
        args.blind_windows,
        args.blind_summary,
        args.clears,
        args.clears_summary,
        args.shipped_windows,
    )
    write_scalars(args.out, scalars)
    print(format_report(args, scalars))
    return 0


def summarise(
    blind_path: Path,
    blind_summary_path: Path,
    clears_path: Path,
    clears_summary_path: Path,
    shipped_path: Path,
) -> dict[str, str]:
    blind = pq.read_table(blind_path)
    blind_summary = json.loads(blind_summary_path.read_text())
    clears = pq.read_table(clears_path)
    clears_summary = json.loads(clears_summary_path.read_text())
    shipped = pq.read_table(shipped_path)

    blind_us = blind.column("blind_us").to_pylist()
    has_gap_row = blind.column("has_gap_row").to_pylist()
    in_frozen = blind.column("in_frozen_window").to_pylist()
    frozen = [(us, gap) for us, gap, flag in zip(blind_us, has_gap_row, in_frozen) if flag]

    widths = sorted(blind_us)
    frozen_widths = sorted(us for us, _ in frozen)
    frozen_with_gap_row = sum(1 for _, gap in frozen if gap)
    frozen_blind_us = sum(frozen_widths)
    frozen_wall_us = (FROZEN_END - FROZEN_START) // _MICROSECOND
    with_gap_row = sum(1 for gap in has_gap_row if gap)

    shipped_starts = shipped.column("start").to_pylist()
    stale_rows = clears.column("stale_rows").to_pylist()
    ceiling = blind_summary["max_id"]

    return {
        "supersedes": str(shipped_path),
        "correction_tables": ",".join(CORRECTION_TABLES),
        "source_blind_windows": str(blind_path),
        "source_blind_summary": str(blind_summary_path),
        "source_clears": str(clears_path),
        "source_clears_summary": str(clears_summary_path),
        "id_ceiling": "none" if ceiling is None else str(ceiling),
        "defect": DEFECT,
        "rows_scanned": str(blind_summary["rows_scanned"]),
        "windows": str(len(widths)),
        "with_gap_row": str(with_gap_row),
        "without_gap_row": str(len(widths) - with_gap_row),
        "total_blind_us": str(sum(widths)),
        "percentiles": NEAREST_RANK,
        "p50_blind_us": str(_percentile(widths, 50)),
        "p90_blind_us": str(_percentile(widths, 90)),
        "max_blind_us": str(max(widths, default=0)),
        "unmatched_gap_rows": str(blind_summary["unmatched_gap_rows"]),
        "frozen_start": FROZEN_START.isoformat(),
        "frozen_end": FROZEN_END.isoformat(),
        "frozen_wall_us": str(frozen_wall_us),
        "frozen_windows": str(len(frozen_widths)),
        "frozen_with_gap_row": str(frozen_with_gap_row),
        "frozen_without_gap_row": str(len(frozen_widths) - frozen_with_gap_row),
        "frozen_total_blind_us": str(frozen_blind_us),
        "frozen_p50_blind_us": str(_percentile(frozen_widths, 50)),
        "frozen_p90_blind_us": str(_percentile(frozen_widths, 90)),
        "frozen_max_blind_us": str(max(frozen_widths, default=0)),
        "frozen_blind_fraction": str(
            (Decimal(frozen_blind_us) / Decimal(frozen_wall_us)).quantize(_FRACTION)
        ),
        "shipped_windows": str(len(shipped_starts)),
        "corrected_windows": str(len(widths)),
        "shipped_frozen_windows": str(
            sum(1 for start in shipped_starts if FROZEN_START <= start < FROZEN_END)
        ),
        "corrected_frozen_windows": str(len(frozen_widths)),
        "clear_days": str(clears_summary["days"]),
        "clears": str(clears_summary["clears"]),
        "clears_terminal": str(clears_summary["terminal"]),
        "clears_no_anchor": str(clears_summary["no_anchor"]),
        "clears_empty_book": str(clears_summary["empty_book"]),
        "clears_mid_life": str(len(stale_rows)),
        "clears_stale_rows": str(sum(stale_rows)),
        "clears_stale_us": str(sum(clears.column("stale_us").to_pylist())),
        "clears_unresolved": str(
            sum(1 for flag in clears.column("unresolved").to_pylist() if flag)
        ),
        "clears_with_stale_rows": str(sum(1 for rows in stale_rows if rows > 0)),
    }


def write_scalars(path: Path, scalars: Mapping[str, str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    rows = [{"name": name, "value": value} for name, value in scalars.items()]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCALARS_SCHEMA), path)


def format_report(args: argparse.Namespace, scalars: Mapping[str, str]) -> str:
    reported = (
        "id_ceiling",
        "rows_scanned",
        "windows",
        "with_gap_row",
        "without_gap_row",
        "total_blind_us",
        "p50_blind_us",
        "p90_blind_us",
        "max_blind_us",
        "unmatched_gap_rows",
        "frozen_windows",
        "frozen_total_blind_us",
        "frozen_max_blind_us",
        "frozen_blind_fraction",
        "shipped_windows",
        "corrected_windows",
        "shipped_frozen_windows",
        "corrected_frozen_windows",
        "clears_mid_life",
        "clears_stale_rows",
        "clears_stale_us",
        "clears_unresolved",
    )
    return "\n".join(
        ["== INVENTORY CORRECTED", f"supersedes={args.shipped_windows}"]
        + [f"{name}={scalars[name]}" for name in reported]
        + [f"out={args.out}"]
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
