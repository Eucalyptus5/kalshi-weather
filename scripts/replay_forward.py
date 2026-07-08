from __future__ import annotations

import argparse
import json
import logging
import resource
import sqlite3
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.replay.artifacts import (  # noqa: E402
    MAX_OPEN_WRITERS,
    BudgetGuard,
    ByteBudget,
    LadderEmitter,
    TouchEmitter,
    bytes_written,
    measure_byte_budget,
    project_artifact_bytes,
    write_inventory,
    write_trades,
)
from bot.replay.forward_pass import (  # noqa: E402
    BARRIER_ROWS,
    READ_BATCH_ROWS,
    ROW_GROUP_ROWS,
    JsonState,
    PassResult,
    SourceRow,
    run_forward_pass,
)
from bot.replay.inventory import (  # noqa: E402
    ExclusionInventory,
    SeqBoundaryDetector,
    TickerInventory,
    build_inventory,
    read_gap_rows,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "state.db"
DEFAULT_OUT = REPO_ROOT / "data" / "replay"
DEFAULT_RAW = REPO_ROOT / "data" / "ws_raw"
DEFAULT_PASS_HOURS = Decimal("24")
OUTLIER_SAMPLES = 20
STATS_NAME = "run_stats.json"

_PRICE_SCALE = -4
_SIZE_SCALE = -2
# received_at is stored as a fixed 26-character timestamp; the read-rate baseline this run is
# compared against summed len(str(received_at)) per row and the two have to match.
_TIMESTAMP_CHARS = 26


class ScaleCensus:
    name = "census"

    def __init__(self) -> None:
        self.rows = 0
        self.snapshot_rows = 0
        self.logical_bytes = 0
        self.price_exponents: Counter[int] = Counter()
        self.size_exponents: Counter[int] = Counter()
        self.outliers: list[dict[str, object]] = []

    def observe(self, row: SourceRow) -> None:
        self.rows += 1
        self.snapshot_rows += int(row.is_snapshot)
        self.logical_bytes += (
            8
            + len(row.ticker)
            + _TIMESTAMP_CHARS
            + 8
            + len(row.side)
            + len(row.price)
            + len(row.size)
            + 1
        )
        price = Decimal(row.price).as_tuple().exponent
        size = Decimal(row.size).as_tuple().exponent
        self.price_exponents[price] += 1
        self.size_exponents[size] += 1
        if (price < _PRICE_SCALE or size < _SIZE_SCALE) and len(self.outliers) < OUTLIER_SAMPLES:
            self.outliers.append(
                {
                    "id": row.id,
                    "ticker": row.ticker,
                    "price": row.price,
                    "size": row.size,
                    "price_exponent": price,
                    "size_exponent": size,
                }
            )

    def state(self) -> JsonState:
        return {
            "rows": self.rows,
            "snapshot_rows": self.snapshot_rows,
            "logical_bytes": self.logical_bytes,
            "price_exponents": {str(k): v for k, v in self.price_exponents.items()},
            "size_exponents": {str(k): v for k, v in self.size_exponents.items()},
            "outliers": self.outliers,
        }

    def restore(self, state: JsonState) -> None:
        self.rows = state["rows"]
        self.snapshot_rows = state["snapshot_rows"]
        self.logical_bytes = state["logical_bytes"]
        self.price_exponents = Counter({int(k): v for k, v in state["price_exponents"].items()})
        self.size_exponents = Counter({int(k): v for k, v in state["size_exponents"].items()})
        self.outliers = list(state["outliers"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="forward-pass replay over ws_book_events")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        help="bound the last id consumed; a pass still starts at id 1 or a checkpoint",
    )
    parser.add_argument(
        "--trades-max-id",
        type=int,
        default=None,
        help="bound the last ws_trades id consumed; a rowid space of its own, not --max-id's",
    )
    parser.add_argument(
        "--pass-hours",
        type=Decimal,
        default=DEFAULT_PASS_HOURS,
        help="expected run length; sizes the WAL the recorder cannot checkpoint meanwhile",
    )
    parser.add_argument("--read-batch", type=int, default=READ_BATCH_ROWS)
    parser.add_argument("--row-group", type=int, default=ROW_GROUP_ROWS)
    parser.add_argument("--barrier-rows", type=int, default=BARRIER_ROWS)
    parser.add_argument(
        "--max-open-writers",
        type=int,
        default=MAX_OPEN_WRITERS,
        help="trade partitions written per scan; the tape is rescanned once per chunk",
    )
    parser.add_argument("--trades", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    budget = measure_byte_budget(args.out, args.raw_dir, args.pass_hours)
    census = ScaleCensus()
    exclusions = ExclusionInventory(read_gap_rows(args.db))
    seq = SeqBoundaryDetector()
    tickers = TickerInventory()
    guard = BudgetGuard(args.out, budget)
    ladder = LadderEmitter()

    cpu_before = _cpu_s()
    read_before = _block_read_bytes()
    result = run_forward_pass(
        args.db,
        args.out,
        [TouchEmitter(), ladder],
        accumulators=[exclusions, seq, tickers, census, guard],
        read_batch_rows=args.read_batch,
        row_group_rows=args.row_group,
        barrier_rows=args.barrier_rows,
        max_id=args.max_id,
    )
    cpu_s = _cpu_s() - cpu_before
    block_bytes = _block_read_bytes() - read_before

    trades_rows = 0
    if args.trades:
        # ws_trades and ws_book_events carry independent rowids, so --max-id cannot bound this.
        trades_rows = write_trades(
            args.db,
            args.out,
            budget,
            row_group_rows=args.row_group,
            max_id=args.trades_max_id,
            max_open_writers=args.max_open_writers,
        )

    stats = _stats(args, result, census, budget, cpu_s, block_bytes, trades_rows)
    write_inventory(
        args.out,
        build_inventory(exclusions, seq, tickers),
        budget,
        ladder=ladder,
        extra={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in stats.items()
        },
    )
    readable, unreadable = _verify(args.out)
    stats["files_readable"] = readable
    stats["files_unreadable"] = unreadable
    stats["artifact_bytes"] = bytes_written(args.out)
    (args.out / STATS_NAME).write_text(json.dumps(stats, indent=1, default=str))
    print(format_report(stats))
    return 1 if unreadable else 0


def format_report(stats: dict[str, object]) -> str:
    lines = ["== FORWARD PASS"]
    for name, value in stats.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{name}={json.dumps(value, default=str)}")
        else:
            lines.append(f"{name}={value}")
    return "\n".join(lines)


def _stats(
    args: argparse.Namespace,
    result: PassResult,
    census: ScaleCensus,
    budget: ByteBudget,
    cpu_s: float,
    block_bytes: int,
    trades_rows: int,
) -> dict[str, object]:
    total_rows, total_bytes = _table_totals(args.db)
    artifact_bytes = bytes_written(args.out)
    levels = sum(len(side) for ladder in result.ladders.values() for side in ladder.levels.values())
    # /proc/self/io is Linux-only; off it the table's own bytes per row is the honest stand-in.
    source_bytes = block_bytes or total_bytes * result.rows // total_rows
    projection = project_artifact_bytes(
        measured_bytes=artifact_bytes,
        measured_rows=result.rows,
        measured_source_bytes=source_bytes,
        total_rows=total_rows,
        total_bytes=total_bytes,
        budget=budget,
    )
    return {
        "db": str(args.db),
        "out": str(args.out),
        "max_id": args.max_id,
        "trades_max_id": args.trades_max_id,
        "read_batch": args.read_batch,
        "row_group": args.row_group,
        "barrier_rows": args.barrier_rows,
        "max_open_writers": args.max_open_writers,
        "rows": result.rows,
        "last_id": result.last_id,
        "barriers": result.barriers,
        "snapshot_rows": census.snapshot_rows,
        "trades_rows": trades_rows,
        "wall_s": round(result.elapsed_s, 3),
        "rows_per_s": round(result.rows / result.elapsed_s, 1),
        "logical_bytes": census.logical_bytes,
        "logical_mb_per_s": round(census.logical_bytes / result.elapsed_s / 1e6, 3),
        "block_read_bytes": block_bytes,
        "block_read_mb_per_s": round(block_bytes / result.elapsed_s / 1e6, 3),
        "process_cpu_s": round(cpu_s, 3),
        "cpu_over_wall": round(cpu_s / result.elapsed_s, 4),
        "peak_rss_bytes": _peak_rss_bytes(),
        "resident_ladders": len(result.ladders),
        "resident_levels": levels,
        "poisoned_tickers": len(result.poisoned_rows),
        "poisoned_rows": sum(result.poisoned_rows.values()),
        "artifact_bytes": artifact_bytes,
        "artifact_bytes_per_row": str(projection.bytes_per_row),
        "table_max_id": total_rows,
        "table_bytes": total_bytes,
        "projected_bytes_by_rows": projection.by_rows,
        "projected_bytes_by_bytes": projection.by_bytes,
        "projected_bytes": projection.projected_bytes,
        "projection_basis": projection.basis,
        "projection_fits": projection.fits,
        "budget_bytes": budget.budget_bytes,
        "projected_wall_s": round(total_rows / (result.rows / result.elapsed_s), 1),
        "price_exponents": {str(k): v for k, v in sorted(census.price_exponents.items())},
        "size_exponents": {str(k): v for k, v in sorted(census.size_exponents.items())},
        "scale_outliers": census.outliers,
    }


def _table_totals(db_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        # _persist_ws_events only ever appends and nothing deletes, so the rowid high-water
        # mark is the row count and costs no scan.
        max_id = conn.execute("SELECT MAX(id) FROM ws_book_events").fetchone()[0]
    finally:
        conn.close()
    wal = db_path.with_name(f"{db_path.name}-wal")
    return max_id, db_path.stat().st_size + (wal.stat().st_size if wal.exists() else 0)


def _verify(out_dir: Path) -> tuple[int, list[str]]:
    readable = 0
    unreadable = []
    for path in sorted(out_dir.rglob("*.parquet")):
        try:
            pq.read_table(path)
        except (pa.ArrowInvalid, OSError) as exc:
            unreadable.append(f"{path}: {exc}")
        else:
            readable += 1
    return readable, unreadable


def _block_read_bytes() -> int:
    path = Path("/proc/self/io")
    if not path.exists():
        return 0
    fields = dict(line.split(": ") for line in path.read_text().splitlines())
    return int(fields["read_bytes"])


def _cpu_s() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on darwin and kilobytes everywhere else.
    return peak if sys.platform == "darwin" else peak * 1024


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()
    rc = run(build_parser().parse_args(argv))
    logging.getLogger(__name__).info(
        "replay_forward rc=%d elapsed_s=%.1f", rc, time.monotonic() - started
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
