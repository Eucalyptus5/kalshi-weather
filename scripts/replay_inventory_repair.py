from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.replay.artifacts import SCALARS_SCHEMA, manifest_roots  # noqa: E402
from bot.replay.forward_pass import _decode_ts  # noqa: E402


UNCHANGED_TABLES = ("windows", "boundaries", "coverage")
_RECEIVED_AT = "SELECT received_at FROM ws_book_events WHERE id = ?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="rewrite a shipped forward-pass scalars table from the drain manifest"
    )
    parser.add_argument("--scalars", type=Path, required=True, help="the shipped scalars parquet")
    parser.add_argument("--manifest", type=Path, required=True, help="append-only drain jsonl")
    parser.add_argument("--db", type=Path, required=True, help="the recorder database, read only")
    parser.add_argument("--out", type=Path, required=True, help="corrected scalars parquet")
    return parser


def run(args: argparse.Namespace) -> int:
    corrected = repair(args.scalars, args.manifest, args.db)
    write_scalars(args.out, corrected)
    print(format_report(args, corrected))
    return 0


def repair(scalars_path: Path, manifest_path: Path, db_path: Path) -> dict[str, str]:
    shipped = read_scalars(scalars_path)
    objects, size, rows = manifest_totals(manifest_path)
    conn = read_only(db_path)
    try:
        bound = book_received_at(conn, int(shipped["max_id"]))
    finally:
        conn.close()

    corrected = dict(shipped)
    for name, roots in (
        ("ladder_roots", sorted(manifest_roots(manifest_path, "ladder"))),
        ("touch_roots", sorted(manifest_roots(manifest_path, "touch"))),
    ):
        if name in shipped:
            corrected[f"{name}_as_shipped"] = shipped[name]
        corrected[name] = ",".join(roots)
        corrected[f"{name}_count"] = str(len(roots))
    # ws_trades carries a rowid space of its own, so the pass could not bound it with the book's
    # ceiling and wrote every trade it had; a consumer has to cut the artifact by arrival time.
    corrected["trades_id_ceiling"] = "none"
    corrected["trades_bound_received_at"] = bound.isoformat()
    corrected["source_scalars"] = str(scalars_path)
    corrected["source_manifest"] = str(manifest_path)
    corrected["unchanged_tables"] = ",".join(UNCHANGED_TABLES)
    corrected["shipped_objects"] = str(objects.total())
    corrected["shipped_bytes"] = str(size.total())
    corrected["shipped_rows"] = str(rows.total())
    for kind in sorted(objects):
        corrected[f"shipped_objects_{kind}"] = str(objects[kind])
        corrected[f"shipped_bytes_{kind}"] = str(size[kind])
        corrected[f"shipped_rows_{kind}"] = str(rows[kind])
    return corrected


def read_scalars(path: Path) -> dict[str, str]:
    table = pq.read_table(path)
    return dict(zip(table.column("name").to_pylist(), table.column("value").to_pylist()))


def write_scalars(path: Path, scalars: Mapping[str, str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    rows = [{"name": name, "value": value} for name, value in scalars.items()]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCALARS_SCHEMA), path)


def manifest_totals(manifest_path: Path) -> tuple[Counter[str], Counter[str], Counter[str]]:
    objects: Counter[str] = Counter()
    size: Counter[str] = Counter()
    rows: Counter[str] = Counter()
    for line in manifest_path.read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        objects[record["kind"]] += 1
        size[record["kind"]] += record["size"]
        rows[record["kind"]] += record["rows"]
    return objects, size, rows


def read_only(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def book_received_at(conn: sqlite3.Connection, row_id: int) -> datetime:
    row = conn.execute(_RECEIVED_AT, (row_id,)).fetchone()
    if row is None:
        raise ValueError(f"no ws_book_events row at id={row_id}")
    return _decode_ts(row[0])


def format_report(args: argparse.Namespace, corrected: Mapping[str, str]) -> str:
    reported = (
        "ladder_roots",
        "ladder_roots_count",
        "touch_roots",
        "touch_roots_count",
        "trades_bound_received_at",
        "shipped_objects",
        "shipped_bytes",
        "shipped_rows",
    )
    return "\n".join(
        ["== INVENTORY REPAIR", f"scalars={args.scalars}", f"manifest={args.manifest}"]
        + [f"{name}={corrected[name]}" for name in reported]
        + [f"out={args.out}"]
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
