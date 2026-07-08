from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.replay.artifacts import SCALARS_SCHEMA
from scripts.replay_inventory_repair import book_received_at, build_parser, read_only, run
from tests.test_artifacts import build_book_db, drain_record, write_manifest


REPO_ROOT = Path(__file__).resolve().parent.parent
DEN = "KXHIGHDEN-26JUL19-B85.5"
BOUND_AT = "2026-08-08 09:14:03.812345"

SHIPPED = {
    "gap_rows": "3",
    "tickers": "20",
    "ladder_scope": "KXHIGH,KXLOW",
    "ladder_roots": "",
    "ladder_depth": "6",
    "bytes_written": "40265",
    "max_id": "5",
    "trades_max_id": "None",
    "trades_rows": "9",
}

MANIFEST = [
    ("KXHIGHAUS-2026-07-17-b000001.parquet", "ladder", 100, 5),
    ("KXHIGHAUS-2026-07-18-b000002.parquet", "ladder", 200, 7),
    ("KXLOWTCHI-2026-07-17-b000001.parquet", "ladder", 300, 11),
    ("KXHIGHAUS-2026-07-17-b000001.parquet", "touch", 400, 13),
    ("KXRAINCHIM-2026-07-17-b000001.parquet", "touch", 500, 17),
    ("KXHIGHDEN-2026-07-17-b000000.parquet", "trades", 600, 19),
    ("KXLOWTCHI-2026-07-17-b000000.parquet", "trades", 700, 23),
]

BOOK_ROWS = [
    (4, DEN, "2026-08-08 09:13:00.000000", 1, "yes", "0.4000", "10.00", 1, BOUND_AT, None),
    (5, DEN, BOUND_AT, 2, "yes", "0.4100", "9.00", 0, BOUND_AT, 1_753_000_000_000),
    (6, DEN, "2026-08-08 09:15:00.000000", 3, "yes", "0.4200", "8.00", 0, BOUND_AT, None),
]


@pytest.fixture
def scalars(tmp_path: Path) -> Path:
    path = tmp_path / "scalars-b000000.parquet"
    rows = [{"name": name, "value": value} for name, value in SHIPPED.items()]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCALARS_SCHEMA), path)
    return path


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    records = [
        drain_record(name, kind) | {"size": size, "rows": rows}
        for name, kind, size, rows in MANIFEST
    ]
    return write_manifest(tmp_path / "manifest.jsonl", [*records[:3], None, *records[3:]])


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return build_book_db(tmp_path / "state.db", BOOK_ROWS)


@pytest.fixture
def repaired(scalars: Path, manifest: Path, db: Path, tmp_path: Path) -> dict[str, str]:
    out = tmp_path / "repaired.parquet"
    assert run(_args(scalars, manifest, db, out)) == 0
    return read_scalars(out)


def _args(scalars: Path, manifest: Path, db: Path, out: Path) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--scalars",
            str(scalars),
            "--manifest",
            str(manifest),
            "--db",
            str(db),
            "--out",
            str(out),
        ]
    )


def read_scalars(path: Path) -> dict[str, str]:
    table = pq.read_table(path)
    return dict(zip(table.column("name").to_pylist(), table.column("value").to_pylist()))


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_inventory_repair.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    for flag in ("--scalars", "--manifest", "--db", "--out"):
        assert flag in result.stdout


def test_the_manifest_replaces_the_root_list_the_drain_emptied(repaired: dict[str, str]) -> None:
    assert repaired["ladder_roots"] == "KXHIGHAUS,KXLOWTCHI"
    assert repaired["ladder_roots_count"] == "2"


def test_the_touch_roots_are_their_own_set(repaired: dict[str, str]) -> None:
    assert repaired["touch_roots"] == "KXHIGHAUS,KXRAINCHIM"
    assert repaired["touch_roots_count"] == "2"
    assert repaired["touch_roots"] != repaired["ladder_roots"]


def test_the_empty_root_list_that_shipped_survives_beside_its_replacement(
    repaired: dict[str, str],
) -> None:
    assert repaired["ladder_roots_as_shipped"] == ""


def test_a_scalar_that_is_only_added_carries_no_shipped_companion(
    repaired: dict[str, str],
) -> None:
    assert "touch_roots_as_shipped" not in repaired


def test_every_other_shipped_scalar_survives_in_order(
    scalars: Path, manifest: Path, db: Path, tmp_path: Path
) -> None:
    out = tmp_path / "repaired.parquet"
    run(_args(scalars, manifest, db, out))

    names = pq.read_table(out).column("name").to_pylist()
    assert names[: len(SHIPPED)] == list(SHIPPED)
    repaired = read_scalars(out)
    for name, value in SHIPPED.items():
        if name != "ladder_roots":
            assert repaired[name] == value


def test_the_trades_bound_is_the_arrival_of_the_row_the_book_stopped_at(
    repaired: dict[str, str],
) -> None:
    assert repaired["trades_id_ceiling"] == "none"
    assert repaired["trades_bound_received_at"] == "2026-08-08T09:14:03.812345+00:00"


def test_the_recorder_database_is_opened_read_only(db: Path) -> None:
    conn = read_only(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM ws_book_events WHERE id = 5")
    finally:
        conn.close()


def test_an_id_the_book_never_reached_is_refused(db: Path) -> None:
    conn = read_only(db)
    try:
        with pytest.raises(ValueError, match="99"):
            book_received_at(conn, 99)
    finally:
        conn.close()


def test_what_shipped_is_counted_from_the_manifest(repaired: dict[str, str]) -> None:
    assert repaired["shipped_objects"] == "7"
    assert repaired["shipped_bytes"] == "2800"
    assert repaired["shipped_rows"] == "95"
    assert repaired["shipped_objects_ladder"] == "3"
    assert repaired["shipped_bytes_ladder"] == "600"
    assert repaired["shipped_rows_ladder"] == "23"
    assert repaired["shipped_objects_touch"] == "2"
    assert repaired["shipped_bytes_touch"] == "900"
    assert repaired["shipped_rows_touch"] == "30"
    assert repaired["shipped_objects_trades"] == "2"
    assert repaired["shipped_bytes_trades"] == "1300"
    assert repaired["shipped_rows_trades"] == "42"


def test_the_local_byte_count_that_shipped_is_left_alone(repaired: dict[str, str]) -> None:
    assert repaired["bytes_written"] == SHIPPED["bytes_written"]
    assert "bytes_written_as_shipped" not in repaired


def test_the_corrected_file_names_what_it_came_from(
    repaired: dict[str, str], scalars: Path, manifest: Path
) -> None:
    assert repaired["source_scalars"] == str(scalars)
    assert repaired["source_manifest"] == str(manifest)
    assert repaired["unchanged_tables"] == "windows,boundaries,coverage"


def test_nothing_already_written_is_overwritten(
    scalars: Path, manifest: Path, db: Path, tmp_path: Path
) -> None:
    out = tmp_path / "repaired.parquet"
    out.write_bytes(b"PAR1")

    with pytest.raises(FileExistsError, match=str(out)):
        run(_args(scalars, manifest, db, out))

    assert out.read_bytes() == b"PAR1"


def test_the_corrected_file_is_a_scalars_table(
    scalars: Path, manifest: Path, db: Path, tmp_path: Path
) -> None:
    out = tmp_path / "repaired.parquet"
    run(_args(scalars, manifest, db, out))

    assert pq.read_table(out).schema.equals(SCALARS_SCHEMA)


def test_the_report_carries_the_counts_and_the_bound(
    scalars: Path, manifest: Path, db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "repaired.parquet"
    run(_args(scalars, manifest, db, out))

    printed = capsys.readouterr().out
    assert "ladder_roots_count=2" in printed
    assert "touch_roots_count=2" in printed
    assert "trades_bound_received_at=2026-08-08T09:14:03.812345+00:00" in printed
    assert f"out={out}" in printed
