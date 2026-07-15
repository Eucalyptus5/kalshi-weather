from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.replay.artifacts import SCALARS_SCHEMA, WINDOWS_SCHEMA
from bot.replay.blind_windows import FROZEN_END, FROZEN_START, BlindWindow, write_blind_windows
from bot.replay.clears import MidLifeClear, write_mid_life_clears
from scripts.replay_inventory_corrected import build_parser, run, summarise


REPO_ROOT = Path(__file__).resolve().parent.parent
UTC = timezone.utc
AUS = "KXHIGHAUS-26JUL30-B85.5"
DEN = "KXHIGHDEN-26JUL30-B85.5"
CLEAR_AT = datetime(2026, 7, 30, 3, 0, 0, tzinfo=UTC)

BLIND_WINDOWS = [
    (FROZEN_START + timedelta(hours=1), 3_920_000, 1),
    (FROZEN_START + timedelta(days=3), 10_000_000, None),
    (FROZEN_START + timedelta(days=9), 12_000_000, None),
    (FROZEN_START - timedelta(days=1), 100_000_000, 2),
    (FROZEN_END, 1_000_000, None),
    (FROZEN_END + timedelta(days=1), 2_000_000, None),
]
BLIND_SUMMARY = {
    "db": "/srv/kalshi/data/state.db",
    "out": "/srv/kalshi/out/blind-b000000.parquet",
    "max_id": 458_915_263,
    "rows_scanned": 458_915_263,
    "windows": 999,
    "with_gap_row": 999,
    "unmatched_gap_rows": 4,
    "percentiles": "nearest_rank",
    "elapsed_s": 1812.4,
}

CLEARS = [(AUS, 10_000_000, 3, False), (DEN, 5_000_000, 0, True)]
CLEARS_SUMMARY = {
    "days": 15,
    "clears": 4812,
    "terminal": 4800,
    "no_anchor": 7,
    "empty_book": 3,
    "mid_life": 1,
    "stale_rows": 99,
    "stale_us": 99,
    "unresolved": 99,
    "with_stale_rows": 99,
}

SHIPPED_STARTS = [FROZEN_START + timedelta(hours=1), FROZEN_START - timedelta(days=2)]


def write_windows(path: Path, specs: Sequence[tuple[datetime, int, int | None]]) -> Path:
    write_blind_windows(
        path,
        [
            BlindWindow(
                boundary_id=index + 1,
                prev_id=index,
                end_id=index + 1,
                burst_messages=1,
                ticker="",
                start=start,
                end=start + timedelta(microseconds=blind_us),
                prev_seq=2,
                seq=1,
                gap_id=gap_id,
                gap_reason=None if gap_id is None else "connection_reset",
                gap_detected_at=None if gap_id is None else start,
            )
            for index, (start, blind_us, gap_id) in enumerate(specs)
        ],
    )
    return path


def write_clears(path: Path, specs: Sequence[tuple[str, int, int, bool]]) -> Path:
    write_mid_life_clears(
        path,
        [
            MidLifeClear(
                ticker=ticker,
                clear_at=CLEAR_AT,
                anchor_at=CLEAR_AT - timedelta(seconds=1),
                yes_levels=2,
                no_levels=1,
                total_levels=3,
                next_row_at=CLEAR_AT + timedelta(seconds=1),
                next_snapshot_at=None if unresolved else CLEAR_AT + timedelta(microseconds=stale),
                stale_end=CLEAR_AT + timedelta(microseconds=stale),
                stale_rows=rows,
                unresolved=unresolved,
                clear_since_anchor=False,
            )
            for ticker, stale, rows, unresolved in specs
        ],
    )
    return path


def write_shipped(path: Path, starts: Sequence[datetime]) -> Path:
    rows = [
        {
            "gap_id": index + 1,
            "ticker": AUS,
            "start": start,
            "end": start + timedelta(seconds=9),
            "detected_at": start + timedelta(seconds=10),
            "last_seq": 41,
            "reason": "connection_reset",
        }
        for index, start in enumerate(starts)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=WINDOWS_SCHEMA), path)
    return path


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=1))
    return path


def read_scalars(path: Path) -> dict[str, str]:
    table = pq.read_table(path)
    return dict(zip(table.column("name").to_pylist(), table.column("value").to_pylist()))


def args_for(paths: Mapping[str, Path], out: Path) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--blind-windows",
            str(paths["blind"]),
            "--blind-summary",
            str(paths["blind_summary"]),
            "--clears",
            str(paths["clears"]),
            "--clears-summary",
            str(paths["clears_summary"]),
            "--shipped-windows",
            str(paths["shipped"]),
            "--out",
            str(out),
        ]
    )


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "blind": write_windows(tmp_path / "blind-b000000.parquet", BLIND_WINDOWS),
        "blind_summary": write_json(tmp_path / "blind.json", BLIND_SUMMARY),
        "clears": write_clears(tmp_path / "clears-b000000.parquet", CLEARS),
        "clears_summary": write_json(tmp_path / "clears.json", CLEARS_SUMMARY),
        "shipped": write_shipped(tmp_path / "windows-b000000.parquet", SHIPPED_STARTS),
    }


@pytest.fixture
def corrected(paths: dict[str, Path], tmp_path: Path) -> dict[str, str]:
    out = tmp_path / "scalars-b000000.parquet"
    assert run(args_for(paths, out)) == 0
    return read_scalars(out)


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_inventory_corrected.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    for flag in (
        "--blind-windows",
        "--blind-summary",
        "--clears",
        "--clears-summary",
        "--shipped-windows",
        "--out",
    ):
        assert flag in result.stdout


def test_a_run_writes_a_scalars_table_and_reports_the_coverage(
    paths: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "scalars-b000000.parquet"

    assert run(args_for(paths, out)) == 0

    assert pq.read_table(out).schema.equals(SCALARS_SCHEMA)
    scalars = read_scalars(out)
    assert scalars["windows"] == "6"
    assert scalars["with_gap_row"] == "2"
    assert scalars["without_gap_row"] == "4"
    assert scalars["total_blind_us"] == "128920000"
    assert scalars["percentiles"] == "nearest_rank"
    assert scalars["p50_blind_us"] == "3920000"
    assert scalars["p90_blind_us"] == "100000000"
    assert scalars["max_blind_us"] == "100000000"
    assert scalars["unmatched_gap_rows"] == "4"
    assert scalars["rows_scanned"] == "458915263"
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "== INVENTORY CORRECTED"
    assert "windows=6" in printed
    assert "total_blind_us=128920000" in printed
    assert f"out={out}" in printed


def test_the_provenance_names_what_it_corrects_and_why(
    corrected: dict[str, str], paths: dict[str, Path]
) -> None:
    assert corrected["supersedes"] == str(paths["shipped"])
    assert corrected["correction_tables"] == "blind_windows,mid_life_clears"
    assert corrected["source_blind_windows"] == str(paths["blind"])
    assert corrected["source_blind_summary"] == str(paths["blind_summary"])
    assert corrected["source_clears"] == str(paths["clears"])
    assert corrected["source_clears_summary"] == str(paths["clears_summary"])
    assert corrected["id_ceiling"] == "458915263"
    assert "ws_gaps" in corrected["defect"]
    assert "\n" not in corrected["defect"]


def test_nothing_already_written_is_overwritten_and_nothing_is_read_first(
    tmp_path: Path,
) -> None:
    out = tmp_path / "scalars-b000000.parquet"
    out.write_bytes(b"PAR1")
    absent = {name: tmp_path / f"{name}.absent" for name in ("blind", "clears", "shipped")}
    absent |= {"blind_summary": tmp_path / "b.absent", "clears_summary": tmp_path / "c.absent"}

    with pytest.raises(FileExistsError, match=str(out)):
        run(args_for(absent, out))

    assert out.read_bytes() == b"PAR1"


def test_the_frozen_subset_counts_only_the_windows_inside_it(corrected: dict[str, str]) -> None:
    assert corrected["frozen_start"] == "2026-07-18T00:00:00+00:00"
    assert corrected["frozen_end"] == "2026-08-02T00:00:00+00:00"
    assert corrected["frozen_windows"] == "3"
    assert corrected["frozen_with_gap_row"] == "1"
    assert corrected["frozen_without_gap_row"] == "2"
    assert corrected["frozen_total_blind_us"] == "25920000"
    assert corrected["frozen_p50_blind_us"] == "10000000"
    assert corrected["frozen_p90_blind_us"] == "12000000"
    assert corrected["frozen_max_blind_us"] == "12000000"
    whole = (
        "windows",
        "with_gap_row",
        "without_gap_row",
        "total_blind_us",
        "p50_blind_us",
        "p90_blind_us",
        "max_blind_us",
    )
    for name in whole:
        assert corrected[name] != corrected[f"frozen_{name}"]


def test_the_blind_fraction_is_frozen_blind_time_over_the_frozen_wall_clock(
    corrected: dict[str, str],
) -> None:
    assert corrected["frozen_wall_us"] == "1296000000000"
    assert corrected["frozen_blind_fraction"] == "0.000020"


def test_the_shipped_reach_is_counted_from_the_shipped_table(
    corrected: dict[str, str], paths: dict[str, Path], tmp_path: Path
) -> None:
    assert corrected["shipped_windows"] == "2"
    assert corrected["corrected_windows"] == "6"
    assert corrected["shipped_frozen_windows"] == "1"
    assert corrected["corrected_frozen_windows"] == "3"

    wider = paths | {
        "shipped": write_shipped(tmp_path / "wider.parquet", [*SHIPPED_STARTS, *SHIPPED_STARTS[:2]])
    }
    out = tmp_path / "wider-scalars.parquet"
    assert run(args_for(wider, out)) == 0

    again = read_scalars(out)
    assert again["shipped_windows"] == "4"
    assert again["shipped_frozen_windows"] == "2"
    assert again["corrected_windows"] == "6"


def test_the_mid_life_figures_come_from_the_table_the_summary_disagrees_with(
    corrected: dict[str, str],
) -> None:
    assert CLEARS_SUMMARY["mid_life"] != len(CLEARS)
    assert corrected["clears_mid_life"] == "2"
    assert corrected["clears_stale_rows"] == "3"
    assert corrected["clears_stale_us"] == "15000000"
    assert corrected["clears_unresolved"] == "1"
    assert corrected["clears_with_stale_rows"] == "1"
    assert corrected["clear_days"] == "15"
    assert corrected["clears"] == "4812"
    assert corrected["clears_terminal"] == "4800"
    assert corrected["clears_no_anchor"] == "7"
    assert corrected["clears_empty_book"] == "3"


def test_a_clears_table_with_no_mid_life_rows_totals_zero(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    empty = paths | {"clears": write_clears(tmp_path / "empty.parquet", [])}
    out = tmp_path / "empty-scalars.parquet"

    assert run(args_for(empty, out)) == 0

    scalars = read_scalars(out)
    assert scalars["clears_mid_life"] == "0"
    assert scalars["clears_stale_rows"] == "0"
    assert scalars["clears_stale_us"] == "0"
    assert scalars["clears_unresolved"] == "0"
    assert scalars["clears_with_stale_rows"] == "0"
    assert scalars["clears_terminal"] == "4800"


def test_every_value_is_a_string(paths: dict[str, Path]) -> None:
    scalars = summarise(
        paths["blind"],
        paths["blind_summary"],
        paths["clears"],
        paths["clears_summary"],
        paths["shipped"],
    )

    assert scalars
    assert all(isinstance(name, str) and isinstance(value, str) for name, value in scalars.items())
