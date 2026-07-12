from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.replay.clears import MID_LIFE_CLEARS_SCHEMA
from scripts.replay_clears import FROZEN_DAYS, build_parser, run
from tests.test_clears import ANCHORED, SECOND, at, build_db, row, write_clears
from tests.test_raw_tape import AUS


REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = "2026-07-30"

BOOK_ROWS = [
    *ANCHORED,
    row(4, at(11 * SECOND), "yes", "0.5000", "-1.00"),
    row(5, at(20 * SECOND), "yes", "0.6000", "4.00", seq=2, snapshot=True),
    row(6, at(30 * SECOND), "yes", "0.6000", "-1.00", seq=3),
]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return build_db(tmp_path / "state.db", BOOK_ROWS)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "ws_raw"
    write_clears(directory, [at(10 * SECOND), at(40 * SECOND)])
    return directory


def args_for(db: Path, raw: Path, out: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        ["--db", str(db), "--raw-dir", str(raw), "--out", str(out), *extra]
    )


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_clears.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    for flag in ("--db", "--raw-dir", "--out", "--summary", "--day"):
        assert flag in result.stdout


def test_the_run_defaults_to_the_fifteen_frozen_days(tmp_path: Path) -> None:
    args = args_for(tmp_path / "state.db", tmp_path / "ws_raw", tmp_path / "clears.parquet")

    assert args.day == []
    assert args.summary is None
    assert len(FROZEN_DAYS) == 15
    assert (FROZEN_DAYS[0], FROZEN_DAYS[-1]) == (date(2026, 7, 18), date(2026, 8, 1))


def test_a_day_is_repeatable_and_parsed_as_a_date(tmp_path: Path) -> None:
    args = args_for(
        tmp_path / "state.db",
        tmp_path / "ws_raw",
        tmp_path / "clears.parquet",
        "--day",
        DAY,
        "--day",
        "2026-07-31",
    )

    assert [day.isoformat() for day in args.day] == [DAY, "2026-07-31"]


def test_a_run_writes_the_table_and_names_every_mid_life_clear(
    db_path: Path, raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "clears.parquet"

    assert run(args_for(db_path, raw_dir, out, "--day", DAY)) == 0

    table = pq.read_table(out)
    assert table.schema.equals(MID_LIFE_CLEARS_SCHEMA)
    assert table.to_pylist() == [
        {
            "ticker": AUS,
            "clear_at": at(10 * SECOND),
            "anchor_at": at(0),
            "yes_levels": 2,
            "no_levels": 1,
            "total_levels": 3,
            "next_row_at": at(11 * SECOND),
            "next_snapshot_at": at(20 * SECOND),
            "stale_end": at(20 * SECOND),
            "stale_us": 10 * SECOND,
            "stale_rows": 1,
            "unresolved": False,
            "clear_since_anchor": False,
        }
    ]
    printed = capsys.readouterr().out
    assert "== MID LIFE CLEARS" in printed
    assert f"scanned_days={DAY}" in printed
    assert "days=1" in printed
    assert "clears=2" in printed
    assert "terminal=1" in printed
    assert "no_anchor=0" in printed
    assert "empty_book=0" in printed
    assert "mid_life=1" in printed
    assert "stale_rows=1" in printed
    assert "stale_us=10000000" in printed
    assert "unresolved=0" in printed
    assert "with_stale_rows=1" in printed
    assert (
        f"clear={AUS} 2026-07-30T03:00:10+00:00 "
        "next_snapshot=2026-07-30T03:00:20+00:00 stale_us=10000000 stale_rows=1"
    ) in printed


def test_a_day_named_twice_is_scanned_once(
    db_path: Path, raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "clears.parquet"

    assert run(args_for(db_path, raw_dir, out, "--day", DAY, "--day", DAY)) == 0

    assert len(pq.read_table(out)) == 1
    printed = capsys.readouterr().out
    assert f"scanned_days={DAY}" in printed
    assert "days=1" in printed
    assert "clears=2" in printed


def test_an_unresolved_clear_reports_no_corrective_snapshot(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = build_db(
        tmp_path / "state.db", [*ANCHORED, row(4, at(11 * SECOND), "yes", "0.5000", "-1.00")]
    )
    out = tmp_path / "clears.parquet"

    assert run(args_for(db, raw_dir, out, "--day", DAY)) == 0

    printed = capsys.readouterr().out
    assert "unresolved=1" in printed
    assert f"clear={AUS} 2026-07-30T03:00:10+00:00 next_snapshot=none " in printed


def test_the_summary_is_written_as_json(db_path: Path, raw_dir: Path, tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"

    run(
        args_for(
            db_path,
            raw_dir,
            tmp_path / "clears.parquet",
            "--day",
            DAY,
            "--summary",
            str(summary_path),
        )
    )

    summary = json.loads(summary_path.read_text())
    assert summary["scanned_days"] == DAY
    assert (summary["days"], summary["clears"], summary["terminal"]) == (1, 2, 1)
    assert (summary["mid_life"], summary["stale_rows"], summary["stale_us"]) == (1, 1, 10 * SECOND)
    assert summary["with_stale_rows"] == 1
