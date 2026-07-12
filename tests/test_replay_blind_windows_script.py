from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.replay.blind_windows import BLIND_WINDOWS_SCHEMA
from scripts.replay_blind_windows import build_parser, run
from tests.test_blind_windows import SECOND, at, book, build_db, gap
from tests.test_raw_tape import frame, write_tape


REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = "2026-07-30"

BOOK_ROWS = [
    book(1, 0, 1),
    book(2, SECOND, 2),
    book(3, 2 * SECOND + 398_909, 1),
    book(4, 3 * SECOND, 2),
]
GAP_ROWS = [gap(1, 2 * SECOND + 500_000)]

CEILING_ROWS = [
    book(1, 0, 1),
    book(2, SECOND, 2),
    book(3, 2 * SECOND, 1),
    book(4, 3 * SECOND, 2),
    book(5, 5 * SECOND, 1),
    book(6, 6 * SECOND, 2),
    book(7, 20 * SECOND, 1),
    book(8, 21 * SECOND, 2),
]
CEILING_GAP_ROWS = [gap(1, 2 * SECOND + 500_000), gap(2, 20 * SECOND + 500_000)]

AGREEING_TAPE = [
    (at(SECOND - 10), frame("orderbook_delta")),
    (at(2 * SECOND + 398_900), frame("orderbook_snapshot")),
    (at(3 * SECOND - 10), frame("orderbook_delta", 2)),
]
FALSIFYING_TAPE = [
    (at(SECOND - 10), frame("orderbook_delta")),
    (at(3 * SECOND - 10), frame("orderbook_delta", 2)),
]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return build_db(tmp_path / "state.db", BOOK_ROWS, GAP_ROWS)


def raw_dir(tmp_path: Path, frames: list[tuple[object, str]]) -> Path:
    directory = tmp_path / "ws_raw"
    write_tape(directory, frames)
    return directory


def args_for(db: Path, out: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(["--db", str(db), "--out", str(out), *extra])


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_blind_windows.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    for flag in ("--db", "--out", "--max-id", "--summary", "--raw-dir", "--validate-day"):
        assert flag in result.stdout


def test_the_parser_defaults(tmp_path: Path) -> None:
    args = args_for(tmp_path / "state.db", tmp_path / "blind.parquet")

    assert args.max_id is None
    assert args.summary is None
    assert args.raw_dir is None
    assert args.validate_day == []


def test_validate_day_is_repeatable_and_parsed_as_a_date(tmp_path: Path) -> None:
    args = args_for(
        tmp_path / "state.db",
        tmp_path / "blind.parquet",
        "--validate-day",
        DAY,
        "--validate-day",
        "2026-07-31",
    )

    assert [day.isoformat() for day in args.validate_day] == [DAY, "2026-07-31"]


def test_a_run_writes_the_table_and_reports_the_counts(
    db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "blind.parquet"

    assert run(args_for(db_path, out)) == 0

    table = pq.read_table(out)
    assert table.schema.equals(BLIND_WINDOWS_SCHEMA)
    assert table.to_pylist() == [
        {
            "boundary_id": 3,
            "prev_id": 2,
            "ticker": "",
            "start": at(SECOND),
            "end": at(2 * SECOND + 398_909),
            "blind_us": 1_398_909,
            "prev_seq": 2,
            "seq": 1,
            "has_gap_row": True,
            "gap_id": 1,
            "gap_reason": "connection_reset",
            "gap_detected_at": at(2 * SECOND + 500_000),
            "in_frozen_window": True,
        }
    ]
    printed = capsys.readouterr().out
    assert "== BLIND WINDOWS" in printed
    assert "windows=1" in printed
    assert "with_gap_row=1" in printed
    assert "without_gap_row=0" in printed
    assert "total_blind_us=1398909" in printed
    assert "percentiles=nearest_rank" in printed
    assert "p50_blind_us=1398909" in printed
    assert "p90_blind_us=1398909" in printed
    assert "max_blind_us=1398909" in printed
    assert "frozen_windows=1" in printed
    assert "unmatched_gap_rows=0" in printed
    assert "rows_scanned=4" in printed


def test_a_gap_row_the_scan_cannot_place_is_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = build_db(tmp_path / "state.db", BOOK_ROWS, [*GAP_ROWS, gap(2, -SECOND)])

    assert run(args_for(db, tmp_path / "blind.parquet")) == 0

    assert "unmatched_gap_rows=1" in capsys.readouterr().out


def test_a_gap_row_above_the_ceiling_reaches_no_window_below_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = build_db(tmp_path / "state.db", CEILING_ROWS, CEILING_GAP_ROWS)
    out = tmp_path / "blind.parquet"

    assert run(args_for(db, out, "--max-id", "6")) == 0

    assert [row["gap_id"] for row in pq.read_table(out).to_pylist()] == [1, None]
    printed = capsys.readouterr().out
    assert "windows=2" in printed
    assert "with_gap_row=1" in printed
    assert "without_gap_row=1" in printed
    assert "unmatched_gap_rows=1" in printed


def test_the_ceiling_bounds_what_the_run_inventories(
    db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "blind.parquet"

    assert run(args_for(db_path, out, "--max-id", "2")) == 0

    assert pq.read_table(out).to_pylist() == []
    assert "windows=0" in capsys.readouterr().out


def test_the_summary_is_written_as_json(db_path: Path, tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"

    run(args_for(db_path, tmp_path / "blind.parquet", "--summary", str(summary_path)))

    summary = json.loads(summary_path.read_text())
    assert summary["windows"] == 1
    assert summary["with_gap_row"] == 1
    assert summary["total_blind_us"] == 1_398_909
    assert summary["percentiles"] == "nearest_rank"
    assert summary["frozen_start"] == "2026-07-18T00:00:00+00:00"
    assert summary["frozen_end"] == "2026-08-02T00:00:00+00:00"
    assert summary["unmatched_gap_rows"] == 0


def test_a_validated_day_reports_the_sample_and_the_three_tallies(
    db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = raw_dir(tmp_path, AGREEING_TAPE)
    out = tmp_path / "blind.parquet"

    rc = run(args_for(db_path, out, "--raw-dir", str(directory), "--validate-day", DAY))

    printed = capsys.readouterr().out
    assert rc == 0
    assert "validated_windows=1" in printed
    assert "tape_agreed=1" in printed
    assert "tape_wider=0" in printed
    assert "tape_extra_frames=0" in printed
    assert "tape_falsified=0" in printed


def test_a_day_named_twice_is_scanned_once(
    db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = raw_dir(tmp_path, AGREEING_TAPE)
    out = tmp_path / "blind.parquet"

    rc = run(
        args_for(
            db_path,
            out,
            "--raw-dir",
            str(directory),
            "--validate-day",
            DAY,
            "--validate-day",
            DAY,
        )
    )

    printed = capsys.readouterr().out
    assert rc == 0
    assert f"validated_days={DAY}" in printed
    assert "validated_windows=1" in printed
    assert "tape_agreed=1" in printed
    assert "tape_wider=0" in printed
    assert "tape_extra_frames=0" in printed


def test_a_falsified_window_is_named_and_fails_the_run(
    db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = raw_dir(tmp_path, FALSIFYING_TAPE)
    out = tmp_path / "blind.parquet"

    rc = run(args_for(db_path, out, "--raw-dir", str(directory), "--validate-day", DAY))

    printed = capsys.readouterr().out
    assert rc == 1
    assert "tape_falsified=1" in printed
    assert f"falsifying=3 {at(SECOND).isoformat()}" in printed


def test_validation_without_a_raw_directory_is_refused(db_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "blind.parquet"

    with pytest.raises(ValueError, match="--raw-dir"):
        run(args_for(db_path, out, "--validate-day", DAY))

    assert not out.exists()
