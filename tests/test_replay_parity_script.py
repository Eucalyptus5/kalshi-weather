from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.replay_parity import DEFAULT_POINTS, build_parser, run
from tests.test_parity import CHI, DEN, at, book, build_db, gap


REPO_ROOT = Path(__file__).resolve().parent.parent

CLEAN = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
    book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
    book(3, CHI, 0.5, 2, "yes", "0.2000", "4.00", True),
    book(4, CHI, 0.5, 2, "no", "0.7000", "6.00", True),
    book(5, DEN, 2.0, 3, "yes", "0.4100", "2.00"),
    book(6, CHI, 3.0, 4, "yes", "0.2100", "1.00"),
    book(7, DEN, 20.0, 5, "yes", "0.3000", "5.00", True),
    book(8, CHI, 20.5, 6, "yes", "0.2500", "5.00", True),
    book(9, DEN, 21.0, 7, "yes", "0.3000", "1.00"),
    book(10, CHI, 21.5, 8, "yes", "0.2500", "1.00"),
]

DELTA_BEFORE_ANY_SNAPSHOT = [
    book(1, CHI, 0.0, 1, "yes", "0.2000", "4.00", True),
    book(2, DEN, 0.5, 2, "yes", "0.4000", "3.00"),
    book(3, DEN, 1.0, 3, "yes", "0.4000", "10.00", True),
    book(4, DEN, 1.0, 3, "no", "0.5500", "7.00", True),
    book(5, DEN, 2.0, 4, "yes", "0.4100", "2.00"),
    book(6, CHI, 3.0, 5, "yes", "0.2100", "1.00"),
]


def _namespace(db_path: Path, points: int = 20) -> argparse.Namespace:
    return argparse.Namespace(db=db_path, points=points)


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_parity.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--db" in result.stdout
    assert "--points" in result.stdout


def test_default_arg_values() -> None:
    args = build_parser().parse_args([])
    assert args.db == REPO_ROOT / "data" / "state.db"
    assert args.points == DEFAULT_POINTS
    assert DEFAULT_POINTS >= 200


def test_points_budget_is_an_int_override() -> None:
    assert build_parser().parse_args(["--points", "512"]).points == 512


def test_a_clean_tape_reports_agreement_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = build_db(tmp_path / "state.db", CLEAN, [gap(1, "", 11.0)])

    rc = run(_namespace(db_path))

    assert rc == 0
    out = capsys.readouterr().out
    assert "== PARITY vs book_state_at" in out
    assert "disagreed=0" in out
    assert "agreed_on_raise=1" in out
    assert "-- by kind" in out
    assert "-- by cohort" in out
    assert "parity_points=" in out
    assert "post_gap=1" in out
    assert out.rstrip().endswith("none")


def test_a_point_the_oracle_cannot_answer_exits_one_and_names_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = build_db(tmp_path / "state.db", DELTA_BEFORE_ANY_SNAPSHOT)

    rc = run(_namespace(db_path))

    out = capsys.readouterr().out
    assert rc == 1
    assert "disagreed=2" in out
    assert DEN in out.split("-- disagreements")[1]
    assert at(0.5).isoformat() in out.split("-- disagreements")[1]
