from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.replay.artifacts import BudgetExceeded, ByteBudget, byte_budget, ws_raw_daily_bytes
from bot.replay.forward_pass import BARRIER_ROWS, READ_BATCH_ROWS, ROW_GROUP_ROWS
from scripts import replay_forward
from scripts.replay_forward import DEFAULT_PASS_HOURS, build_parser, run
from tests.test_artifacts import build_db


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tape(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    build_db(path)
    return path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "ws_raw"
    directory.mkdir()
    (directory / "2026-07-18.jsonl.gz").write_bytes(b"x" * 4096)
    return directory


@pytest.fixture(autouse=True)
def statvfs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the filesystem read is stood in for; the budget arithmetic and --pass-hours stay real,
    # otherwise the suite passes or fails on how full the machine running it happens to be.
    def measured(path: Path, raw: Path, pass_hours) -> ByteBudget:
        return byte_budget(
            f_bavail=22_000_000,
            f_frsize=4096,
            f_blocks=50_000_000,
            ws_raw_daily_bytes=ws_raw_daily_bytes(raw),
            pass_hours=pass_hours,
        )

    monkeypatch.setattr(replay_forward, "measure_byte_budget", measured)


def _args(tape: Path, raw_dir: Path, out: Path, *extra: str):
    return build_parser().parse_args(
        ["--db", str(tape), "--raw-dir", str(raw_dir), "--out", str(out), *extra]
    )


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_forward.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--db" in result.stdout
    assert "--out" in result.stdout
    assert "--max-id" in result.stdout
    assert "--trades-max-id" in result.stdout
    assert "--pass-hours" in result.stdout


def test_default_arg_values() -> None:
    args = build_parser().parse_args([])
    assert args.db == REPO_ROOT / "data" / "state.db"
    assert args.raw_dir == REPO_ROOT / "data" / "ws_raw"
    assert args.max_id is None
    assert args.trades_max_id is None
    assert args.pass_hours == DEFAULT_PASS_HOURS
    assert args.read_batch == READ_BATCH_ROWS
    assert args.row_group == ROW_GROUP_ROWS
    assert args.barrier_rows == BARRIER_ROWS
    assert args.trades is True


def test_a_run_emits_readable_artifacts_and_reports_its_rates(
    tape: Path, raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"

    rc = run(_args(tape, raw_dir, out))

    assert rc == 0
    printed = capsys.readouterr().out
    assert "== FORWARD PASS" in printed
    assert "rows=10" in printed
    assert "rows_per_s=" in printed
    assert "cpu_over_wall=" in printed
    assert "files_unreadable=[]" in printed
    stats = json.loads((out / "run_stats.json").read_text())
    assert stats["rows"] == 10
    assert stats["files_readable"] == len(list(out.rglob("*.parquet")))
    assert stats["files_unreadable"] == []
    assert stats["trades_rows"] == 2
    assert stats["peak_rss_bytes"] > 0
    assert stats["artifact_bytes"] > 0


def test_the_ladder_artifact_holds_only_the_scoped_roots(
    tape: Path, raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    run(_args(tape, raw_dir, out))

    roots = {path.name.split("-")[0] for path in (out / "ladder").glob("*.parquet")}
    assert roots == {"KXHIGHDEN", "KXHIGHCHI"}
    assert {path.name.split("-")[0] for path in (out / "touch").glob("*.parquet")} == {
        "KXHIGHDEN",
        "KXHIGHCHI",
        "KXRAINCHIM",
    }


def test_the_scale_census_counts_every_exponent_it_saw(
    tape: Path, raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    run(_args(tape, raw_dir, out))

    stats = json.loads((out / "run_stats.json").read_text())
    assert stats["price_exponents"] == {"-4": 8, "-1": 2}
    assert stats["size_exponents"] == {"-2": 8, "0": 2}
    assert stats["scale_outliers"] == []
    assert stats["logical_bytes"] > 0


def test_the_id_ceiling_bounds_the_run(tape: Path, raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"

    run(_args(tape, raw_dir, out, "--max-id", "5", "--no-trades"))

    stats = json.loads((out / "run_stats.json").read_text())
    assert stats["rows"] == 5
    assert stats["last_id"] == 5
    assert stats["trades_rows"] == 0
    assert not (out / "trades").exists()


def test_the_book_ceiling_alone_leaves_the_trades_artifact_unbounded(
    tape: Path, raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    run(_args(tape, raw_dir, out, "--max-id", "1"))

    stats = json.loads((out / "run_stats.json").read_text())
    assert stats["rows"] == 1
    assert stats["trades_rows"] == 2
    assert stats["trades_max_id"] is None


def test_the_trades_ceiling_bounds_the_trades_artifact_alone(
    tape: Path, raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    run(_args(tape, raw_dir, out, "--trades-max-id", "1"))

    stats = json.loads((out / "run_stats.json").read_text())
    assert stats["rows"] == 10
    assert stats["max_id"] is None
    assert stats["trades_max_id"] == 1
    assert stats["trades_rows"] == 1
    assert [p.name for p in (out / "trades").glob("*.parquet")] == [
        "KXHIGHDEN-2026-07-19-b000000.parquet"
    ]


def test_a_negative_budget_stops_before_anything_is_written(
    tape: Path, raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    with pytest.raises(BudgetExceeded):
        run(_args(tape, raw_dir, out, "--pass-hours", "1000000"))

    assert list(out.rglob("*.parquet")) == []


def test_the_inventory_carries_the_run_stats_and_the_realized_roots(
    tape: Path, raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    run(_args(tape, raw_dir, out))

    scalars = pq.read_table(out / "inventory" / "scalars-b000000.parquet").to_pydict()
    values = dict(zip(scalars["name"], scalars["value"]))
    assert values["ladder_roots"] == "KXHIGHCHI,KXHIGHDEN"
    assert values["ladder_depth"] == "6"
    assert values["rows"] == "10"
    assert values["resident_ladders"] == "3"
    assert float(values["rows_per_s"]) > 0
