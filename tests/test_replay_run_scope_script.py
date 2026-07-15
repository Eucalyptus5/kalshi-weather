import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.replay.artifacts import COVERAGE_SCHEMA
from bot.replay.run_scope import D_EVAL
from scripts.replay_run_scope import build_parser, run
from tests.test_run_scope import (
    BAND_US,
    BLIND_ROWS,
    HOUR,
    SCOPE_END,
    SCOPE_START,
    SECOND,
    TAPE_FIRST,
    TAPE_LAST,
    coverage_rows,
    write_blind,
)


ARTIFACTS = ("exclusions.parquet", "event_days.parquet", "split.json", "summary.json")


def write_coverage(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=COVERAGE_SCHEMA), path)
    return path


def args_for(blind: Path, coverage: Path, out: Path) -> argparse.Namespace:
    return build_parser().parse_args(
        ["--blind-windows", str(blind), "--coverage", str(coverage), "--out", str(out)]
    )


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        write_blind(tmp_path / "blind.parquet", BLIND_ROWS),
        write_coverage(tmp_path / "coverage.parquet", coverage_rows()),
        tmp_path / "scope",
    )


def test_a_run_writes_every_artifact_and_reports_the_scope(
    inputs: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    blind, coverage, out = inputs

    assert run(args_for(blind, coverage, out)) == 0

    for name in ARTIFACTS:
        assert (out / name).exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cities"] == 2
    assert summary["event_days"] == 34
    assert summary["covered_days"] == 32
    assert summary["evaluable_days"] == 2 * D_EVAL
    assert summary["in_scope_days"] == 2 * D_EVAL
    assert summary["tape_first"] == TAPE_FIRST.isoformat()
    assert summary["tape_last"] == TAPE_LAST.isoformat()
    assert summary["scope_start"] == SCOPE_START.isoformat()
    assert summary["scope_end"] == SCOPE_END.isoformat()
    assert summary["boundary_event_day"] == "2026-07-28"
    assert summary["exclusions"] == 19
    assert summary["recorded_gap_intervals"] == 2
    assert summary["recorded_gap_us"] == 30 * SECOND + 1748 * SECOND
    assert summary["subscription_wide_intervals"] == 1
    assert summary["subscription_wide_us"] == 10 * SECOND
    assert summary["resubscribe_blind_intervals"] == 1
    assert summary["resubscribe_blind_us"] == 9 * SECOND
    assert summary["quiet_band_intervals"] == 15
    assert summary["quiet_band_us"] == 14 * BAND_US + HOUR
    assert summary["union_excluded_us"] == 14 * BAND_US + HOUR + 49 * SECOND
    assert summary["padded_boundary_id"] == 51
    assert summary["padded_gap_id"] == 5
    assert summary["split_sha256"] == json.loads((out / "split.json").read_text())["sha256"]

    printed = capsys.readouterr().out
    assert "== RUN SCOPE" in printed
    assert "in_scope_days=28" in printed
    assert f"split_sha256={summary['split_sha256']}" in printed


def test_a_run_without_the_named_incident_is_refused(tmp_path: Path) -> None:
    blind = write_blind(tmp_path / "blind.parquet", BLIND_ROWS[:4])
    coverage = write_coverage(tmp_path / "coverage.parquet", coverage_rows())
    out = tmp_path / "scope"

    with pytest.raises(ValueError, match="padded=0"):
        run(args_for(blind, coverage, out))

    assert not out.exists()


def test_the_parser_requires_both_inputs_and_the_output(tmp_path: Path) -> None:
    args = args_for(tmp_path / "b.parquet", tmp_path / "c.parquet", tmp_path / "scope")

    assert args.blind_windows == tmp_path / "b.parquet"
    assert args.coverage == tmp_path / "c.parquet"
    assert args.out == tmp_path / "scope"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--out", str(tmp_path)])
