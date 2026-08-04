import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.replay.artifacts import COVERAGE_SCHEMA
from bot.replay.run_scope import D_EVAL, split_lengths
from scripts.replay_run_scope import PADDED_INCIDENT, STATION_SETS, build_parser, run
from tests.test_run_scope import (
    BAND_US,
    BLIND_ROWS,
    HOLE_DAY,
    HOLE_ROW,
    HOUR,
    LONG_DAYS,
    RESTART_DAYS,
    SCOPE_END,
    SCOPE_START,
    SECOND,
    TAPE_FIRST,
    TAPE_LAST,
    coverage_rows,
    spread_rows,
    write_blind,
)


UTC = timezone.utc
ARTIFACTS = ("exclusions.parquet", "event_days.parquet", "split.json", "summary.json")
START = "2026-07-19"
ANCHOR_FIRST = datetime(2026, 7, 18, tzinfo=UTC)
ANCHOR_LAST = datetime(2026, 8, 25, tzinfo=UTC)


def write_coverage(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=COVERAGE_SCHEMA), path)
    return path


def long_coverage() -> list[dict[str, object]]:
    anchor = {
        "ticker": "KXRAINNYCM-26JUL19-T0.1",
        "rows": 3,
        "first_received_at": ANCHOR_FIRST,
        "last_received_at": ANCHOR_LAST,
    }
    return [anchor, *spread_rows(LONG_DAYS)]


def args_for(blind: Path, coverage: Path, out: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--blind-windows",
            str(blind),
            "--coverage",
            str(coverage),
            "--out",
            str(out),
            "--start",
            START,
            *extra,
        ]
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
    assert summary["cohort"] == "high"
    assert summary["d_eval"] == D_EVAL
    assert summary["scope_open"] == datetime(2026, 7, 19, tzinfo=UTC).isoformat()
    assert summary["cities"] == 2
    assert summary["event_days"] == 34
    assert summary["covered_days"] == 32
    assert summary["evaluable_days"] == 2 * D_EVAL
    assert summary["in_scope_days"] == 2 * D_EVAL
    assert summary["over_tolerance_days"] == 0
    assert summary["worst_outage_us"] == 1757 * SECOND
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


def test_a_window_clear_of_the_incident_is_not_asked_to_find_it(tmp_path: Path) -> None:
    blind = write_blind(tmp_path / "blind.parquet", [*BLIND_ROWS[:4], HOLE_ROW])
    coverage = write_coverage(tmp_path / "coverage.parquet", long_coverage())
    out = tmp_path / "scope"

    assert run(args_for(blind, coverage, out)) == 0

    split = json.loads((out / "split.json").read_text())
    assert split["discovery_days"][0] == RESTART_DAYS[0].isoformat()
    assert split["last_evaluable_event_day"] == RESTART_DAYS[-1].isoformat()
    assert HOLE_DAY.isoformat() not in split["discovery_days"] + split["holdout_days"]
    assert datetime.fromisoformat(split["scope_start"]) > PADDED_INCIDENT
    summary = json.loads((out / "summary.json").read_text())
    assert summary["over_tolerance_days"] == 2
    assert summary["in_scope_days"] == 2 * D_EVAL


def test_a_partial_accrual_is_refused_before_anything_is_written(
    inputs: tuple[Path, Path, Path],
) -> None:
    blind, coverage, out = inputs

    with pytest.raises(ValueError, match="whole multiple"):
        run(args_for(blind, coverage, out, "--d-eval", "1"))

    assert not out.exists()


def test_a_longer_accrual_splits_by_the_same_two_thirds(tmp_path: Path) -> None:
    blind = write_blind(tmp_path / "blind.parquet", BLIND_ROWS)
    coverage = write_coverage(tmp_path / "coverage.parquet", long_coverage())
    out = tmp_path / "scope"

    assert run(args_for(blind, coverage, out, "--d-eval", "28")) == 0

    split = json.loads((out / "split.json").read_text())
    d_disc, d_hold = split_lengths(28)
    assert (split["d_eval"], split["d_disc"], split["d_hold"]) == (28, d_disc, d_hold)
    assert len(split["discovery_days"]) == 18
    assert len(split["holdout_days"]) == 10
    assert split["boundary_event_day"] == (LONG_DAYS[0] + timedelta(days=18)).isoformat()


def test_a_both_ladder_run_freezes_the_low_roots_its_coverage_carries(tmp_path: Path) -> None:
    rows = [*coverage_rows()]
    for day in LONG_DAYS[:17]:
        rows.append(
            {
                "ticker": f"KXLOWTDEN-{day.strftime('%y%b%d').upper()}-T60",
                "rows": 4,
                "first_received_at": datetime(day.year, day.month, day.day, 9, tzinfo=UTC),
                "last_received_at": datetime(day.year, day.month, day.day, 10, tzinfo=UTC),
            }
        )
    blind = write_blind(tmp_path / "blind.parquet", BLIND_ROWS)
    coverage = write_coverage(tmp_path / "coverage.parquet", rows)
    out = tmp_path / "scope"

    assert run(args_for(blind, coverage, out, "--cohort", "both")) == 0

    split = json.loads((out / "split.json").read_text())
    assert split["cities"] == ["KXHIGHNY", "KXHIGHTSFO", "KXLOWTDEN"]
    assert json.loads((out / "summary.json").read_text())["cohort"] == "both"


def test_the_parser_requires_both_inputs_the_output_and_a_named_start(tmp_path: Path) -> None:
    args = args_for(tmp_path / "b.parquet", tmp_path / "c.parquet", tmp_path / "scope")

    assert args.blind_windows == tmp_path / "b.parquet"
    assert args.coverage == tmp_path / "c.parquet"
    assert args.out == tmp_path / "scope"
    assert args.start == datetime(2026, 7, 19, tzinfo=UTC)
    assert args.d_eval == D_EVAL
    assert args.cohort == "high"
    assert args.incident == PADDED_INCIDENT
    assert sorted(STATION_SETS) == ["both", "high", "low"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--blind-windows",
                "b.parquet",
                "--coverage",
                "c.parquet",
                "--out",
                str(tmp_path),
            ]
        )
