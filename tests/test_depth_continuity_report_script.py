import argparse
import json
from pathlib import Path

import pytest

from bot.lag.depth_continuity import RESULTS_NAME, SAMPLE_MAX
from bot.lag.run_manifest import MANIFEST_NAME
from scripts.depth_continuity_report import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    build_parser,
    format_report,
    run,
)
from tests.test_depth_continuity import (
    AGREEING_ROWS,
    AGREEING_SNAPSHOTS,
    LEG_A,
    SEED,
    SERIES,
    ladder_artifacts,
    run_paths,
    snapshot,
    snapshots_db,
    stamp,
)
from tests.test_tape_studies import SHORT_SAMPLES, write_rtt_samples


RUN_ID = "2026-08-13-d3"
REQUIRED = (
    "--run-id",
    "--preregistration",
    "--run-scope",
    "--artifacts",
    "--db",
    "--rtt-samples",
    "--floor-source",
    "--seed",
)


def argv_for(paths: dict[str, Path], run_root: Path) -> list[str]:
    return [
        "--run-id",
        RUN_ID,
        "--preregistration",
        str(paths["preregistration"]),
        "--run-scope",
        str(paths["run_scope"]),
        "--artifacts",
        str(paths["artifacts"]),
        "--db",
        str(paths["db"]),
        "--rtt-samples",
        str(paths["rtt_samples"]),
        "--floor-source",
        "RTT_read",
        "--seed",
        str(SEED),
        "--run-root",
        str(run_root),
        "--repo",
        str(paths["repo"]),
    ]


def args_for(paths: dict[str, Path], run_root: Path) -> argparse.Namespace:
    return build_parser().parse_args(argv_for(paths, run_root))


def without(argv: list[str], flag: str) -> list[str]:
    index = argv.index(flag)
    return argv[:index] + argv[index + 2 :]


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return run_paths(tmp_path, AGREEING_SNAPSHOTS) | {
        "artifacts": ladder_artifacts(tmp_path, AGREEING_ROWS)
    }


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "tape_studies"


def results_of(run_root: Path) -> dict:
    return json.loads((run_root / RUN_ID / RESULTS_NAME).read_text())


def test_a_complete_run_writes_the_manifest_and_the_results_beside_it(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(args_for(paths, run_root)) == 0

    manifest = json.loads((run_root / RUN_ID / MANIFEST_NAME).read_text())
    results = results_of(run_root)
    assert results["manifest_sha256"] == manifest["sha256"]
    assert results["latency_floor_source"] == manifest["latency_floor_source"]
    assert results["run_id"] == RUN_ID
    assert results["elapsed_s"] >= 0
    assert capsys.readouterr().out == format_report(results) + "\n"


def test_the_report_leads_with_the_row_accounting(paths: dict[str, Path], run_root: Path) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    lines = format_report(results).splitlines()
    assert lines[0] == f"== D3 REST CONTINUITY CHECK  run_id={RUN_ID}  reported only, no gate"
    assert lines[3] == "== ROWS"
    assert lines[4] == (
        "  returned=1 era_dropped=0 out_of_window=0 excluded=0 no_coverage=0 null_depth=0 "
        "compared=1"
    )
    assert lines.index("== ROWS") < lines.index("== POOLED") < lines.index("== BY CITY")
    assert lines.index("== BY CITY") < lines.index(f"== DISAGREEMENTS  0 of at most {SAMPLE_MAX}")


def test_the_report_carries_no_gate_and_no_interval(paths: dict[str, Path], run_root: Path) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)
    assert "  this reading carries no verdict and evaluates no threshold" in report
    assert "PASS" not in report
    assert "CLOSED" not in report
    assert "ci_level" not in json.dumps(results)


def test_a_disagreeing_snapshot_reaches_the_printed_sample(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["db"] = snapshots_db(tmp_path, [snapshot(LEG_A, stamp(11), yes_depth=9)], name="off.db")

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)
    assert results["pooled"]["depth_agree_fraction"] == "0"
    assert results["by_city"][SERIES]["depth_agree_truncated_fraction"] == "0"
    assert len(results["disagreements"]) == 1
    assert f"  {LEG_A} " in report
    assert "rest yes=0.4/9" in report
    assert "ws yes=0.4/5" in report


def test_a_run_whose_floor_is_unavailable_writes_nothing(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_rtt_samples(paths["rtt_samples"], SHORT_SAMPLES)

    assert run(args_for(paths, run_root)) == 1

    captured = capsys.readouterr()
    assert "latency_floor" in captured.err
    assert captured.out == ""
    assert not run_root.exists()


def test_a_second_run_under_the_same_id_refuses_to_overwrite(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0
    digest = results_of(run_root)["manifest_sha256"]

    with pytest.raises(FileExistsError, match=MANIFEST_NAME):
        run(args_for(paths, run_root))

    assert results_of(run_root)["manifest_sha256"] == digest


@pytest.mark.parametrize("flag", REQUIRED)
def test_every_input_the_run_is_read_under_is_required(
    paths: dict[str, Path], run_root: Path, flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(without(argv_for(paths, run_root), flag))

    assert excinfo.value.code != 0
    assert flag in capsys.readouterr().err


def test_the_run_root_and_repo_default_to_the_tree_the_script_ships_in(
    paths: dict[str, Path], run_root: Path
) -> None:
    argv = without(without(argv_for(paths, run_root), "--run-root"), "--repo")

    args = build_parser().parse_args(argv)

    assert args.run_root == DEFAULT_RUN_ROOT
    assert args.repo == REPO_ROOT
    assert FLOOR_SOURCES == ("L", "RTT_read")
