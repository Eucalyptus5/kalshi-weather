import argparse
import json
from pathlib import Path

import pytest

from bot.lag.depth_map_run import ALL_LEGS, ATM, CONFIRMED, REFUTED, RESULTS_NAME
from bot.lag.run_manifest import MANIFEST_NAME
from scripts.depth_report import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    build_parser,
    format_report,
    run,
)
from tests.test_depth_map_run import (
    DEEP_ROWS,
    DEEP_TRADES,
    EVENT_DATE,
    LEG_A,
    SEED,
    SERIES,
    THIN_ROWS,
    ladder_artifacts,
    run_paths,
)
from tests.test_tape_studies import SHORT_SAMPLES, write_rtt_samples


RUN_ID = "2026-08-13-d2"
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
    return run_paths(tmp_path) | {
        "artifacts": ladder_artifacts(tmp_path, DEEP_ROWS, trades=DEEP_TRADES)
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


def test_the_report_leads_with_the_verdict_and_both_headline_figures(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    lines = format_report(results).splitlines()
    depth = results["headline"]["median_touch_depth_contracts"]
    assert lines[0] == f"== D2 LIQUIDITY AND DEPTH MAP  run_id={RUN_ID}  reported only, no gate"
    assert lines[3] == "== VERDICT"
    assert results["verdict"] == REFUTED
    assert lines[4] == f"  the small-capacity presumption is {REFUTED.upper()}"
    assert f"median touch depth yes={depth['yes']} no={depth['no']} contracts" in lines[5]
    assert "events=1 replenished=1" in lines[6]


def test_a_thin_book_reports_the_confirmed_verdict(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["artifacts"] = ladder_artifacts(tmp_path, THIN_ROWS, name="thin", trades=DEEP_TRADES)

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)
    assert results["verdict"] == CONFIRMED
    assert f"  the small-capacity presumption is {CONFIRMED.upper()}" in report
    assert f"  {SERIES} {EVENT_DATE.isoformat()} atm={LEG_A}" in report


def test_the_report_lays_out_both_universes_and_leaves_the_cube_in_the_results(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)
    lines = report.splitlines()
    all_legs = next(index for index, line in enumerate(lines) if line.startswith("== ALL LEGS"))
    atm = next(index for index, line in enumerate(lines) if line.startswith("== AT THE MONEY"))
    assert all_legs < atm < lines.index("== EXCLUSIONS") < lines.index("== COVERAGE")
    assert (
        lines[all_legs]
        == f"== ALL LEGS  cube cells={len(results[ALL_LEGS]['cube'])} in results.json"
    )
    assert results[ATM]["cube"]
    assert not any(key in report for key in results[ATM]["cube"])


def test_a_run_whose_floor_is_unavailable_writes_nothing(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_rtt_samples(paths["rtt_samples"], SHORT_SAMPLES)

    assert run(args_for(paths, run_root)) == 1

    captured = capsys.readouterr()
    assert "latency_floor" in captured.err
    assert captured.out == ""
    assert not run_root.exists()


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
