import argparse
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME
from bot.lag.settlement_price import SIZE
from bot.lag.settlement_run import (
    ALPHA_F2,
    BOOTSTRAP_SEED,
    DISCOVERY_N_MIN,
    RESULTS_NAME,
    UNDERPOWERED,
)
from bot.lag.tape_studies import SELF_CHARGED_BAR_SOURCE
from bot.replay.analysis_stations import HIGH, LOW
from bot.replay.run_scope import DISCOVERY, HOLDOUT
from scripts.f2_report import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    UNIDENTIFIABLE,
    build_parser,
    format_report,
    run,
)
from scripts.q4_report import write_settles_cache
from tests.test_settlement_run import (
    CROSSING_DAYS,
    ON_OR_AFTER,
    RUN_ID,
    SERIES,
    WINDOW,
    observations,
    run_paths,
    settles,
)
from tests.test_tape_studies import SHORT_SAMPLES, write_rtt_samples


SCRIPT = REPO_ROOT / "scripts" / "f2_report.py"
REQUIRED = (
    "--run-id",
    "--preregistration",
    "--run-scope",
    "--artifacts",
    "--closes",
    "--settlement-sources",
    "--observations",
    "--settles",
    "--rtt-samples",
    "--floor-source",
)
SECTIONS = (
    "== STRADDLES",
    "== ENTRIES",
    "== DELTAS",
    "== SETTLEMENT SOURCE",
    "== DISCOVERY",
    "== HOLDOUT",
    "== GATE",
    "== REPLICATION",
    "== SCREEN",
    "== COVERAGE",
)
FORBIDDEN = ("bar", "alpha", "size", "seed")
REPORTED = (
    ("deltas", "abs_delta_gt_1"),
    ("settlement_source", "days_on_or_after_boundary"),
    ("entries", "open_at_first_reading_n"),
    ("entries", "entry_at_close_n"),
)


def write_arrivals(path: Path, days: Sequence[date] = WINDOW) -> Path:
    rows = [
        {
            "station": item.station,
            "source": item.source,
            "obs_time": item.valid_time.isoformat(),
            "tmpf": str(item.temp_f),
            "received_at": item.publication_time.isoformat(),
        }
        for readings in observations(days).values()
        for item in readings
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def write_settles(path: Path, days: Sequence[date] = WINDOW) -> Path:
    write_settles_cache(path, settles(days))
    return path


def script_paths(tmp_path: Path) -> dict[str, Path]:
    return run_paths(tmp_path) | {
        "observations": write_arrivals(tmp_path / "arrivals.jsonl"),
        "settles": write_settles(tmp_path / "acis_settles.json"),
    }


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
        "--closes",
        str(paths["closes"]),
        "--settlement-sources",
        str(paths["settlement_sources"]),
        "--observations",
        str(paths["observations"]),
        "--settles",
        str(paths["settles"]),
        "--rtt-samples",
        str(paths["rtt_samples"]),
        "--floor-source",
        "RTT_read",
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
    return script_paths(tmp_path)


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "tape_studies"


def results_of(run_root: Path) -> dict:
    return json.loads((run_root / RUN_ID / RESULTS_NAME).read_text())


def test_a_complete_run_writes_the_manifest_and_the_results_beside_it(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    manifest = json.loads((run_root / RUN_ID / MANIFEST_NAME).read_text())
    assert results["manifest_sha256"] == manifest["sha256"]
    assert results["bootstrap_seed"] == BOOTSTRAP_SEED
    assert results["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert manifest["cohort"] == HIGH
    assert capsys.readouterr().out == format_report(results) + "\n"


def test_the_results_carry_the_readout_the_question_asked_for(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["verdict"] == UNDERPOWERED
    assert results["bar"] == "0"
    assert results["bar_source"] == SELF_CHARGED_BAR_SOURCE
    assert results["bar_is_strict"] is True
    assert results["alpha"] == ALPHA_F2
    assert results["size"] == str(SIZE)
    assert results["discovery"]["split"] == DISCOVERY
    assert results["holdout"]["split"] == HOLDOUT
    assert results["gate"]["n_min"] == DISCOVERY_N_MIN
    assert results["gate"]["powered"] is False
    assert results["cities"] == [SERIES]


def test_the_report_opens_on_the_verdict_and_lays_its_sections_out_in_order(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    lines = format_report(results_of(run_root)).splitlines()
    at = [lines.index(name) for name in SECTIONS]

    assert lines[0] == f"== F2 SETTLEMENT LAG  run_id={RUN_ID}  verdict={UNDERPOWERED}"
    assert at == sorted(at)


def test_the_report_carries_every_figure_the_question_turns_on(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)

    assert f"abs_delta_gt_1={results['deltas']['abs_delta_gt_1']}" in report
    assert f"days_on_or_after_boundary={ON_OR_AFTER}" in report
    assert f"open_at_first_reading_n={results['entries']['open_at_first_reading_n']}" in report
    assert f"entry_at_close_n={results['entries']['entry_at_close_n']}" in report
    assert f"n={CROSSING_DAYS} " in report


@pytest.mark.parametrize(("block", "key"), REPORTED)
def test_a_report_missing_one_of_those_figures_cannot_be_written(
    paths: dict[str, Path], run_root: Path, block: str, key: str
) -> None:
    assert run(args_for(paths, run_root)) == 0
    results = results_of(run_root)
    del results[block][key]

    with pytest.raises(KeyError, match=key):
        format_report(results)


def test_the_report_states_the_unidentifiable_instant_once(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)

    assert results["identifiable_ex_ante"] is False
    assert report.count("identifiable_ex_ante") == 1
    assert f"identifiable_ex_ante=False: {UNIDENTIFIABLE}" in report


def test_the_bar_the_alpha_the_size_and_the_seed_are_not_command_line_inputs() -> None:
    parser = build_parser()
    named = {action.dest for action in parser._actions} | {
        option for action in parser._actions for option in action.option_strings
    }

    assert named
    for forbidden in FORBIDDEN:
        assert [name for name in named if forbidden in name] == []
    assert "closes" in named
    assert "settlement_sources" in named


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


def test_the_run_reads_the_high_ladder_unless_it_is_told_otherwise(
    paths: dict[str, Path], run_root: Path
) -> None:
    argv = argv_for(paths, run_root)

    assert build_parser().parse_args(argv).cohort == HIGH
    assert build_parser().parse_args([*argv, "--cohort", LOW]).cohort == LOW
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([*argv, "--cohort", "both"])
    assert excinfo.value.code != 0


def test_a_run_whose_floor_is_unavailable_writes_nothing(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_rtt_samples(paths["rtt_samples"], SHORT_SAMPLES)

    assert run(args_for(paths, run_root)) != 0

    captured = capsys.readouterr()
    assert "latency_floor" in captured.err
    assert captured.out == ""
    assert not run_root.exists()
