import argparse
import json
from pathlib import Path

import pytest

from bot.backtest.historical_open_meteo import CALIBRATION_PATH
from bot.lag.forecast_class_run import (
    ALPHA_F4,
    BLEND_LABEL,
    BOOTSTRAP_SEED,
    CLASS_REPORTED_ONLY,
    DISCOVERY_N_MIN,
    GATING_LEAD,
    LEAD_REPORTED_ONLY,
    RESULTS_NAME,
    SIGMA_BAND_REPORTED_ONLY,
    UNDERPOWERED,
)
from bot.lag.forecast_entry import SCREEN_RULE, SIZE, TICK_RULE, WALKED_TICK_REPORTED_ONLY
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME
from bot.lag.tape_studies import SELF_CHARGED_BAR_SOURCE
from bot.replay.analysis_stations import HIGH, LOW
from bot.replay.run_scope import DISCOVERY, HOLDOUT
from scripts.f4_report import (
    DEFAULT_INPUTS,
    DEFAULT_MARKETS,
    DEFAULT_PREREGISTRATION,
    DEFAULT_RUN_ROOT,
    DEFAULT_SAMPLE,
    REPO_ROOT,
    build_parser,
    format_report,
    run,
)
from tests.test_forecast_class_run import (
    ALL_MEMBERS,
    BROKEN_DAY,
    RUN_ID,
    write_corpus,
)
from tests.test_tape_studies import argument_flags


SCRIPT = REPO_ROOT / "scripts" / "f4_report.py"
FORBIDDEN = ("bar", "alpha", "size", "seed")
SECTIONS = (
    "== BLEND",
    "== DISCOVERY",
    "== HOLDOUT",
    "== GATE",
    "== REPLICATION",
    "== CLASSES (reported only, gates nothing)",
    "== LEAD 36H (reported only, gates nothing)",
    "== WALKED TICK (reported only, gates nothing)",
    "== SIGMA BAND (reported only, gates nothing)",
    "== DEPTH",
    "== SCREEN",
    "== ENTRIES",
    "== LADDER SUMS",
    "== COVERAGE",
)
REPORTED = (
    ("blend", "sha256"),
    ("blend", "fitted_on_event_days"),
    ("gate", "passed"),
    ("screen", "candidates"),
    ("entries", "not_blendable_legs"),
    ("ladder_sums", "failures"),
    ("sigma_band", "bands"),
    ("depth", "kept"),
    ("lead_36h", "classes"),
    ("walked_tick", "tick_rule"),
    ("briers", BLEND_LABEL),
)


def argv_for(paths: dict[str, Path], run_root: Path) -> list[str]:
    return [
        "--run-id",
        RUN_ID,
        "--preregistration",
        str(paths["preregistration"]),
        "--sample",
        str(paths["sample"]),
        "--classes",
        str(paths["classes"]),
        "--markets",
        str(paths["markets"]),
        "--calibration",
        str(paths["calibration"]),
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


@pytest.fixture(scope="module")
def reported(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("f4-report")
    paths = write_corpus(root, broken=BROKEN_DAY)
    run_root = root / "tape_studies"
    assert run(args_for(paths, run_root)) == 0
    return {
        "root": root,
        "paths": paths,
        "run_root": run_root,
        "results": json.loads((run_root / RUN_ID / RESULTS_NAME).read_text()),
    }


def test_a_complete_run_writes_the_manifest_and_the_results_beside_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = write_corpus(tmp_path)
    run_root = tmp_path / "tape_studies"

    assert run(args_for(paths, run_root)) == 0

    results = json.loads((run_root / RUN_ID / RESULTS_NAME).read_text())
    manifest = json.loads((run_root / RUN_ID / MANIFEST_NAME).read_text())
    assert results["manifest_sha256"] == manifest["sha256"]
    assert results["bootstrap_seed"] == BOOTSTRAP_SEED
    assert results["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert manifest["bootstrap_seed"] == BOOTSTRAP_SEED
    assert manifest["cohort"] == HIGH
    assert capsys.readouterr().out == format_report(results) + "\n"


def test_the_results_carry_the_readout_the_question_asked_for(reported: dict) -> None:
    results = reported["results"]

    assert results["verdict"] == UNDERPOWERED
    assert results["bar"] == "0"
    assert results["bar_source"] == SELF_CHARGED_BAR_SOURCE
    assert results["bar_is_strict"] is True
    assert results["alpha"] == ALPHA_F4
    assert results["size"] == str(SIZE)
    assert results["screen_rule"] == SCREEN_RULE
    assert results["tick_rule"] == TICK_RULE
    assert results["gating_lead_hours"] == GATING_LEAD
    assert results["discovery"]["split"] == DISCOVERY
    assert results["holdout"]["split"] == HOLDOUT
    assert results["gate"]["n_min"] == DISCOVERY_N_MIN
    assert results["gate"]["powered"] is False


def test_the_report_opens_on_the_verdict_and_lays_its_sections_out_in_order(
    reported: dict,
) -> None:
    lines = format_report(reported["results"]).splitlines()
    at = [lines.index(name) for name in SECTIONS]

    assert lines[0] == f"== F4 FORECAST CLASS  run_id={RUN_ID}  verdict={UNDERPOWERED}"
    assert at == sorted(at)


def test_the_report_carries_every_figure_the_question_turns_on(reported: dict) -> None:
    results = reported["results"]
    blend = results["blend"]
    report = format_report(results)

    assert f"blend_fitted_on_event_days={blend['fitted_on_event_days']}" in report
    assert f"blend_weights_sha256={blend['sha256']}" in report
    assert f"estimate={results['gate']['estimate']}" in report
    assert f"rule={SCREEN_RULE}" in report
    assert f"screen_rule={SCREEN_RULE}" in report
    assert f"tick_rule={TICK_RULE}" in report
    assert LEAD_REPORTED_ONLY in report
    assert CLASS_REPORTED_ONLY in report
    assert SIGMA_BAND_REPORTED_ONLY in report
    assert WALKED_TICK_REPORTED_ONLY in report
    for member in ALL_MEMBERS:
        brier = results["briers"][member]
        assert f"brier={brier['brier']}" in report
        assert f"baseline_brier={brier['baseline_brier']}" in report
    for multiplier in ("0.5x=", "1x=", "2x="):
        assert multiplier in report
    assert "candidates prints_p10=" in report
    assert "kept prints_p10=" in report
    assert results["ladder_sums"]["failed"] > 0
    for failure in results["ladder_sums"]["failures"]:
        assert failure in report


@pytest.mark.parametrize(("block", "key"), REPORTED)
def test_a_report_missing_one_of_those_figures_cannot_be_written(
    reported: dict, block: str, key: str
) -> None:
    results = json.loads(json.dumps(reported["results"]))
    del results[block][key]

    with pytest.raises(KeyError, match=key):
        format_report(results)


def test_the_bar_the_alpha_the_size_and_the_seed_are_not_command_line_inputs() -> None:
    flags = argument_flags(SCRIPT.read_text())

    assert flags
    for forbidden in FORBIDDEN:
        assert [flag for flag in flags if forbidden in flag] == []
    assert "sample" in flags
    assert "classes" in flags
    assert "markets" in flags
    assert "calibration" in flags


def test_the_run_id_is_the_one_input_with_no_default() -> None:
    with pytest.raises(SystemExit) as refused:
        build_parser().parse_args([])

    assert refused.value.code != 0


def test_the_defaults_name_the_tree_the_script_ships_in() -> None:
    args = build_parser().parse_args(["--run-id", RUN_ID])

    assert args.run_root == DEFAULT_RUN_ROOT
    assert args.repo == REPO_ROOT
    assert args.sample == DEFAULT_SAMPLE
    assert args.classes == DEFAULT_INPUTS
    assert args.markets == DEFAULT_MARKETS
    assert args.calibration == CALIBRATION_PATH
    assert args.cohort == HIGH
    # The manifest records the path it was handed, so an absolute default would put this host's
    # checkout inside every digest.
    assert args.preregistration == DEFAULT_PREREGISTRATION
    assert not DEFAULT_PREREGISTRATION.is_absolute()


def test_the_run_reads_the_high_ladder_unless_it_is_told_otherwise() -> None:
    argv = ["--run-id", RUN_ID]

    assert build_parser().parse_args(argv).cohort == HIGH
    assert build_parser().parse_args([*argv, "--cohort", LOW]).cohort == LOW
    with pytest.raises(SystemExit) as refused:
        build_parser().parse_args([*argv, "--cohort", "both"])
    assert refused.value.code != 0


def test_a_run_whose_preregistration_is_missing_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = write_corpus(tmp_path)
    run_root = tmp_path / "tape_studies"
    argv = [*without(argv_for(paths, run_root), "--preregistration")]

    assert run(build_parser().parse_args(argv)) != 0

    captured = capsys.readouterr()
    assert "preregistration_sha256" in captured.err
    assert captured.out == ""
    assert not run_root.exists()
