import argparse
import ast
import json
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.fee_floor import (
    MAKER_RATE_SOURCE as PUBLISHED_MAKER_RATE_SOURCE,
    PUBLISHED_MAKER_RATE,
)
from bot.lag.maker_edge import HORIZONS_S, PRIMARY_HORIZON_S
from bot.lag.maker_edge_run import (
    ALPHA,
    MAKER_RATE,
    MAKER_RATE_SOURCE,
    NO_ESTIMATE,
    NO_GATE_ESTIMATE,
    RESULTS_NAME,
    UNDERPOWERED,
)
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME
from bot.lag.tape_studies import SELF_CHARGED_BAR_SOURCE
from bot.replay.analysis_stations import HIGH, LOW
from bot.replay.run_scope import DISCOVERY, HOLDOUT, QUIET_BAND, RECORDED_GAP
from scripts.f1_report import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    build_parser,
    format_report,
    run,
)
from tests.test_maker_edge_run import (
    DISCOVERY_DAY,
    HOLDOUT_BAND,
    HOLDOUT_DAY,
    LOW_SERIES,
    MARK_BAND,
    SEED,
    SERIES,
    T62,
    T64,
    artifacts_dir,
    closes_dir,
    run_paths,
    scope_dir,
)
from tests.test_tape_studies import SHORT_SAMPLES, argument_flags, write_rtt_samples


RUN_ID = "2026-08-19-f1"
SCRIPT = REPO_ROOT / "scripts" / "f1_report.py"
REQUIRED = (
    "--run-id",
    "--preregistration",
    "--run-scope",
    "--artifacts",
    "--closes",
    "--rtt-samples",
    "--floor-source",
    "--seed",
)
SECTIONS = (
    "== FILLS",
    "== DISCOVERY (primary)",
    "== HOLDOUT (primary)",
    "== GATE",
    "== REPLICATION",
    "== HORIZON CURVE (discovery)",
    "== PUBLISHED-RATE SENSITIVITY (reported, not gating)",
    "== EXCLUSIONS",
    "== COVERAGE",
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
        "--closes",
        str(paths["closes"]),
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
    return run_paths(tmp_path)


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
    assert results["bootstrap_seed"] == SEED
    assert results["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert manifest["fee_maker_rate"] == str(MAKER_RATE)
    assert manifest["fee_maker_rate"] != str(PUBLISHED_MAKER_RATE)
    assert manifest["fee_maker_rate_source"] == MAKER_RATE_SOURCE
    assert manifest["fee_maker_rate_source"] != PUBLISHED_MAKER_RATE_SOURCE
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
    assert results["alpha"] == ALPHA
    assert results["discovery"]["split"] == DISCOVERY
    assert results["discovery"]["market_days"] == 2
    assert results["holdout"]["split"] == HOLDOUT
    assert results["holdout"]["market_days"] == 1
    assert results["gate"]["threshold"] == "0"
    assert results["gate"]["n_min"] == 200
    assert results["gate"]["powered"] is False
    assert results["fills"]["yes_fills"] == 3
    assert results["fills"]["no_fills"] == 3
    assert results["cities"] == [SERIES]
    assert results["markets_per_city_day"] == {
        f"{SERIES} {DISCOVERY_DAY.isoformat()}": 2,
        f"{SERIES} {HOLDOUT_DAY.isoformat()}": 1,
    }
    assert T62.startswith(SERIES) and T64.startswith(SERIES)


def test_the_report_opens_on_the_verdict_and_lays_its_sections_out_in_order(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    lines = format_report(results_of(run_root)).splitlines()
    at = [lines.index(name) for name in SECTIONS]

    assert lines[0] == f"== F1 MAKER EDGE  run_id={RUN_ID}  verdict={UNDERPOWERED}"
    assert at == sorted(at)


def test_the_gate_block_reads_the_gates_own_estimate_and_its_predicates(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    gate = results["gate"]
    lines = format_report(results).splitlines()
    at = lines.index("== GATE")

    assert lines[at + 1].startswith(f"  estimate={gate['estimate']}  threshold={gate['threshold']}")
    assert f"passed={gate['passed']}" in lines[at + 2]


def test_a_gate_with_no_estimate_reads_its_own_reason_and_not_the_replications(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["run_scope"] = scope_dir(
        tmp_path, bands=((QUIET_BAND, *MARK_BAND), (RECORDED_GAP, *HOLDOUT_BAND)), name="banded"
    )

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    lines = format_report(results).splitlines()
    at = lines.index("== GATE")
    replication_at = lines.index("== REPLICATION")

    assert results["gate"] is None
    assert results["replication_skipped"] == NO_ESTIMATE
    assert lines[at + 1] == f"  not evaluated: {NO_GATE_ESTIMATE}"
    assert "replicate" not in lines[at + 1]
    assert lines[replication_at + 1] == f"  not evaluated: {NO_ESTIMATE}"


def test_each_split_block_reads_its_own_split_and_prints_in_fixed_point(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    discovery = results["discovery"]
    lines = format_report(results).splitlines()
    at = lines.index("== DISCOVERY (primary)")
    holdout_at = lines.index("== HOLDOUT (primary)")

    assert discovery["market_days"] != results["holdout"]["market_days"]
    assert f"market_days={discovery['market_days']}" in lines[at + 1]
    assert f"market_days={results['holdout']['market_days']}" in lines[holdout_at + 1]
    assert str(discovery["p_value"]) != f"{discovery['p_value']:.5f}"
    assert f"p_value={discovery['p_value']:.5f}" in lines[at + 1]
    assert f"degenerate={discovery['degenerate']}" in lines[at + 2]


def test_the_published_rate_prints_beside_the_gating_figure_without_gating(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    sensitivity = results["published_rate_sensitivity"]
    report = format_report(results)
    lines = report.splitlines()
    at = lines.index("== PUBLISHED-RATE SENSITIVITY (reported, not gating)")

    assert sensitivity["gating"] is False
    assert sensitivity["rate"] == str(PUBLISHED_MAKER_RATE)
    assert sensitivity["rate"] != results["maker_rate"]
    assert sensitivity["horizon_s"] == PRIMARY_HORIZON_S
    assert Decimal(sensitivity["discovery_edge_cents_per_contract"]) < Decimal(
        results["discovery"]["edge_cents_per_contract"]
    )
    assert lines[at + 1].startswith(f"  rate={sensitivity['rate']} ")
    assert next(line for line in lines if line.startswith("bar=")).endswith(
        f"maker_rate={results['maker_rate']}"
    )
    assert "not gating" in report


def test_the_curve_reports_every_frozen_horizon_on_discovery(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    curve = results["horizon_curve"]
    assert [item["horizon_s"] for item in curve] == list(HORIZONS_S)
    assert [item["window_cap_s"] for item in curve] == [301, 310, 360, 600]
    assert {item["split"] for item in curve} == {DISCOVERY}
    assert (
        next(item for item in curve if item["horizon_s"] == PRIMARY_HORIZON_S)
        == (results["discovery"])
    )


def test_a_run_whose_books_never_two_side_reports_its_drops(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["artifacts"] = artifacts_dir(tmp_path, one_sided=True, name="one_sided")

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    far = next(item for item in results["horizon_curve"] if item["horizon_s"] == 300)
    assert results["discovery"]["no_mid_drops"] == 0
    assert Decimal(results["discovery"]["no_mid_fraction"]) == 0
    assert far["no_mid_drops"] == 4
    assert Decimal(far["no_mid_fraction"]) == 1
    assert far["edge_cents_per_contract"] is None
    assert far["market_days"] == 0
    assert "no_mid_drops=4" in format_report(results)


def test_a_run_whose_floor_is_unavailable_writes_nothing(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_rtt_samples(paths["rtt_samples"], SHORT_SAMPLES)

    assert run(args_for(paths, run_root)) != 0

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


def test_the_same_seed_reads_the_same_p_value_twice(paths: dict[str, Path], tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert run(args_for(paths, first)) == 0
    assert run(args_for(paths, second)) == 0

    assert results_of(first)["discovery"]["p_value"] == results_of(second)["discovery"]["p_value"]
    assert results_of(first)["holdout"]["p_value"] == results_of(second)["holdout"]["p_value"]


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


def test_naming_the_other_ladder_refuses_rather_than_sweeping_nothing(
    paths: dict[str, Path], run_root: Path
) -> None:
    named = build_parser().parse_args([*argv_for(paths, run_root), "--cohort", LOW])

    with pytest.raises(ValueError, match="holds no low series"):
        run(named)

    assert not run_root.exists()


def test_a_two_ladder_scope_is_read_under_the_cohort_the_run_names(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["run_scope"] = scope_dir(tmp_path, series=(SERIES, LOW_SERIES), name="both")
    paths["closes"] = closes_dir(tmp_path, roots=(SERIES, LOW_SERIES))
    args = args_for(paths, run_root)

    assert args.cohort == HIGH
    assert run(args) == 0
    assert results_of(run_root)["cities"] == [SERIES]


def test_the_bar_and_the_regime_are_not_command_line_inputs() -> None:
    flags = argument_flags(SCRIPT.read_text())

    assert flags
    assert [flag for flag in flags if "bar" in flag] == []
    assert [flag for flag in flags if "rate" in flag] == []
    assert "closes" in flags
    assert isinstance(ast.parse(SCRIPT.read_text()), ast.Module)
