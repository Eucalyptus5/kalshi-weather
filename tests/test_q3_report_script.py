import argparse
import json
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME
from bot.lag.taker_flow import HORIZONS_S, PRIMARY_HORIZON_S
from bot.lag.taker_flow_run import RESULTS_NAME, UNDERPOWERED
from bot.replay.run_scope import DISCOVERY, HOLDOUT, RESUBSCRIBE_BLIND
from scripts.q3_report import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    build_parser,
    format_report,
    run,
)
from tests.test_taker_flow_run import (
    DAY_TICKER,
    DISCOVERY_DAY,
    HOLDOUT_DAY,
    NEXT_TICKER,
    SEED,
    SERIES,
    artifacts_dir,
    scope_dir,
)
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    SHORT_SAMPLES,
    seeded_repo,
    write_preregistration,
    write_rtt_samples,
)


RUN_ID = "2026-08-12-q3"
REQUIRED = (
    "--run-id",
    "--preregistration",
    "--run-scope",
    "--artifacts",
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
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "artifacts": artifacts_dir(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
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

    assert (run_root / RUN_ID / MANIFEST_NAME).exists()
    results = results_of(run_root)
    manifest = json.loads((run_root / RUN_ID / MANIFEST_NAME).read_text())
    assert results["manifest_sha256"] == manifest["sha256"]
    assert results["bootstrap_seed"] == SEED
    assert results["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert capsys.readouterr().out == format_report(results) + "\n"


def test_the_results_carry_the_readout_the_question_asked_for(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["verdict"] == UNDERPOWERED
    assert Decimal(results["discovery"]["mean_net_cents"]) == Decimal("1.4")
    assert results["discovery"]["split"] == DISCOVERY
    assert Decimal(results["holdout"]["mean_net_cents"]) == Decimal("1.4")
    assert results["holdout"]["split"] == HOLDOUT
    assert results["gate"]["economic"] is True
    assert results["gate"]["powered"] is False
    assert results["replication"]["replicated"] is False
    assert results["prints"] == {
        "in_scope_pooled": 6,
        "in_scope_discovery": 4,
        "in_scope_holdout": 2,
        "screened_pooled": 3,
        "screened_discovery": 2,
        "screened_holdout": 1,
        "empty_side": 1,
        "duplicate_trade_id": 1,
        "out_of_scope": 1,
        "out_of_window": 2,
    }
    assert results["exclusions"]["by_class"][RESUBSCRIBE_BLIND] == 1
    assert results["exclusions"]["excluded_fraction"] == "0.25"
    assert results["cities"] == [SERIES]
    assert results["tickers_per_city_day"] == {
        f"{SERIES} {DISCOVERY_DAY.isoformat()}": 1,
        f"{SERIES} {HOLDOUT_DAY.isoformat()}": 1,
    }
    assert DAY_TICKER.startswith(SERIES) and NEXT_TICKER.startswith(SERIES)


def test_the_curve_reports_every_frozen_horizon_on_discovery(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    curve = results_of(run_root)["horizon_curve"]
    assert [item["horizon_s"] for item in curve] == list(HORIZONS_S)
    assert {item["split"] for item in curve} == {DISCOVERY}
    assert [Decimal(item["mean_net_cents"]) for item in curve] == [
        Decimal("-2.6"),
        Decimal("-1.6"),
        Decimal("1.4"),
        Decimal("6.4"),
    ]
    primary = next(item for item in curve if item["horizon_s"] == PRIMARY_HORIZON_S)
    assert primary == results_of(run_root)["discovery"]


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
