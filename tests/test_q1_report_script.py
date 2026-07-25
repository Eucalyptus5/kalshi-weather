import argparse
import json
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.ladder_consistency import (
    CITY_DAY_MIN_DISCOVERY,
    CITY_DAY_MIN_HOLDOUT,
    DEPTH_MIN,
    EXCESS_BAR,
    MONOTONICITY,
    SUM_BUY,
    SUM_SELL,
)
from bot.lag.ladder_run import CLOSED, RESULTS_NAME
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME
from bot.replay.run_scope import DISCOVERY, HOLDOUT, RESUBSCRIBE_BLIND
from scripts.q1_report import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    build_parser,
    format_report,
    run,
)
from tests.test_ladder_run import (
    DISCOVERY_DAY,
    EXCESS,
    HOLDOUT_DAY,
    SEED,
    SERIES,
    artifacts_dir,
    excluded_artifacts,
    five_leg_artifacts,
    quiet_artifacts,
    scope_dir,
)
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    SHORT_SAMPLES,
    seeded_repo,
    write_preregistration,
    write_rtt_samples,
)


RUN_ID = "2026-08-13-q1"
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

    manifest = json.loads((run_root / RUN_ID / MANIFEST_NAME).read_text())
    results = results_of(run_root)
    assert results["manifest_sha256"] == manifest["sha256"]
    assert results["bootstrap_seed"] == SEED
    assert results["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert results["t_persist_s"] == str(manifest["t_persist_s"])
    assert results["latency_floor_source"] == manifest["latency_floor_source"]
    assert capsys.readouterr().out == format_report(results) + "\n"


def test_the_results_carry_the_readout_the_question_asked_for(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["verdict"] == CLOSED
    assert results["excess_bar"] == str(EXCESS_BAR)
    assert results["depth_min"] == str(DEPTH_MIN)
    assert results["universe"] == {"read": "recorded", "series": [SERIES]}
    assert Decimal(results["discovery"]["median_excess_cents"]) == EXCESS
    assert results["discovery"]["split"] == DISCOVERY
    assert results["discovery"]["n_city_days"] == 1
    assert results["discovery"]["population"] == 1
    assert results["discovery"]["episodes"] == 1
    assert Decimal(results["holdout"]["median_excess_cents"]) == EXCESS
    assert results["holdout"]["split"] == HOLDOUT
    assert results["gate"]["economic"] is True
    assert results["gate"]["powered"] is False
    assert results["gate"]["n_min"] == CITY_DAY_MIN_DISCOVERY
    assert results["gate"]["n_unit"] == "city event-days"
    assert results["replication"]["holdout_n_min"] == CITY_DAY_MIN_HOLDOUT
    assert results["replication"]["replicated"] is False
    assert results["rows"] == 16
    assert results["cities"] == [SERIES]
    assert results["tickers_per_city_day"] == {
        f"{SERIES} {DISCOVERY_DAY.isoformat()}": 6,
        f"{SERIES} {HOLDOUT_DAY.isoformat()}": 6,
    }
    assert results["ladders"] == {
        "in_scope": 2,
        "complete": 2,
        "incomplete": 0,
        "incomplete_keys": [],
    }


def test_a_rare_edge_closes_the_question_without_calling_it_underpowered(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["discovery"]["n_city_days"] < CITY_DAY_MIN_DISCOVERY
    assert results["verdict"] == CLOSED
    assert "UNDERPOWERED" not in json.dumps(results)
    assert "UNDERPOWERED" not in format_report(results)


def test_the_episode_counts_and_distributions_read_per_stream_and_pooled(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["episodes"]["pooled"] == {
        "found": 2,
        "tradeable": 2,
        "kept": 2,
        "censored": 0,
        "incomplete_states": 30,
    }
    assert results["episodes"][MONOTONICITY]["found"] == 2
    assert results["episodes"][SUM_BUY]["found"] == 0
    assert results["episodes"][SUM_SELL]["found"] == 0
    every = results["distributions"]["all"]
    kept = results["distributions"]["kept_tradeable"]
    assert every["pooled"]["magnitude_cents"] == {
        "count": 2,
        "min": "10",
        "p25": "10",
        "median": "10",
        "p75": "10",
        "p90": "10",
        "max": "10",
    }
    assert every["pooled"]["depth_contracts"]["median"] == "100"
    assert every["pooled"]["duration_s"]["median"] == "60"
    assert kept[MONOTONICITY]["magnitude_cents"]["count"] == 2
    assert kept[SUM_BUY]["magnitude_cents"] == {
        "count": 0,
        "min": None,
        "p25": None,
        "median": None,
        "p75": None,
        "p90": None,
        "max": None,
    }


def test_the_report_leads_with_the_cent_figure_and_its_interval(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    lines = format_report(results).splitlines()
    assert lines[0].startswith(f"== Q1 LADDER CONSISTENCY  run_id={RUN_ID}")
    assert lines[1].startswith(f"median_excess_cents={results['discovery']['median_excess_cents']}")
    assert "ci0.95=[7.1200, 7.1200]" in lines[1]
    assert "n=1 city event-days" in lines[1]
    assert lines.index("== GATE") < lines.index("== EPISODES")


def test_a_run_with_no_tradeable_episode_still_writes_a_verdict(
    paths: dict[str, Path], run_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths["artifacts"] = quiet_artifacts(tmp_path)

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["verdict"] == CLOSED
    assert results["gate"] is None
    assert results["replication"] is None
    assert results["replication_skipped"]
    assert results["discovery"]["median_excess_cents"] is None
    assert results["discovery"]["p_value"] is None
    assert results["discovery"]["n_city_days"] == 0
    assert results["exclusions"]["excluded_fraction"] is None
    assert capsys.readouterr().out == format_report(results) + "\n"


def test_an_excluded_episode_is_reported_against_its_class(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["artifacts"] = excluded_artifacts(tmp_path)

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["exclusions"]["candidates"] == 1
    assert results["exclusions"]["excluded"] == 1
    assert results["exclusions"]["by_class"][RESUBSCRIBE_BLIND] == 1
    assert Decimal(results["exclusions"]["excluded_fraction"]) == Decimal("1")
    assert results["discovery"]["n_city_days"] == 0


def test_an_incomplete_ladder_is_named_in_the_results(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["artifacts"] = five_leg_artifacts(tmp_path)

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["ladders"] == {
        "in_scope": 2,
        "complete": 1,
        "incomplete": 1,
        "incomplete_keys": [f"{SERIES} {DISCOVERY_DAY.isoformat()}"],
    }
    assert f"incomplete {SERIES} {DISCOVERY_DAY.isoformat()}" in format_report(results)


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
