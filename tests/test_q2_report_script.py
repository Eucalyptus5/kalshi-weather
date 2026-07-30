import argparse
import json
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.lead_lag import CORRIDORS, PAIRS
from bot.lag.lead_lag_run import CI_LEVEL, CORRIDOR_DAY_MIN_REPORT, FORWARD, RESULTS_NAME, REVERSE
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME
from scripts.q2_report import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    build_parser,
    format_report,
    run,
)
from tests.test_lead_lag_run import (
    DISCOVERY_DAY,
    FLOOR_DAYS,
    LEAD_S,
    SEED,
    artifacts_dir,
    floor_artifacts,
    floor_scope_dir,
    scope_dir,
)
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    SHORT_SAMPLES,
    seeded_repo,
    write_preregistration,
    write_rtt_samples,
)


RUN_ID = "2026-08-13-q2"
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
    assert results["latency_floor_source"] == manifest["latency_floor_source"]
    assert capsys.readouterr().out == format_report(results) + "\n"


def test_the_report_says_plainly_that_it_carries_no_verdict(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)
    lines = report.splitlines()
    assert lines[0] == (f"== Q2 CROSS-CITY LEAD-LAG  run_id={RUN_ID}  reported only, no gate")
    assert "PASS" not in report
    assert "CLOSED" not in report
    assert "PASS" not in json.dumps(results)
    assert "CLOSED" not in json.dumps(results)


def test_a_reading_under_the_floor_declines_the_estimate(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)
    assert results[FORWARD]["corridor_days"] < CORRIDOR_DAY_MIN_REPORT
    assert results[FORWARD]["episodes"] == 1
    assert results[FORWARD]["median_lead_s"] is None
    assert results[FORWARD]["interval"] is None
    assert results[REVERSE]["median_lead_s"] is None
    assert (
        f"  corridor_days=1 of a ceiling of {len(CORRIDORS)} corridor-days, under the "
        f"reporting floor of {CORRIDOR_DAY_MIN_REPORT}: the estimate is declined"
    ) in report
    assert "median_lead_s=None" in report


def test_a_run_at_the_floor_prints_the_median_and_its_interval(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["run_scope"] = floor_scope_dir(tmp_path)
    paths["artifacts"] = floor_artifacts(tmp_path)

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = format_report(results)
    assert results["corridor_day_ceiling"] == len(CORRIDORS) * len(FLOOR_DAYS)
    assert results[FORWARD]["corridor_days"] == CORRIDOR_DAY_MIN_REPORT
    assert Decimal(results[FORWARD]["median_lead_s"]) == LEAD_S
    assert results[FORWARD]["interval"]["low"] is None
    assert results[FORWARD]["interval"]["high"] is None
    assert Decimal(results[REVERSE]["median_lead_s"]) == LEAD_S
    assert f"median_lead_s={results[FORWARD]['median_lead_s']}" in report
    assert f"ci{CI_LEVEL}=[none admitted, none admitted]" in report
    assert "declined" not in report
    assert f"    gulf {FLOOR_DAYS[0].isoformat()} 1" in report


def test_the_report_lays_out_every_pair_and_both_readings(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    lines = format_report(results_of(run_root)).splitlines()
    assert lines.index("== FORWARD") < lines.index("== REVERSE")
    assert lines.index("== REVERSE") < lines.index("== PAIRS")
    assert sum(1 for line in lines if line.startswith("  DEN->OKC gulf")) == 1
    assert sum(1 for line in lines if line.strip().startswith("DEN->OKC")) == 3
    assert len(PAIRS) == 14


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


def test_the_same_seed_reads_the_same_interval_twice(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    paths["run_scope"] = floor_scope_dir(tmp_path)
    paths["artifacts"] = floor_artifacts(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert run(args_for(paths, first)) == 0
    assert run(args_for(paths, second)) == 0

    assert results_of(first)[FORWARD]["interval"] == results_of(second)[FORWARD]["interval"]
    assert results_of(first)[REVERSE]["interval"] == results_of(second)[REVERSE]["interval"]


def test_the_atm_leg_and_the_rows_read_are_named_in_the_results(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["rows"] == 5
    assert results["city_event_days"] == 2
    assert results["no_atm_series"] == []
    assert list(results["atm_ticker_per_city_day"]) == [
        f"KXHIGHDEN {DISCOVERY_DAY.isoformat()}",
        f"KXHIGHTOKC {DISCOVERY_DAY.isoformat()}",
    ]


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
