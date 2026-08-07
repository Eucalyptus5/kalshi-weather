import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from bot.lag.near_lock import REPORTED, STRATUM
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME
from bot.lag.taker_flow import PRIMARY_HORIZON_S
from bot.lag.taker_flow_run import CLOSED, PASS, RESULTS_NAME, UNDERPOWERED
from bot.replay.analysis_stations import HIGH, LOW
from bot.replay.run_scope import DISCOVERY, HOLDOUT
from scripts.q3_near_lock import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    build_parser,
    export_observations,
    format_report,
    run,
)
from tests.test_near_lock import (
    CROSSING_PUBLISHED,
    CROSSING_VALID,
    LAX,
    MIA,
    NEXT_PUBLISHED,
    NEXT_VALID,
    STATION,
    cool_observations,
    crossing_observations,
    wide_artifacts,
    wide_observations,
    wide_scope_dir,
)
from tests.test_taker_flow_run import (
    DISCOVERY_DAY,
    SEED,
    SERIES,
    artifacts_dir,
    both_ladder_scope_dir,
    scope_dir,
)
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    SHORT_SAMPLES,
    seeded_repo,
    write_preregistration,
    write_rtt_samples,
)


RUN_ID = "2026-08-12-q3-near-lock"
REQUIRED = (
    "--run-id",
    "--preregistration",
    "--run-scope",
    "--artifacts",
    "--state-db",
    "--observations",
    "--rtt-samples",
    "--floor-source",
    "--seed",
)
ARRIVAL_ROWS = (
    (STATION, "metar", "2026-07-18 12:00:00.000000", "60", "2026-07-18 12:01:00.000000"),
    (STATION, "metar", "2026-07-18 17:00:00.000000", "72", "2026-07-18 18:00:00.000000"),
    (STATION, "metar", "2026-07-19 17:00:00.000000", "72", "2026-07-19 18:00:00.000000"),
)


def write_state_db(path: Path, rows: tuple[tuple[str, str, str, str, str], ...]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE ws_obs_arrivals ("
        "id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, station VARCHAR(16) NOT NULL, "
        "source VARCHAR(16) NOT NULL, obs_time DATETIME NOT NULL, tmpf VARCHAR NOT NULL, "
        "received_at DATETIME NOT NULL, created_at DATETIME NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX ix_ws_obs_arrivals_station_obs_time ON ws_obs_arrivals (station, obs_time)"
    )
    conn.executemany(
        "INSERT INTO ws_obs_arrivals (station, source, obs_time, tmpf, received_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [row + (row[4],) for row in rows],
    )
    conn.commit()
    conn.close()
    return path


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
        "--state-db",
        str(paths["state_db"]),
        "--observations",
        str(paths["observations"]),
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
        "state_db": write_state_db(tmp_path / "state.db", ARRIVAL_ROWS),
        "observations": tmp_path / "arrivals.jsonl",
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
    }


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "tape_studies"


def results_of(run_root: Path) -> dict:
    return json.loads((run_root / RUN_ID / RESULTS_NAME).read_text())


def test_the_export_reads_the_arrivals_the_locks_are_detected_from(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    exported = export_observations(paths["state_db"], tmp_path / "export.jsonl")

    rows = [json.loads(line) for line in (tmp_path / "export.jsonl").read_text().splitlines()]
    assert exported == len(ARRIVAL_ROWS)
    assert rows[1] == {
        "station": STATION,
        "source": "metar",
        "obs_time": CROSSING_VALID.isoformat(),
        "tmpf": "72",
        "received_at": CROSSING_PUBLISHED.isoformat(),
    }
    assert rows[2]["obs_time"] == NEXT_VALID.isoformat()
    assert rows[2]["received_at"] == NEXT_PUBLISHED.isoformat()


def test_a_complete_run_writes_the_manifest_the_export_and_the_results(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    manifest = json.loads((run_root / RUN_ID / MANIFEST_NAME).read_text())
    assert paths["observations"].exists()
    assert results["manifest_sha256"] == manifest["sha256"]
    assert results["bootstrap_seed"] == SEED
    assert results["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert results["stratum"] == STRATUM
    assert results["gating"] is False
    assert capsys.readouterr().out == format_report(results) + "\n"


def test_the_stratum_reads_only_the_prints_inside_a_lock_window(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["locks"] == {
        "markets": 2,
        "locked": 2,
        "ambiguous": 0,
        "ambiguous_fraction": "0",
        "no_lock": 0,
        "no_observations": 0,
    }
    assert results["prints"]["in_window_discovery"] == 1
    assert results["prints"]["in_window_holdout"] == 1
    assert results["prints"]["outside_lock_window"] == 4
    assert results["prints"]["on_a_market_that_never_locked"] == 0
    assert results["prints"]["empty_side"] == 1
    assert results["prints"]["duplicate_trade_id"] == 1
    assert results["prints"]["out_of_scope"] == 1
    assert results["discovery"]["split"] == DISCOVERY
    assert results["holdout"]["split"] == HOLDOUT
    assert results["discovery"]["horizon_s"] == PRIMARY_HORIZON_S


def test_the_coverage_names_only_the_cities_the_stratum_reads(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["run_scope"] = wide_scope_dir(tmp_path)
    paths["artifacts"] = wide_artifacts(tmp_path)
    paths["observations"] = wide_observations(tmp_path)

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["cities"] == [SERIES]
    assert list(results["tickers_per_city_day"]) == [f"{SERIES} {DISCOVERY_DAY.isoformat()}"]
    assert results["prints"]["on_a_market_that_never_locked"] == 0
    assert results["prints"]["on_a_series_outside_the_lock_universe"] == 2
    assert MIA not in json.dumps(results)
    assert LAX not in json.dumps(results)


def test_a_stratum_under_two_hundred_prints_reports_underpowered_and_no_estimate(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["stratum_status"] == UNDERPOWERED
    assert results["discovery"]["status"] == UNDERPOWERED
    assert results["discovery"]["n_prints"] < results["stratum_n_min"]
    assert results["discovery"]["mean_net_cents"] is None
    assert results["discovery"]["p_value"] is None
    assert results["holdout"]["mean_net_cents"] is None
    assert REPORTED not in json.dumps(results)


def test_the_results_carry_no_verdict_the_question_could_be_read_off(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    report = capsys.readouterr().out
    assert "verdict" not in results
    assert "gate" not in results
    assert "replication" not in results
    assert PASS not in json.dumps(results)
    assert CLOSED not in json.dumps(results)
    assert results["reported_only"] in report
    assert "nothing here decides Q3" in report


def test_a_run_where_nothing_locked_reads_no_prints_at_all(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["observations"] = cool_observations(tmp_path)

    assert run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    assert results["locks"]["locked"] == 0
    assert results["locks"]["no_lock"] == 2
    assert results["locks"]["ambiguous_fraction"] is None
    assert results["prints"]["in_window_pooled"] == 0
    assert results["prints"]["on_a_market_that_never_locked"] == 6
    assert results["discovery"]["n_prints"] == 0
    assert results["stratum_status"] == UNDERPOWERED


def test_an_export_already_on_disk_is_read_instead_of_the_database(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["observations"] = crossing_observations(tmp_path)
    paths["state_db"] = tmp_path / "absent.db"

    assert run(args_for(paths, run_root)) == 0

    assert not paths["state_db"].exists()
    assert results_of(run_root)["locks"]["locked"] == 2


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


def test_the_cohort_is_optional_and_never_names_both(
    paths: dict[str, Path], run_root: Path
) -> None:
    argv = argv_for(paths, run_root)

    assert build_parser().parse_args(argv).cohort is None
    assert build_parser().parse_args([*argv, "--cohort", HIGH]).cohort == HIGH
    assert build_parser().parse_args([*argv, "--cohort", LOW]).cohort == LOW
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([*argv, "--cohort", "both"])
    assert excinfo.value.code != 0


def test_a_two_ladder_scope_is_read_under_the_cohort_the_run_names(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["run_scope"] = both_ladder_scope_dir(tmp_path)

    with pytest.raises(ValueError, match="names no cohort"):
        run(args_for(paths, run_root))

    named = build_parser().parse_args([*argv_for(paths, run_root), "--cohort", HIGH])

    assert run(named) == 0
    assert results_of(run_root)["cities"] == [SERIES]
