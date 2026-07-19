import argparse
import json
from pathlib import Path

import pytest

from bot.lag.r0_universe import freeze_digest
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import MANIFEST_NAME
from scripts.tape_report import (
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    build_parser,
    format_report,
    run,
)
from tests.test_tape_studies import (
    CONSUMED,
    RUN_ID,
    SEED,
    SHORT_SAMPLES,
    TOUCH,
    run_input_paths,
    write_rtt_samples,
)


SECOND_RUN_ID = "2026-08-12-q2"
REQUIRED = (
    "--run-id",
    "--preregistration",
    "--run-scope",
    "--artifacts",
    "--rtt-samples",
    "--floor-source",
    "--seed",
)


def argv_for(
    paths: dict[str, Path],
    run_root: Path,
    *,
    run_id: str = RUN_ID,
    floor_source: str = "RTT_read",
) -> list[str]:
    return [
        "--run-id",
        run_id,
        "--preregistration",
        str(paths["preregistration"]),
        "--run-scope",
        str(paths["run_scope"]),
        "--artifacts",
        str(paths["artifacts"]),
        "--rtt-samples",
        str(paths["rtt_samples"]),
        "--floor-source",
        floor_source,
        "--seed",
        str(SEED),
        "--run-root",
        str(run_root),
        "--repo",
        str(paths["repo"]),
    ]


def args_for(paths: dict[str, Path], run_root: Path, **overrides: str) -> argparse.Namespace:
    return build_parser().parse_args(argv_for(paths, run_root, **overrides))


def without(argv: list[str], flag: str) -> list[str]:
    index = argv.index(flag)
    return argv[:index] + argv[index + 2 :]


def manifest_of(run_root: Path, run_id: str = RUN_ID) -> dict:
    return json.loads((run_root / run_id / MANIFEST_NAME).read_text())


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return run_input_paths(tmp_path)


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "tape_studies"


def test_a_complete_run_writes_the_manifest_it_reports(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(args_for(paths, run_root)) == 0

    manifest = manifest_of(run_root)
    printed = capsys.readouterr().out
    payload = dict(manifest)
    digest = payload.pop("sha256")
    assert freeze_digest(payload) == digest
    assert payload["run_id"] == RUN_ID
    assert payload["bootstrap_seed"] == SEED
    assert payload["row_counts"][TOUCH] == CONSUMED[TOUCH]
    assert printed == format_report(manifest) + "\n"


def test_a_run_whose_floor_is_unavailable_writes_nothing(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_rtt_samples(paths["rtt_samples"], SHORT_SAMPLES)

    assert run(args_for(paths, run_root)) != 0

    captured = capsys.readouterr()
    assert "latency_floor" in captured.err
    assert f"usable samples {SHORT_SAMPLES} short of" in captured.err
    assert captured.out == ""
    assert not run_root.exists()


def test_a_second_run_under_the_same_id_refuses_to_overwrite(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0
    digest = manifest_of(run_root)["sha256"]

    with pytest.raises(FileExistsError, match=MANIFEST_NAME):
        run(args_for(paths, run_root))

    assert manifest_of(run_root)["sha256"] == digest


@pytest.mark.parametrize("flag", REQUIRED)
def test_every_input_the_manifest_records_is_required(
    paths: dict[str, Path], run_root: Path, flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(without(argv_for(paths, run_root), flag))

    assert excinfo.value.code != 0
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("value", FLOOR_SOURCES)
def test_the_stated_floor_source_is_the_one_recorded(
    paths: dict[str, Path], run_root: Path, value: str
) -> None:
    assert run(args_for(paths, run_root, floor_source=value)) == 0

    assert manifest_of(run_root)["latency_floor_source"] == value


def test_a_floor_source_outside_the_two_the_run_can_state_is_refused(
    paths: dict[str, Path], run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert FLOOR_SOURCES == ("L", "RTT_read")
    assert {FloorSource(value) for value in FLOOR_SOURCES} == set(FloorSource)

    with pytest.raises(SystemExit) as excinfo:
        args_for(paths, run_root, floor_source="ws")

    assert excinfo.value.code != 0
    assert "--floor-source" in capsys.readouterr().err


def test_the_dirty_flag_follows_the_tree_the_run_came_from(
    paths: dict[str, Path], run_root: Path
) -> None:
    assert run(args_for(paths, run_root)) == 0
    clean = manifest_of(run_root)

    (paths["repo"] / "scratch.py").write_text("x = 1\n")
    assert run(args_for(paths, run_root, run_id=SECOND_RUN_ID)) == 0
    dirty = manifest_of(run_root, SECOND_RUN_ID)

    assert clean["git_dirty"] is False
    assert dirty["git_dirty"] is True
    assert dirty["git_head"] == clean["git_head"]


def test_the_run_root_and_repo_default_to_the_tree_the_script_ships_in(
    paths: dict[str, Path], run_root: Path
) -> None:
    argv = without(without(argv_for(paths, run_root), "--run-root"), "--repo")

    args = build_parser().parse_args(argv)

    assert args.run_root == DEFAULT_RUN_ROOT
    assert args.repo == REPO_ROOT
    assert DEFAULT_RUN_ROOT == REPO_ROOT / "data" / "tape_studies"


def test_the_seed_reaches_the_manifest_as_an_integer(
    paths: dict[str, Path], run_root: Path
) -> None:
    args = args_for(paths, run_root)

    assert run(args) == 0
    assert manifest_of(run_root)["bootstrap_seed"] == SEED
