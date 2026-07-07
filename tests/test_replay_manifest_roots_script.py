from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.replay_manifest_roots import DEFAULT_KIND, build_parser, run
from tests.test_artifacts import drain_record, write_manifest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    return write_manifest(
        tmp_path / "manifest.jsonl",
        [
            drain_record("KXHIGHAUS-2026-07-17-b000001.parquet", "ladder"),
            drain_record("KXHIGHAUS-2026-07-18-b000002.parquet", "ladder"),
            drain_record("KXLOWTCHI-2026-07-17-b000001.parquet", "ladder"),
            drain_record("KXRAINCHIM-2026-07-17-b000001.parquet", "touch"),
        ],
    )


@pytest.fixture
def backlog(tmp_path: Path) -> Path:
    directory = tmp_path / "ladder"
    directory.mkdir()
    for name in ("KXHIGHDEN-2026-07-19-b000004.parquet", "KXLOWTCHI-2026-07-19-b000004.parquet"):
        (directory / name).write_bytes(b"PAR1")
    return directory


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_manifest_roots.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "manifest" in result.stdout
    assert "--artifact-dir" in result.stdout
    assert "--kind" in result.stdout


def test_default_arg_values(manifest: Path) -> None:
    args = build_parser().parse_args([str(manifest)])
    assert args.manifest == manifest
    assert args.artifact_dir is None
    assert args.kind == DEFAULT_KIND


def test_the_manifest_alone_carries_only_what_the_drain_shipped(
    manifest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run(build_parser().parse_args([str(manifest)]))

    assert rc == 0
    printed = capsys.readouterr().out
    assert "roots=KXHIGHAUS,KXLOWTCHI" in printed
    assert "count=2" in printed


def test_the_backlog_still_on_disk_joins_the_roots_the_drain_shipped(
    manifest: Path, backlog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run(build_parser().parse_args([str(manifest), "--artifact-dir", str(backlog)]))

    assert rc == 0
    printed = capsys.readouterr().out
    assert "roots=KXHIGHAUS,KXHIGHDEN,KXLOWTCHI" in printed
    assert "count=3" in printed


def test_the_kind_selects_which_half_of_the_manifest_counts(
    manifest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run(build_parser().parse_args([str(manifest), "--kind", "touch"]))

    assert rc == 0
    printed = capsys.readouterr().out
    assert "roots=KXRAINCHIM" in printed
    assert "count=1" in printed
