import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from bot.lag.fee_floor import FeeSource
from bot.lag.r0_universe import R0Universe, freeze_digest, universe_payload
from bot.lag.read_rtt import (
    FloorSource,
    InadequateSamples,
    LatencyFloor,
    ReadSample,
    derive_latency_floor,
)


BOOTSTRAP_RESAMPLES = 10_000
MANIFEST_NAME = "manifest.json"
BORROWED_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")

SUPPLIED_FIELDS = (
    ("accrual_start", "accrual_start"),
    ("accrual_end", "accrual_end"),
    ("row_counts", "row_counts"),
    ("universe", "r0_fraction_invalid_max"),
    ("fee", "fee_source"),
    ("floor", "latency_floor"),
    ("bootstrap_seed", "bootstrap_seed"),
)


class ManifestIncomplete(RuntimeError):
    """A field the manifest must record has no value, so the run aborts."""

    def __init__(self, fields: Sequence[str], detail: str) -> None:
        super().__init__(f"run aborted, unavailable: {', '.join(fields)}: {detail}")
        self.fields = tuple(fields)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunInputs:
    run_id: str
    preregistration: Path
    repo: Path
    accrual_start: datetime | None
    accrual_end: datetime | None
    row_counts: Mapping[str, int] | None
    universe: R0Universe | None
    fee: FeeSource | None
    floor: LatencyFloor | None
    bootstrap_seed: int | None


@dataclass(frozen=True, slots=True)
class GitState:
    head: str
    dirty: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class Manifest:
    run_id: str
    preregistration: Path
    preregistration_sha256: str
    accrual_start: datetime
    accrual_end: datetime
    row_counts: Mapping[str, int]
    git: GitState
    r0_fraction_invalid_max: Decimal
    r0_universe_sha256: str
    fee: FeeSource
    floor: LatencyFloor
    bootstrap_seed: int


def preregistration_sha256(path: Path) -> str:
    if not path.is_file():
        raise ManifestIncomplete(("preregistration_sha256",), f"{path} is not on disk")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # An inherited GIT_DIR, GIT_WORK_TREE or GIT_INDEX_FILE points git at another tree's state.
    env = {key: value for key, value in os.environ.items() if key not in BORROWED_GIT_ENV}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env)


def git_state(repo: Path) -> GitState:
    toplevel = _git(repo, "rev-parse", "--show-toplevel")
    if toplevel.returncode != 0:
        raise ManifestIncomplete(("git_head",), toplevel.stderr.strip())
    root = Path(toplevel.stdout.strip()).resolve()
    if root != repo.resolve():
        raise ManifestIncomplete(("git_head",), f"{repo} is not the root of the tree at {root}")
    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise ManifestIncomplete(("git_head",), head.stderr.strip())
    dirty = _git(repo, "status", "--porcelain")
    if dirty.returncode != 0:
        raise ManifestIncomplete(("git_dirty",), dirty.stderr.strip())
    return GitState(head=head.stdout.strip(), dirty=bool(dirty.stdout.strip()))


def resolve_latency_floor(samples: list[ReadSample], source: FloorSource) -> LatencyFloor:
    try:
        return derive_latency_floor(samples, source)
    except InadequateSamples as exc:
        raise ManifestIncomplete(("latency_floor",), str(exc)) from exc


def build_manifest(inputs: RunInputs) -> Manifest:
    if not inputs.run_id:
        raise ManifestIncomplete(("run_id",), "a run with no id has no run directory")
    if inputs.run_id in (".", "..") or Path(inputs.run_id).name != inputs.run_id:
        raise ManifestIncomplete(
            ("run_id",), f"{inputs.run_id} does not name one directory under the run root"
        )

    missing = []
    for attribute, field in SUPPLIED_FIELDS:
        value = getattr(inputs, attribute)
        # An empty count map records nothing consumed rather than a run that consumed nothing.
        if value is None or value == {}:
            missing.append(field)
    if missing:
        raise ManifestIncomplete(missing, "the run supplied no value")

    return Manifest(
        run_id=inputs.run_id,
        preregistration=inputs.preregistration,
        preregistration_sha256=preregistration_sha256(inputs.preregistration),
        accrual_start=inputs.accrual_start,
        accrual_end=inputs.accrual_end,
        row_counts=inputs.row_counts,
        git=git_state(inputs.repo),
        r0_fraction_invalid_max=inputs.universe.fraction_invalid_max,
        r0_universe_sha256=freeze_digest(universe_payload(inputs.universe)),
        fee=inputs.fee,
        floor=inputs.floor,
        bootstrap_seed=inputs.bootstrap_seed,
    )


def fee_payload(fee: FeeSource) -> dict[str, str | bool]:
    payload: dict[str, str | bool] = {}
    for field in fields(fee):
        value = getattr(fee, field.name)
        key = field.name if field.name.startswith("fee") else f"fee_{field.name}"
        payload[key] = str(value) if isinstance(value, Decimal) else value
    return payload


def manifest_payload(manifest: Manifest) -> dict:
    return {
        "run_id": manifest.run_id,
        "preregistration_path": str(manifest.preregistration),
        "preregistration_sha256": manifest.preregistration_sha256,
        "accrual_start": manifest.accrual_start.isoformat(),
        "accrual_end": manifest.accrual_end.isoformat(),
        "row_counts": dict(manifest.row_counts),
        "git_head": manifest.git.head,
        "git_dirty": manifest.git.dirty,
        "r0_fraction_invalid_max": str(manifest.r0_fraction_invalid_max),
        "r0_universe_sha256": manifest.r0_universe_sha256,
        **fee_payload(manifest.fee),
        "latency_floor_source": manifest.floor.source.value,
        "latency_floor_s": manifest.floor.floor_s,
        "t_persist_s": manifest.floor.t_persist_s,
        "latency_floor_samples": manifest.floor.n_usable,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": manifest.bootstrap_seed,
    }


def write_manifest(root: Path, inputs: RunInputs) -> str:
    manifest = build_manifest(inputs)
    path = root / manifest.run_id / MANIFEST_NAME
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = manifest_payload(manifest)
    digest = freeze_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=1))
    return digest
