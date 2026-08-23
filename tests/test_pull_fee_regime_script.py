from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bot.lag.fee_regime import (
    PLAIN_FEE_MULTIPLIER,
    PLAIN_FEE_TYPE,
    PLAIN_REGIME,
    check_fee_regime,
    read_fee_regime,
)
from scripts.pull_fee_regime import REPO_ROOT, build_parser, main
from tests.test_fee_regime import DEN, MIA, NY, OBSERVED_AT, ROOTS, SFO, plain_bodies, transport_for
from tests.test_tape_studies import argument_flags


SCRIPT = REPO_ROOT / "scripts" / "pull_fee_regime.py"


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies = plain_bodies()
    client = httpx.AsyncClient

    def factory(**kwargs: object) -> httpx.AsyncClient:
        return client(transport=transport_for(bodies))

    monkeypatch.setattr("bot.lag.fee_regime.httpx.AsyncClient", factory)


def argv_parts(out: Path) -> dict[str, list[str]]:
    return {
        "--roots": [DEN, NY, SFO, MIA],
        "--out": [str(out)],
        "--observed-at": [OBSERVED_AT.isoformat()],
    }


def argv_for(out: Path) -> list[str]:
    return [item for flag, values in argv_parts(out).items() for item in (flag, *values)]


def without(out: Path, flag: str) -> list[str]:
    parts = argv_parts(out)
    del parts[flag]
    return [item for name, values in parts.items() for item in (name, *values)]


def test_the_sweep_writes_the_sidecar_and_prints_its_digest(
    tmp_path: Path, offline: None, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "fee_regime.json"

    assert main(argv_for(out)) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["sha256"] == read_fee_regime(out).sha256
    assert printed["path"] == str(out)
    assert printed["observed_at"] == OBSERVED_AT.isoformat()
    assert sorted(printed["series"]) == sorted(ROOTS)
    assert printed["series"][DEN] == {
        "fee_type": PLAIN_FEE_TYPE,
        "fee_multiplier": PLAIN_FEE_MULTIPLIER,
    }


def test_the_frozen_sidecar_clears_the_wire(tmp_path: Path, offline: None) -> None:
    out = tmp_path / "fee_regime.json"

    assert main(argv_for(out)) == 0

    assert check_fee_regime(read_fee_regime(out), ROOTS) == PLAIN_REGIME


def test_a_second_sweep_refuses_to_overwrite_the_frozen_sidecar(
    tmp_path: Path, offline: None
) -> None:
    out = tmp_path / "fee_regime.json"
    assert main(argv_for(out)) == 0

    with pytest.raises(FileExistsError):
        main(argv_for(out))


@pytest.mark.parametrize("flag", ("--roots", "--out", "--observed-at"))
def test_the_sweep_will_not_run_without_the_flags_that_pin_it(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(without(tmp_path / "fee_regime.json", flag))


def test_the_sweep_names_no_rate_on_the_command_line() -> None:
    flags = argument_flags(SCRIPT.read_text())

    assert flags == {"roots", "out", "observed_at"}
    assert [flag for flag in flags if "rate" in flag] == []
