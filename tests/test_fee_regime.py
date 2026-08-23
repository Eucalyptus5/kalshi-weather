from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from bot.lag.fee_regime import (
    PLAIN_FEE_MULTIPLIER,
    PLAIN_FEE_TYPE,
    PLAIN_REGIME,
    FeeRegime,
    FeeRegimeMoved,
    check_fee_regime,
    pull_fee_regime,
    read_fee_regime,
)


UTC = timezone.utc
OBSERVED_AT = datetime(2026, 8, 19, 17, 30, tzinfo=UTC)
DEN = "KXHIGHDEN"
NY = "KXHIGHNY"
SFO = "KXHIGHTSFO"
MIA = "KXLOWTMIA"
ROOTS = (DEN, NY, SFO, MIA)
MAKER_FEE_TYPE = "quadratic_with_maker_fees"


def series_body(root: str, fee_type: str = PLAIN_FEE_TYPE, fee_multiplier: int = 1) -> dict:
    return {
        "series": {
            "ticker": root,
            "category": "Climate and Weather",
            "fee_type": fee_type,
            "fee_multiplier": fee_multiplier,
            "settlement_sources": [{"name": "The Weather Company"}],
        }
    }


def transport_for(bodies: Mapping[str, dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        root = request.url.path.rsplit("/", 1)[-1]
        if root not in bodies:
            return httpx.Response(404, json={"error": {"code": "not_found"}})
        return httpx.Response(200, json=bodies[root])

    return httpx.MockTransport(handler)


def plain_bodies() -> dict[str, dict]:
    return {root: series_body(root) for root in ROOTS}


def frozen(path: Path, bodies: Mapping[str, dict]) -> FeeRegime:
    pull_fee_regime(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))
    return read_fee_regime(path)


@pytest.fixture
def plain(tmp_path: Path) -> FeeRegime:
    return frozen(tmp_path / "fee_regime.json", plain_bodies())


def test_the_frozen_regime_is_the_plain_quadratic_one() -> None:
    assert PLAIN_FEE_TYPE == "quadratic"
    assert PLAIN_FEE_MULTIPLIER == 1
    assert PLAIN_REGIME == "plain_quadratic"


def test_a_sweep_of_plain_series_freezes_every_root_it_asked_for(plain: FeeRegime) -> None:
    assert sorted(plain.series) == sorted(ROOTS)
    assert {item.fee_type for item in plain.series.values()} == {PLAIN_FEE_TYPE}
    assert {item.fee_multiplier for item in plain.series.values()} == {PLAIN_FEE_MULTIPLIER}
    assert plain.observed_at == OBSERVED_AT


def test_the_sidecar_reproduces_its_digest_on_a_second_read(tmp_path: Path) -> None:
    path = tmp_path / "fee_regime.json"
    bodies = plain_bodies()

    written = pull_fee_regime(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))

    assert read_fee_regime(path).sha256 == written
    assert read_fee_regime(path).sha256 == read_fee_regime(path).sha256


def test_the_sweep_refuses_to_overwrite_a_frozen_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "fee_regime.json"
    bodies = plain_bodies()
    pull_fee_regime(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))

    with pytest.raises(FileExistsError):
        pull_fee_regime(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))


def test_a_sidecar_carrying_no_sha256_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "fee_regime.json"
    bodies = plain_bodies()
    pull_fee_regime(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))
    payload = json.loads(path.read_text())
    payload.pop("sha256")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="sha256"):
        read_fee_regime(path)


def test_a_tampered_sidecar_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "fee_regime.json"
    bodies = plain_bodies()
    pull_fee_regime(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))
    path.write_text(path.read_text().replace(PLAIN_FEE_TYPE, MAKER_FEE_TYPE))

    with pytest.raises(ValueError, match="sha256"):
        read_fee_regime(path)


def test_a_series_the_venue_refuses_is_never_frozen(tmp_path: Path) -> None:
    path = tmp_path / "fee_regime.json"

    with pytest.raises(httpx.HTTPStatusError):
        pull_fee_regime([DEN, "KXHIGHNOWHERE"], OBSERVED_AT, path, transport_for(plain_bodies()))


def test_a_plain_sweep_clears_the_wire(plain: FeeRegime) -> None:
    assert check_fee_regime(plain, ROOTS) == PLAIN_REGIME
    assert check_fee_regime(plain, (DEN,)) == PLAIN_REGIME


def test_a_maker_fee_series_trips_the_wire(tmp_path: Path) -> None:
    bodies = plain_bodies() | {NY: series_body(NY, fee_type=MAKER_FEE_TYPE)}
    regime = frozen(tmp_path / "fee_regime.json", bodies)

    with pytest.raises(FeeRegimeMoved) as excinfo:
        check_fee_regime(regime, ROOTS)

    assert NY in str(excinfo.value)
    assert MAKER_FEE_TYPE in str(excinfo.value)
    assert check_fee_regime(regime, (DEN,)) == PLAIN_REGIME


def test_a_zeroed_multiplier_trips_the_wire(tmp_path: Path) -> None:
    bodies = plain_bodies() | {SFO: series_body(SFO, fee_multiplier=0)}
    regime = frozen(tmp_path / "fee_regime.json", bodies)

    with pytest.raises(FeeRegimeMoved) as excinfo:
        check_fee_regime(regime, ROOTS)

    assert SFO in str(excinfo.value)
    assert "fee_multiplier=0" in str(excinfo.value)


def test_a_doubled_multiplier_trips_the_wire(tmp_path: Path) -> None:
    bodies = plain_bodies() | {MIA: series_body(MIA, fee_multiplier=2)}
    regime = frozen(tmp_path / "fee_regime.json", bodies)

    with pytest.raises(FeeRegimeMoved) as excinfo:
        check_fee_regime(regime, ROOTS)

    assert MIA in str(excinfo.value)
    assert "fee_multiplier=2" in str(excinfo.value)


def test_a_root_the_sidecar_does_not_name_trips_the_wire(plain: FeeRegime) -> None:
    with pytest.raises(FeeRegimeMoved) as excinfo:
        check_fee_regime(plain, (*ROOTS, "KXHIGHCHI"))

    assert "KXHIGHCHI" in str(excinfo.value)


def test_the_wire_names_every_offending_root_not_only_the_first(tmp_path: Path) -> None:
    bodies = plain_bodies() | {
        DEN: series_body(DEN, fee_type=MAKER_FEE_TYPE),
        NY: series_body(NY, fee_multiplier=2),
    }
    regime = frozen(tmp_path / "fee_regime.json", bodies)

    with pytest.raises(FeeRegimeMoved) as excinfo:
        check_fee_regime(regime, (*ROOTS, "KXHIGHCHI"))

    message = str(excinfo.value)
    assert DEN in message
    assert NY in message
    assert "KXHIGHCHI" in message
    assert excinfo.value.offenders == tuple(sorted(excinfo.value.offenders))
    assert len(excinfo.value.offenders) == 3
