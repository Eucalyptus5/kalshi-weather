from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from bot.lag.r0_universe import freeze_digest


SERIES_URL = "https://api.elections.kalshi.com/trade-api/v2/series"
PLAIN_FEE_TYPE = "quadratic"
PLAIN_FEE_MULTIPLIER = 1
PLAIN_REGIME = "plain_quadratic"


class FeeRegimeMoved(RuntimeError):
    """A series no longer carries the fee regime the run was frozen under, so the run aborts."""

    def __init__(self, offenders: Sequence[str]) -> None:
        super().__init__(
            "run aborted, the frozen fee regime no longer holds: " + "; ".join(offenders)
        )
        self.offenders = tuple(offenders)


@dataclass(frozen=True, slots=True)
class SeriesFee:
    root: str
    fee_type: str
    fee_multiplier: object


@dataclass(frozen=True, slots=True)
class FeeRegime:
    observed_at: datetime
    series: Mapping[str, SeriesFee]
    sha256: str


# The observation is a run-level input the caller stamps and freezes: a rate read at run time
# would price old tape at today's regime and leave the manifest digest irreproducible.
@dataclass(frozen=True, slots=True)
class FeeRegimeCheck:
    result: str
    observed_at: datetime
    sha256: str


async def fetch_series_fee(root: str, client: httpx.AsyncClient) -> SeriesFee:
    response = await client.get(f"{SERIES_URL}/{root}")
    response.raise_for_status()
    series = response.json()["series"]
    return SeriesFee(
        root=series["ticker"],
        fee_type=series["fee_type"],
        fee_multiplier=series["fee_multiplier"],
    )


async def _fetch_roots(
    roots: Sequence[str], transport: httpx.AsyncBaseTransport | None
) -> list[SeriesFee]:
    async with httpx.AsyncClient(transport=transport) as client:
        return [await fetch_series_fee(root, client) for root in roots]


def regime_payload(observed_at: datetime, series: Sequence[SeriesFee]) -> dict:
    return {
        "observed_at": observed_at.isoformat(),
        "series": [
            {
                "root": item.root,
                "fee_type": item.fee_type,
                "fee_multiplier": item.fee_multiplier,
            }
            for item in sorted(series, key=lambda item: item.root)
        ],
    }


def pull_fee_regime(
    roots: Sequence[str],
    observed_at: datetime,
    path: Path,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = regime_payload(observed_at, asyncio.run(_fetch_roots(roots, transport)))
    digest = freeze_digest(payload)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=1))
    return digest


def read_fee_regime(path: Path) -> FeeRegime:
    payload = json.loads(path.read_text())
    if "sha256" not in payload:
        raise ValueError(f"{path} carries no sha256")
    stored = payload.pop("sha256")
    digest = freeze_digest(payload)
    if digest != stored:
        raise ValueError(f"{path} does not match the sha256 it carries")
    return FeeRegime(
        observed_at=datetime.fromisoformat(payload["observed_at"]),
        series={
            row["root"]: SeriesFee(
                root=row["root"],
                fee_type=row["fee_type"],
                fee_multiplier=row["fee_multiplier"],
            )
            for row in payload["series"]
        },
        sha256=digest,
    )


def check_fee_regime(regime: FeeRegime, roots: Iterable[str]) -> str:
    offenders = []
    for root in sorted(set(roots)):
        carried = regime.series.get(root)
        if carried is None:
            offenders.append(f"{root} is not named in the frozen sidecar")
        elif carried.fee_type != PLAIN_FEE_TYPE or carried.fee_multiplier != PLAIN_FEE_MULTIPLIER:
            offenders.append(
                f"{root} carries fee_type={carried.fee_type} "
                f"fee_multiplier={carried.fee_multiplier}"
            )
    if offenders:
        raise FeeRegimeMoved(offenders)
    return PLAIN_REGIME
