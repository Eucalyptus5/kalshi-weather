from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from bot.backtest.backfill import fetch_settled
from bot.backtest.normalize import CanonicalSnapshot
from bot.lag.r0_universe import freeze_digest


WINDOW_OPEN_HOURS = 24
WINDOW_CLOSE_HOURS = 12
PLACEMENT_STEP_S = 900
PLACEMENTS = 48


@dataclass(frozen=True, slots=True)
class MarketClose:
    ticker: str
    event_ticker: str
    close_time: datetime
    floor_strike: int | None
    cap_strike: int | None
    strike_type: str | None
    status: str
    result: str


@dataclass(frozen=True, slots=True)
class CloseSidecar:
    root: str
    markets: Mapping[str, MarketClose]
    voided: tuple[str, ...]
    sha256: str


# The window spans exactly PLACEMENTS steps, so rounding its open up to the clock grid pushes the
# last instant back by the same amount and it stays strictly inside the 12-hour edge. The count is
# therefore fixed by the pre-registration and never depends on where the close sits in a step.
def placement_grid(close: datetime) -> tuple[datetime, ...]:
    opened = int((close - timedelta(hours=WINDOW_OPEN_HOURS)).timestamp())
    first = -(-opened // PLACEMENT_STEP_S) * PLACEMENT_STEP_S
    return tuple(
        datetime.fromtimestamp(first + PLACEMENT_STEP_S * index, tz=timezone.utc)
        for index in range(PLACEMENTS)
    )


def close_of(sidecar: CloseSidecar, ticker: str) -> datetime:
    market = sidecar.markets.get(ticker)
    if market is None:
        raise ValueError(f"{ticker} is not named in the {sidecar.root} close sidecar")
    return market.close_time


def grid_for(sidecar: CloseSidecar, ticker: str) -> tuple[datetime, ...]:
    return placement_grid(close_of(sidecar, ticker))


def sidecar_payload(root: str, markets: Sequence[MarketClose], voided: Sequence[str]) -> dict:
    return {
        "root": root,
        "markets": [
            {
                "ticker": market.ticker,
                "event_ticker": market.event_ticker,
                "close_time": market.close_time.isoformat(),
                "floor_strike": market.floor_strike,
                "cap_strike": market.cap_strike,
                "strike_type": market.strike_type,
                "status": market.status,
                "result": market.result,
            }
            for market in sorted(markets, key=lambda item: item.ticker)
        ],
        "voided": sorted(voided),
    }


def write_sidecar(path: Path, root: str, snapshots: Sequence[CanonicalSnapshot]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    markets = []
    voided = []
    for snapshot in snapshots:
        if not snapshot.result:
            voided.append(snapshot.ticker)
            continue
        markets.append(
            MarketClose(
                ticker=snapshot.ticker,
                event_ticker=snapshot.event_ticker,
                close_time=snapshot.close_time.astimezone(timezone.utc),
                floor_strike=snapshot.floor_strike,
                cap_strike=snapshot.cap_strike,
                strike_type=snapshot.strike_type,
                status=snapshot.status,
                result=snapshot.result,
            )
        )
    payload = sidecar_payload(root, markets, voided)
    digest = freeze_digest(payload)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=1))
    return digest


def read_sidecar(path: Path) -> CloseSidecar:
    payload = json.loads(path.read_text())
    if "sha256" not in payload:
        raise ValueError(f"{path} carries no sha256")
    stored = payload.pop("sha256")
    digest = freeze_digest(payload)
    if digest != stored:
        raise ValueError(f"{path} does not match the sha256 it carries")
    return CloseSidecar(
        root=payload["root"],
        markets={
            row["ticker"]: MarketClose(
                ticker=row["ticker"],
                event_ticker=row["event_ticker"],
                close_time=datetime.fromisoformat(row["close_time"]),
                floor_strike=row["floor_strike"],
                cap_strike=row["cap_strike"],
                strike_type=row["strike_type"],
                status=row["status"],
                result=row["result"],
            )
            for row in payload["markets"]
        },
        voided=tuple(payload["voided"]),
        sha256=digest,
    )


async def _fetch_roots(
    roots: Sequence[str],
    min_ts: int,
    max_ts: int,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, list[CanonicalSnapshot]]:
    async with httpx.AsyncClient(transport=transport) as client:
        return {root: await fetch_settled(root, min_ts, max_ts, client) for root in roots}


def pull_closes(
    roots: Sequence[str],
    min_ts: int,
    max_ts: int,
    directory: Path,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, str]:
    pulled = asyncio.run(_fetch_roots(roots, min_ts, max_ts, transport))
    return {
        root: write_sidecar(directory / f"{root}.json", root, snapshots)
        for root, snapshots in pulled.items()
    }
