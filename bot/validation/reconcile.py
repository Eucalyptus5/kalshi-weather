from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import httpx

from bot.execution.paper import PaperTrade, TradeSide
from bot.markets.parser import ParsedTicker
from bot.validation.scoring import realized_pnl_for_trade

ACIS_URL = "https://data.rcc-acis.org/StnData"

_ACIS_FETCH_TIMEOUT_SECONDS: float = 30.0


@dataclass(frozen=True, slots=True)
class Reconciliation:
    observed_high: Decimal
    yes_settled: bool
    won: bool
    realized_pnl: Decimal


class ACISClient:
    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        user_agent: str = "kalshi-weather-bot/0.1",
    ) -> None:
        self._owns_http = http_client is None
        if http_client is not None:
            self._http = http_client
        else:
            self._http = httpx.AsyncClient(
                timeout=10.0,
                headers={"User-Agent": user_agent},
            )
        self._user_agent = user_agent

    async def fetch_daily_high(self, station: str, settled_date: date) -> Decimal | None:
        params = {
            "sid": station,
            "sdate": settled_date.isoformat(),
            "edate": settled_date.isoformat(),
            "elems": "maxt",
            "output": "json",
        }
        headers = {"User-Agent": self._user_agent} if not self._owns_http else None
        response = await asyncio.wait_for(
            self._http.get(ACIS_URL, params=params, headers=headers),
            timeout=_ACIS_FETCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            return None
        value = rows[0][1]
        if value == "M":
            return None
        if not isinstance(value, str) or not value.lstrip("-").isdigit():
            return None
        return Decimal(value)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


def settle_bracket(parsed: ParsedTicker, observed_high: Decimal) -> bool:
    if not parsed.is_bracket:
        raise ValueError(f"settle_bracket called on non-bracket ticker: {parsed.raw}")
    low, high = parsed.strikes
    return low <= observed_high < high


def settle_tail(parsed: ParsedTicker, observed_high: Decimal) -> bool:
    if not parsed.is_tail:
        raise ValueError(f"settle_tail called on non-tail ticker: {parsed.raw}")
    (strike,) = parsed.strikes
    if parsed.kind == "below":
        return observed_high < strike
    return observed_high >= strike


def reconcile_trade(
    trade: PaperTrade,
    parsed: ParsedTicker,
    observed_high: Decimal,
) -> Reconciliation:
    if parsed.is_bracket:
        yes_settled = settle_bracket(parsed, observed_high)
    else:
        yes_settled = settle_tail(parsed, observed_high)

    if trade.side is TradeSide.BUY_YES:
        won = yes_settled
    else:
        won = not yes_settled

    realized = realized_pnl_for_trade(
        trade.side,
        trade.simulated_price,
        trade.contracts,
        trade.fee_dollars,
        won,
    )
    return Reconciliation(
        observed_high=observed_high,
        yes_settled=yes_settled,
        won=won,
        realized_pnl=realized,
    )
