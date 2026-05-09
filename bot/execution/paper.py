from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from bot.execution.fees import taker_fee


class TradeSide(Enum):
    BUY_YES = "buy_yes"
    SELL_YES = "sell_yes"


@dataclass(frozen=True, slots=True)
class Orderbook:
    yes_ask: Decimal
    yes_bid: Decimal


@dataclass(frozen=True, slots=True)
class TradeIntent:
    market_ticker: str
    side: TradeSide
    contracts: int
    fair_yes: Decimal
    strategy: str


@dataclass(frozen=True, slots=True)
class PaperTrade:
    intended_at: datetime
    market_ticker: str
    side: TradeSide
    contracts: int
    simulated_price: Decimal
    fee_dollars: Decimal
    fair_at_entry: Decimal
    strategy: str


def simulate_taker_fill(intent: TradeIntent, book: Orderbook, now: datetime) -> PaperTrade:
    if intent.contracts <= 0:
        raise ValueError(f"contracts must be > 0, got {intent.contracts}")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    if intent.side is TradeSide.BUY_YES:
        simulated_price = book.yes_ask
    else:
        simulated_price = book.yes_bid

    fee_dollars = taker_fee(intent.contracts, simulated_price)

    return PaperTrade(
        intended_at=now,
        market_ticker=intent.market_ticker,
        side=intent.side,
        contracts=intent.contracts,
        simulated_price=simulated_price,
        fee_dollars=fee_dollars,
        fair_at_entry=intent.fair_yes,
        strategy=intent.strategy,
    )
