from __future__ import annotations

from decimal import Decimal

from bot.execution.paper import TradeSide
from bot.kalshi_client import KalshiMarket, KalshiOrderbook


def cost_per_contract_from_book(side: TradeSide, book: KalshiOrderbook) -> Decimal:
    if side is TradeSide.BUY_YES:
        return book.yes_ask
    return book.no_ask


def cost_per_contract_from_market_legacy(side: TradeSide, market: KalshiMarket) -> Decimal:
    if side is TradeSide.BUY_YES:
        return market.yes_ask
    return Decimal("1") - market.yes_bid
