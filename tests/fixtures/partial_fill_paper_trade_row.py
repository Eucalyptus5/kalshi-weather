from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from bot.execution.fees import taker_fee
from bot.execution.paper import Orderbook, PaperTrade, TradeIntent, TradeSide
from bot.storage.sqlite import PaperTradeRow


_DENVER_TICKER = "KXHIGHDEN-26MAY20-B54.5"
_INTENDED_AT = datetime(2026, 5, 19, 16, 1, tzinfo=timezone.utc)


def make_partial_fill_row() -> PaperTradeRow:
    return PaperTradeRow(
        intended_at=_INTENDED_AT,
        market_ticker=_DENVER_TICKER,
        side="sell_yes",
        contracts=1,
        simulated_price=Decimal("0.99"),
        fee_dollars=taker_fee(1, Decimal("0.99")),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        attempted_contracts=7194,
    )


def make_partial_fill_intent_and_trade() -> tuple[TradeIntent, PaperTrade, Orderbook]:
    intent = TradeIntent(
        market_ticker=_DENVER_TICKER,
        side=TradeSide.SELL_YES,
        contracts=7194,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    book = Orderbook(
        yes_ask=Decimal("1.00"),
        yes_bid=Decimal("0.99"),
        yes_ask_depth=1,
        yes_bid_depth=1,
        snapshot_at=_INTENDED_AT - timedelta(seconds=1),
    )
    trade = PaperTrade(
        intended_at=_INTENDED_AT,
        market_ticker=_DENVER_TICKER,
        side=TradeSide.SELL_YES,
        contracts=1,
        simulated_price=Decimal("0.99"),
        fee_dollars=taker_fee(1, Decimal("0.99")),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        attempted_contracts=7194,
    )
    return intent, trade, book
