from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bot.backtest.engine import BacktestOrder
from bot.backtest.normalize import CanonicalSnapshot
from bot.execution.fees import taker_fee
from bot.execution.paper import TradeSide
from bot.validation.scoring import realized_pnl_for_trade


@dataclass(frozen=True, slots=True)
class BacktestFill:
    gross_pnl: Decimal
    fee_dollars: Decimal
    net_pnl: Decimal


def settle_order(order: BacktestOrder, snap: CanonicalSnapshot, result: str) -> BacktestFill:
    fill_price = snap.yes_ask if order.action is TradeSide.BUY_YES else snap.yes_bid
    won = (result == "yes") == (order.action is TradeSide.BUY_YES)

    fee_dollars = taker_fee(order.contracts, fill_price)
    gross_pnl = realized_pnl_for_trade(order.action, fill_price, order.contracts, Decimal("0"), won)
    net_pnl = realized_pnl_for_trade(order.action, fill_price, order.contracts, fee_dollars, won)
    return BacktestFill(gross_pnl=gross_pnl, fee_dollars=fee_dollars, net_pnl=net_pnl)
