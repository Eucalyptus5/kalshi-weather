from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from bot.execution.order_placer import DemoOrder
from bot.execution.paper import TradeIntent, trade_side_for_demo
from bot.storage.sqlite import DemoOrder as DemoOrderRow
from bot.storage.sqlite import PaperTradeRow

_EXCLUDED_FROM_VALUES = frozenset({"id", "created_at"})


def yes_frame_price(side: str, price: Decimal) -> Decimal:
    if side == "no":
        return Decimal("1") - price
    return price


def paper_trade_row_from_demo_order(
    order: DemoOrder, intent: TradeIntent, intended_at: datetime
) -> PaperTradeRow | None:
    if (
        order.filled_contracts <= 0
        or order.avg_yes_fill_price_dollars is None
        or order.avg_yes_fill_price_dollars <= Decimal("0")
    ):
        return None
    return PaperTradeRow(
        intended_at=intended_at,
        market_ticker=intent.market_ticker,
        side=trade_side_for_demo(order.side_kalshi).value,
        contracts=order.filled_contracts,
        simulated_price=yes_frame_price(order.side_kalshi, order.avg_yes_fill_price_dollars),
        fee_dollars=order.fee_dollars,
        fair_at_entry=intent.fair_yes,
        strategy=intent.strategy,
        attempted_contracts=order.filled_contracts,
        ensemble_spread_sigma_t=None,
        lead_time_hours=None,
        nbm_divergence=None,
        demo_order_client_id=order.client_order_id,
    )


def _paper_trade_row_from_demo_row(demo_row: DemoOrderRow, now: datetime) -> PaperTradeRow | None:
    if demo_row.filled_contracts == 0:
        return None
    if demo_row.avg_fill_price is None or demo_row.avg_fill_price <= Decimal("0"):
        return None
    if (
        demo_row.strategy is None
        or demo_row.fair_at_entry is None
        or demo_row.intended_at is None
        or demo_row.requested_yes_price_dollars is None
    ):
        return None
    fee = demo_row.fee_dollars if demo_row.fee_dollars is not None else Decimal("0")
    return PaperTradeRow(
        intended_at=demo_row.intended_at,
        market_ticker=demo_row.market_ticker,
        side=trade_side_for_demo(demo_row.side).value,
        contracts=demo_row.filled_contracts,
        simulated_price=yes_frame_price(demo_row.side, demo_row.avg_fill_price),
        fee_dollars=fee,
        fair_at_entry=demo_row.fair_at_entry,
        strategy=demo_row.strategy,
        attempted_contracts=demo_row.filled_contracts,
        ensemble_spread_sigma_t=None,
        lead_time_hours=None,
        nbm_divergence=None,
        demo_order_client_id=demo_row.client_order_id,
    )


def paper_trade_row_values(row: PaperTradeRow) -> dict[str, object]:
    return {
        column.name: getattr(row, column.name)
        for column in PaperTradeRow.__table__.columns
        if column.name not in _EXCLUDED_FROM_VALUES
    }


def _demo_order_values(
    order: DemoOrder, intent: TradeIntent, now_pre_post: datetime
) -> dict[str, object]:
    return {
        "client_order_id": order.client_order_id,
        "exchange_order_id": order.exchange_order_id or None,
        "market_ticker": order.ticker,
        "strategy": intent.strategy,
        "side": order.side_kalshi,
        "requested_contracts": order.requested_contracts,
        "filled_contracts": order.filled_contracts,
        "requested_yes_price_dollars": order.requested_yes_price_dollars,
        "fair_at_entry": intent.fair_yes,
        "intended_at": now_pre_post,
        "avg_fill_price": order.avg_yes_fill_price_dollars,
        "fee_dollars": order.fee_dollars,
        "status": order.status,
        "placed_at": order.placed_at,
        "last_status_at": order.placed_at,
    }
