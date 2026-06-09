from datetime import datetime, timezone
from decimal import Decimal

from bot.backtest.engine import BacktestOrder
from bot.backtest.normalize import CanonicalSnapshot
from bot.backtest.pnl import settle_order
from bot.execution.fees import taker_fee
from bot.execution.paper import TradeSide


_AS_OF = datetime(2025, 11, 4, 17, 0, tzinfo=timezone.utc)


def make_snap(ticker: str, *, yes_ask: Decimal, yes_bid: Decimal, result: str) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0],
        series_ticker=ticker.split("-", 1)[0],
        status="settled",
        result=result,
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=Decimal("1") - yes_bid,
        no_bid=Decimal("1") - yes_ask,
        last_price=yes_ask,
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
    )


def make_order(ticker: str, action: TradeSide, *, price_per_contract: Decimal) -> BacktestOrder:
    return BacktestOrder(
        market_ticker=ticker,
        as_of=_AS_OF,
        strategy="edge",
        action=action,
        contracts=1,
        fair_yes=Decimal("0.95"),
        price_per_contract=price_per_contract,
        order_dollars=price_per_contract,
        depth_at_price=3000,
        depth_source="snapshot",
        lead_bucket="1d",
    )


def test_buy_yes_winning_bracket():
    ticker = "KXHIGHDEN-25NOV04-B61.5"
    snap = make_snap(ticker, yes_ask=Decimal("0.87"), yes_bid=Decimal("0.85"), result="yes")
    order = make_order(ticker, TradeSide.BUY_YES, price_per_contract=Decimal("0.87"))

    fill = settle_order(order, snap, "yes")

    assert fill.gross_pnl == Decimal("0.13")
    assert fill.fee_dollars == taker_fee(1, Decimal("0.87"))
    assert fill.fee_dollars == Decimal("0.007917")
    assert fill.net_pnl == Decimal("0.122083")


def test_sell_yes_tail_winning():
    ticker = "KXHIGHDEN-25NOV04-T70"
    snap = make_snap(ticker, yes_ask=Decimal("0.14"), yes_bid=Decimal("0.12"), result="no")
    order = make_order(
        ticker, TradeSide.SELL_YES, price_per_contract=Decimal("1") - Decimal("0.12")
    )

    fill = settle_order(order, snap, "no")

    assert fill.gross_pnl == Decimal("0.12")
