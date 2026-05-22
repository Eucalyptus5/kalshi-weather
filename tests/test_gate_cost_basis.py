from __future__ import annotations

from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal

from bot.execution.gate_cost_basis import (
    cost_per_contract_from_book,
    cost_per_contract_from_market_legacy,
)
from bot.execution.paper import TradeSide
from bot.kalshi_client import KalshiMarket, KalshiOrderbook


def _book() -> KalshiOrderbook:
    return KalshiOrderbook(
        ticker="KXHIGHDEN-26MAY08-B66.5",
        yes_ask=Decimal("0.85"),
        yes_bid=Decimal("0.20"),
        no_ask=Decimal("0.80"),
        no_bid=Decimal("0.15"),
        yes_ask_depth=10,
        yes_bid_depth=10,
        no_ask_depth=10,
        no_bid_depth=10,
        snapshot_at=datetime(2026, 5, 6, 12, 0, tzinfo=_timezone.utc),
    )


def _market() -> KalshiMarket:
    return KalshiMarket(
        ticker="KXHIGHDEN-26MAY08-B66.5",
        event_ticker="KXHIGHDEN-26MAY08",
        series="KXHIGHDEN",
        status="active",
        close_time=datetime(2026, 5, 8, 23, 0, tzinfo=_timezone.utc),
        yes_ask=Decimal("0.85"),
        yes_bid=Decimal("0.25"),
    )


def test_from_book_buy_yes_uses_yes_ask() -> None:
    assert cost_per_contract_from_book(TradeSide.BUY_YES, _book()) == Decimal("0.85")


def test_from_book_sell_yes_uses_no_ask() -> None:
    assert cost_per_contract_from_book(TradeSide.SELL_YES, _book()) == Decimal("0.80")


def test_legacy_buy_yes_uses_market_yes_ask() -> None:
    assert cost_per_contract_from_market_legacy(TradeSide.BUY_YES, _market()) == Decimal("0.85")


def test_legacy_sell_yes_uses_yes_bid_complement() -> None:
    assert cost_per_contract_from_market_legacy(TradeSide.SELL_YES, _market()) == Decimal("0.75")
