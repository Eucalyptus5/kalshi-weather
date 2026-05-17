from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from bot.execution.fees import taker_fee
from bot.execution.paper import Orderbook, PaperTrade, TradeIntent, TradeSide, simulate_taker_fill


def _intent(**overrides: object) -> TradeIntent:
    base: dict[str, object] = {
        "market_ticker": "KXHIGHDEN-26MAY05-T80",
        "side": TradeSide.BUY_YES,
        "contracts": 10,
        "fair_yes": Decimal("0.50"),
        "strategy": "edge",
    }
    base.update(overrides)
    return TradeIntent(**base)  # type: ignore[arg-type]


def _book(**overrides: object) -> Orderbook:
    base: dict[str, object] = {
        "yes_ask": Decimal("0.40"),
        "yes_bid": Decimal("0.38"),
    }
    base.update(overrides)
    return Orderbook(**base)  # type: ignore[arg-type]


def _now() -> datetime:
    return datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)


def test_buy_yes_fills_at_yes_ask() -> None:
    trade = simulate_taker_fill(_intent(), _book(), _now())
    assert trade.simulated_price == Decimal("0.40")


def test_buy_yes_fee_uses_simulated_price() -> None:
    trade = simulate_taker_fill(_intent(contracts=10), _book(yes_ask=Decimal("0.40")), _now())
    assert trade.fee_dollars == taker_fee(10, Decimal("0.40"))


def test_buy_yes_preserves_intent_fields() -> None:
    now = _now()
    intent = _intent(
        market_ticker="KXHIGHDEN-26MAY05-T80",
        contracts=10,
        fair_yes=Decimal("0.50"),
        strategy="edge",
    )
    trade = simulate_taker_fill(intent, _book(), now)
    assert trade.market_ticker == "KXHIGHDEN-26MAY05-T80"
    assert trade.side is TradeSide.BUY_YES
    assert trade.contracts == 10
    assert trade.fair_at_entry == Decimal("0.50")
    assert trade.strategy == "edge"
    assert trade.intended_at == now


def test_sell_yes_fills_at_yes_bid() -> None:
    trade = simulate_taker_fill(
        _intent(side=TradeSide.SELL_YES, contracts=20, fair_yes=Decimal("0.20")),
        _book(yes_ask=Decimal("0.32"), yes_bid=Decimal("0.30")),
        _now(),
    )
    assert trade.simulated_price == Decimal("0.30")


def test_sell_yes_fee_uses_bid() -> None:
    trade = simulate_taker_fill(
        _intent(side=TradeSide.SELL_YES, contracts=20, fair_yes=Decimal("0.20")),
        _book(yes_ask=Decimal("0.32"), yes_bid=Decimal("0.30")),
        _now(),
    )
    assert trade.fee_dollars == taker_fee(20, Decimal("0.30"))


def test_decimal_hygiene() -> None:
    trade = simulate_taker_fill(_intent(), _book(), _now())
    assert isinstance(trade.simulated_price, Decimal)
    assert isinstance(trade.fee_dollars, Decimal)
    assert isinstance(trade.fair_at_entry, Decimal)
    assert isinstance(trade.intended_at, datetime)
    assert trade.intended_at.tzinfo is not None


def test_zero_contracts_raises() -> None:
    with pytest.raises(ValueError):
        simulate_taker_fill(_intent(contracts=0), _book(), _now())


def test_negative_contracts_raises() -> None:
    with pytest.raises(ValueError):
        simulate_taker_fill(_intent(contracts=-5), _book(), _now())


def test_naive_datetime_raises() -> None:
    naive_now = datetime(2026, 5, 5, 12, 0)
    with pytest.raises(ValueError):
        simulate_taker_fill(_intent(), _book(), naive_now)


def test_tail_price_buy() -> None:
    trade = simulate_taker_fill(
        _intent(contracts=10, fair_yes=Decimal("0.03")),
        _book(yes_ask=Decimal("0.05"), yes_bid=Decimal("0.03")),
        _now(),
    )
    assert trade.simulated_price == Decimal("0.05")
    assert trade.fee_dollars == taker_fee(10, Decimal("0.05"))


def test_paper_trade_is_frozen() -> None:
    trade = simulate_taker_fill(_intent(), _book(), _now())
    assert isinstance(trade, PaperTrade)
    with pytest.raises(Exception):
        trade.contracts = 99  # type: ignore[misc]


# bot.main.evaluate_strategies feeds one orderbook-anchored cost_per_contract scalar to the cap gate and the overlay update.
# bot.storage.positions.open_exposures reconstructs cross-cycle max-loss from paper_trades.simulated_price.
# If a future slippage haircut alters simulate_taker_fill, the equality pins below break; updating them silently
# decouples the in-cycle cap overlay from the cross-cycle exposure reconstruction.
@pytest.mark.parametrize(
    "side,yes_ask,yes_bid",
    [
        (TradeSide.BUY_YES, Decimal("0.40"), Decimal("0.38")),
        (TradeSide.BUY_YES, Decimal("0.05"), Decimal("0.03")),
        (TradeSide.BUY_YES, Decimal("0.987654"), Decimal("0.987650")),
        (TradeSide.SELL_YES, Decimal("0.40"), Decimal("0.38")),
        (TradeSide.SELL_YES, Decimal("0.05"), Decimal("0.03")),
        (TradeSide.SELL_YES, Decimal("0.987654"), Decimal("0.987650")),
    ],
)
def test_simulate_taker_fill_orderbook_verbatim_for_cap_overlay_coupling(
    side: TradeSide, yes_ask: Decimal, yes_bid: Decimal
) -> None:
    book = _book(yes_ask=yes_ask, yes_bid=yes_bid)
    trade = simulate_taker_fill(_intent(side=side), book, _now())
    expected = yes_ask if side is TradeSide.BUY_YES else yes_bid
    assert trade.simulated_price == expected
    assert trade.simulated_price.compare_total(expected) == Decimal("0")
