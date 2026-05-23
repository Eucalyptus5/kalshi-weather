from __future__ import annotations

from decimal import Decimal

from bot.execution.fees import FEE_QUANTUM, taker_fee


# real kalshi weather spreads run 3-8c but vary per market and time-of-day;
# the constant is a conservative placeholder until per-book spread reads land.
_HALF_TICK: Decimal = Decimal("0.005")
_ADVERSE_SELECTION_PER_UNIT: Decimal = Decimal("0.01")
_ADVERSE_SELECTION_CAP: Decimal = Decimal("0.05")


def required_edge(price: Decimal, depth_at_price: int, contracts: int) -> Decimal:
    fee = taker_fee(1, price)
    if contracts <= depth_at_price:
        adverse_selection = Decimal("0")
    else:
        overflow = Decimal(contracts - depth_at_price)
        denom = Decimal(max(depth_at_price, 1))
        raw = _ADVERSE_SELECTION_PER_UNIT * overflow / denom
        adverse_selection = min(raw, _ADVERSE_SELECTION_CAP)
    total = fee + _HALF_TICK + adverse_selection
    return total.quantize(FEE_QUANTUM)
