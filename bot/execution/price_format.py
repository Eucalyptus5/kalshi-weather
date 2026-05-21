from __future__ import annotations

from decimal import Decimal


PRICE_QUANTUM: Decimal = Decimal("0.0001")


def format_price_dollars(price: Decimal) -> str:
    if price < Decimal("0") or price > Decimal("1"):
        raise ValueError(f"price must be in [0, 1], got {price}")
    return f"{price.quantize(PRICE_QUANTUM):f}"
