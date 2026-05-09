from decimal import ROUND_CEILING, Decimal


TAKER_RATE: Decimal = Decimal("0.07")
MAKER_RATE: Decimal = Decimal("0.0175")
# kalshi 6-decimal fixed-point precision (march 2026 _dollars migration)
FEE_QUANTUM: Decimal = Decimal("0.000001")


def _validate(contracts: int, price: Decimal) -> None:
    if contracts < 0:
        raise ValueError(f"contracts must be >= 0, got {contracts}")
    if price < Decimal("0") or price > Decimal("1"):
        raise ValueError(f"price must be in [0, 1], got {price}")


def taker_fee(contracts: int, price: Decimal) -> Decimal:
    _validate(contracts, price)
    raw = TAKER_RATE * Decimal(contracts) * price * (Decimal("1") - price)
    return raw.quantize(FEE_QUANTUM, rounding=ROUND_CEILING)


def maker_fee(contracts: int, price: Decimal) -> Decimal:
    _validate(contracts, price)
    raw = MAKER_RATE * Decimal(contracts) * price * (Decimal("1") - price)
    return raw.quantize(FEE_QUANTUM, rounding=ROUND_CEILING)
