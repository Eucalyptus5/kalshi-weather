from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from bot.execution import fees


PUBLISHED_TAKER_RATE: Decimal = Decimal("0.07")
CENT: Decimal = Decimal("0.01")
THRESHOLD_SOURCE = "published_formula"
FEE_MODULE = "bot.execution.fees.taker_fee"


@dataclass(frozen=True, slots=True)
class FeeSource:
    threshold_source: str
    fee_module: str
    fee_module_quantum: Decimal
    fee_module_corrected: bool


# Deliberately duplicates fees.taker_fee: that module quantizes to FEE_QUANTUM (1e-6), four
# orders finer than the cent an account settles to, and correcting it is a separate decision.
def published_taker_fee(contracts: int, price: Decimal) -> Decimal:
    raw = PUBLISHED_TAKER_RATE * Decimal(contracts) * price * (Decimal("1") - price)
    return raw.quantize(CENT, rounding=ROUND_CEILING)


def quantum_is_corrected(quantum: Decimal) -> bool:
    return quantum == CENT


def fee_source() -> FeeSource:
    return FeeSource(
        threshold_source=THRESHOLD_SOURCE,
        fee_module=FEE_MODULE,
        fee_module_quantum=fees.FEE_QUANTUM,
        fee_module_corrected=quantum_is_corrected(fees.FEE_QUANTUM),
    )
