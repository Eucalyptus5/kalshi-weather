from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from bot.execution import fees


PUBLISHED_TAKER_RATE: Decimal = Decimal("0.07")
PUBLISHED_MAKER_RATE: Decimal = Decimal("0.0175")
CENT: Decimal = Decimal("0.01")
THRESHOLD_SOURCE = "published_formula"
FEE_MODULE = "bot.execution.fees.taker_fee"
MAKER_RATE_SOURCE = "published_formula"


@dataclass(frozen=True, slots=True)
class FeeSource:
    threshold_source: str
    fee_module: str
    fee_module_quantum: Decimal
    fee_module_corrected: bool
    maker_rate: Decimal
    maker_rate_source: str


# Deliberately duplicates fees.taker_fee rather than delegating to it: this is the oracle that
# module is checked against, and delegating would make fee_module_corrected a tautology. It takes
# the size as a Decimal because the wire prints fractional counts that fees.taker_fee cannot say.
def published_taker_fee(contracts: Decimal, price: Decimal) -> Decimal:
    raw = PUBLISHED_TAKER_RATE * contracts * price * (Decimal("1") - price)
    return raw.quantize(CENT, rounding=ROUND_CEILING)


# The rate is a parameter rather than a module constant: which fee regime priced a run is a
# finding, not a build-time decision, and the caller records its choice in FeeSource.
def published_maker_fee(contracts: Decimal, price: Decimal, rate: Decimal) -> Decimal:
    raw = rate * contracts * price * (Decimal("1") - price)
    return raw.quantize(CENT, rounding=ROUND_CEILING)


def quantum_is_corrected(quantum: Decimal) -> bool:
    return quantum == CENT


def fee_source() -> FeeSource:
    return FeeSource(
        threshold_source=THRESHOLD_SOURCE,
        fee_module=FEE_MODULE,
        fee_module_quantum=fees.FEE_QUANTUM,
        fee_module_corrected=quantum_is_corrected(fees.FEE_QUANTUM),
        maker_rate=PUBLISHED_MAKER_RATE,
        maker_rate_source=MAKER_RATE_SOURCE,
    )
