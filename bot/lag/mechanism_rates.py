from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Context, Decimal


DECODE_DEFECT = "decode_defect"
ROUNDING_DIFFERENCE = "rounding_difference"
WINDOW_DIFFERENCE = "window_difference"
MISSING_OBSERVATION = "missing_observation"

MECHANISM_ORDER: tuple[str, ...] = (
    DECODE_DEFECT,
    ROUNDING_DIFFERENCE,
    WINDOW_DIFFERENCE,
    MISSING_OBSERVATION,
)
THRESHOLDS: Mapping[str, Decimal] = {
    DECODE_DEFECT: Decimal("0.005"),
    ROUNDING_DIFFERENCE: Decimal("0.005"),
    WINDOW_DIFFERENCE: Decimal("0.005"),
    MISSING_OBSERVATION: Decimal("0.10"),
}

REPORT_QUANTUM: Decimal = Decimal("0.000001")
# The quotient carries its own context so the reported figure is the same at every ambient
# precision, and truncates so it can never read as matched on a row the exact predicate missed.
REPORT_CONTEXT: Context = Context(prec=28, rounding=ROUND_DOWN)


@dataclass(frozen=True, slots=True, kw_only=True)
class MechanismRate:
    mechanism: str
    numerator: int
    denominator: int
    rate: Decimal
    threshold: Decimal
    matched: bool


# Stated as a multiplication because nothing in this repo pins the decimal context precision, and
# a quotient compared against the threshold would answer differently at different precisions.
# n / d >= t is n >= t * d for any positive d, and the second form is exact.
def mechanism_rate(mechanism: str, numerator: int, denominator: int) -> MechanismRate:
    if mechanism not in THRESHOLDS:
        raise ValueError(f"unknown mechanism: {mechanism}")
    if denominator == 0:
        raise ValueError(f"{mechanism} has no denominator: a rate over nothing is not a rate")
    if numerator > denominator:
        raise ValueError(
            f"{mechanism} numerator {numerator} exceeds denominator {denominator}",
        )
    threshold = THRESHOLDS[mechanism]
    quotient = REPORT_CONTEXT.divide(Decimal(numerator), Decimal(denominator))
    return MechanismRate(
        mechanism=mechanism,
        numerator=numerator,
        denominator=denominator,
        rate=quotient.quantize(REPORT_QUANTUM, context=REPORT_CONTEXT),
        threshold=threshold,
        matched=Decimal(numerator) >= threshold * Decimal(denominator),
    )


# Strictly greater on both sides: the venue's greater strike type settles yes above the strike, so
# a reading that lands exactly on a strike is below it.
def separating_strikes(first: Decimal, second: Decimal, strikes: Sequence[int]) -> tuple[int, ...]:
    return tuple(strike for strike in sorted(strikes) if (first > strike) != (second > strike))


def implies_opposite_sides(first: Decimal, second: Decimal, strikes: Sequence[int]) -> bool:
    return bool(separating_strikes(first, second, strikes))


def first_matching(rates: Sequence[MechanismRate]) -> MechanismRate | None:
    by_mechanism = {rate.mechanism: rate for rate in rates}
    for mechanism in MECHANISM_ORDER:
        rate = by_mechanism.get(mechanism)
        if rate is not None and rate.matched:
            return rate
    return None
