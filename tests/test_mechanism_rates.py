from __future__ import annotations

import decimal
from decimal import Decimal

import pytest

from bot.lag.mechanism_rates import (
    DECODE_DEFECT,
    MECHANISM_ORDER,
    MISSING_OBSERVATION,
    ROUNDING_DIFFERENCE,
    THRESHOLDS,
    WINDOW_DIFFERENCE,
    MechanismRate,
    first_matching,
    implies_opposite_sides,
    mechanism_rate,
    separating_strikes,
)


SWEEP_DENOMINATORS = (3, 7, 28, 200, 201, 300, 1000, 1400)


def sweep_pairs() -> tuple[tuple[str, int, int], ...]:
    pairs: list[tuple[str, int, int]] = []
    for mechanism in MECHANISM_ORDER:
        for denominator in SWEEP_DENOMINATORS:
            coarse = range(0, denominator + 1, max(1, denominator // 8))
            fine = range(0, min(denominator, 20) + 1)
            for numerator in sorted({*coarse, *fine, denominator}):
                pairs.append((mechanism, numerator, denominator))
    return tuple(pairs)


def test_the_order_is_the_preregistered_one() -> None:
    assert MECHANISM_ORDER == (
        "decode_defect",
        "rounding_difference",
        "window_difference",
        "missing_observation",
    )
    assert (DECODE_DEFECT, ROUNDING_DIFFERENCE, WINDOW_DIFFERENCE, MISSING_OBSERVATION) == (
        MECHANISM_ORDER
    )


def test_the_thresholds_are_the_preregistered_ones() -> None:
    assert THRESHOLDS == {
        DECODE_DEFECT: Decimal("0.005"),
        ROUNDING_DIFFERENCE: Decimal("0.005"),
        WINDOW_DIFFERENCE: Decimal("0.005"),
        MISSING_OBSERVATION: Decimal("0.10"),
    }
    assert set(THRESHOLDS) == set(MECHANISM_ORDER)


@pytest.mark.parametrize(
    "mechanism, numerator, denominator, matched",
    [
        (DECODE_DEFECT, 5, 1000, True),
        (DECODE_DEFECT, 4, 1000, False),
        (DECODE_DEFECT, 1, 200, True),
        (DECODE_DEFECT, 1, 201, False),
        (ROUNDING_DIFFERENCE, 1, 200, True),
        (ROUNDING_DIFFERENCE, 1, 201, False),
        (WINDOW_DIFFERENCE, 5, 1000, True),
        (WINDOW_DIFFERENCE, 4, 1000, False),
        (MISSING_OBSERVATION, 100, 1000, True),
        (MISSING_OBSERVATION, 99, 1000, False),
        (MISSING_OBSERVATION, 1, 10, True),
    ],
)
def test_the_threshold_boundary_is_inclusive(
    mechanism: str, numerator: int, denominator: int, matched: bool
) -> None:
    assert mechanism_rate(mechanism, numerator, denominator).matched is matched


def test_the_historical_figure_the_check_remeasures() -> None:
    hit = mechanism_rate(MISSING_OBSERVATION, 6, 28)

    assert hit.matched is True
    assert hit.rate == Decimal("0.214285")
    assert hit.threshold == Decimal("0.10")
    assert mechanism_rate(MISSING_OBSERVATION, 2, 28).matched is False


@pytest.mark.parametrize("mechanism, numerator, denominator", sweep_pairs())
def test_the_reported_rate_never_disagrees_with_the_exact_predicate(
    mechanism: str, numerator: int, denominator: int
) -> None:
    hit = mechanism_rate(mechanism, numerator, denominator)

    assert (hit.rate >= hit.threshold) is hit.matched
    assert hit.rate.as_tuple().exponent == -6


@pytest.mark.parametrize("precision", [6, 50])
def test_the_predicate_does_not_move_with_the_ambient_decimal_precision(precision: int) -> None:
    reference = {
        (mechanism, numerator, denominator): mechanism_rate(mechanism, numerator, denominator)
        for mechanism, numerator, denominator in sweep_pairs()
    }

    with decimal.localcontext() as ctx:
        ctx.prec = precision
        for key, expected in reference.items():
            assert mechanism_rate(*key) == expected


def test_a_denominator_of_zero_raises_naming_the_mechanism() -> None:
    with pytest.raises(ValueError, match=WINDOW_DIFFERENCE):
        mechanism_rate(WINDOW_DIFFERENCE, 0, 0)


def test_a_numerator_over_the_denominator_raises() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        mechanism_rate(DECODE_DEFECT, 9, 8)


def test_an_unknown_mechanism_raises() -> None:
    with pytest.raises(ValueError, match="daylight_saving"):
        mechanism_rate("daylight_saving", 1, 2)


def test_an_exact_hit_on_a_strike_sits_below_it() -> None:
    strikes = [87, 88, 89, 90, 91]

    assert separating_strikes(Decimal("88.0"), Decimal("90.0"), strikes) == (88, 89)
    assert 87 not in separating_strikes(Decimal("88.0"), Decimal("90.0"), strikes)
    assert 90 not in separating_strikes(Decimal("88.0"), Decimal("90.0"), strikes)
    assert 91 not in separating_strikes(Decimal("88.0"), Decimal("90.0"), strikes)
    assert separating_strikes(Decimal("88.0"), Decimal("87.5"), strikes) == ()
    assert separating_strikes(Decimal("88.0"), Decimal("88.2"), strikes) == (88,)


def test_separating_strikes_comes_back_ascending_and_order_free() -> None:
    strikes = [91, 88, 90, 87, 89]

    assert separating_strikes(Decimal("87.0"), Decimal("91.0"), strikes) == (87, 88, 89, 90)
    assert separating_strikes(Decimal("91.0"), Decimal("87.0"), strikes) == (87, 88, 89, 90)


def test_equal_readings_separate_nothing() -> None:
    strikes = [87, 88, 89]

    assert separating_strikes(Decimal("88.6"), Decimal("88.6"), strikes) == ()
    assert implies_opposite_sides(Decimal("88.6"), Decimal("88.6"), strikes) is False
    assert implies_opposite_sides(Decimal("88.6"), Decimal("89.4"), strikes) is True
    assert implies_opposite_sides(Decimal("88.6"), Decimal("89.4"), []) is False


def test_the_first_matching_row_is_read_in_the_preregistered_order() -> None:
    decode = mechanism_rate(DECODE_DEFECT, 5, 1000)
    missing = mechanism_rate(MISSING_OBSERVATION, 6, 28)
    window = mechanism_rate(WINDOW_DIFFERENCE, 4, 1000)

    assert first_matching([missing, window, decode]) is decode
    assert first_matching([decode, missing]) is decode
    assert first_matching([missing]) is missing
    assert first_matching([window]) is None
    assert first_matching([]) is None


def test_the_rate_row_is_frozen_and_slotted() -> None:
    hit = mechanism_rate(DECODE_DEFECT, 5, 1000)

    assert isinstance(hit, MechanismRate)
    with pytest.raises(AttributeError):
        hit.matched = False
    with pytest.raises(TypeError):
        MechanismRate(DECODE_DEFECT, 5, 1000, Decimal("0.005"), Decimal("0.005"), True)
