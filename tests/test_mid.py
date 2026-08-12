from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from bot.lag.mid import mid2, mid2_array, ticks, two_sided, two_sided_array


def test_mid_is_the_doubled_tick_sum_of_the_two_quotes() -> None:
    assert mid2(ticks(Decimal("0.94")), ticks(Decimal("0.04"))) == 19_000


def test_a_half_tick_mid_stays_exact_because_it_is_left_doubled() -> None:
    assert mid2(3_333, 6_600) == 6_733


def test_the_mid_is_an_integer_not_a_quotient() -> None:
    value = mid2(4_000, 5_000)

    assert isinstance(value, int)
    assert value == 9_000


def test_a_zero_no_bid_is_never_two_sided() -> None:
    assert not two_sided(9_900, 0)
    assert not two_sided(9_999, 0, yes_depth=500, no_depth=500)


def test_a_zero_yes_bid_is_never_two_sided() -> None:
    assert not two_sided(0, 9_900)
    assert not two_sided(0, 9_999, yes_depth=500, no_depth=500)


def test_a_book_priced_on_both_sides_but_empty_on_one_is_not_two_sided() -> None:
    assert not two_sided(4_000, 5_000, yes_depth=30, no_depth=0)
    assert not two_sided(4_000, 5_000, yes_depth=0, no_depth=30)
    assert two_sided(4_000, 5_000, yes_depth=30, no_depth=30)


def test_depth_left_unrecorded_falls_back_to_the_prices() -> None:
    assert two_sided(4_000, 5_000)
    assert not two_sided(4_000, 0)


def test_the_array_forms_agree_with_the_scalar_forms() -> None:
    yes_bid = np.array([9_900, 0, 4_000, 4_000, 3_333], dtype=np.int64)
    no_bid = np.array([0, 9_900, 5_000, 5_000, 6_600], dtype=np.int64)
    yes_depth = np.array([500, 500, 30, 0, 30], dtype=np.int64)
    no_depth = np.array([500, 500, 30, 30, 30], dtype=np.int64)

    mids = mid2_array(yes_bid, no_bid)
    flags = two_sided_array(yes_bid, no_bid, yes_depth=yes_depth, no_depth=no_depth)

    assert mids.tolist() == [mid2(int(y), int(n)) for y, n in zip(yes_bid, no_bid, strict=True)]
    assert flags.tolist() == [
        two_sided(int(y), int(n), yes_depth=int(yd), no_depth=int(nd))
        for y, n, yd, nd in zip(yes_bid, no_bid, yes_depth, no_depth, strict=True)
    ]


def test_the_array_form_agrees_with_the_scalar_form_when_depth_is_unrecorded() -> None:
    yes_bid = np.array([9_900, 0, 4_000], dtype=np.int64)
    no_bid = np.array([0, 9_900, 5_000], dtype=np.int64)

    flags = two_sided_array(yes_bid, no_bid)

    assert flags.tolist() == [
        two_sided(int(y), int(n)) for y, n in zip(yes_bid, no_bid, strict=True)
    ]


def test_a_price_off_the_tick_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="off the 10000-per-dollar grid"):
        ticks(Decimal("0.123456"))


def test_a_four_decimal_price_is_on_the_grid() -> None:
    assert ticks(Decimal("0.9500")) == 9_500
    assert ticks(Decimal("1.00")) == 10_000
    assert ticks(Decimal("0")) == 0
