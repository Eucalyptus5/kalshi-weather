from __future__ import annotations

from decimal import Decimal

import pytest

from bot.execution.fees import FEE_QUANTUM, MAKER_RATE, TAKER_RATE, maker_fee, taker_fee


@pytest.mark.parametrize(
    "contracts, price, expected",
    [
        (1, Decimal("0.5"), Decimal("0.02")),
        (100, Decimal("0.5"), Decimal("1.75")),
        (1, Decimal("0.20"), Decimal("0.02")),
        (1, Decimal("0.80"), Decimal("0.02")),
        (1, Decimal("0.10"), Decimal("0.01")),
        (1, Decimal("0.90"), Decimal("0.01")),
        (0, Decimal("0.5"), Decimal("0")),
    ],
)
def test_taker_golden_table(contracts: int, price: Decimal, expected: Decimal) -> None:
    assert taker_fee(contracts, price) == expected


def test_round_trip_taker_at_mid_is_four_cents() -> None:
    assert 2 * taker_fee(1, Decimal("0.5")) == Decimal("0.04")


def test_round_trip_taker_at_twenty_eighty_is_four_cents() -> None:
    assert 2 * taker_fee(1, Decimal("0.20")) == Decimal("0.04")


def test_maker_golden_mid() -> None:
    assert maker_fee(1, Decimal("0.5")) == Decimal("0.01")


def test_round_trip_maker_at_mid_is_two_cents() -> None:
    assert 2 * maker_fee(1, Decimal("0.5")) == Decimal("0.02")


def test_maker_rate_is_exactly_a_quarter_of_taker_rate() -> None:
    assert 4 * MAKER_RATE == TAKER_RATE


@pytest.mark.parametrize(
    "price",
    [Decimal("0.30"), Decimal("0.5"), Decimal("0.10"), Decimal("0.90")],
)
def test_four_maker_fees_are_at_least_a_taker_fee(price: Decimal) -> None:
    assert 4 * maker_fee(1, price) >= taker_fee(1, price)


def test_independent_ceilings_break_the_four_to_one_fee_identity() -> None:
    assert maker_fee(1, Decimal("0.5")) == Decimal("0.01")
    assert taker_fee(1, Decimal("0.5")) == Decimal("0.02")
    assert 4 * maker_fee(1, Decimal("0.5")) != taker_fee(1, Decimal("0.5"))


def test_ceiling_rounds_up_a_hair_over_the_cent_boundary() -> None:
    assert taker_fee(27, Decimal("0.10")) == Decimal("0.18")
    assert taker_fee(27, Decimal("0.10")) != Decimal("0.17")


def test_an_exact_cent_is_not_pushed_to_the_next_cent() -> None:
    assert taker_fee(4, Decimal("0.50")) == Decimal("0.07")
    assert taker_fee(100, Decimal("0.50")) == Decimal("1.75")


@pytest.mark.parametrize("contracts", [-1, -100])
def test_negative_contracts_raises_taker(contracts: int) -> None:
    with pytest.raises(ValueError):
        taker_fee(contracts, Decimal("0.5"))


@pytest.mark.parametrize("contracts", [-1, -100])
def test_negative_contracts_raises_maker(contracts: int) -> None:
    with pytest.raises(ValueError):
        maker_fee(contracts, Decimal("0.5"))


@pytest.mark.parametrize(
    "price",
    [Decimal("-0.01"), Decimal("-1"), Decimal("1.01"), Decimal("2")],
)
def test_out_of_range_price_raises_taker(price: Decimal) -> None:
    with pytest.raises(ValueError):
        taker_fee(1, price)


@pytest.mark.parametrize(
    "price",
    [Decimal("-0.01"), Decimal("1.01")],
)
def test_out_of_range_price_raises_maker(price: Decimal) -> None:
    with pytest.raises(ValueError):
        maker_fee(1, price)


def test_price_zero_accepted() -> None:
    assert taker_fee(1, Decimal("0")) == Decimal("0")
    assert maker_fee(1, Decimal("0")) == Decimal("0")


def test_price_one_accepted() -> None:
    assert taker_fee(1, Decimal("1")) == Decimal("0")
    assert maker_fee(1, Decimal("1")) == Decimal("0")


def test_constants() -> None:
    assert TAKER_RATE == Decimal("0.07")
    assert MAKER_RATE == Decimal("0.0175")
    assert FEE_QUANTUM == Decimal("0.01")


def test_taker_returns_decimal() -> None:
    assert isinstance(taker_fee(10, Decimal("0.40")), Decimal)


def test_maker_returns_decimal() -> None:
    assert isinstance(maker_fee(10, Decimal("0.40")), Decimal)


def test_taker_fee_at_0905_is_a_cent() -> None:
    assert taker_fee(1, Decimal("0.905")) == Decimal("0.01")


def test_tails_boundary_modeled_headroom_is_a_cent() -> None:
    sell_price = Decimal("1") - Decimal("0.095")
    fee = taker_fee(1, sell_price)
    assert fee == Decimal("0.01")
    headroom = Decimal("0.025") - fee - Decimal("0.005")
    assert headroom == Decimal("0.010")
