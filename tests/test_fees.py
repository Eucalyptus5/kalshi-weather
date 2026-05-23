from __future__ import annotations

from decimal import Decimal

import pytest

from bot.execution.fees import FEE_QUANTUM, MAKER_RATE, TAKER_RATE, maker_fee, taker_fee


@pytest.mark.parametrize(
    "contracts, price, expected",
    [
        (1, Decimal("0.5"), Decimal("0.017500")),
        (100, Decimal("0.5"), Decimal("1.750000")),
        (1, Decimal("0.20"), Decimal("0.011200")),
        (1, Decimal("0.80"), Decimal("0.011200")),
        (1, Decimal("0.10"), Decimal("0.006300")),
        (1, Decimal("0.90"), Decimal("0.006300")),
        (0, Decimal("0.5"), Decimal("0")),
    ],
)
def test_taker_golden_table(contracts: int, price: Decimal, expected: Decimal) -> None:
    assert taker_fee(contracts, price) == expected


def test_round_trip_taker_at_mid_is_three_and_a_half_cents() -> None:
    assert 2 * taker_fee(1, Decimal("0.5")) == Decimal("0.035000")


def test_round_trip_taker_at_twenty_eighty_is_about_two_point_two_cents() -> None:
    rt = 2 * taker_fee(1, Decimal("0.20"))
    assert Decimal("0.022") <= rt <= Decimal("0.024")
    assert rt == Decimal("0.022400")


def test_maker_golden_mid() -> None:
    assert maker_fee(1, Decimal("0.5")) == Decimal("0.004375")


def test_round_trip_maker_at_mid_is_about_eighty_eight_hundredths_of_a_cent() -> None:
    assert 2 * maker_fee(1, Decimal("0.5")) == Decimal("0.008750")


@pytest.mark.parametrize(
    "price",
    [Decimal("0.30"), Decimal("0.5"), Decimal("0.10"), Decimal("0.90")],
)
def test_maker_is_quarter_of_taker(price: Decimal) -> None:
    assert 4 * maker_fee(1, price) == taker_fee(1, price)


def test_ceiling_rounds_up_at_six_decimal_boundary() -> None:
    assert taker_fee(1, Decimal("0.123456789")) == Decimal("0.007576")


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
    assert FEE_QUANTUM == Decimal("0.000001")


def test_taker_returns_decimal() -> None:
    assert isinstance(taker_fee(10, Decimal("0.40")), Decimal)


def test_maker_returns_decimal() -> None:
    assert isinstance(maker_fee(10, Decimal("0.40")), Decimal)


def test_taker_fee_at_0905_is_six_thousand_nineteen_millionths() -> None:
    assert taker_fee(1, Decimal("0.905")) == Decimal("0.006019")


def test_tails_boundary_modeled_headroom_is_zero_point_zero_one_three_nine_eight_one() -> None:
    sell_price = Decimal("1") - Decimal("0.095")
    fee = taker_fee(1, sell_price)
    assert fee == Decimal("0.006019")
    headroom = Decimal("0.025") - fee - Decimal("0.005")
    assert headroom == Decimal("0.013981")
