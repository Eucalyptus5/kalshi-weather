from __future__ import annotations

from decimal import Decimal

import pytest

from bot.execution.price_format import PRICE_QUANTUM, format_price_dollars


def test_format_pads_two_decimal_input_to_four() -> None:
    assert format_price_dollars(Decimal("0.85")) == "0.8500"


def test_str_decimal_counter_example_does_not_pad() -> None:
    assert str(Decimal("0.85")) == "0.85"


def test_format_rounds_to_quantum() -> None:
    result = format_price_dollars(Decimal("0.12345"))
    assert result.count(".") == 1
    fractional = result.split(".")[1]
    assert len(fractional) == 4


def test_format_one_minus_yes_bid_complement() -> None:
    yes_bid = Decimal("0.0750")
    assert format_price_dollars(Decimal("1") - yes_bid) == "0.9250"


@pytest.mark.parametrize("bad", [Decimal("-0.01"), Decimal("1.01")])
def test_format_rejects_out_of_range(bad: Decimal) -> None:
    with pytest.raises(ValueError):
        format_price_dollars(bad)


def test_format_accepts_endpoints() -> None:
    assert format_price_dollars(Decimal("0")) == "0.0000"
    assert format_price_dollars(Decimal("1")) == "1.0000"


def test_quantum_is_four_decimal_places() -> None:
    assert PRICE_QUANTUM == Decimal("0.0001")
