from __future__ import annotations

from decimal import Decimal

import pytest

from bot.execution import fees
from bot.execution.fees import FEE_QUANTUM, taker_fee
from bot.lag.fee_floor import (
    CENT,
    PUBLISHED_TAKER_RATE,
    FeeSource,
    fee_source,
    published_taker_fee,
    quantum_is_corrected,
)


@pytest.mark.parametrize(
    "contracts, price, expected",
    [
        (100, Decimal("0.05"), Decimal("0.34")),
        (1, Decimal("0.07"), Decimal("0.01")),
        (10, Decimal("0.07"), Decimal("0.05")),
        (1, Decimal("0.50"), Decimal("0.02")),
        (4, Decimal("0.50"), Decimal("0.07")),
        (100, Decimal("0.50"), Decimal("1.75")),
        (1000, Decimal("0.20"), Decimal("11.20")),
        (27, Decimal("0.10"), Decimal("0.18")),
        (0, Decimal("0.50"), Decimal("0")),
        (1, Decimal("0"), Decimal("0")),
        (1, Decimal("1"), Decimal("0")),
        (500, Decimal("0"), Decimal("0")),
        (500, Decimal("1"), Decimal("0")),
    ],
)
def test_published_floor_golden_table(contracts: int, price: Decimal, expected: Decimal) -> None:
    assert published_taker_fee(contracts, price) == expected


def test_hundred_at_five_cents_costs_thirty_four_cents_not_a_dollar() -> None:
    aggregate = published_taker_fee(100, Decimal("0.05"))
    per_contract_ceiling = 100 * published_taker_fee(1, Decimal("0.05"))

    assert aggregate == Decimal("0.34")
    assert per_contract_ceiling == Decimal("1.00")
    assert aggregate != per_contract_ceiling


def test_one_contract_at_seven_cents_pays_fourteen_percent_of_premium() -> None:
    fee = published_taker_fee(1, Decimal("0.07"))

    assert fee == Decimal("0.01")
    assert (fee / Decimal("0.07")).quantize(CENT) == Decimal("0.14")


def test_ten_contracts_at_seven_cents_pay_half_a_cent_each() -> None:
    one = published_taker_fee(1, Decimal("0.07"))
    ten = published_taker_fee(10, Decimal("0.07"))

    assert ten == Decimal("0.05")
    assert ten / Decimal("10") == Decimal("0.005")
    assert ten / Decimal("10") == one / Decimal("2")


def test_a_hundredth_of_a_cent_over_a_boundary_rounds_up_to_the_next_cent() -> None:
    assert published_taker_fee(27, Decimal("0.10")) == Decimal("0.18")
    assert published_taker_fee(27, Decimal("0.10")) != Decimal("0.17")


def test_an_exact_cent_boundary_is_not_pushed_to_the_next_cent() -> None:
    assert published_taker_fee(4, Decimal("0.50")) == Decimal("0.07")
    assert published_taker_fee(100, Decimal("0.50")) == Decimal("1.75")


@pytest.mark.parametrize("contracts", [0, 1, 10, 250])
@pytest.mark.parametrize("price", [Decimal("0.03"), Decimal("0.41"), Decimal("0.88")])
def test_every_floor_lands_on_the_cent(contracts: int, price: Decimal) -> None:
    fee = published_taker_fee(contracts, price)

    assert isinstance(fee, Decimal)
    assert fee.as_tuple().exponent == -2


@pytest.mark.parametrize("price", [Decimal("0.07"), Decimal("0.20"), Decimal("0.63")])
def test_floor_never_falls_below_a_cent_on_a_real_fill(price: Decimal) -> None:
    assert published_taker_fee(1, price) >= CENT


@pytest.mark.parametrize(
    "contracts",
    [0, 1, 3, 4, 10, 27, 100, 250, 1000],
)
@pytest.mark.parametrize(
    "price",
    [
        Decimal("0"),
        Decimal("0.01"),
        Decimal("0.05"),
        Decimal("0.07"),
        Decimal("0.10"),
        Decimal("0.20"),
        Decimal("0.39"),
        Decimal("0.50"),
        Decimal("0.63"),
        Decimal("0.87"),
        Decimal("0.905"),
        Decimal("0.99"),
        Decimal("1"),
    ],
)
def test_the_module_agrees_with_the_published_floor(contracts: int, price: Decimal) -> None:
    assert taker_fee(contracts, price) == published_taker_fee(contracts, price)


def test_the_module_reads_a_cent_where_the_floor_reads_a_cent() -> None:
    assert taker_fee(1, Decimal("0.07")) == Decimal("0.01")
    assert published_taker_fee(1, Decimal("0.07")) == Decimal("0.01")


def test_published_rate_is_the_schedule_rate() -> None:
    assert PUBLISHED_TAKER_RATE == Decimal("0.07")


def test_fee_source_names_the_published_formula_not_the_execution_module() -> None:
    source = fee_source()

    assert isinstance(source, FeeSource)
    assert source.threshold_source == "published_formula"
    assert source.fee_module == "bot.execution.fees.taker_fee"


def test_fee_source_reads_the_live_module_quantum_as_corrected() -> None:
    source = fee_source()

    assert source.fee_module_quantum == FEE_QUANTUM
    assert source.fee_module_quantum == CENT
    assert source.fee_module_corrected is True


def test_fee_source_follows_the_module_quantum_back_to_uncorrected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fees, "FEE_QUANTUM", Decimal("0.000001"))
    source = fee_source()

    assert source.fee_module_quantum == Decimal("0.000001")
    assert source.fee_module_corrected is False


@pytest.mark.parametrize(
    "quantum, corrected",
    [
        (Decimal("0.000001"), False),
        (Decimal("0.0001"), False),
        (Decimal("0.01"), True),
        (Decimal("0.010"), True),
        (Decimal("0.1"), False),
        (Decimal("1"), False),
    ],
)
def test_correction_predicate_is_decided_by_the_quantum(quantum: Decimal, corrected: bool) -> None:
    assert quantum_is_corrected(quantum) is corrected
