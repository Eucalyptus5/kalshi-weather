from __future__ import annotations

import inspect
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

import pytest

from bot.execution import fees
from bot.execution.fees import FEE_QUANTUM, taker_fee
from bot.lag.fee_floor import (
    BAR_CONTEXT,
    CENT,
    MAKER_RATE_SOURCE,
    PUBLISHED_MAKER_RATE,
    PUBLISHED_TAKER_RATE,
    TICK_CENTS,
    FeeSource,
    economic_bar_cents_per_contract,
    fee_source,
    published_maker_fee,
    published_taker_fee,
    quantum_is_corrected,
)
from bot.lag.maker_headroom import PRICE_GRID


FOUR_DP = Decimal("0.0001")
HALF = Decimal("0.50")
BAR_SIZE = Decimal("26")
STATED_REGIME = ("maker_rate", "maker_rate_source")
ZERO_RATE = Decimal("0")
ZERO_RATE_SOURCE = "kalshi_series_metadata"
AMBIENT_PRECISIONS = (20, 28, 50)
PINNED_BAR = Decimal("2.769230769230769230769230769")
# Half-even and round-down agree on the bar at size 26 / price 0.50, so that case says nothing
# about which rounding is pinned. This pair is one of the inputs where the two part.
SPLIT_SIZE = Decimal("3")
SPLIT_PRICE = Decimal("0.06")
SPLIT_BAR = Decimal("1.666666666666666666666666667")


def _per_contract_cents(contracts: Decimal, price: Decimal, rate: Decimal) -> Decimal:
    aggregate = published_maker_fee(contracts, price, rate)
    return (Decimal(100) * aggregate / contracts).quantize(FOUR_DP)


@pytest.mark.parametrize(
    "contracts, price, expected",
    [
        (Decimal("100"), Decimal("0.05"), Decimal("0.34")),
        (Decimal("1"), Decimal("0.07"), Decimal("0.01")),
        (Decimal("10"), Decimal("0.07"), Decimal("0.05")),
        (Decimal("1"), Decimal("0.50"), Decimal("0.02")),
        (Decimal("4"), Decimal("0.50"), Decimal("0.07")),
        (Decimal("100"), Decimal("0.50"), Decimal("1.75")),
        (Decimal("1000"), Decimal("0.20"), Decimal("11.20")),
        (Decimal("27"), Decimal("0.10"), Decimal("0.18")),
        (Decimal("0"), Decimal("0.50"), Decimal("0")),
        (Decimal("1"), Decimal("0"), Decimal("0")),
        (Decimal("1"), Decimal("1"), Decimal("0")),
        (Decimal("500"), Decimal("0"), Decimal("0")),
        (Decimal("500"), Decimal("1"), Decimal("0")),
        (Decimal("0.01"), Decimal("0.50"), Decimal("0.01")),
        (Decimal("0.39"), Decimal("0.50"), Decimal("0.01")),
        (Decimal("1.24"), Decimal("0.50"), Decimal("0.03")),
        (Decimal("46.51"), Decimal("0.50"), Decimal("0.82")),
        (Decimal("60015.36"), Decimal("0.50"), Decimal("1050.27")),
    ],
)
def test_published_floor_golden_table(
    contracts: Decimal, price: Decimal, expected: Decimal
) -> None:
    assert published_taker_fee(contracts, price) == expected


def test_hundred_at_five_cents_costs_thirty_four_cents_not_a_dollar() -> None:
    aggregate = published_taker_fee(Decimal("100"), Decimal("0.05"))
    per_contract_ceiling = 100 * published_taker_fee(Decimal("1"), Decimal("0.05"))

    assert aggregate == Decimal("0.34")
    assert per_contract_ceiling == Decimal("1.00")
    assert aggregate != per_contract_ceiling


def test_one_contract_at_seven_cents_pays_fourteen_percent_of_premium() -> None:
    fee = published_taker_fee(Decimal("1"), Decimal("0.07"))

    assert fee == Decimal("0.01")
    assert (fee / Decimal("0.07")).quantize(CENT) == Decimal("0.14")


def test_ten_contracts_at_seven_cents_pay_half_a_cent_each() -> None:
    one = published_taker_fee(Decimal("1"), Decimal("0.07"))
    ten = published_taker_fee(Decimal("10"), Decimal("0.07"))

    assert ten == Decimal("0.05")
    assert ten / Decimal("10") == Decimal("0.005")
    assert ten / Decimal("10") == one / Decimal("2")


def test_a_hundredth_of_a_cent_over_a_boundary_rounds_up_to_the_next_cent() -> None:
    assert published_taker_fee(Decimal("27"), Decimal("0.10")) == Decimal("0.18")
    assert published_taker_fee(Decimal("27"), Decimal("0.10")) != Decimal("0.17")


def test_an_exact_cent_boundary_is_not_pushed_to_the_next_cent() -> None:
    assert published_taker_fee(Decimal("4"), Decimal("0.50")) == Decimal("0.07")
    assert published_taker_fee(Decimal("100"), Decimal("0.50")) == Decimal("1.75")


@pytest.mark.parametrize(
    "contracts", [Decimal("0"), Decimal("0.01"), Decimal("1"), Decimal("10.09"), Decimal("250")]
)
@pytest.mark.parametrize("price", [Decimal("0.03"), Decimal("0.41"), Decimal("0.88")])
def test_every_floor_lands_on_the_cent(contracts: Decimal, price: Decimal) -> None:
    fee = published_taker_fee(contracts, price)

    assert isinstance(fee, Decimal)
    assert fee.as_tuple().exponent == -2


@pytest.mark.parametrize("price", [Decimal("0.07"), Decimal("0.20"), Decimal("0.63")])
def test_floor_never_falls_below_a_cent_on_a_real_fill(price: Decimal) -> None:
    assert published_taker_fee(Decimal("1"), price) >= CENT


@pytest.mark.parametrize("contracts", [Decimal("0.01"), Decimal("0.39"), Decimal("0.99")])
def test_a_fill_under_one_contract_still_pays_the_cent(contracts: Decimal) -> None:
    assert published_taker_fee(contracts, Decimal("0.50")) >= CENT


def test_a_size_far_under_a_contract_pays_the_whole_cent_anyway() -> None:
    raw = PUBLISHED_TAKER_RATE * Decimal("0.39") * Decimal("0.5") * Decimal("0.5")

    assert raw == Decimal("0.0068250")
    assert published_taker_fee(Decimal("0.39"), Decimal("0.50")) == CENT


def test_a_fractional_size_prices_between_the_two_contracts_it_sits_between() -> None:
    rate = PUBLISHED_TAKER_RATE * Decimal("0.5") * Decimal("0.5")
    one = rate * Decimal("11")
    between = rate * Decimal("11.09")
    two = rate * Decimal("12")

    assert one < between < two
    assert published_taker_fee(Decimal("11"), Decimal("0.50")) == Decimal("0.20")
    assert published_taker_fee(Decimal("11.09"), Decimal("0.50")) == Decimal("0.20")
    assert published_taker_fee(Decimal("11.43"), Decimal("0.50")) == Decimal("0.21")
    assert published_taker_fee(Decimal("12"), Decimal("0.50")) == Decimal("0.21")


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
    assert taker_fee(contracts, price) == published_taker_fee(Decimal(contracts), price)


def test_the_module_reads_a_cent_where_the_floor_reads_a_cent() -> None:
    assert taker_fee(1, Decimal("0.07")) == Decimal("0.01")
    assert published_taker_fee(Decimal("1"), Decimal("0.07")) == Decimal("0.01")


def test_published_rate_is_the_schedule_rate() -> None:
    assert PUBLISHED_TAKER_RATE == Decimal("0.07")


def test_fee_source_names_the_published_formula_not_the_execution_module() -> None:
    source = fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE)

    assert isinstance(source, FeeSource)
    assert source.threshold_source == "published_formula"
    assert source.fee_module == "bot.execution.fees.taker_fee"


def test_fee_source_reads_the_live_module_quantum_as_corrected() -> None:
    source = fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE)

    assert source.fee_module_quantum == FEE_QUANTUM
    assert source.fee_module_quantum == CENT
    assert source.fee_module_corrected is True


def test_fee_source_follows_the_module_quantum_back_to_uncorrected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fees, "FEE_QUANTUM", Decimal("0.000001"))
    source = fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE)

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


def test_fee_source_carries_the_published_maker_rate_and_its_provenance() -> None:
    source = fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE)

    assert source.maker_rate == PUBLISHED_MAKER_RATE
    assert source.maker_rate == Decimal("0.0175")
    assert source.maker_rate_source == MAKER_RATE_SOURCE


def test_fee_source_carries_a_stated_zero_regime_instead_of_the_published_one() -> None:
    source = fee_source(maker_rate=ZERO_RATE, maker_rate_source=ZERO_RATE_SOURCE)

    assert source.maker_rate == ZERO_RATE
    assert source.maker_rate_source == ZERO_RATE_SOURCE
    assert PUBLISHED_MAKER_RATE == Decimal("0.0175")


def test_fee_source_states_no_default_regime() -> None:
    parameters = inspect.signature(fee_source).parameters

    assert tuple(parameters) == STATED_REGIME
    for name in STATED_REGIME:
        assert parameters[name].default is inspect.Parameter.empty
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_fee_source_refuses_to_price_a_run_that_states_no_regime() -> None:
    with pytest.raises(TypeError, match="maker_rate"):
        fee_source()


@pytest.mark.parametrize("price", PRICE_GRID)
def test_a_zero_maker_regime_charges_nothing_across_the_grid(price: Decimal) -> None:
    fee = published_maker_fee(BAR_SIZE, price, ZERO_RATE)

    assert fee == Decimal("0.00")
    assert fee.as_tuple().exponent == -2


@pytest.mark.parametrize(
    "contracts, aggregate, per_contract",
    [
        (Decimal("12"), Decimal("0.06"), Decimal("0.5000")),
        (Decimal("12.36"), Decimal("0.06"), Decimal("0.4854")),
        (Decimal("13"), Decimal("0.06"), Decimal("0.4615")),
        (Decimal("14"), Decimal("0.07"), Decimal("0.5000")),
        (Decimal("15"), Decimal("0.07"), Decimal("0.4667")),
        (Decimal("16"), Decimal("0.07"), Decimal("0.4375")),
        (Decimal("17"), Decimal("0.08"), Decimal("0.4706")),
        (Decimal("18"), Decimal("0.08"), Decimal("0.4444")),
        (Decimal("19"), Decimal("0.09"), Decimal("0.4737")),
        (Decimal("20"), Decimal("0.09"), Decimal("0.4500")),
        (Decimal("21"), Decimal("0.10"), Decimal("0.4762")),
        (Decimal("22"), Decimal("0.10"), Decimal("0.4545")),
        (Decimal("23"), Decimal("0.11"), Decimal("0.4783")),
        (Decimal("24"), Decimal("0.11"), Decimal("0.4583")),
        (Decimal("25"), Decimal("0.11"), Decimal("0.4400")),
        (Decimal("26"), Decimal("0.12"), Decimal("0.4615")),
    ],
)
def test_maker_fee_golden_table_at_half_price(
    contracts: Decimal, aggregate: Decimal, per_contract: Decimal
) -> None:
    fee = published_maker_fee(contracts, HALF, PUBLISHED_MAKER_RATE)

    assert fee == aggregate
    assert (Decimal(100) * fee / contracts).quantize(FOUR_DP) == per_contract


def test_the_half_price_band_bottoms_at_16_and_the_non_half_cells_top_at_23() -> None:
    per_contract = {
        n: _per_contract_cents(Decimal(n), HALF, PUBLISHED_MAKER_RATE) for n in range(12, 27)
    }

    assert per_contract[16] == Decimal("0.4375")
    assert min(per_contract.values()) == Decimal("0.4375")
    non_half = [value for value in per_contract.values() if value != Decimal("0.5000")]
    assert max(non_half) == Decimal("0.4783")
    assert per_contract[23] == Decimal("0.4783")


def test_exactly_twelve_and_fourteen_land_on_a_half_cent_and_nothing_else_does() -> None:
    per_contract = {
        n: _per_contract_cents(Decimal(n), HALF, PUBLISHED_MAKER_RATE) for n in range(12, 27)
    }

    at_half = sorted(n for n, value in per_contract.items() if value == Decimal("0.5000"))
    assert at_half == [12, 14]
    assert all(value <= Decimal("0.5000") for value in per_contract.values())


@pytest.mark.parametrize(
    "contracts", [Decimal("12"), Decimal("12.36"), Decimal("19"), Decimal("26")]
)
def test_maker_fee_is_symmetric_around_the_midpoint(contracts: Decimal) -> None:
    low_tail = published_maker_fee(contracts, Decimal("0.05"), PUBLISHED_MAKER_RATE)
    high_tail = published_maker_fee(contracts, Decimal("0.95"), PUBLISHED_MAKER_RATE)

    assert low_tail == high_tail


@pytest.mark.parametrize(
    "contracts, aggregate, per_contract, headroom",
    [
        (Decimal("12"), Decimal("0.01"), Decimal("0.0833"), Decimal("0.4167")),
        (Decimal("12.36"), Decimal("0.02"), Decimal("0.1618"), Decimal("0.3382")),
        (Decimal("13"), Decimal("0.02"), Decimal("0.1538"), Decimal("0.3462")),
        (Decimal("14"), Decimal("0.02"), Decimal("0.1429"), Decimal("0.3571")),
        (Decimal("15"), Decimal("0.02"), Decimal("0.1333"), Decimal("0.3667")),
        (Decimal("16"), Decimal("0.02"), Decimal("0.1250"), Decimal("0.3750")),
        (Decimal("17"), Decimal("0.02"), Decimal("0.1176"), Decimal("0.3824")),
        (Decimal("18"), Decimal("0.02"), Decimal("0.1111"), Decimal("0.3889")),
        (Decimal("19"), Decimal("0.02"), Decimal("0.1053"), Decimal("0.3947")),
        (Decimal("20"), Decimal("0.02"), Decimal("0.1000"), Decimal("0.4000")),
        (Decimal("21"), Decimal("0.02"), Decimal("0.0952"), Decimal("0.4048")),
        (Decimal("22"), Decimal("0.02"), Decimal("0.0909"), Decimal("0.4091")),
        (Decimal("23"), Decimal("0.02"), Decimal("0.0870"), Decimal("0.4130")),
        (Decimal("24"), Decimal("0.02"), Decimal("0.0833"), Decimal("0.4167")),
        (Decimal("25"), Decimal("0.03"), Decimal("0.1200"), Decimal("0.3800")),
        (Decimal("26"), Decimal("0.03"), Decimal("0.1154"), Decimal("0.3846")),
    ],
)
def test_maker_fee_golden_table_at_the_tails(
    contracts: Decimal, aggregate: Decimal, per_contract: Decimal, headroom: Decimal
) -> None:
    fee = published_maker_fee(contracts, Decimal("0.05"), PUBLISHED_MAKER_RATE)
    cents = (Decimal(100) * fee / contracts).quantize(FOUR_DP)

    assert fee == aggregate
    assert cents == per_contract
    assert (Decimal("0.5") - cents).quantize(FOUR_DP) == headroom


def test_tail_headroom_spans_from_thirteen_to_twelve_and_twenty_four() -> None:
    headroom = {
        n: (Decimal("0.5") - _per_contract_cents(Decimal(n), Decimal("0.05"), PUBLISHED_MAKER_RATE))
        for n in range(12, 27)
    }

    assert headroom[13] == Decimal("0.3462")
    assert min(headroom.values()) == Decimal("0.3462")
    assert headroom[12] == Decimal("0.4167")
    assert headroom[24] == Decimal("0.4167")
    assert max(headroom.values()) == Decimal("0.4167")
    assert {n for n, value in headroom.items() if value == Decimal("0.4167")} == {12, 24}

    at_fractional = Decimal("0.5") - _per_contract_cents(
        Decimal("12.36"), Decimal("0.05"), PUBLISHED_MAKER_RATE
    )
    assert at_fractional == Decimal("0.3382")
    assert at_fractional < min(headroom.values())


def test_rate_is_a_genuine_parameter_not_a_hidden_constant() -> None:
    low = published_maker_fee(Decimal("20"), HALF, Decimal("0.0175"))
    high = published_maker_fee(Decimal("20"), HALF, Decimal("0.07"))

    assert low != high


@pytest.mark.parametrize(
    "contracts, price",
    [
        (Decimal("1"), Decimal("0.07")),
        (Decimal("10"), Decimal("0.07")),
        (Decimal("100"), Decimal("0.50")),
        (Decimal("27"), Decimal("0.10")),
        (Decimal("0.39"), Decimal("0.50")),
    ],
)
def test_maker_fee_at_the_taker_rate_agrees_with_the_taker_floor(
    contracts: Decimal, price: Decimal
) -> None:
    assert published_maker_fee(contracts, price, PUBLISHED_TAKER_RATE) == published_taker_fee(
        contracts, price
    )


@pytest.mark.parametrize(
    "size, price, expected",
    [
        (BAR_SIZE, Decimal("0.50"), Decimal("2.7692")),
        (BAR_SIZE, Decimal("0.05"), Decimal("1.3462")),
        (BAR_SIZE, Decimal("0.95"), Decimal("1.3462")),
    ],
)
def test_economic_bar_golden_table(size: Decimal, price: Decimal, expected: Decimal) -> None:
    assert economic_bar_cents_per_contract(size, price).quantize(FOUR_DP) == expected


def test_the_bar_is_the_per_contract_fee_plus_one_tick() -> None:
    fee = Decimal(100) * published_taker_fee(BAR_SIZE, HALF) / BAR_SIZE

    assert published_taker_fee(BAR_SIZE, HALF) == Decimal("0.46")
    assert fee.quantize(FOUR_DP) == Decimal("1.7692")
    assert economic_bar_cents_per_contract(BAR_SIZE, HALF) - fee == TICK_CENTS
    assert TICK_CENTS == Decimal("1")


def test_the_bar_is_symmetric_across_the_price_grid() -> None:
    tail = economic_bar_cents_per_contract(BAR_SIZE, Decimal("0.05"))
    mirror = economic_bar_cents_per_contract(BAR_SIZE, Decimal("0.95"))
    middle = economic_bar_cents_per_contract(BAR_SIZE, HALF)

    assert tail == mirror
    assert ((middle - TICK_CENTS) / (tail - TICK_CENTS)).quantize(FOUR_DP) == Decimal("5.1111")


def test_a_stated_zero_size_derives_a_bar_of_zero_with_no_tick() -> None:
    assert economic_bar_cents_per_contract(Decimal("0"), Decimal("0")) == Decimal("0")
    assert economic_bar_cents_per_contract(Decimal("0"), HALF) == Decimal("0")


def _bar_at(prec: int, size: Decimal, price: Decimal) -> Decimal:
    with localcontext(prec=prec):
        return economic_bar_cents_per_contract(size, price)


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_bar_reads_the_same_figure_at_every_ambient_precision(prec: int) -> None:
    bar = _bar_at(prec, BAR_SIZE, HALF)

    assert bar == PINNED_BAR
    assert str(bar) == str(PINNED_BAR)


# manifest_payload serialises the bar with str(), so a Decimal("0.00") would compare equal to
# Decimal("0") here and still move the digest.
def test_the_stated_zero_size_bar_serialises_as_a_bare_zero() -> None:
    assert str(economic_bar_cents_per_contract(Decimal("0"), HALF)) == "0"


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_bar_rounds_half_even_where_the_roundings_part(prec: int) -> None:
    bar = _bar_at(prec, SPLIT_SIZE, SPLIT_PRICE)

    assert bar == SPLIT_BAR
    assert str(bar) == str(SPLIT_BAR)


# The value tests above pin the rounding only to the half-* family: swapping the context to
# ROUND_HALF_UP or ROUND_HALF_DOWN leaves every other test in the suite green, since separating
# those from half-even needs an exact tie at the 28th significant digit that no input here reaches.
# This assertion is the only thing holding half-even.
def test_the_bar_context_carries_the_ambient_default_rounding() -> None:
    assert BAR_CONTEXT.rounding == ROUND_HALF_EVEN
