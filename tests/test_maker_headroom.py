from __future__ import annotations

import decimal
from decimal import Decimal

import pytest

from bot.lag.fee_floor import PUBLISHED_MAKER_RATE
from bot.lag.maker_headroom import (
    BAND,
    HALF_TICK_CAPTURE_CENTS,
    MODELLED_NO,
    MODELLED_NO_SIZE,
    MODELLED_YES,
    MODELLED_YES_SIZE,
    NO_MAKER_FEE_RATE,
    PRICE_GRID,
    SENSITIVITY_BAND,
    HeadroomCell,
    closed_on_arithmetic,
    headroom_grid,
    leaves_positive_headroom,
    maker_fee_cents_per_contract,
)


HALF = Decimal("0.50")
SWAMPING_RATE = Decimal("0.25")
MIRRORED = (
    (Decimal("0.05"), Decimal("0.95")),
    (Decimal("0.10"), Decimal("0.90")),
    (Decimal("0.25"), Decimal("0.75")),
)


@pytest.fixture
def published_cells() -> tuple[HeadroomCell, ...]:
    return headroom_grid(PUBLISHED_MAKER_RATE)


@pytest.fixture
def free_cells() -> tuple[HeadroomCell, ...]:
    return headroom_grid(NO_MAKER_FEE_RATE)


def column(cells: tuple[HeadroomCell, ...], price: Decimal) -> list[HeadroomCell]:
    return [cell for cell in cells if cell.price == price]


def band_headroom(cells: tuple[HeadroomCell, ...], price: Decimal) -> dict[Decimal, Decimal]:
    return {
        cell.size: cell.headroom_cents_per_contract
        for cell in column(cells, price)
        if cell.size_kind == BAND
    }


def without_price(cell: HeadroomCell) -> tuple[object, ...]:
    return (
        cell.size,
        cell.size_kind,
        cell.aggregate_fee,
        cell.fee_cents_per_contract,
        cell.headroom_cents_per_contract,
        cell.leaves_headroom,
    )


@pytest.mark.parametrize(
    "size_kind, size, headroom",
    [
        (MODELLED_YES, Decimal("12.36"), Decimal("0.0146")),
        (MODELLED_NO, Decimal("26"), Decimal("0.0385")),
        (BAND, Decimal("12"), Decimal("0.0000")),
        (BAND, Decimal("13"), Decimal("0.0385")),
        (BAND, Decimal("14"), Decimal("0.0000")),
        (BAND, Decimal("15"), Decimal("0.0333")),
        (BAND, Decimal("16"), Decimal("0.0625")),
        (BAND, Decimal("17"), Decimal("0.0294")),
        (BAND, Decimal("18"), Decimal("0.0556")),
        (BAND, Decimal("19"), Decimal("0.0263")),
        (BAND, Decimal("20"), Decimal("0.0500")),
        (BAND, Decimal("21"), Decimal("0.0238")),
        (BAND, Decimal("22"), Decimal("0.0455")),
        (BAND, Decimal("23"), Decimal("0.0217")),
        (BAND, Decimal("24"), Decimal("0.0417")),
        (BAND, Decimal("25"), Decimal("0.0600")),
        (BAND, Decimal("26"), Decimal("0.0385")),
    ],
)
def test_headroom_golden_table_at_the_half(
    published_cells: tuple[HeadroomCell, ...],
    size_kind: str,
    size: Decimal,
    headroom: Decimal,
) -> None:
    matches = [
        cell
        for cell in column(published_cells, HALF)
        if cell.size == size and cell.size_kind == size_kind
    ]

    assert len(matches) == 1
    assert matches[0].headroom_cents_per_contract == headroom
    assert matches[0].fee_cents_per_contract == HALF_TICK_CAPTURE_CENTS - headroom


def test_exactly_twelve_and_fourteen_are_charged_the_whole_half_tick(
    published_cells: tuple[HeadroomCell, ...],
) -> None:
    charged = [
        cell for cell in published_cells if cell.fee_cents_per_contract == HALF_TICK_CAPTURE_CENTS
    ]

    assert [(cell.size_kind, cell.size, cell.price) for cell in charged] == [
        (BAND, Decimal("12"), HALF),
        (BAND, Decimal("14"), HALF),
    ]
    assert all(not cell.leaves_headroom for cell in charged)


def test_the_other_thirteen_band_cells_at_the_half_span_the_stated_range(
    published_cells: tuple[HeadroomCell, ...],
) -> None:
    headroom = band_headroom(published_cells, HALF)
    positive = [value for value in headroom.values() if value > 0]

    assert len(positive) == 13
    assert min(positive) == Decimal("0.0217")
    assert max(positive) == Decimal("0.0625")


@pytest.mark.parametrize("price", [Decimal("0.05"), Decimal("0.95")])
def test_the_tails_span_from_thirteen_to_twelve_and_twenty_four(
    published_cells: tuple[HeadroomCell, ...], price: Decimal
) -> None:
    headroom = band_headroom(published_cells, price)

    assert min(headroom.values()) == Decimal("0.3462")
    assert headroom[Decimal("13")] == Decimal("0.3462")
    assert max(headroom.values()) == Decimal("0.4167")
    assert {size for size, value in headroom.items() if value == Decimal("0.4167")} == {
        Decimal("12"),
        Decimal("24"),
    }


@pytest.mark.parametrize("price", [Decimal("0.05"), Decimal("0.95")])
def test_the_modelled_yes_size_sits_under_the_whole_band_at_the_tails(
    published_cells: tuple[HeadroomCell, ...], price: Decimal
) -> None:
    modelled = [cell for cell in column(published_cells, price) if cell.size_kind == MODELLED_YES]

    assert len(modelled) == 1
    assert modelled[0].headroom_cents_per_contract == Decimal("0.3382")
    assert modelled[0].headroom_cents_per_contract < min(
        band_headroom(published_cells, price).values()
    )


@pytest.mark.parametrize("low, high", MIRRORED)
def test_the_price_grid_is_symmetric_about_a_half(
    published_cells: tuple[HeadroomCell, ...], low: Decimal, high: Decimal
) -> None:
    left = [without_price(cell) for cell in column(published_cells, low)]
    right = [without_price(cell) for cell in column(published_cells, high)]

    assert left == right


def test_no_cell_is_negative_and_exactly_two_are_zero(
    published_cells: tuple[HeadroomCell, ...],
) -> None:
    headroom = [cell.headroom_cents_per_contract for cell in published_cells]

    assert min(headroom) == Decimal("0.0000")
    assert sum(1 for value in headroom if value == 0) == 2


def test_the_rounding_never_flips_the_verdict(
    published_cells: tuple[HeadroomCell, ...], free_cells: tuple[HeadroomCell, ...]
) -> None:
    for cell in published_cells + free_cells:
        assert (cell.headroom_cents_per_contract > 0) is cell.leaves_headroom


@pytest.mark.parametrize("precision", [6, 50])
def test_the_verdict_does_not_move_with_the_ambient_precision(
    published_cells: tuple[HeadroomCell, ...], precision: int
) -> None:
    with decimal.localcontext() as ctx:
        ctx.prec = precision
        cells = headroom_grid(PUBLISHED_MAKER_RATE)
        verdict = closed_on_arithmetic(cells)

    assert [cell.leaves_headroom for cell in cells] == [
        cell.leaves_headroom for cell in published_cells
    ]
    assert verdict is closed_on_arithmetic(published_cells)


def test_neither_published_regime_closes_the_question_on_arithmetic(
    published_cells: tuple[HeadroomCell, ...], free_cells: tuple[HeadroomCell, ...]
) -> None:
    assert closed_on_arithmetic(published_cells) is False
    assert closed_on_arithmetic(free_cells) is False


def test_a_grid_with_no_positive_cell_closes_the_question() -> None:
    cells = [
        HeadroomCell(
            size=size,
            size_kind=BAND,
            price=HALF,
            aggregate_fee=Decimal("9.99"),
            fee_cents_per_contract=Decimal("1.0000"),
            headroom_cents_per_contract=Decimal("-0.5000"),
            leaves_headroom=False,
        )
        for size in SENSITIVITY_BAND
    ]

    assert closed_on_arithmetic(cells) is True


def test_a_rate_that_swamps_the_half_tick_closes_the_question() -> None:
    cells = headroom_grid(SWAMPING_RATE)

    assert maker_fee_cents_per_contract(
        MODELLED_NO_SIZE, Decimal("0.05"), SWAMPING_RATE
    ) == Decimal("1.1923")
    assert all(cell.headroom_cents_per_contract < 0 for cell in cells)
    assert closed_on_arithmetic(cells) is True


def test_the_free_regime_charges_nothing_and_leaves_the_whole_half_tick(
    free_cells: tuple[HeadroomCell, ...],
) -> None:
    assert NO_MAKER_FEE_RATE == Decimal("0")
    assert all(cell.aggregate_fee == Decimal("0.00") for cell in free_cells)
    assert all(cell.fee_cents_per_contract == Decimal("0.0000") for cell in free_cells)
    assert all(cell.headroom_cents_per_contract == Decimal("0.5000") for cell in free_cells)
    assert all(cell.leaves_headroom for cell in free_cells)


def test_the_grid_carries_seventeen_rows_of_seven_prices(
    published_cells: tuple[HeadroomCell, ...],
) -> None:
    rows = list(dict.fromkeys((cell.size_kind, cell.size) for cell in published_cells))

    assert len(published_cells) == 119
    assert rows == [
        (MODELLED_YES, MODELLED_YES_SIZE),
        (MODELLED_NO, MODELLED_NO_SIZE),
        *((BAND, size) for size in SENSITIVITY_BAND),
    ]
    assert all(
        [cell.price for cell in published_cells[index : index + 7]] == list(PRICE_GRID)
        for index in range(0, 119, 7)
    )


def test_the_size_kinds_split_seven_seven_and_a_hundred_and_five(
    published_cells: tuple[HeadroomCell, ...],
) -> None:
    counts = {kind: 0 for kind in (MODELLED_YES, MODELLED_NO, BAND)}
    for cell in published_cells:
        counts[cell.size_kind] += 1

    assert counts == {MODELLED_YES: 7, MODELLED_NO: 7, BAND: 105}


def test_the_grid_constants_are_the_preregistered_ones() -> None:
    assert HALF_TICK_CAPTURE_CENTS == Decimal("0.50")
    assert MODELLED_YES_SIZE == Decimal("12.36")
    assert MODELLED_NO_SIZE == Decimal("26")
    assert SENSITIVITY_BAND == tuple(Decimal(n) for n in range(12, 27))
    assert PRICE_GRID == (
        Decimal("0.05"),
        Decimal("0.10"),
        Decimal("0.25"),
        Decimal("0.50"),
        Decimal("0.75"),
        Decimal("0.90"),
        Decimal("0.95"),
    )


@pytest.mark.parametrize(
    "size, price, positive",
    [
        (Decimal("12"), HALF, False),
        (Decimal("14"), HALF, False),
        (Decimal("12.36"), HALF, True),
        (Decimal("16"), HALF, True),
        (Decimal("26"), Decimal("0.05"), True),
    ],
)
def test_the_predicate_runs_on_multiplication_not_a_rounded_figure(
    size: Decimal, price: Decimal, positive: bool
) -> None:
    assert leaves_positive_headroom(size, price, PUBLISHED_MAKER_RATE) is positive
