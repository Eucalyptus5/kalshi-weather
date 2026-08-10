from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from bot.lag.fee_floor import published_maker_fee


HALF_TICK_CAPTURE_CENTS: Decimal = Decimal("0.50")
NO_MAKER_FEE_RATE: Decimal = Decimal("0")
MODELLED_YES_SIZE: Decimal = Decimal("12.36")
MODELLED_NO_SIZE: Decimal = Decimal("26")
MODELLED_YES = "modelled_yes"
MODELLED_NO = "modelled_no"
BAND = "band"
REPORT_QUANTUM: Decimal = Decimal("0.0001")

SENSITIVITY_BAND: tuple[Decimal, ...] = tuple(Decimal(size) for size in range(12, 27))
PRICE_GRID: tuple[Decimal, ...] = (
    Decimal("0.05"),
    Decimal("0.10"),
    Decimal("0.25"),
    Decimal("0.50"),
    Decimal("0.75"),
    Decimal("0.90"),
    Decimal("0.95"),
)
GRID_ROWS: tuple[tuple[Decimal, str], ...] = (
    (MODELLED_YES_SIZE, MODELLED_YES),
    (MODELLED_NO_SIZE, MODELLED_NO),
    *((size, BAND) for size in SENSITIVITY_BAND),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class HeadroomCell:
    size: Decimal
    size_kind: str
    price: Decimal
    aggregate_fee: Decimal
    fee_cents_per_contract: Decimal
    headroom_cents_per_contract: Decimal
    leaves_headroom: bool


def maker_fee_cents_per_contract(contracts: Decimal, price: Decimal, rate: Decimal) -> Decimal:
    aggregate = published_maker_fee(contracts, price, rate)
    return (Decimal(100) * aggregate / contracts).quantize(REPORT_QUANTUM)


# Stated as a multiplication because the per-contract figure above divides at the ambient decimal
# precision, which nothing in this repo pins. 100 * fee / size < 0.50 is 200 * fee < size for any
# positive size, and the second form is exact at every precision.
def leaves_positive_headroom(contracts: Decimal, price: Decimal, rate: Decimal) -> bool:
    return Decimal(200) * published_maker_fee(contracts, price, rate) < contracts


def headroom_grid(rate: Decimal) -> tuple[HeadroomCell, ...]:
    cells = []
    for size, size_kind in GRID_ROWS:
        for price in PRICE_GRID:
            fee_cents = maker_fee_cents_per_contract(size, price, rate)
            cells.append(
                HeadroomCell(
                    size=size,
                    size_kind=size_kind,
                    price=price,
                    aggregate_fee=published_maker_fee(size, price, rate),
                    fee_cents_per_contract=fee_cents,
                    headroom_cents_per_contract=HALF_TICK_CAPTURE_CENTS - fee_cents,
                    leaves_headroom=leaves_positive_headroom(size, price, rate),
                )
            )
    return tuple(cells)


def closed_on_arithmetic(cells: Sequence[HeadroomCell]) -> bool:
    return not any(cell.leaves_headroom for cell in cells)
