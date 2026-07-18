from __future__ import annotations

from decimal import Decimal

import pytest

from bot.execution.fees import taker_fee
from bot.execution.friction import required_edge


def test_required_edge_tails_small_size() -> None:
    floor = required_edge(Decimal("0.07"), 10, 5)
    fee = taker_fee(1, Decimal("0.07"))
    assert fee == Decimal("0.01")
    assert floor == fee + Decimal("0.005")
    assert floor == Decimal("0.015")


def test_required_edge_edge_no_walk() -> None:
    floor = required_edge(Decimal("0.39"), 15, 10)
    fee = taker_fee(1, Decimal("0.39"))
    assert fee == Decimal("0.02")
    assert floor == fee + Decimal("0.005")
    assert floor == Decimal("0.025")


def test_required_edge_edge_walks_book() -> None:
    floor = required_edge(Decimal("0.39"), 15, 100)
    fee = taker_fee(1, Decimal("0.39"))
    assert floor == fee + Decimal("0.005") + Decimal("0.05")


def test_required_edge_zero_depth_max_haircut() -> None:
    floor = required_edge(Decimal("0.07"), 0, 10)
    fee = taker_fee(1, Decimal("0.07"))
    assert floor == fee + Decimal("0.005") + Decimal("0.05")


def test_required_edge_symmetric_in_price() -> None:
    a = required_edge(Decimal("0.07"), 100, 1)
    b = required_edge(Decimal("0.93"), 100, 1)
    assert a == b


def test_required_edge_returns_decimal() -> None:
    assert isinstance(required_edge(Decimal("0.5"), 10, 1), Decimal)


@pytest.mark.parametrize("price", [Decimal("0.05"), Decimal("0.39"), Decimal("0.81")])
def test_required_edge_fee_matches_helper(price: Decimal) -> None:
    floor = required_edge(price, 999, 1)
    assert floor - Decimal("0.005") == taker_fee(1, price)


def test_tails_boundary_edge_clears_friction_floor_with_safety_margin() -> None:
    floor = required_edge(Decimal("0.905"), 10, 10)
    gross_edge = Decimal("0.025")
    assert gross_edge - floor > Decimal("0.005")


def test_tails_boundary_clears_at_depth_one_corner() -> None:
    floor = required_edge(Decimal("0.905"), 1, 1)
    gross_edge = Decimal("0.025")
    assert gross_edge > floor
