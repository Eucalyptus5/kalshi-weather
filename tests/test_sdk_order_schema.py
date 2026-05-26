from __future__ import annotations

import pytest
from kalshi_python_async.models.order import Order
from pydantic import ValidationError


def _required_payload(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "order_id": "ex-1",
        "user_id": "user-1",
        "client_order_id": "cid-1",
        "ticker": "KXHIGHDEN-26MAY22-T70",
        "side": "yes",
        "action": "buy",
        "type": "limit",
        "status": "executed",
        "yes_price": 50,
        "no_price": 50,
        "yes_price_dollars": "0.5000",
        "no_price_dollars": "0.5000",
        "fill_count": 5,
        "remaining_count": 0,
        "initial_count": 5,
        "taker_fees": 1,
        "maker_fees": 0,
        "taker_fill_cost": 250,
        "maker_fill_cost": 0,
        "taker_fill_cost_dollars": "2.5000",
        "maker_fill_cost_dollars": "0.0000",
        "queue_position": 0,
    }
    base.update(overrides)
    return base


def test_order_construction_succeeds_with_taker_and_maker_fees_dollars_omitted() -> None:
    payload = _required_payload()
    payload.pop("taker_fees_dollars", None)
    payload.pop("maker_fees_dollars", None)
    order = Order(**payload)
    assert order.taker_fees == 1
    assert order.taker_fees_dollars is None
    assert order.maker_fees_dollars is None


def test_order_construction_fails_when_taker_fill_cost_dollars_omitted() -> None:
    payload = _required_payload()
    payload.pop("taker_fill_cost_dollars")
    with pytest.raises(ValidationError):
        Order(**payload)


def test_order_construction_fails_when_maker_fill_cost_dollars_omitted() -> None:
    payload = _required_payload()
    payload.pop("maker_fill_cost_dollars")
    with pytest.raises(ValidationError):
        Order(**payload)
