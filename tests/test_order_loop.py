from __future__ import annotations

import logging

import httpx
import pytest

from bot.execution.order_loop import place_orders_resilient


async def test_place_orders_resilient_continues_after_raise_on_iter_4(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items = [1, 2, 3, 4, 5, 6]

    async def placer(item: int) -> str | None:
        if item == 4:
            raise httpx.NetworkError("boom")
        return f"r{item}"

    caplog.set_level(logging.ERROR, logger="bot.execution.order_loop")
    results = await place_orders_resilient(items, placer)

    assert results == ["r1", "r2", "r3", "r5", "r6"]
    exception_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(exception_records) == 1
    assert "place_order_demo_raised" in exception_records[0].getMessage()


async def test_place_orders_resilient_filters_none_returns_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items = [1, 2, 3, 4, 5]

    async def placer(item: int) -> str | None:
        if item in (2, 4):
            return None
        return f"r{item}"

    caplog.set_level(logging.ERROR, logger="bot.execution.order_loop")
    results = await place_orders_resilient(items, placer)

    assert results == ["r1", "r3", "r5"]
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


async def test_place_orders_resilient_continues_through_runtime_error_from_host_guard(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items = [1, 2, 3]

    async def placer(item: int) -> str | None:
        if item == 2:
            raise RuntimeError("absolute URL forbidden on signed call")
        return f"r{item}"

    caplog.set_level(logging.ERROR, logger="bot.execution.order_loop")
    results = await place_orders_resilient(items, placer)

    assert results == ["r1", "r3"]
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1


async def test_place_orders_resilient_returns_empty_on_empty_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def placer(item: int) -> str | None:
        return "unreachable"

    caplog.set_level(logging.ERROR, logger="bot.execution.order_loop")
    results = await place_orders_resilient([], placer)

    assert results == []
    assert caplog.records == []


async def test_place_orders_resilient_returns_empty_when_all_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items = [1, 2, 3]

    async def placer(item: int) -> str | None:
        raise httpx.ReadTimeout("slow")

    caplog.set_level(logging.ERROR, logger="bot.execution.order_loop")
    results = await place_orders_resilient(items, placer)

    assert results == []
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 3
