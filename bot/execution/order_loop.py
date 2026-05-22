from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


async def place_orders_resilient(
    items: Sequence[T],
    placer: Callable[[T], Awaitable[R | None]],
) -> list[R]:
    results: list[R] = []
    for item in items:
        try:
            result = await placer(item)
        except Exception:
            logger.exception("place_order_demo_raised item=%r", item)
            continue
        if result is not None:
            results.append(result)
    return results
