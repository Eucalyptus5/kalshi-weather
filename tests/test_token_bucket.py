from __future__ import annotations

import asyncio
import time

import pytest

from bot.execution.token_bucket import TokenBucket


@pytest.mark.parametrize("bad_capacity", [0, -1])
def test_token_bucket_constructor_rejects_non_positive_capacity(bad_capacity: int) -> None:
    with pytest.raises(ValueError):
        TokenBucket(capacity=bad_capacity, refill_per_second=10)


@pytest.mark.parametrize("bad_refill", [0, -1])
def test_token_bucket_constructor_rejects_non_positive_refill(bad_refill: int) -> None:
    with pytest.raises(ValueError):
        TokenBucket(capacity=10, refill_per_second=bad_refill)


async def test_token_bucket_acquire_validates_cost_bounds() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=10)
    await bucket.aopen()
    try:
        with pytest.raises(ValueError):
            await bucket.acquire(cost=0)
        with pytest.raises(ValueError):
            await bucket.acquire(cost=11)
    finally:
        await bucket.aclose()


async def test_token_bucket_default_cost_throttles_writes() -> None:
    bucket = TokenBucket(capacity=100, refill_per_second=100)
    await bucket.aopen()
    try:
        start = time.monotonic()
        for _ in range(10):
            await bucket.acquire()
        within_first = time.monotonic() - start
        assert within_first < 0.5

        blocker_start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - blocker_start
        assert elapsed >= 0.005
    finally:
        await bucket.aclose()


async def test_token_bucket_cost_one_does_not_block_cheap_reads() -> None:
    bucket = TokenBucket(capacity=100, refill_per_second=100)
    await bucket.aopen()
    try:
        start = time.monotonic()
        for _ in range(100):
            await bucket.acquire(cost=1)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
    finally:
        await bucket.aclose()


async def test_token_bucket_legacy_one_token_per_call_admits_too_much() -> None:
    bucket = TokenBucket(capacity=100, refill_per_second=100)
    await bucket.aopen()
    try:
        start = time.monotonic()
        for _ in range(100):
            await bucket.acquire(cost=1)
        elapsed = time.monotonic() - start
        rate = 100 / max(elapsed, 1e-9)
        assert rate > 200
    finally:
        await bucket.aclose()


async def test_aclose_unblocks_max_cost_parked_waiter() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=1)
    await bucket.aopen()

    drained = asyncio.Event()
    second_done = asyncio.Event()

    async def first() -> None:
        await bucket.acquire(cost=10)
        drained.set()

    async def second() -> str:
        await drained.wait()
        try:
            await bucket.acquire(cost=10)
            return "acquired"
        except RuntimeError:
            second_done.set()
            return "closed"

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())

    await drained.wait()
    await asyncio.sleep(0.05)
    await bucket.aclose()

    result = await asyncio.wait_for(second_task, timeout=1.0)
    assert result == "closed"
    await first_task


async def test_token_bucket_acquire_raises_after_close() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=10)
    await bucket.aopen()
    await bucket.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await bucket.acquire(cost=1)


async def test_token_bucket_aclose_is_idempotent() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=10)
    await bucket.aopen()
    await bucket.aclose()
    await bucket.aclose()


async def test_token_bucket_gather_resolves_on_close() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=1)
    await bucket.aopen()

    primed = asyncio.Event()

    async def primer() -> None:
        await bucket.acquire(cost=10)
        primed.set()

    primer_task = asyncio.create_task(primer())
    await primed.wait()

    async def waiter() -> str:
        try:
            await bucket.acquire(cost=10)
            return "acquired"
        except RuntimeError:
            return "closed"

    tasks = [asyncio.create_task(waiter()) for _ in range(3)]
    await asyncio.sleep(0.05)
    await bucket.aclose()
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    assert results == ["closed", "closed", "closed"]
    await primer_task


async def test_aclose_yields_mixed_shape_for_four_waiters() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=1)
    await bucket.aopen()

    drained = asyncio.Event()

    async def holder() -> str:
        await bucket.acquire(cost=10)
        drained.set()
        return "acquired"

    async def waiter() -> str:
        await drained.wait()
        try:
            await bucket.acquire(cost=10)
            return "acquired"
        except RuntimeError:
            return "closed"

    tasks = [asyncio.create_task(holder())]
    tasks.extend(asyncio.create_task(waiter()) for _ in range(3))

    await drained.wait()
    await asyncio.sleep(0.05)
    await bucket.aclose()
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    assert results == ["acquired", "closed", "closed", "closed"]


async def test_token_bucket_acquire_after_close_via_event_during_wait() -> None:
    bucket = TokenBucket(capacity=1, refill_per_second=1)
    await bucket.aopen()
    await bucket.acquire(cost=1)

    async def waiter() -> str:
        try:
            await bucket.acquire(cost=1)
            return "acquired"
        except RuntimeError:
            return "closed"

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    await bucket.aclose()
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result == "closed"
