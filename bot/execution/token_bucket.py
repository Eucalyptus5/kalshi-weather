from __future__ import annotations

import asyncio


class TokenBucket:
    def __init__(self, capacity: int, refill_per_second: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if refill_per_second < 1:
            raise ValueError(f"refill_per_second must be >= 1, got {refill_per_second}")
        self._capacity = capacity
        self._refill_interval = 1.0 / refill_per_second
        self._sema = asyncio.Semaphore(capacity)
        self._closed_event = asyncio.Event()
        self._refill_task: asyncio.Task[None] | None = None

    async def aopen(self) -> None:
        self._refill_task = asyncio.create_task(self._refill_loop())

    async def acquire(self, cost: int = 10) -> None:
        if cost < 1 or cost > self._capacity:
            raise ValueError(
                f"cost must satisfy 1 <= cost <= capacity ({self._capacity}), got {cost}"
            )
        if self._closed_event.is_set():
            raise RuntimeError("bucket closed")
        for _ in range(cost):
            sema_task = asyncio.create_task(self._sema.acquire())
            close_task = asyncio.create_task(self._closed_event.wait())
            done, pending = await asyncio.wait(
                {sema_task, close_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, BaseException):
                    pass
            if close_task in done:
                if sema_task in done and not sema_task.cancelled():
                    self._sema.release()
                raise RuntimeError("bucket closed")
            if self._closed_event.is_set():
                self._sema.release()
                raise RuntimeError("bucket closed")

    async def aclose(self) -> None:
        if self._closed_event.is_set():
            return
        self._closed_event.set()
        if self._refill_task is not None:
            self._refill_task.cancel()
            try:
                await self._refill_task
            except asyncio.CancelledError:
                pass
            self._refill_task = None

    async def _refill_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._refill_interval)
                if self._sema.locked():
                    self._sema.release()
        except asyncio.CancelledError:
            return
