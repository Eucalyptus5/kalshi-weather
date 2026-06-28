from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Final

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_DEDUP_WINDOW_SECONDS: Final[float] = 60.0
_DEFAULT_BACKOFF_AFTER_N: Final[int] = 3
_DEFAULT_BACKOFF_CAP_MULTIPLIER: Final[float] = 4.0
_FAILURE_RECOVERY_SLEEP_SECONDS: Final[float] = 5.0
_DEFAULT_SUSTAINED_THRESHOLD_CEILING_SECONDS: Final[float] = 21600.0

_monotonic: Callable[[], float] = time.monotonic


_LOOP_PAIRS = [
    ("_market_loop", "market_loop"),
    ("_eval_loop", "eval_loop"),
    ("_order_reconcile_loop", "order_reconcile_loop"),
    ("_portfolio_snapshot_loop", "portfolio_snapshot_loop"),
    ("_settlement_loop", "settlement_loop"),
    ("_calibration_refit_loop", "calibration_loop"),
    ("_forecast_loop", "forecast_loop"),
    ("_ws_recorder_loop", "ws_recorder_loop"),
    ("_obs_arrival_loop", "obs_arrival_loop"),
]


class LoopSkipped(Exception): ...


def _default_sustained_threshold(interval_seconds: float) -> float:
    return min(max(300.0, 3.0 * interval_seconds), _DEFAULT_SUSTAINED_THRESHOLD_CEILING_SECONDS)


def _error_key(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        # path only, no query: min_ts changes per call and would defeat dedup
        return f"http_{exc.response.status_code}:{exc.request.url.path}"
    return f"{type(exc).__name__}:{str(exc)[:80]}"


async def _sleep_or_stop(stop: asyncio.Event, timeout: float) -> None:
    if timeout <= 0:
        await asyncio.sleep(0)
        return
    try:
        await asyncio.wait_for(stop.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass


async def _resolve(value: float | Awaitable[float]) -> float:
    if inspect.isawaitable(value):
        return await value
    return value


async def run_loop(
    *,
    name: str,
    body: Callable[[], Awaitable[None]],
    interval_seconds: float,
    stop: asyncio.Event,
    sustained_failure_threshold_seconds: float | None = None,
    dedup_window_seconds: float = _DEFAULT_DEDUP_WINDOW_SECONDS,
    backoff_after_n_failures: int = _DEFAULT_BACKOFF_AFTER_N,
    backoff_cap_multiplier: float = _DEFAULT_BACKOFF_CAP_MULTIPLIER,
    next_delay: Callable[[bool], float | Awaitable[float]] | None = None,
    failure_delay: Callable[[], float | Awaitable[float]] | None = None,
) -> None:
    sustained_threshold = (
        sustained_failure_threshold_seconds
        if sustained_failure_threshold_seconds is not None
        else _default_sustained_threshold(interval_seconds)
    )
    backoff_cap_seconds = min(backoff_cap_multiplier * interval_seconds, sustained_threshold)

    consecutive_failures = 0
    streak_started_at = 0.0
    sustained_alarmed = False
    last_error_key: str | None = None
    dedup_first_seen_at = 0.0

    async def _success_sleep() -> float:
        if next_delay is None:
            return interval_seconds
        try:
            return await _resolve(next_delay(False))
        except Exception:
            logger.error("loop_next_delay_failed name=%s", name, exc_info=True)
            return interval_seconds

    async def _failure_sleep() -> float:
        if failure_delay is None:
            base = _FAILURE_RECOVERY_SLEEP_SECONDS
        else:
            try:
                base = await _resolve(failure_delay())
            except Exception:
                logger.error("loop_failure_delay_failed name=%s", name, exc_info=True)
                base = _FAILURE_RECOVERY_SLEEP_SECONDS
        base = min(base, backoff_cap_seconds)
        if consecutive_failures > backoff_after_n_failures:
            base = base * (2 ** (consecutive_failures - backoff_after_n_failures))
        return min(base, backoff_cap_seconds)

    while not stop.is_set():
        try:
            await body()
        except LoopSkipped:
            await _sleep_or_stop(stop, await _success_sleep())
            continue
        except Exception as exc:
            now = _monotonic()
            consecutive_failures += 1
            key = _error_key(exc)

            if consecutive_failures == 1:
                streak_started_at = now
                sustained_alarmed = False

            if key != last_error_key:
                last_error_key = key
                dedup_first_seen_at = now
                logger.exception("loop_iteration_failed name=%s", name)
            else:
                while now - dedup_first_seen_at >= dedup_window_seconds:
                    dedup_first_seen_at += dedup_window_seconds
                    logger.warning(
                        "loop_iteration_failed_repeated name=%s consecutive=%d last_error_key=%s",
                        name,
                        consecutive_failures,
                        key,
                    )

            if not sustained_alarmed and now - streak_started_at >= sustained_threshold:
                sustained_alarmed = True
                logger.error(
                    "loop_sustained_failure name=%s duration_sec=%s consecutive=%d "
                    "last_error_key=%s",
                    name,
                    now - streak_started_at,
                    consecutive_failures,
                    key,
                )

            await _sleep_or_stop(stop, await _failure_sleep())
            continue

        if consecutive_failures > 0:
            logger.info(
                "loop_recovered name=%s consecutive_failures_before_recovery=%d "
                "outage_duration_sec=%s",
                name,
                consecutive_failures,
                _monotonic() - streak_started_at,
            )
            consecutive_failures = 0
            streak_started_at = 0.0
            sustained_alarmed = False
            last_error_key = None
            dedup_first_seen_at = 0.0

        await _sleep_or_stop(stop, await _success_sleep())
