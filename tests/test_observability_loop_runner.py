from __future__ import annotations

import asyncio
import logging
import re
from unittest.mock import Mock

import httpx
import pytest

import bot.observability.loop_runner as loop_runner
from bot.observability.loop_runner import (
    LoopSkipped,
    _default_sustained_threshold,
    _sleep_or_stop,
    run_loop,
)

_LOG = "bot.observability.loop_runner"


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _http_error(status: int, path: str = "/portfolio/fills") -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://demo-api.kalshi.co{path}?min_ts=123")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


class _SleepShim:
    def __init__(
        self, clock: FakeClock, stop_after: int | None = None, stop: asyncio.Event | None = None
    ) -> None:
        self.clock = clock
        self.durations: list[float] = []
        self.stop_after = stop_after
        self.stop = stop
        self.calls = 0

    async def __call__(self, stop: asyncio.Event, timeout: float) -> None:
        self.durations.append(timeout)
        self.calls += 1
        if timeout > 0:
            self.clock.advance(timeout)
        if self.stop_after is not None and self.calls >= self.stop_after and self.stop is not None:
            self.stop.set()
        await asyncio.sleep(0)


async def test_first_failure_logs_full_error_with_stack_trace(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=1, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    body = Mock(side_effect=_http_error(500))

    async def call_body() -> None:
        body()

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(name="test_loop", body=call_body, interval_seconds=30.0, stop=stop)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].getMessage().startswith("loop_iteration_failed name=")
    assert errors[0].exc_info is not None


async def test_second_failure_within_dedup_window_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=2, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    err = _http_error(500)

    async def call_body() -> None:
        raise err

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=30.0,
        stop=stop,
        dedup_window_seconds=60.0,
        failure_delay=lambda: 1.0,
    )

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1


async def test_failure_past_dedup_window_emits_one_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=3, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        raise _http_error(500)

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=30.0,
        stop=stop,
        dedup_window_seconds=60.0,
        failure_delay=lambda: 31.0,
    )

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert re.search(
        r"loop_iteration_failed_repeated name=test_loop consecutive=3 "
        r"last_error_key=http_500:/portfolio/fills",
        warns[0].getMessage(),
    )


async def test_dedup_warn_cadence_tracks_wall_clock(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=10, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        raise _http_error(500)

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=30.0,
        stop=stop,
        dedup_window_seconds=60.0,
        backoff_cap_multiplier=4.0,
        sustained_failure_threshold_seconds=1_000_000.0,
        failure_delay=lambda: 120.0,
    )

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert 17 <= len(warns) <= 20


async def test_different_error_key_resets_window_not_streak(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    keys = [_http_error(500), _http_error(503), _http_error(503), _http_error(503)]

    shim = _SleepShim(clock, stop_after=len(keys), stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    seq = iter(keys)

    async def call_body() -> None:
        raise next(seq)

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=30.0,
        stop=stop,
        dedup_window_seconds=60.0,
        sustained_failure_threshold_seconds=1_000_000.0,
        failure_delay=lambda: 61.0,
    )

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns
    assert "consecutive=" in warns[0].getMessage()
    nums = [int(re.search(r"consecutive=(\d+)", w.getMessage()).group(1)) for w in warns]
    assert max(nums) >= 3


async def test_alternating_keys_fire_sustained_alarm_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=30, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    toggle = {"n": 0}

    async def call_body() -> None:
        toggle["n"] += 1
        raise _http_error(500 if toggle["n"] % 2 == 0 else 503)

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=30.0,
        stop=stop,
        sustained_failure_threshold_seconds=300.0,
        failure_delay=lambda: 60.0,
    )

    alarms = [r for r in caplog.records if "loop_sustained_failure" in r.getMessage()]
    assert len(alarms) == 1


async def test_sustained_alarm_fires_exactly_once_per_streak(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=40, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        raise _http_error(500)

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=30.0,
        stop=stop,
        sustained_failure_threshold_seconds=300.0,
        failure_delay=lambda: 60.0,
    )

    alarms = [r for r in caplog.records if "loop_sustained_failure" in r.getMessage()]
    assert len(alarms) == 1
    assert re.search(
        r"loop_sustained_failure name=test_loop duration_sec=\S+ consecutive=\d+ "
        r"last_error_key=http_500:/portfolio/fills",
        alarms[0].getMessage(),
    )


def test_default_sustained_threshold_capped_at_ceiling() -> None:
    assert _default_sustained_threshold(30.0) == 300.0
    assert _default_sustained_threshold(3600.0) == 10800.0
    assert _default_sustained_threshold(21600.0) == 21600.0
    assert _default_sustained_threshold(86400.0) == 21600.0


async def test_backoff_cap_clamped_to_sustained_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=25, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        raise _http_error(500)

    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=21600.0,
        stop=stop,
        sustained_failure_threshold_seconds=300.0,
    )

    assert max(shim.durations) <= 300.0


async def test_failure_delay_override_clamped_by_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=1, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        raise _http_error(500)

    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=60.0,
        stop=stop,
        sustained_failure_threshold_seconds=300.0,
        failure_delay=lambda: 600.0,
    )

    cap = min(4.0 * 60.0, 300.0)
    assert shim.durations[0] == pytest.approx(cap)
    assert shim.durations[0] != pytest.approx(5.0)
    assert shim.durations[0] != pytest.approx(600.0)


async def test_small_failure_delay_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=1, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        raise _http_error(500)

    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=60.0,
        stop=stop,
        failure_delay=lambda: 0.05,
    )

    assert shim.durations[0] == pytest.approx(0.05)


async def test_settlement_migration_paces_failures_at_success_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bot.main import SETTLEMENT_INTERVAL_SECONDS

    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    six_hours = 6 * 3600.0

    class _BoundedShim(_SleepShim):
        async def __call__(self, ev: asyncio.Event, timeout: float) -> None:
            self.durations.append(timeout)
            if timeout > 0:
                self.clock.advance(timeout)
            if self.clock.now >= six_hours:
                stop.set()
            await asyncio.sleep(0)

    shim = _BoundedShim(clock)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    calls = {"n": 0}

    async def call_body() -> None:
        calls["n"] += 1
        raise _http_error(500, path="/markets")

    cap = min(4.0 * SETTLEMENT_INTERVAL_SECONDS, six_hours)
    await run_loop(
        name="settlement_loop",
        body=call_body,
        interval_seconds=SETTLEMENT_INTERVAL_SECONDS,
        stop=stop,
        failure_delay=lambda: SETTLEMENT_INTERVAL_SECONDS,
    )

    assert shim.durations[0] == pytest.approx(min(SETTLEMENT_INTERVAL_SECONDS, cap))
    assert calls["n"] <= 2


async def test_calibration_body_sleeps_before_refit(monkeypatch: pytest.MonkeyPatch) -> None:
    import bot.main as bot_main

    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    monkeypatch.setattr(bot_main, "_seconds_until_next_refit", lambda now: 1234.5)

    refit_mock = Mock(return_value=bot_main._empty_calibration_maps())
    monkeypatch.setattr(bot_main, "refit_all", refit_mock)

    refit_at: list[float] = []

    def record_refit(*args: object, **kwargs: object) -> object:
        refit_at.append(clock.now)
        return bot_main._empty_calibration_maps()

    refit_mock.side_effect = record_refit

    stop = asyncio.Event()

    async def leading_sleep(ev: asyncio.Event, timeout: float) -> None:
        if timeout > 0:
            clock.advance(timeout)
        await asyncio.sleep(0)

    async def wrapper_sleep(ev: asyncio.Event, timeout: float) -> None:
        ev.set()
        await asyncio.sleep(0)

    monkeypatch.setattr(bot_main, "_sleep_or_stop", leading_sleep)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", wrapper_sleep)

    app = _make_calibration_app(monkeypatch)
    await bot_main._calibration_refit_loop(app, stop)

    assert refit_at == [1234.5]


async def test_calibration_body_honors_stop_during_leading_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bot.main as bot_main

    monkeypatch.setattr(bot_main, "_seconds_until_next_refit", lambda now: 60.0)
    refit_mock = Mock(return_value=bot_main._empty_calibration_maps())
    monkeypatch.setattr(bot_main, "refit_all", refit_mock)

    app = _make_calibration_app(monkeypatch)
    stop = asyncio.Event()

    async def stopper() -> None:
        await asyncio.sleep(0.01)
        stop.set()

    start = asyncio.get_running_loop().time()
    await asyncio.gather(bot_main._calibration_refit_loop(app, stop), stopper())
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 0.5
    refit_mock.assert_not_called()


async def test_zero_next_delay_with_raising_body_does_not_tight_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=50, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    iters = {"n": 0}

    async def call_body() -> None:
        iters["n"] += 1
        raise RuntimeError("nope")

    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=60.0,
        stop=stop,
        next_delay=lambda _: 0.0,
    )

    assert iters["n"] < 200
    assert any(d > 0 for d in shim.durations)


async def test_next_delay_raising_does_not_kill_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=3, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    iters = {"n": 0}

    async def call_body() -> None:
        iters["n"] += 1

    def boom(_: bool) -> float:
        raise RuntimeError("delay boom")

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=60.0,
        stop=stop,
        next_delay=boom,
    )

    assert iters["n"] == 3
    delay_errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "loop_next_delay_failed name=test_loop" in r.getMessage()
    ]
    assert len(delay_errors) == 3
    assert all(r.exc_info is not None for r in delay_errors)


async def test_failure_delay_raising_does_not_kill_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=2, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        raise _http_error(500)

    def boom() -> float:
        raise RuntimeError("failure delay boom")

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=60.0,
        stop=stop,
        failure_delay=boom,
    )

    fd_errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and "loop_failure_delay_failed name=test_loop" in r.getMessage()
    ]
    assert len(fd_errors) == 2
    assert shim.durations[0] == pytest.approx(loop_runner._FAILURE_RECOVERY_SLEEP_SECONDS)


def test_loop_skipped_is_public_api() -> None:
    assert issubclass(LoopSkipped, Exception)
    assert not issubclass(LoopSkipped, KeyboardInterrupt)
    assert not issubclass(LoopSkipped, SystemExit)


async def test_loop_skipped_after_failure_streak_no_recovered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=5, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    seq = [
        _http_error(500),
        _http_error(500),
        _http_error(500),
        LoopSkipped(),
        _http_error(500),
    ]
    it = iter(seq)

    async def call_body() -> None:
        raise next(it)

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=60.0,
        stop=stop,
        dedup_window_seconds=60.0,
        failure_delay=lambda: 1.0,
    )

    recovered = [r for r in caplog.records if "loop_recovered" in r.getMessage()]
    assert recovered == []
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1


async def test_recovery_emits_once_and_resets_state(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=7, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    seq = [
        _http_error(500),
        _http_error(500),
        _http_error(500),
        _http_error(500),
        _http_error(500),
        None,
        _http_error(500),
    ]
    it = iter(seq)

    async def call_body() -> None:
        item = next(it)
        if item is not None:
            raise item

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=60.0,
        stop=stop,
        dedup_window_seconds=60.0,
        failure_delay=lambda: 1.0,
    )

    recovered = [r for r in caplog.records if "loop_recovered" in r.getMessage()]
    assert len(recovered) == 1
    assert re.search(
        r"loop_recovered name=test_loop consecutive_failures_before_recovery=5 "
        r"outage_duration_sec=\S+",
        recovered[0].getMessage(),
    )
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2


async def test_backoff_multiplier_kicks_in_after_n(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=6, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        raise _http_error(500)

    base = loop_runner._FAILURE_RECOVERY_SLEEP_SECONDS
    interval = 60.0
    cap = min(4.0 * interval, 1_000_000.0)
    await run_loop(
        name="test_loop",
        body=call_body,
        interval_seconds=interval,
        stop=stop,
        sustained_failure_threshold_seconds=1_000_000.0,
        backoff_after_n_failures=3,
    )

    assert shim.durations[0] == pytest.approx(min(base, cap))
    assert shim.durations[1] == pytest.approx(min(base, cap))
    assert shim.durations[2] == pytest.approx(min(base, cap))
    assert shim.durations[3] == pytest.approx(min(base * 2, cap))
    assert shim.durations[4] == pytest.approx(min(base * 4, cap))


async def test_successful_iteration_no_prior_streak_emits_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(loop_runner, "_monotonic", clock)
    stop = asyncio.Event()
    shim = _SleepShim(clock, stop_after=1, stop=stop)
    monkeypatch.setattr(loop_runner, "_sleep_or_stop", shim)

    async def call_body() -> None:
        return None

    caplog.set_level(logging.INFO, logger=_LOG)
    await run_loop(name="test_loop", body=call_body, interval_seconds=60.0, stop=stop)

    assert caplog.records == []


async def test_stop_event_short_circuits_inter_iteration_sleep() -> None:
    stop = asyncio.Event()
    calls = {"n": 0}

    async def call_body() -> None:
        calls["n"] += 1
        stop.set()

    start = asyncio.get_running_loop().time()
    await run_loop(name="test_loop", body=call_body, interval_seconds=3600.0, stop=stop)
    elapsed = asyncio.get_running_loop().time() - start

    assert calls["n"] == 1
    assert elapsed < 0.5


async def test_base_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = asyncio.Event()

    async def call_body() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        await run_loop(name="test_loop", body=call_body, interval_seconds=60.0, stop=stop)


async def test_system_exit_propagates() -> None:
    stop = asyncio.Event()

    async def call_body() -> None:
        raise SystemExit

    with pytest.raises(SystemExit):
        await run_loop(name="test_loop", body=call_body, interval_seconds=60.0, stop=stop)


async def test_sleep_or_stop_returns_on_stop() -> None:
    stop = asyncio.Event()
    stop.set()
    start = asyncio.get_running_loop().time()
    await _sleep_or_stop(stop, 3600.0)
    assert asyncio.get_running_loop().time() - start < 0.5


async def test_sleep_or_stop_zero_timeout_yields() -> None:
    stop = asyncio.Event()
    await _sleep_or_stop(stop, 0.0)
    await _sleep_or_stop(stop, -1.0)


def _make_calibration_app(monkeypatch: pytest.MonkeyPatch) -> object:
    import bot.main as bot_main
    from bot.storage.sqlite import Base, make_engine, make_session_factory

    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)

    class _Settings:
        mode = "paper"
        log_level = "INFO"

    class _Noop:
        async def aclose(self) -> None:
            return None

    app = bot_main.App(
        settings=_Settings(),  # type: ignore[arg-type]
        engine=engine,
        session_factory=sf,
        meteo=_Noop(),  # type: ignore[arg-type]
        kalshi=_Noop(),  # type: ignore[arg-type]
        kalshi_read=_Noop(),  # type: ignore[arg-type]
        acis=_Noop(),  # type: ignore[arg-type]
        series_list=("KXHIGHDEN",),
    )
    return app
