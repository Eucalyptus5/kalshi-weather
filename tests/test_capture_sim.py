from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal
from typing import Literal

from bot.execution.fees import taker_fee
from bot.lag.capture_sim import (
    CaptureResult,
    LatencyStack,
    simulate_capture,
)
from bot.lag.event_study import OrderbookSnapshotRow
from bot.lag.lock_events import LockEvent


UTC = _timezone.utc


def _event(
    ticker: str,
    *,
    side: Literal["yes", "no"] = "yes",
    t0: datetime,
    strike: Decimal | str | int = 85,
    crossing: Decimal | str | int | None = None,
    lock_ambiguous: bool = False,
) -> LockEvent:
    strike_d = Decimal(str(strike))
    crossing_d = Decimal(str(crossing)) if crossing is not None else strike_d + Decimal("2")
    return LockEvent(
        ticker=ticker,
        side_locked=side,
        t0=t0,
        strike=strike_d,
        crossing_temp_f=crossing_d,
        lock_ambiguous=lock_ambiguous,
    )


def _snap(
    ticker: str,
    t: datetime,
    *,
    yes_bid: Decimal | str = "0.50",
    yes_ask: Decimal | str = "0.50",
    no_bid: Decimal | str | None = None,
    no_ask: Decimal | str | None = None,
    yes_ask_depth: int | None = None,
    yes_bid_depth: int | None = None,
    no_ask_depth: int | None = None,
    no_bid_depth: int | None = None,
) -> OrderbookSnapshotRow:
    return OrderbookSnapshotRow(
        ticker=ticker,
        snapshot_at=t,
        yes_bid=Decimal(str(yes_bid)),
        yes_ask=Decimal(str(yes_ask)),
        no_bid=Decimal(str(no_bid)) if no_bid is not None else None,
        no_ask=Decimal(str(no_ask)) if no_ask is not None else None,
        yes_ask_depth=yes_ask_depth,
        yes_bid_depth=yes_bid_depth,
        no_ask_depth=no_ask_depth,
        no_bid_depth=no_bid_depth,
    )


def test_clean_yes_lock_fills_full_depth_and_pays() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(
            ticker,
            t0 - timedelta(seconds=30),
            yes_bid="0.30",
            yes_ask="0.40",
            yes_ask_depth=10,
        ),
        _snap(
            ticker,
            t0 + timedelta(seconds=125),
            yes_bid="0.93",
            yes_ask="0.95",
            yes_ask_depth=5,
        ),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("100"),
    )

    expected_fee = taker_fee(5, Decimal("0.40"))
    assert isinstance(result, CaptureResult)
    assert result.ticker == ticker
    assert result.side_locked == "yes"
    assert result.stale_price == Decimal("0.40")
    assert result.fillable_depth == 5
    assert result.contracts_filled == 5
    assert result.notional_spent == Decimal("5") * Decimal("0.40")
    assert result.payoff == Decimal("5")
    assert result.fees_paid == expected_fee
    assert result.pnl == Decimal("5") - (Decimal("5") * Decimal("0.40")) - expected_fee
    assert result.depth_lower_bound is False


def test_no_lock_fills_depth_and_pays() -> None:
    ticker = "KXHIGHCHI-26JUN17-T75-80"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, side="no", t0=t0, strike=80, crossing=82)
    snaps = [
        _snap(
            ticker,
            t0 - timedelta(seconds=20),
            yes_bid="0.65",
            yes_ask="0.75",
            no_bid="0.25",
            no_ask="0.30",
            no_ask_depth=20,
        ),
        _snap(
            ticker,
            t0 + timedelta(seconds=125),
            yes_bid="0.97",
            yes_ask="0.99",
            no_bid="0.01",
            no_ask="0.02",
            no_ask_depth=8,
        ),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("83"),
        notional_cap=Decimal("50"),
    )

    expected_fee = taker_fee(8, Decimal("0.30"))
    assert result.side_locked == "no"
    assert result.stale_price == Decimal("0.30")
    assert result.fillable_depth == 8
    assert result.contracts_filled == 8
    assert result.notional_spent == Decimal("8") * Decimal("0.30")
    assert result.payoff == Decimal("8")
    assert result.fees_paid == expected_fee
    assert result.pnl == Decimal("8") - (Decimal("8") * Decimal("0.30")) - expected_fee


def test_no_lock_strict_gt_boundary_pays_zero() -> None:
    ticker = "KXHIGHCHI-26JUN17-T75-80"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, side="no", t0=t0, strike=80, crossing=82)
    snaps = [
        _snap(
            ticker,
            t0 - timedelta(seconds=20),
            no_bid="0.25",
            no_ask="0.30",
            no_ask_depth=20,
        ),
        _snap(
            ticker,
            t0 + timedelta(seconds=125),
            no_bid="0.01",
            no_ask="0.02",
            no_ask_depth=8,
        ),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("80"),
        notional_cap=Decimal("50"),
    )

    expected_fee = taker_fee(8, Decimal("0.30"))
    assert result.contracts_filled == 8
    assert result.payoff == Decimal("0")
    assert result.notional_spent == Decimal("8") * Decimal("0.30")
    assert result.fees_paid == expected_fee
    assert result.pnl == -(Decimal("8") * Decimal("0.30")) - expected_fee


def test_yes_lock_mislock_pays_zero() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 - timedelta(seconds=30), yes_ask="0.40", yes_ask_depth=10),
        _snap(ticker, t0 + timedelta(seconds=125), yes_ask="0.95", yes_ask_depth=5),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("84"),
        notional_cap=Decimal("100"),
    )

    expected_fee = taker_fee(5, Decimal("0.40"))
    assert result.contracts_filled == 5
    assert result.payoff == Decimal("0")
    assert result.notional_spent == Decimal("5") * Decimal("0.40")
    assert result.pnl == -(Decimal("5") * Decimal("0.40")) - expected_fee


def test_no_fillable_snapshot_at_t0_plus_delay() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 - timedelta(seconds=10), yes_ask="0.40", yes_ask_depth=12),
        _snap(ticker, t0 + timedelta(seconds=60), yes_ask="0.60", yes_ask_depth=9),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("100"),
    )

    assert result.stale_price == Decimal("0.40")
    assert result.fillable_depth == 0
    assert result.contracts_filled == 0
    assert result.notional_spent == Decimal("0")
    assert result.payoff == Decimal("0")
    assert result.fees_paid == Decimal("0")
    assert result.pnl == Decimal("0")


def test_no_stale_snapshot_before_t0() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 + timedelta(seconds=1), yes_ask="0.40", yes_ask_depth=10),
        _snap(ticker, t0 + timedelta(seconds=60), yes_ask="0.60", yes_ask_depth=8),
        _snap(ticker, t0 + timedelta(seconds=125), yes_ask="0.95", yes_ask_depth=5),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("100"),
    )

    assert result.stale_price is None
    assert result.contracts_filled == 0
    assert result.fillable_depth == 0
    assert result.notional_spent == Decimal("0")
    assert result.payoff == Decimal("0")
    assert result.fees_paid == Decimal("0")
    assert result.pnl == Decimal("0")


def test_notional_cap_binds() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 - timedelta(seconds=30), yes_ask="0.40", yes_ask_depth=1000),
        _snap(ticker, t0 + timedelta(seconds=125), yes_ask="0.95", yes_ask_depth=1000),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("10"),
    )

    assert result.contracts_filled == 25


def test_depth_cap_binds() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 - timedelta(seconds=30), yes_ask="0.40", yes_ask_depth=3),
        _snap(ticker, t0 + timedelta(seconds=125), yes_ask="0.95", yes_ask_depth=3),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("100"),
    )

    assert result.contracts_filled == 3


def test_latency_stack_defaults_and_overrides() -> None:
    assert LatencyStack().total_s == 125
    assert LatencyStack(decision_s=0).total_s == 120
    assert LatencyStack(obs_publication_s=0, poll_interval_s=0, decision_s=0).total_s == 0


def test_custom_latency_stack_changes_depth_read() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 - timedelta(seconds=10), yes_ask="0.40", yes_ask_depth=50),
        _snap(ticker, t0 + timedelta(seconds=30), yes_ask="0.60", yes_ask_depth=20),
        _snap(ticker, t0 + timedelta(seconds=125), yes_ask="0.95", yes_ask_depth=3),
    ]

    default_result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("1000"),
    )
    assert default_result.fillable_depth == 3
    assert default_result.contracts_filled == 3

    short_stack = LatencyStack(obs_publication_s=10, poll_interval_s=10, decision_s=10)
    short_result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("1000"),
        latency_stack=short_stack,
    )
    assert short_result.fillable_depth == 20
    assert short_result.contracts_filled == 20


def test_depth_from_t0_anti_test() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0, yes_ask="0.40", yes_ask_depth=65),
        _snap(ticker, t0 + timedelta(seconds=125), yes_ask="0.95", yes_ask_depth=3),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("1000"),
    )

    assert result.fillable_depth == 3
    assert result.contracts_filled <= 3


def test_snapshot_unreliable_forwards_in_no_fill() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 + timedelta(seconds=1), yes_ask="0.40", yes_ask_depth=10),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("100"),
        snapshot_unreliable=True,
    )

    assert result.stale_price is None
    assert result.contracts_filled == 0
    assert result.depth_lower_bound is True


def test_decimal_strict_on_result() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 - timedelta(seconds=30), yes_ask="0.40", yes_ask_depth=10),
        _snap(ticker, t0 + timedelta(seconds=125), yes_ask="0.95", yes_ask_depth=5),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("100"),
    )

    assert type(result.pnl) is Decimal
    assert type(result.notional_spent) is Decimal
    assert type(result.fees_paid) is Decimal


def test_stale_price_zero_short_circuit() -> None:
    ticker = "KXHIGHDEN-26JUN17-T85"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, t0=t0, strike=85, crossing=87)
    snaps = [
        _snap(ticker, t0 - timedelta(seconds=30), yes_ask="0", yes_ask_depth=10),
        _snap(ticker, t0 + timedelta(seconds=125), yes_ask="0.95", yes_ask_depth=5),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("87"),
        notional_cap=Decimal("100"),
    )

    assert result.stale_price == Decimal("0")
    assert result.contracts_filled == 0
    assert result.notional_spent == Decimal("0")
    assert result.payoff == Decimal("0")
    assert result.fees_paid == Decimal("0")
    assert result.pnl == Decimal("0")


def test_no_ask_missing_on_no_lock() -> None:
    ticker = "KXHIGHCHI-26JUN17-T75-80"
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(ticker, side="no", t0=t0, strike=80, crossing=82)
    snaps = [
        _snap(ticker, t0 - timedelta(seconds=20), yes_ask="0.75", no_ask=None),
        _snap(
            ticker,
            t0 + timedelta(seconds=125),
            yes_ask="0.99",
            no_ask="0.02",
            no_ask_depth=8,
        ),
    ]

    result = simulate_capture(
        ev,
        snaps,
        settle_price=Decimal("83"),
        notional_cap=Decimal("50"),
    )

    assert result.stale_price is None
    assert result.contracts_filled == 0
    assert result.notional_spent == Decimal("0")
    assert result.payoff == Decimal("0")
    assert result.fees_paid == Decimal("0")
    assert result.pnl == Decimal("0")


def test_orderbook_snapshot_row_backwards_compat() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    row = OrderbookSnapshotRow(
        ticker="KXHIGHDEN-26JUN17-T85",
        snapshot_at=t0,
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.50"),
    )

    assert row.no_bid is None
    assert row.no_ask is None
    assert row.yes_ask_depth is None
    assert row.yes_bid_depth is None
    assert row.no_ask_depth is None
    assert row.no_bid_depth is None
    assert row.yes_bid == Decimal("0.40")
    assert row.yes_ask == Decimal("0.50")
