from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Literal

from bot.execution.fees import taker_fee
from bot.lag.event_study import OrderbookSnapshotRow
from bot.lag.lock_events import LockEvent


_ZERO = Decimal("0")
_ONE = Decimal("1")


@dataclass(frozen=True, slots=True)
class LatencyStack:
    obs_publication_s: int = 60
    poll_interval_s: int = 60
    decision_s: int = 5

    @property
    def total_s(self) -> int:
        return self.obs_publication_s + self.poll_interval_s + self.decision_s


@dataclass(frozen=True, slots=True)
class CaptureResult:
    ticker: str
    side_locked: Literal["yes", "no"]
    stale_price: Decimal | None
    contracts_filled: int
    fillable_depth: int
    notional_spent: Decimal
    payoff: Decimal
    fees_paid: Decimal
    pnl: Decimal
    depth_lower_bound: bool


def simulate_capture(
    event: LockEvent,
    snapshots: list[OrderbookSnapshotRow],
    *,
    settle_price: Decimal,
    notional_cap: Decimal,
    latency_stack: LatencyStack | None = None,
    snapshot_unreliable: bool = False,
) -> CaptureResult:
    ticker_snaps = sorted(
        (s for s in snapshots if s.ticker == event.ticker),
        key=lambda s: s.snapshot_at,
    )

    stale_snap = None
    for s in ticker_snaps:
        if s.snapshot_at <= event.t0:
            stale_snap = s
        else:
            break

    if stale_snap is None:
        return CaptureResult(
            ticker=event.ticker,
            side_locked=event.side_locked,
            stale_price=None,
            contracts_filled=0,
            fillable_depth=0,
            notional_spent=_ZERO,
            payoff=_ZERO,
            fees_paid=_ZERO,
            pnl=_ZERO,
            depth_lower_bound=snapshot_unreliable,
        )

    delay_s = (latency_stack or LatencyStack()).total_s
    t_decision = event.t0 + timedelta(seconds=delay_s)
    fillable_snap = next(
        (s for s in ticker_snaps if s.snapshot_at >= t_decision),
        None,
    )

    if event.side_locked == "yes":
        stale_price = stale_snap.yes_ask
        fillable_depth = (fillable_snap.yes_ask_depth or 0) if fillable_snap is not None else 0
    else:
        stale_price = stale_snap.no_ask
        fillable_depth = (fillable_snap.no_ask_depth or 0) if fillable_snap is not None else 0

    if stale_price is None or stale_price <= _ZERO:
        return CaptureResult(
            ticker=event.ticker,
            side_locked=event.side_locked,
            stale_price=stale_price,
            contracts_filled=0,
            fillable_depth=fillable_depth,
            notional_spent=_ZERO,
            payoff=_ZERO,
            fees_paid=_ZERO,
            pnl=_ZERO,
            depth_lower_bound=snapshot_unreliable,
        )

    cap_by_notional = int(notional_cap / stale_price)
    contracts_filled = min(fillable_depth, cap_by_notional)

    if contracts_filled <= 0:
        return CaptureResult(
            ticker=event.ticker,
            side_locked=event.side_locked,
            stale_price=stale_price,
            contracts_filled=0,
            fillable_depth=fillable_depth,
            notional_spent=_ZERO,
            payoff=_ZERO,
            fees_paid=_ZERO,
            pnl=_ZERO,
            depth_lower_bound=snapshot_unreliable,
        )

    if event.side_locked == "yes":
        payoff_per_contract = _ONE if settle_price >= event.strike else _ZERO
    else:
        payoff_per_contract = _ONE if settle_price > event.strike else _ZERO

    notional_spent = Decimal(contracts_filled) * stale_price
    payoff = Decimal(contracts_filled) * payoff_per_contract
    fees_paid = taker_fee(contracts_filled, stale_price)
    pnl = payoff - notional_spent - fees_paid

    return CaptureResult(
        ticker=event.ticker,
        side_locked=event.side_locked,
        stale_price=stale_price,
        contracts_filled=contracts_filled,
        fillable_depth=fillable_depth,
        notional_spent=notional_spent,
        payoff=payoff,
        fees_paid=fees_paid,
        pnl=pnl,
        depth_lower_bound=snapshot_unreliable,
    )
