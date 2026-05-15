from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class TailsAction(Enum):
    SELL_YES = "sell_yes"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class TailsContext:
    yes_ask: Decimal
    yes_bid: Decimal
    no_bid: Decimal
    fair_yes: Decimal
    close_time: datetime
    now: datetime
    bankroll: Decimal
    is_same_day: bool


@dataclass(frozen=True, slots=True)
class TailsSignal:
    action: TailsAction
    contracts: int
    notional_dollars: Decimal
    reason: str


ASK_THRESHOLD: Decimal = Decimal("0.10")
FAIR_THRESHOLD: Decimal = Decimal("0.07")
MIN_MINUTES_TO_CLOSE: int = 60
DEFAULT_KELLY_FRACTION: Decimal = Decimal("0.15")
DEFAULT_POSITION_CAP: Decimal = Decimal("50")


def _skip(reason: str) -> TailsSignal:
    return TailsSignal(
        action=TailsAction.SKIP,
        contracts=0,
        notional_dollars=Decimal("0"),
        reason=reason,
    )


def evaluate(
    ctx: TailsContext,
    *,
    kelly_fraction: Decimal = DEFAULT_KELLY_FRACTION,
    position_cap: Decimal = DEFAULT_POSITION_CAP,
) -> TailsSignal:
    if ctx.yes_ask <= ASK_THRESHOLD:
        return _skip("ask_too_low")
    if ctx.no_bid <= Decimal("0"):
        return _skip("no_no_bid")
    if ctx.fair_yes >= FAIR_THRESHOLD:
        return _skip("fair_too_high")
    if ctx.is_same_day:
        return _skip("same_day")

    seconds_to_close = (ctx.close_time - ctx.now).total_seconds()
    minutes_to_close = Decimal(str(seconds_to_close)) / Decimal("60")
    if minutes_to_close <= Decimal(MIN_MINUTES_TO_CLOSE):
        return _skip("too_close_to_settle")

    fair_no = Decimal("1") - ctx.fair_yes
    edge = fair_no - ctx.no_bid
    if edge <= Decimal("0"):
        return _skip("negative_edge")

    edge_ratio = edge / (Decimal("1") - ctx.no_bid)
    notional_target = kelly_fraction * ctx.bankroll * edge_ratio
    notional = min(notional_target, position_cap)

    contracts = int(notional / ctx.no_bid)
    if contracts < 1:
        return _skip("below_min_size")

    actual_notional = ctx.no_bid * Decimal(contracts)
    return TailsSignal(
        action=TailsAction.SELL_YES,
        contracts=contracts,
        notional_dollars=actual_notional,
        reason="trade",
    )
