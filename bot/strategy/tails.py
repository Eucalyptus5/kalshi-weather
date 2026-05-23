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
    no_cost_per_contract: Decimal | None = None
    yes_bid_depth: int | None = None


@dataclass(frozen=True, slots=True)
class TailsSignal:
    action: TailsAction
    contracts: int
    notional_dollars: Decimal
    reason: str


ASK_THRESHOLD: Decimal = Decimal("0.10")
FAIR_THRESHOLD: Decimal = Decimal("0.07")
# load-bearing: tuning below 0.095 flips the real-spread margin negative against
# the friction gate's real-fee-plus-real-half-tick boundary.
YES_BID_FLOOR: Decimal = FAIR_THRESHOLD + Decimal("0.025")
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
    mode: str = "paper",
    kelly_fraction: Decimal = DEFAULT_KELLY_FRACTION,
    position_cap: Decimal = DEFAULT_POSITION_CAP,
    contracts_cap: int | None = None,
) -> TailsSignal:
    if mode == "demo" and ctx.no_cost_per_contract is None:
        raise RuntimeError(
            "demo mode requires book-derived cost basis; "
            "_build_intents failed to thread book.no_ask"
        )
    if ctx.yes_ask <= ASK_THRESHOLD:
        return _skip("ask_too_low")
    if ctx.no_bid <= Decimal("0"):
        return _skip("no_no_bid")
    if ctx.yes_bid < YES_BID_FLOOR:
        return _skip("no_yes_bid")
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

    if ctx.yes_bid_depth is not None and ctx.yes_bid_depth <= 0:
        return _skip("yes_bid_depth_zero")

    cost_per_contract = (
        ctx.no_cost_per_contract if ctx.no_cost_per_contract is not None else ctx.no_bid
    )
    if cost_per_contract == Decimal("1"):
        return _skip("cost_basis_unusable")

    edge_ratio = edge / (Decimal("1") - cost_per_contract)
    notional_target = kelly_fraction * ctx.bankroll * edge_ratio
    notional = min(notional_target, position_cap)

    contracts = int(notional / cost_per_contract)
    if contracts_cap is not None:
        contracts = min(contracts, contracts_cap)
    if ctx.yes_bid_depth is not None and contracts > ctx.yes_bid_depth:
        contracts = ctx.yes_bid_depth
    if contracts < 1:
        return _skip("below_min_size")

    actual_notional = cost_per_contract * Decimal(contracts)
    return TailsSignal(
        action=TailsAction.SELL_YES,
        contracts=contracts,
        notional_dollars=actual_notional,
        reason="trade",
    )
