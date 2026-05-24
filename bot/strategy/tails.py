from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from bot.execution.paper import TradeSide
from bot.strategy.sizing import TAILS_ACTIVE_KELLY_FRAC, compute_stake_contracts


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
    ensemble_spread: Decimal
    sigma_T_median: Decimal
    event_budget_remaining: Decimal
    market_budget_remaining: Decimal
    depth_at_price: int
    price_per_contract: Decimal
    no_cost_per_contract: Decimal | None = None


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


def _skip(reason: str) -> TailsSignal:
    return TailsSignal(
        action=TailsAction.SKIP,
        contracts=0,
        notional_dollars=Decimal("0"),
        reason=reason,
    )


def evaluate(ctx: TailsContext, *, mode: str = "paper") -> TailsSignal:
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

    if ctx.depth_at_price <= 0:
        return _skip("yes_bid_depth_zero")

    if ctx.price_per_contract == Decimal("1"):
        return _skip("cost_basis_unusable")

    contracts = compute_stake_contracts(
        side=TradeSide.SELL_YES,
        q=ctx.fair_yes,
        p=ctx.yes_bid,
        sigma_T=ctx.ensemble_spread,
        sigma_T_median=ctx.sigma_T_median,
        kelly_frac=TAILS_ACTIVE_KELLY_FRAC,
        bankroll=ctx.bankroll,
        event_budget_remaining=ctx.event_budget_remaining,
        market_budget_remaining=ctx.market_budget_remaining,
        depth_at_price=ctx.depth_at_price,
        price_per_contract=ctx.price_per_contract,
    )
    if contracts < 1:
        return _skip("below_min_size")

    actual_notional = ctx.price_per_contract * Decimal(contracts)
    return TailsSignal(
        action=TailsAction.SELL_YES,
        contracts=contracts,
        notional_dollars=actual_notional,
        reason="trade",
    )
