from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from bot.execution.paper import TradeSide
from bot.strategy.sizing import EDGE_KELLY_FRAC, compute_stake_contracts


class EdgeAction(Enum):
    BUY_YES = "buy_yes"
    SELL_YES = "sell_yes"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class EdgeContext:
    yes_ask: Decimal
    yes_bid: Decimal
    fair_yes: Decimal
    ensemble_spread: Decimal
    bankroll: Decimal
    is_same_day: bool
    is_blacklisted: bool
    nbm_divergence: Decimal | None
    sigma_T_median: Decimal
    event_budget_remaining: Decimal
    market_budget_remaining: Decimal
    depth_at_price: int
    price_per_contract: Decimal
    no_cost_per_contract: Decimal | None = None


@dataclass(frozen=True, slots=True)
class EdgeSignal:
    action: EdgeAction
    contracts: int
    notional_dollars: Decimal
    reason: str


EDGE_THRESHOLD: Decimal = Decimal("0.08")
DIRECTION_CUSHION: Decimal = Decimal("0.04")
NBM_DIVERGENCE_LIMIT: Decimal = Decimal("5")
DEFAULT_MIN_SPREAD: Decimal = Decimal("1.0")


def _skip(reason: str) -> EdgeSignal:
    return EdgeSignal(
        action=EdgeAction.SKIP,
        contracts=0,
        notional_dollars=Decimal("0"),
        reason=reason,
    )


def evaluate(
    ctx: EdgeContext,
    *,
    mode: str = "paper",
    min_spread: Decimal = DEFAULT_MIN_SPREAD,
) -> EdgeSignal:
    if mode == "demo" and ctx.no_cost_per_contract is None:
        raise RuntimeError(
            "demo mode requires book-derived cost basis; "
            "_build_intents failed to thread book.no_ask"
        )
    mid = (ctx.yes_ask + ctx.yes_bid) / Decimal("2")
    if ctx.is_blacklisted:
        return _skip("blacklisted")
    if abs(ctx.fair_yes - mid) <= EDGE_THRESHOLD:
        return _skip("edge_too_small")
    if ctx.ensemble_spread <= min_spread:
        return _skip("spread_too_tight")
    if ctx.is_same_day:
        return _skip("same_day")
    if ctx.nbm_divergence is not None and ctx.nbm_divergence > NBM_DIVERGENCE_LIMIT:
        return _skip("nbm_diverged")

    if ctx.fair_yes > ctx.yes_ask + DIRECTION_CUSHION:
        action = EdgeAction.BUY_YES
        side = TradeSide.BUY_YES
        p = ctx.yes_ask
        reason = "trade_buy"
    elif ctx.fair_yes < ctx.yes_bid - DIRECTION_CUSHION:
        action = EdgeAction.SELL_YES
        side = TradeSide.SELL_YES
        p = ctx.yes_bid
        reason = "trade_sell"
    else:
        return _skip("no_direction")

    if ctx.depth_at_price <= 0:
        return _skip("depth_zero_clamp")

    contracts = compute_stake_contracts(
        side=side,
        q=ctx.fair_yes,
        p=p,
        sigma_T=ctx.ensemble_spread,
        sigma_T_median=ctx.sigma_T_median,
        kelly_frac=EDGE_KELLY_FRAC,
        bankroll=ctx.bankroll,
        event_budget_remaining=ctx.event_budget_remaining,
        market_budget_remaining=ctx.market_budget_remaining,
        depth_at_price=ctx.depth_at_price,
        price_per_contract=ctx.price_per_contract,
    )
    if contracts < 1:
        return _skip("below_min_size")

    actual_notional = ctx.price_per_contract * Decimal(contracts)
    return EdgeSignal(
        action=action,
        contracts=contracts,
        notional_dollars=actual_notional,
        reason=reason,
    )
