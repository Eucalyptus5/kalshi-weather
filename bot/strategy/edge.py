from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


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
KELLY_MULTIPLIER: Decimal = Decimal("0.15")


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
    min_spread: Decimal = DEFAULT_MIN_SPREAD,
    kelly_multiplier: Decimal = KELLY_MULTIPLIER,
) -> EdgeSignal:
    mid = (ctx.yes_ask + ctx.yes_bid) / Decimal("2")
    if abs(ctx.fair_yes - mid) <= EDGE_THRESHOLD:
        return _skip("edge_too_small")
    if ctx.ensemble_spread <= min_spread:
        return _skip("spread_too_tight")
    if ctx.is_blacklisted:
        return _skip("blacklisted")
    if ctx.is_same_day:
        return _skip("same_day")
    if ctx.nbm_divergence is not None and ctx.nbm_divergence > NBM_DIVERGENCE_LIMIT:
        return _skip("nbm_diverged")

    if ctx.fair_yes > ctx.yes_ask + DIRECTION_CUSHION:
        action = EdgeAction.BUY_YES
        cost_per_contract = ctx.yes_ask
        kelly_fraction_full = (ctx.fair_yes - ctx.yes_ask) / (Decimal("1") - ctx.yes_ask)
        reason = "trade_buy"
    elif ctx.fair_yes < ctx.yes_bid - DIRECTION_CUSHION:
        action = EdgeAction.SELL_YES
        cost_per_contract = Decimal("1") - ctx.yes_bid
        kelly_fraction_full = (ctx.yes_bid - ctx.fair_yes) / ctx.yes_bid
        reason = "trade_sell"
    else:
        return _skip("no_direction")

    size_dollars = kelly_multiplier * kelly_fraction_full * ctx.bankroll
    contracts = int(size_dollars / cost_per_contract)
    if contracts < 1:
        return _skip("below_min_size")

    actual_notional = cost_per_contract * Decimal(contracts)
    return EdgeSignal(
        action=action,
        contracts=contracts,
        notional_dollars=actual_notional,
        reason=reason,
    )
