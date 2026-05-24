from __future__ import annotations

import logging
from decimal import Decimal

from bot.execution.paper import TradeSide

logger = logging.getLogger(__name__)


EDGE_KELLY_FRAC: Decimal = Decimal("0.25")
TAILS_KELLY_FRAC: Decimal = Decimal("0.10")
# tails sizing is capped at the pre-calibration value until tails calibration lands; the
# active multiplier reads through the min(...) so a copy-paste typo cannot uncap it.
TAILS_KELLY_FRAC_PRE_CALIBRATION: Decimal = Decimal("0.05")
TAILS_ACTIVE_KELLY_FRAC: Decimal = min(TAILS_KELLY_FRAC, TAILS_KELLY_FRAC_PRE_CALIBRATION)
SIGMA_T_FLOOR: Decimal = Decimal("0.01")
DEPTH_FRACTION_CAP: Decimal = Decimal("0.5")


# values measured against the per-lead-time forecast snapshot; replace with a rolling
# median once tails calibration is instrumented. key 168 mirrors 144 (no underlying data).
SIGMA_T_MEDIAN_BY_LEAD_H: dict[int, Decimal] = {
    0: Decimal("1.60"),
    24: Decimal("2.02"),
    48: Decimal("2.46"),
    72: Decimal("2.81"),
    96: Decimal("3.31"),
    120: Decimal("3.86"),
    144: Decimal("4.47"),
    168: Decimal("4.47"),
}


def sigma_t_median_for_lead(lead_hours: int) -> Decimal:
    keys = sorted(SIGMA_T_MEDIAN_BY_LEAD_H.keys())
    if lead_hours <= keys[0]:
        return SIGMA_T_MEDIAN_BY_LEAD_H[keys[0]]
    if lead_hours >= keys[-1]:
        return SIGMA_T_MEDIAN_BY_LEAD_H[keys[-1]]
    chosen = keys[0]
    for k in keys:
        if k <= lead_hours:
            chosen = k
        else:
            break
    return SIGMA_T_MEDIAN_BY_LEAD_H[chosen]


def compute_stake_contracts(
    *,
    side: TradeSide,
    q: Decimal,
    p: Decimal,
    sigma_T: Decimal,
    sigma_T_median: Decimal,
    kelly_frac: Decimal,
    bankroll: Decimal,
    event_budget_remaining: Decimal,
    market_budget_remaining: Decimal,
    depth_at_price: int,
    price_per_contract: Decimal,
) -> int:
    """Return the stake in contracts. kelly_frac = 0 disables sizing."""
    if p <= Decimal("0") or p >= Decimal("1"):
        return 0
    if price_per_contract <= Decimal("0"):
        return 0

    if sigma_T < SIGMA_T_FLOOR:
        sigma_T = SIGMA_T_FLOOR

    if side is TradeSide.BUY_YES:
        f_full = (q - p) / (Decimal("1") - p)
    else:
        f_full = (p - q) / p
    if f_full <= Decimal("0"):
        return 0

    shrinkage = min(Decimal("1"), sigma_T_median / sigma_T)
    f_scaled = f_full * shrinkage * kelly_frac
    kelly_dollars = f_scaled * bankroll

    budget = max(Decimal("0"), min(event_budget_remaining, market_budget_remaining))
    dollars_after_budget = min(kelly_dollars, budget)

    depth_dollars = DEPTH_FRACTION_CAP * Decimal(depth_at_price) * price_per_contract
    dollars_final = min(dollars_after_budget, depth_dollars)

    contracts = int(dollars_final / price_per_contract)
    if contracts < 1:
        return 0

    logger.debug(
        "compute_stake_contracts side=%s contracts=%d dollars_final=%s",
        side.value,
        contracts,
        dollars_final,
    )
    return contracts
