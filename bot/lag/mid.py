from __future__ import annotations

from decimal import Decimal

import numpy as np

from bot.lag.ladder_consistency import PRICE_TICKS


def ticks(price: Decimal) -> int:
    scaled = price * PRICE_TICKS
    whole = int(scaled)
    if scaled != whole:
        raise ValueError(f"price {price} is off the {PRICE_TICKS}-per-dollar grid")
    return whole


def mid2(yes_bid: int, no_bid: int) -> int:
    return yes_bid + PRICE_TICKS - no_bid


def mid2_array(yes_bid: np.ndarray, no_bid: np.ndarray) -> np.ndarray:
    return yes_bid + PRICE_TICKS - no_bid


# yes_ask is stored as one minus the NO bid, so an empty NO book prices the ask at 1.00 against no
# size at all, and a book empty on both sides mids at exactly 0.50 by construction. A zero best bid
# is the tape reporting an empty side, not a one-tick market. Depth is None where the tape does not
# record it: the touch column sets read prices only.
def two_sided(
    yes_bid: int,
    no_bid: int,
    *,
    yes_depth: int | None,
    no_depth: int | None,
) -> bool:
    live_yes = yes_bid > 0 and (yes_depth is None or yes_depth > 0)
    live_no = no_bid > 0 and (no_depth is None or no_depth > 0)
    return live_yes and live_no


def two_sided_array(
    yes_bid: np.ndarray,
    no_bid: np.ndarray,
    *,
    yes_depth: np.ndarray | None,
    no_depth: np.ndarray | None,
) -> np.ndarray:
    live = (yes_bid > 0) & (no_bid > 0)
    if yes_depth is not None:
        live &= yes_depth > 0
    if no_depth is not None:
        live &= no_depth > 0
    return live
