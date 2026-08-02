from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from bot.markets.observation_window import observation_window
from bot.markets.parser import ParsedTicker
from bot.observations.metar import StationObservation


YES: Literal["yes"] = "yes"
NO: Literal["no"] = "no"


def is_low_ladder(series: str) -> bool:
    if series.startswith("KXLOW"):
        return True
    if series.startswith("KXHIGH"):
        return False
    raise ValueError(f"{series} is neither a high nor a low temperature ladder")


@dataclass(frozen=True, slots=True)
class LockEvent:
    ticker: str
    side_locked: Literal["yes", "no"]
    t0: datetime
    strike: Decimal
    crossing_temp_f: Decimal
    lock_ambiguous: bool
    basis_valid: bool = True

    @property
    def series(self) -> str:
        return self.ticker.split("-", 1)[0]


def detect_lock_events(
    market: ParsedTicker,
    obs: list[StationObservation],
    *,
    tz_name: str,
    rounding_margin_f: Decimal = Decimal("1.0"),
) -> list[LockEvent]:
    """Return [] or a single-element list with the first running-extreme crossing of the strike."""
    low = is_low_ladder(market.series)
    if market.kind not in ("above", "below", "bracket"):
        return []

    start_utc, end_utc = observation_window(tz_name, market.event_date)
    in_window = [o for o in obs if start_utc <= o.valid_time < end_utc]
    in_window.sort(key=lambda o: o.valid_time)

    if low:
        return _falling_locks(market, in_window, rounding_margin_f)
    return _rising_locks(market, in_window, rounding_margin_f)


# A market that wins at exact equality is satisfied there and one that dies at exact equality is
# still alive, so on both ladders the locked side reads inclusively and the losing side strictly.
def _rising_locks(
    market: ParsedTicker, in_window: Sequence[StationObservation], margin: Decimal
) -> list[LockEvent]:
    side = YES if market.kind == "above" else NO
    strike = market.strikes[1] if market.kind == "bracket" else market.strikes[0]
    bar = strike + margin

    running_max_f = Decimal("-Infinity")
    for ob in in_window:
        new_max = max(running_max_f, ob.temp_f)
        if new_max == running_max_f:
            continue
        running_max_f = new_max

        clean = running_max_f >= bar if side == YES else running_max_f > bar
        ambiguous = (not clean) and (
            running_max_f >= strike if side == YES else running_max_f > strike
        )
        if not (clean or ambiguous):
            continue
        return [
            LockEvent(
                ticker=market.raw,
                side_locked=side,
                t0=ob.publication_time,
                strike=strike,
                crossing_temp_f=running_max_f,
                lock_ambiguous=ambiguous,
            )
        ]

    return []


def _falling_locks(
    market: ParsedTicker, in_window: Sequence[StationObservation], margin: Decimal
) -> list[LockEvent]:
    side = YES if market.kind == "below" else NO
    strike = market.strikes[0]
    bar = strike - margin

    running_min_f = Decimal("Infinity")
    for ob in in_window:
        new_min = min(running_min_f, ob.temp_f)
        if new_min == running_min_f:
            continue
        running_min_f = new_min

        clean = running_min_f <= bar if side == YES else running_min_f < bar
        ambiguous = (not clean) and (
            running_min_f <= strike if side == YES else running_min_f < strike
        )
        if not (clean or ambiguous):
            continue
        return [
            LockEvent(
                ticker=market.raw,
                side_locked=side,
                t0=ob.publication_time,
                strike=strike,
                crossing_temp_f=running_min_f,
                lock_ambiguous=ambiguous,
            )
        ]

    return []
