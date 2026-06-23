from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from bot.markets.observation_window import observation_window
from bot.markets.parser import ParsedTicker
from bot.observations.metar import StationObservation


@dataclass(frozen=True, slots=True)
class LockEvent:
    ticker: str
    side_locked: Literal["yes", "no"]
    t0: datetime
    strike: Decimal
    crossing_temp_f: Decimal
    lock_ambiguous: bool
    basis_valid: bool = True


def detect_lock_events(
    market: ParsedTicker,
    obs: list[StationObservation],
    *,
    tz_name: str,
    rounding_margin_f: Decimal = Decimal("1.0"),
) -> list[LockEvent]:
    """Return [] or a single-element list with the first running-max crossing of the strike."""
    if market.kind not in ("above", "below", "bracket"):
        return []

    start_utc, end_utc = observation_window(tz_name, market.event_date)
    in_window = [o for o in obs if start_utc <= o.valid_time < end_utc]
    in_window.sort(key=lambda o: o.valid_time)

    running_max_f = Decimal("-Infinity")

    for ob in in_window:
        new_max = max(running_max_f, ob.temp_f)
        if new_max == running_max_f:
            continue
        running_max_f = new_max

        if market.kind == "above":
            strike = market.strikes[0]
            clean = running_max_f >= strike + rounding_margin_f
            ambiguous = (not clean) and running_max_f >= strike
            if not (clean or ambiguous):
                continue
            return [
                LockEvent(
                    ticker=market.raw,
                    side_locked="yes",
                    t0=ob.publication_time,
                    strike=strike,
                    crossing_temp_f=running_max_f,
                    lock_ambiguous=ambiguous,
                )
            ]

        if market.kind == "below":
            strike = market.strikes[0]
            clean = running_max_f > strike + rounding_margin_f
            ambiguous = (not clean) and running_max_f > strike
            if not (clean or ambiguous):
                continue
            return [
                LockEvent(
                    ticker=market.raw,
                    side_locked="no",
                    t0=ob.publication_time,
                    strike=strike,
                    crossing_temp_f=running_max_f,
                    lock_ambiguous=ambiguous,
                )
            ]

        low, high = market.strikes
        clean = running_max_f > high + rounding_margin_f
        ambiguous = (not clean) and running_max_f > high
        if not (clean or ambiguous):
            continue
        return [
            LockEvent(
                ticker=market.raw,
                side_locked="no",
                t0=ob.publication_time,
                strike=high,
                crossing_temp_f=running_max_f,
                lock_ambiguous=ambiguous,
            )
        ]

    return []
