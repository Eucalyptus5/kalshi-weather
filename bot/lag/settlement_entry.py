from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from bot.lag.mechanism_rates import separating_strikes
from bot.lag.placement_grid import CloseSidecar, close_of
from bot.lag.settlement_straddle import Straddle, event_ticker_of, settling_row
from bot.markets.observation_window import observation_window
from bot.observations.metar import StationObservation


CROSSING = "crossing"
OPEN = "open"
MAX = "max"
YES = "yes"


@dataclass(frozen=True, slots=True, kw_only=True)
class StraddleEntry:
    entry_instant: datetime
    strike: int
    settlement_side: str
    ticker: str
    close_time: datetime
    entry_at_or_before_close: bool
    instant_class: str
    identifiable_ex_ante: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class EntryCounts:
    n: int
    entry_at_close_n: int
    open_at_first_reading_n: int
    at_or_before_close_n: int
    close_minus_entry_s: Mapping[int, int]


# The window is half-open: the next event day's window opens exactly where this one closes, and a
# boundary minute counted twice would hand one reading to two days. A walk that never separates,
# or one with nothing to walk, contradicts the straddle's own guarantee over that same window, so
# it is a mismatch between the readings and the record rather than a day without a crossing.
def entry_of(
    straddle: Straddle,
    observations: Sequence[StationObservation],
    sidecar: CloseSidecar,
) -> StraddleEntry:
    if straddle.extreme != MAX:
        raise ValueError(
            f"{straddle.station} {straddle.event_date} carries extreme {straddle.extreme!r}"
        )
    start, end = observation_window(straddle.timezone, straddle.event_date)
    window = sorted(
        (reading for reading in observations if start <= reading.valid_time < end),
        key=lambda reading: reading.valid_time,
    )
    if not window:
        raise ValueError(
            f"{straddle.station} {straddle.event_date} has no readings inside its window"
        )

    running = window[0].temp_f
    for index, observation in enumerate(window):
        running = max(running, observation.temp_f)
        separating = separating_strikes(running, straddle.acis_f, straddle.separating_strikes)
        if separating:
            instant = observation.valid_time
            strike = separating[0]
            instant_class = OPEN if index == 0 else CROSSING
            break
    else:
        raise ValueError(
            f"{straddle.station} {straddle.event_date} never separates inside its window"
        )

    event_ticker = event_ticker_of(straddle.root, straddle.event_date)
    ticker = settling_row(sidecar, event_ticker, straddle.acis_f).ticker
    close_time = close_of(sidecar, ticker)
    # The side is the settlement source's own, so it is yes on the row that settles at the official
    # reading; deriving it from the observer would invert it. That reading is a daily extreme that
    # does not exist until the window has closed, so nobody standing at the instant can see it.
    return StraddleEntry(
        entry_instant=instant,
        strike=strike,
        settlement_side=YES,
        ticker=ticker,
        close_time=close_time,
        entry_at_or_before_close=instant <= close_time,
        instant_class=instant_class,
        identifiable_ex_ante=False,
    )


def entry_counts(records: Sequence[StraddleEntry]) -> EntryCounts:
    differences = Counter(
        (record.close_time - record.entry_instant) // timedelta(seconds=1) for record in records
    )
    return EntryCounts(
        n=sum(1 for record in records if record.instant_class == CROSSING),
        entry_at_close_n=sum(1 for record in records if record.entry_instant == record.close_time),
        open_at_first_reading_n=sum(1 for record in records if record.instant_class == OPEN),
        at_or_before_close_n=sum(1 for record in records if record.entry_at_or_before_close),
        close_minus_entry_s=dict(differences),
    )
