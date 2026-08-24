from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from bot.lag.mechanism_rates import separating_strikes
from bot.lag.placement_grid import CloseSidecar, MarketClose


BETWEEN = "between"
GREATER = "greater"
LESS = "less"

MONTH_CODES: tuple[str, ...] = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Straddle:
    root: str
    station: str
    event_date: date
    extreme: str
    timezone: str
    observed_f: Decimal
    acis_f: Decimal
    cell_edges: tuple[int, ...]
    separating_strikes: tuple[int, ...]


def event_ticker_of(root: str, event_date: date) -> str:
    month = MONTH_CODES[event_date.month - 1]
    return f"{root}-{event_date.year % 100:02d}{month}{event_date.day:02d}"


def event_day_rows(sidecar: CloseSidecar, event_ticker: str) -> tuple[MarketClose, ...]:
    rows = tuple(
        market for market in sidecar.markets.values() if market.event_ticker == event_ticker
    )
    if not rows:
        raise ValueError(f"{event_ticker} is not named in the {sidecar.root} close sidecar")
    return tuple(sorted(rows, key=lambda market: market.ticker))


# The edges are the boundaries between listed rows, not a union of strikes: the less row's yes-set
# ends one degree below its cap, and the greater row is unbounded above so it contributes nothing
# its neighbour's cap does not already carry.
def cell_edges(sidecar: CloseSidecar, event_ticker: str) -> tuple[int, ...]:
    edges: set[int] = set()
    for row in event_day_rows(sidecar, event_ticker):
        if row.strike_type == BETWEEN:
            edges.add(row.cap_strike)
        elif row.strike_type == LESS:
            edges.add(row.cap_strike - 1)
    return tuple(sorted(edges))


# The less row settles strictly below its cap and the greater row strictly above its floor. Under
# either non-strict form an event-day's rows double-cover the reading sitting on that boundary;
# under these they partition the whole line. Every match is collected rather than short-circuited
# so the partition is checked here instead of resting on where the tickers happen to sort.
def settling_row(sidecar: CloseSidecar, event_ticker: str, reading: Decimal) -> MarketClose:
    matched = [
        row
        for row in event_day_rows(sidecar, event_ticker)
        if (row.strike_type == BETWEEN and row.floor_strike <= reading <= row.cap_strike)
        or (row.strike_type == GREATER and reading > row.floor_strike)
        or (row.strike_type == LESS and reading < row.cap_strike)
    ]
    if len(matched) > 1:
        claimed = ", ".join(row.ticker for row in matched)
        raise ValueError(f"{event_ticker} settles at {reading} on more than one row: {claimed}")
    if not matched:
        raise ValueError(f"no {event_ticker} row settles at {reading}")
    return matched[0]


def straddle_of(
    sidecar: CloseSidecar,
    *,
    root: str,
    station: str,
    event_date: date,
    timezone: str,
    observed_f: Decimal,
    acis_f: Decimal,
) -> Straddle | None:
    edges = cell_edges(sidecar, event_ticker_of(root, event_date))
    separating = separating_strikes(observed_f, acis_f, edges)
    if not separating:
        return None
    return Straddle(
        root=root,
        station=station,
        event_date=event_date,
        extreme="max",
        timezone=timezone,
        observed_f=observed_f,
        acis_f=acis_f,
        cell_edges=edges,
        separating_strikes=separating,
    )


def city_event_days(records: Sequence[Straddle]) -> int:
    return len({(record.root, record.event_date) for record in records})
