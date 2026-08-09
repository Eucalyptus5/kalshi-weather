from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from itertools import pairwise

import numpy as np
import pyarrow as pa

from bot.lag.fee_floor import published_taker_fee
from bot.lag.ladder_consistency import PRICE_TICKS


# The smallest whole-cent move that clears a round trip at mid with a tick to spare:
# round_trip_fee_cents(1, 0.50) is 4. The bar is economic, never read off this tape.
MOVE_BAR_CENTS: Decimal = Decimal("5")
WINDOW_S: int = 600

# Written out rather than derived: most codes are the station in bot.main.STATIONS with its leading
# K stripped, CHI is not (its station is KMDW), IAH is not either (KXHIGHTHOU settles on KHOU,
# Hobby), and the root suffix repeats neither (DFW records as KXHIGHTDAL). A rule that got any of
# them wrong would corrupt every pair silently.
CITY_SERIES: Mapping[str, str] = {
    "DEN": "KXHIGHDEN",
    "OKC": "KXHIGHTOKC",
    "DFW": "KXHIGHTDAL",
    "AUS": "KXHIGHAUS",
    "SAT": "KXHIGHTSATX",
    "IAH": "KXHIGHTHOU",
    "MSY": "KXHIGHTNOLA",
    "ATL": "KXHIGHTATL",
    "MSP": "KXHIGHTMIN",
    "CHI": "KXHIGHCHI",
    "DCA": "KXHIGHTDC",
    "PHL": "KXHIGHPHIL",
    "NYC": "KXHIGHNY",
    "BOS": "KXHIGHTBOS",
    "PHX": "KXHIGHTPHX",
    "LAS": "KXHIGHTLV",
    "SFO": "KXHIGHTSFO",
    "LAX": "KXHIGHLAX",
}

CORRIDORS: Mapping[str, tuple[str, ...]] = {
    "gulf": ("DEN", "OKC", "DFW", "AUS", "SAT", "IAH", "MSY", "ATL"),
    "northeast": ("MSP", "CHI", "DCA", "PHL", "NYC", "BOS"),
    "southwest": ("PHX", "LAS"),
    "california": ("SFO", "LAX"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Pair:
    corridor: str
    upstream: str
    downstream: str


PAIRS: tuple[Pair, ...] = tuple(
    Pair(corridor=corridor, upstream=upstream, downstream=downstream)
    for corridor, chain in CORRIDORS.items()
    for upstream, downstream in pairwise(chain)
)

_TICKS_PER_CENT = PRICE_TICKS // 100
_DOUBLED_TICKS_PER_CENT = 2 * _TICKS_PER_CENT
_BAR2 = int(MOVE_BAR_CENTS * _DOUBLED_TICKS_PER_CENT)
_MICROS_PER_S = 1_000_000
_WINDOW_US = WINDOW_S * _MICROS_PER_S
_CENTS_PER_DOLLAR = Decimal(100)
_ROUND_TRIP_LEGS = Decimal(2)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROSECOND = timedelta(microseconds=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class AtmSeries:
    series: str
    event_date: date
    ticker: str
    received_us: np.ndarray
    mid2: np.ndarray
    two_sided: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class Crossing:
    anchor_us: int
    cross_us: int
    deadline_us: int
    direction: int
    move2: int


@dataclass(frozen=True, slots=True, kw_only=True)
class FollowerView:
    base_us: int
    base_mid2: int
    received_us: np.ndarray
    mid2: np.ndarray
    two_sided: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class Response:
    cross_us: int
    move2: int
    lead_s: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class Episode:
    corridor: str
    upstream: str
    downstream: str
    leader: str
    follower: str
    event_date: date
    direction: int
    anchor: datetime
    leader_cross: datetime
    follower_cross: datetime
    lead_s: Decimal
    leader_move_cents: Decimal
    follower_move_cents: Decimal

    def evidence_span(self) -> tuple[datetime, datetime]:
        return self.anchor, self.follower_cross


# Two taker legs charged once each on the aggregate, in cents per the whole fill.
def round_trip_fee_cents(contracts: Decimal, price: Decimal) -> Decimal:
    return _ROUND_TRIP_LEGS * _CENTS_PER_DOLLAR * published_taker_fee(contracts, price)


# The leg is chosen once per city event-day on its median distance from half, and never switches
# inside the day: a series that hops between legs manufactures moves that no book made.
def atm_series(
    series: str,
    event_date: date,
    table: pa.Table,
    *,
    window_start: datetime,
    window_end: datetime,
) -> AtmSeries | None:
    ids = table.column("id").to_numpy()
    if np.any(np.diff(ids) < 0):
        raise ValueError(f"{series} {event_date} touch rows are not in id order")

    tickers = table.column("ticker").to_pylist()
    legs = sorted(set(tickers))
    seats = {ticker: seat for seat, ticker in enumerate(legs)}
    leg = np.fromiter((seats[ticker] for ticker in tickers), dtype=np.int64, count=len(tickers))

    received_us = table.column("received_at").cast(pa.int64()).to_numpy()
    yes_bid = _tick_column(table, "yes_bid")
    no_bid = _tick_column(table, "no_bid")
    mid2 = yes_bid + PRICE_TICKS - no_bid
    # The pass stores yes_ask as 1 - no_bid, so an empty NO book reads back as a live-looking 1.00
    # at zero depth and a book empty on both sides reads as a mid of exactly 0.5 by construction.
    two_sided = (yes_bid > 0) & (no_bid > 0)

    inside = (received_us >= _micros(window_start)) & (received_us <= _micros(window_end))
    quoted = inside & two_sided

    picked = -1
    nearest = 0
    for seat in range(len(legs)):
        rows = np.flatnonzero(quoted & (leg == seat))
        if rows.size == 0:
            continue
        # An even row count puts the median on a half tick, so double again to keep int() exact.
        distance = int(2 * np.median(np.abs(mid2[rows] - PRICE_TICKS)))
        if picked < 0 or distance < nearest:
            picked, nearest = seat, distance
    if picked < 0:
        return None

    kept = np.flatnonzero(inside & (leg == picked))
    return AtmSeries(
        series=series,
        event_date=event_date,
        ticker=legs[picked],
        received_us=received_us[kept],
        mid2=mid2[kept],
        two_sided=two_sided[kept],
    )


# The next anchor sits strictly past the crossing, so definition windows never overlap and one
# sustained ramp emits one episode per crossing rather than one per state.
def leader_crossings(series: AtmSeries) -> list[Crossing]:
    states = np.flatnonzero(series.two_sided)
    stamps = series.received_us[states]
    mids = series.mid2[states]
    crossings: list[Crossing] = []
    anchor = 0
    while anchor < states.size:
        deadline = int(stamps[anchor]) + _WINDOW_US
        limit = int(np.searchsorted(stamps, deadline, side="right"))
        moves = mids[anchor + 1 : limit] - mids[anchor]
        hits = np.flatnonzero(np.abs(moves) >= _BAR2)
        if hits.size == 0:
            anchor += 1
            continue
        at = anchor + 1 + int(hits[0])
        move = int(moves[hits[0]])
        crossings.append(
            Crossing(
                anchor_us=int(stamps[anchor]),
                cross_us=int(stamps[at]),
                deadline_us=deadline,
                direction=1 if move > 0 else -1,
                move2=move,
            )
        )
        anchor = int(np.searchsorted(stamps, stamps[at], side="right"))
    return crossings


# The only read of the follower's full series. Everything downstream takes the view, which holds
# nothing at or before the crossing, so no later code can measure a response off the prices that
# defined the episode.
def follower_view(series: AtmSeries, crossing: Crossing) -> FollowerView | None:
    before = np.flatnonzero(series.two_sided & (series.received_us <= crossing.cross_us))
    after = np.flatnonzero(series.received_us > crossing.cross_us)
    if before.size == 0 or after.size == 0:
        return None
    base = int(before[-1])
    return FollowerView(
        base_us=int(series.received_us[base]),
        base_mid2=int(series.mid2[base]),
        received_us=series.received_us[after],
        mid2=series.mid2[after],
        two_sided=series.two_sided[after],
    )


def follower_response(view: FollowerView, crossing: Crossing) -> Response | None:
    rows = np.flatnonzero(view.two_sided & (view.received_us <= crossing.deadline_us))
    hits = np.flatnonzero((view.mid2[rows] - view.base_mid2) * crossing.direction >= _BAR2)
    if hits.size == 0:
        return None
    at = int(rows[hits[0]])
    return Response(
        cross_us=int(view.received_us[at]),
        move2=int(view.mid2[at]) - view.base_mid2,
        lead_s=Decimal(int(view.received_us[at]) - crossing.cross_us) / _MICROS_PER_S,
    )


def pair_episodes(
    pair: Pair, leader: AtmSeries, follower: AtmSeries, *, reverse: bool
) -> list[Episode]:
    if leader.event_date != follower.event_date:
        raise ValueError(
            f"{leader.series} and {follower.series} carry different event dates: "
            f"{leader.event_date} and {follower.event_date}"
        )
    lead_city, follow_city = (
        (pair.downstream, pair.upstream) if reverse else (pair.upstream, pair.downstream)
    )
    if leader.series != CITY_SERIES[lead_city] or follower.series != CITY_SERIES[follow_city]:
        raise ValueError(
            f"{lead_city} leading {follow_city} reads {CITY_SERIES[lead_city]} then "
            f"{CITY_SERIES[follow_city]}, got {leader.series} then {follower.series}"
        )

    episodes: list[Episode] = []
    for crossing in leader_crossings(leader):
        view = follower_view(follower, crossing)
        if view is None:
            continue
        response = follower_response(view, crossing)
        if response is None:
            continue
        episodes.append(
            Episode(
                corridor=pair.corridor,
                upstream=pair.upstream,
                downstream=pair.downstream,
                leader=lead_city,
                follower=follow_city,
                event_date=leader.event_date,
                direction=crossing.direction,
                anchor=_stamp(crossing.anchor_us),
                leader_cross=_stamp(crossing.cross_us),
                follower_cross=_stamp(response.cross_us),
                lead_s=response.lead_s,
                leader_move_cents=Decimal(crossing.move2) / _DOUBLED_TICKS_PER_CENT,
                follower_move_cents=Decimal(response.move2) / _DOUBLED_TICKS_PER_CENT,
            )
        )
    return episodes


def _micros(stamp: datetime) -> int:
    return (stamp - _EPOCH) // _MICROSECOND


def _stamp(micros: int) -> datetime:
    return _EPOCH + timedelta(microseconds=micros)


def _tick_column(table: pa.Table, name: str) -> np.ndarray:
    return np.array(
        [_ticks(Decimal(value)) for value in table.column(name).to_pylist()], dtype=np.int64
    )


def _ticks(price: Decimal) -> int:
    scaled = price * PRICE_TICKS
    ticks = int(scaled)
    if scaled != ticks:
        raise ValueError(f"price {price} is off the {PRICE_TICKS}-per-dollar grid")
    return ticks
