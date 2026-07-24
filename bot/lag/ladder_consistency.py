from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal

import numpy as np
import pyarrow as pa

from bot.lag.fee_floor import published_taker_fee
from bot.markets.parser import parse_ticker, resolve_event_kinds


LADDER_LEGS: int = 6
DEPTH_MIN: Decimal = Decimal("10")
EXCESS_BAR: Decimal = Decimal("2")
CITY_DAY_MIN_DISCOVERY: int = 30
CITY_DAY_MIN_HOLDOUT: int = 15
SPREAD_TICKS_PER_LEG: Decimal = Decimal("1")

SUM_BUY = "sum_buy"
SUM_SELL = "sum_sell"
MONOTONICITY = "monotonicity"

STREAMS: tuple[str, ...] = (SUM_BUY, SUM_SELL, MONOTONICITY)
STREAM_FAMILY: Mapping[str, str] = {SUM_BUY: "sum", SUM_SELL: "sum", MONOTONICITY: "monotonicity"}
STREAM_LEGS: Mapping[str, int] = {SUM_BUY: LADDER_LEGS, SUM_SELL: LADDER_LEGS, MONOTONICITY: 2}

# The artifact stores four decimals of dollar and two of contract, so the whole sweep runs in
# int64 and Decimal is kept for the money boundary.
PRICE_TICKS: int = 10_000
SIZE_UNITS: int = 100

_TICKS_PER_CENT = PRICE_TICKS // 100
_CENTS_PER_DOLLAR = Decimal(100)
_MICROS_PER_S = 1_000_000
_ZERO = Decimal(0)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Ladder:
    series: str
    event_date: date
    legs: tuple[str, ...]
    below: int
    above: int


@dataclass(frozen=True, slots=True, kw_only=True)
class LadderTape:
    ladder: Ladder
    received_us: np.ndarray
    yes_bid: np.ndarray
    yes_bid_depth: np.ndarray
    yes_ask: np.ndarray
    yes_ask_depth: np.ndarray
    live: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class Episode:
    series: str
    event_date: date
    stream: str
    family: str
    legs: int
    start: datetime
    end: datetime
    duration_s: Decimal
    magnitude_cents: Decimal
    signed_magnitude_cents: Decimal
    depth: Decimal
    fee_floor_cents: Decimal
    excess_cents: Decimal
    states: int
    tradeable: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class LadderResult:
    ladder: Ladder
    episodes: tuple[Episode, ...]
    rows: int
    censored: Mapping[str, int]
    incomplete_states: Mapping[str, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class _StreamStates:
    live: np.ndarray
    magnitude: np.ndarray
    depth: np.ndarray
    prices: np.ndarray


# One taker leg each rather than a round trip: both families enter as taker on every leg and carry
# the package to settlement, which charges no taker fee. The size is part of the read because
# published_taker_fee rounds the whole fill up to a cent once, so a floor taken at one contract is
# roughly twice the truth at ten.
def fee_floor_cents(*, contracts: Decimal, prices: Sequence[Decimal]) -> Decimal:
    if contracts <= _ZERO:
        raise ValueError(f"contracts must be positive, got {contracts}")
    if not prices:
        raise ValueError("a fee floor needs at least one leg price")
    fees = sum((published_taker_fee(contracts, price) for price in prices), _ZERO)
    return _CENTS_PER_DOLLAR * fees / contracts + len(prices) * SPREAD_TICKS_PER_LEG


def build_ladder(series: str, event_date: date, tickers: Sequence[str]) -> Ladder | None:
    if len(tickers) != LADDER_LEGS:
        return None
    parsed = []
    for ticker in tickers:
        try:
            leg = parse_ticker(ticker)
        except ValueError:
            return None
        if leg.series != series or leg.event_date != event_date:
            return None
        parsed.append(leg)
    # The tails tie with their neighbouring bracket on the first strike, and the full strike tuple
    # is what separates them: a one-strike tail sorts inside the bracket that shares its low.
    resolved = sorted(resolve_event_kinds(parsed), key=lambda leg: leg.strikes)
    kinds = [leg.kind for leg in resolved]
    if kinds.count("below") != 1 or kinds.count("above") != 1:
        return None
    return Ladder(
        series=series,
        event_date=event_date,
        legs=tuple(leg.raw for leg in resolved),
        below=kinds.index("below"),
        above=kinds.index("above"),
    )


def build_ladder_tape(ladder: Ladder, table: pa.Table) -> LadderTape:
    ids = table.column("id").to_numpy()
    if np.any(np.diff(ids) < 0):
        raise ValueError(f"{ladder.series} {ladder.event_date} touch rows are not in id order")
    seats = {ticker: seat for seat, ticker in enumerate(ladder.legs)}
    tickers = table.column("ticker").to_pylist()
    leg = np.empty(len(tickers), dtype=np.int64)
    for row, ticker in enumerate(tickers):
        if ticker not in seats:
            raise ValueError(f"{ticker} is not a leg of {ladder.series} {ladder.event_date}")
        leg[row] = seats[ticker]

    rows = table.num_rows
    order = np.arange(rows, dtype=np.int64)
    carried = np.empty((rows, LADDER_LEGS), dtype=np.int64)
    live = np.empty((rows, LADDER_LEGS), dtype=bool)
    for seat in range(LADDER_LEGS):
        last = np.where(leg == seat, order, -1)
        np.maximum.accumulate(last, out=last)
        live[:, seat] = last >= 0
        carried[:, seat] = np.maximum(last, 0)

    return LadderTape(
        ladder=ladder,
        received_us=table.column("received_at").cast(pa.int64()).to_numpy(),
        yes_bid=_units(table, tickers, "yes_bid", PRICE_TICKS)[carried],
        yes_bid_depth=_units(table, tickers, "yes_bid_depth", SIZE_UNITS)[carried],
        yes_ask=_units(table, tickers, "yes_ask", PRICE_TICKS)[carried],
        yes_ask_depth=_units(table, tickers, "yes_ask_depth", SIZE_UNITS)[carried],
        live=live,
    )


def ladder_episodes(tape: LadderTape, *, t_persist_s: Decimal) -> LadderResult:
    persist_us = int((t_persist_s * _MICROS_PER_S).to_integral_value(rounding=ROUND_CEILING))
    episodes: list[Episode] = []
    censored: dict[str, int] = {}
    incomplete: dict[str, int] = {}
    for stream in STREAMS:
        states = _stream_states(tape, stream)
        incomplete[stream] = int(np.count_nonzero(~states.live))
        index = np.flatnonzero(states.live)
        # An empty side stores a zero price against a zero size, so a leg with nothing resting is a
        # leg the package cannot take at all, whatever the stored prices sum to.
        holds = (states.magnitude[index] > 0) & (states.depth[index] > 0)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], holds.astype(np.int8), [0]))))
        dropped = 0
        for first, stop in zip(edges[0::2], edges[1::2], strict=True):
            if stop == holds.size:
                dropped += 1
                continue
            episodes.append(
                _episode(tape, states, stream, index[first:stop], int(index[stop]), persist_us)
            )
        censored[stream] = dropped
    return LadderResult(
        ladder=tape.ladder,
        episodes=tuple(episodes),
        rows=int(tape.received_us.size),
        censored=censored,
        incomplete_states=incomplete,
    )


def _stream_states(tape: LadderTape, stream: str) -> _StreamStates:
    if stream == SUM_BUY:
        return _StreamStates(
            live=tape.live.all(axis=1),
            magnitude=PRICE_TICKS - tape.yes_ask.sum(axis=1),
            depth=tape.yes_ask_depth.min(axis=1),
            prices=tape.yes_ask,
        )
    if stream == SUM_SELL:
        return _StreamStates(
            live=tape.live.all(axis=1),
            magnitude=tape.yes_bid.sum(axis=1) - PRICE_TICKS,
            depth=tape.yes_bid_depth.min(axis=1),
            prices=tape.yes_bid,
        )
    below = tape.yes_bid[:, tape.ladder.below]
    above = tape.yes_bid[:, tape.ladder.above]
    return _StreamStates(
        live=tape.live[:, tape.ladder.below] & tape.live[:, tape.ladder.above],
        magnitude=below + above - PRICE_TICKS,
        depth=np.minimum(
            tape.yes_bid_depth[:, tape.ladder.below], tape.yes_bid_depth[:, tape.ladder.above]
        ),
        # The lower nested claim is the NO side of the below tail, lifted at one minus its YES bid.
        prices=np.column_stack((PRICE_TICKS - below, above)),
    )


def _episode(
    tape: LadderTape,
    states: _StreamStates,
    stream: str,
    rows: np.ndarray,
    close: int,
    persist_us: int,
) -> Episode:
    legs = STREAM_LEGS[stream]
    magnitude = states.magnitude[rows]
    depth = states.depth[rows]
    duration_us = int(tape.received_us[close] - tape.received_us[rows[0]])
    scored = duration_us >= persist_us and int(magnitude.min()) > legs * _TICKS_PER_CENT
    at, floor, excess = _worst_state(magnitude, depth, states.prices[rows], exact=scored)
    cents = Decimal(int(magnitude[at])) / _TICKS_PER_CENT
    contracts = Decimal(int(depth[at])) / SIZE_UNITS
    return Episode(
        series=tape.ladder.series,
        event_date=tape.ladder.event_date,
        stream=stream,
        family=STREAM_FAMILY[stream],
        legs=legs,
        start=_stamp(int(tape.received_us[rows[0]])),
        end=_stamp(int(tape.received_us[close])),
        duration_s=Decimal(duration_us) / _MICROS_PER_S,
        magnitude_cents=cents,
        signed_magnitude_cents=cents if stream == SUM_BUY else -cents,
        depth=contracts,
        fee_floor_cents=floor,
        excess_cents=excess,
        states=int(rows.size),
        tradeable=duration_us >= persist_us and excess > _ZERO and contracts >= DEPTH_MIN,
    )


# The episode is summarised at its worst state rather than its first or its widest: excess is the
# minimum over the run of magnitude less the floor, and the magnitude, depth and floor reported are
# the ones at that state. An episode tradeable under this reading was tradeable at every instant it
# held, and the reading takes no threshold off the tape. A run that cannot clear the floor anywhere
# - shorter than t_persist, or never wider than the one tick per leg the floor already charges -
# skips the per-state fee arithmetic and reports its thinnest magnitude instead.
def _worst_state(
    magnitude: np.ndarray, depth: np.ndarray, prices: np.ndarray, *, exact: bool
) -> tuple[int, Decimal, Decimal]:
    if not exact:
        at = int(np.argmin(magnitude))
        floor = _floor_at(depth[at], prices[at])
        return at, floor, Decimal(int(magnitude[at])) / _TICKS_PER_CENT - floor
    at = 0
    worst_floor = _floor_at(depth[0], prices[0])
    worst = Decimal(int(magnitude[0])) / _TICKS_PER_CENT - worst_floor
    for position in range(1, magnitude.size):
        floor = _floor_at(depth[position], prices[position])
        excess = Decimal(int(magnitude[position])) / _TICKS_PER_CENT - floor
        if excess < worst:
            at, worst_floor, worst = position, floor, excess
    return at, worst_floor, worst


def _floor_at(depth: np.int64, prices: np.ndarray) -> Decimal:
    return fee_floor_cents(
        contracts=Decimal(int(depth)) / SIZE_UNITS,
        prices=[Decimal(int(price)) / PRICE_TICKS for price in prices],
    )


def _stamp(micros: int) -> datetime:
    return _EPOCH + timedelta(microseconds=micros)


def _units(table: pa.Table, tickers: Sequence[str], field: str, scale: int) -> np.ndarray:
    values = table.column(field).to_pylist()
    return np.array(
        [_unit(ticker, field, value, scale) for ticker, value in zip(tickers, values, strict=True)],
        dtype=np.int64,
    )


def _unit(ticker: str, field: str, value: str, scale: int) -> int:
    scaled = Decimal(value) * scale
    units = int(scaled)
    if scaled != units:
        raise ValueError(f"{ticker} {field}={value} is off the 1/{scale} grid")
    return units
