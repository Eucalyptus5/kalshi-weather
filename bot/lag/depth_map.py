from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from itertools import accumulate

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from bot.lag.ladder_consistency import PRICE_TICKS
from bot.lag.tape_studies import Intervals, RunScope
from bot.replay.artifacts import LADDER_DEPTH


SLIPPAGE_BAR_CENTS: Decimal = Decimal("1")
CUMULATIVE_TICKS: int = 5
REPLENISH_FRACTION: Decimal = Decimal("0.5")
REPLENISH_WINDOW_S: int = 60
FINAL_HOURS: int = 6
TOUCH_DEPTH_BAR: Decimal = Decimal("200")
BUCKET_HOURS: int = 6

YES_SIDE: int = 1
NO_SIDE: int = -1
NO_MATCH: int = 0

_TICKS_PER_CENT = PRICE_TICKS // 100
_MICROS_PER_HOUR = 3_600_000_000
_HOURS_PER_DAY = 24
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROSECOND = timedelta(microseconds=1)
_BEFORE_EVERYTHING = np.iinfo(np.int64).min
_FOLD_PAIRS = 1_000_000
_AGGRESSOR = {"yes": "no", "no": "yes"}


@dataclass(frozen=True, slots=True)
class WalkCapacity:
    capacity: np.ndarray
    censored: np.ndarray
    exhausted: np.ndarray


@dataclass(frozen=True, slots=True)
class ScreenedCells:
    kept: np.ndarray
    out_of_window: int
    excluded: int
    by_class: dict[str, int]


def scaled_units(column: pa.Array, decimals: int) -> np.ndarray:
    points = pc.count_substring(column, pattern=".")
    tail = pc.subtract(pc.utf8_length(column), pc.add(pc.find_substring(column, pattern="."), 1))
    off_grid = pc.or_(pc.not_equal(points, 1), pc.not_equal(tail, decimals))
    if pc.any(off_grid).as_py():
        offender = column.filter(off_grid)[0].as_py()
        raise ValueError(f"{offender!r} is not a decimal quantised to {decimals} places")
    # Stripping the point rather than scaling a parsed value is what keeps this exact: a float
    # round trip does not land every stored price back on its own grid.
    stripped = pc.replace_substring(column, pattern=".", replacement="")
    return np.asarray(pc.cast(stripped, pa.int64()))


def level_units(column: pa.ListArray, decimals: int, width: int) -> np.ndarray:
    offsets = np.asarray(column.offsets, dtype=np.int64)
    counts = np.diff(offsets)
    widest = int(counts.max(initial=0))
    if widest > width:
        raise ValueError(f"a row carries {widest} levels, past the stored width of {width}")
    values = scaled_units(column.flatten(), decimals)
    dense = np.zeros((counts.size, width), dtype=np.int64)
    rows = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    levels = np.arange(values.size, dtype=np.int64) - np.repeat(offsets[:-1] - offsets[0], counts)
    dense[rows, levels] = values
    return dense


# The artifact bounds the stored ladder at six levels and prices sit on a one-cent grid, so the
# top six always span five cents: this figure is never censored by that bound.
def cumulative_depth(prices: np.ndarray, sizes: np.ndarray, ticks: int) -> np.ndarray:
    band = ticks * _TICKS_PER_CENT
    return np.sum(np.where(prices >= prices[:, :1] - band, sizes, 0), axis=1)


def walk_capacity(
    prices: np.ndarray, sizes: np.ndarray, levels: np.ndarray, bar_units: int
) -> WalkCapacity:
    width = prices.shape[1]
    distance = prices[:, :1] - prices
    quantity = np.cumsum(sizes, axis=1)
    cost = np.cumsum(distance * sizes, axis=1)
    crosses = cost > bar_units * quantity
    crossed = crosses.any(axis=1)
    at = np.argmax(crosses, axis=1)
    rows = np.arange(prices.shape[0])
    # The best level costs nothing, so a crossing never lands on it and the level under one is
    # always real. Its own distance carries the crossing, which holds that distance strictly above
    # the bar and the denominator strictly above zero. Off a crossing the term is discarded.
    below = at - 1
    step = distance[rows, at]
    reachable = (step * quantity[rows, below] - cost[rows, below]) // (step - bar_units)
    return WalkCapacity(
        capacity=np.where(crossed, reachable, quantity[:, -1]),
        censored=~crossed & (levels > width),
        exhausted=~crossed & (levels <= width),
    )


# The stored price and size lists are cut to LADDER_DEPTH but the level count is written before
# the cut: a book that is genuinely six deep rests a true zero at every price it does not name.
def depth_at_price(row: Mapping[str, object], side: str, price: Decimal) -> tuple[Decimal, bool]:
    censored = row[f"{side}_levels"] > LADDER_DEPTH
    for level, size in zip(row[f"{side}_prices"], row[f"{side}_sizes"], strict=True):
        if Decimal(level) == price:
            return Decimal(size), censored
    return Decimal("0"), censored


# taker_side names the side the aggressor bought, and a YES buy lifts the yes ask, which is stored
# as the complement of the NO bid: it consumes resting NO. Price cannot route on its own, since
# no_price is written as the complement of yes_price on the same row and so matches both quotes.
def volume_at_price(
    prints: pa.Table,
    ticker: str,
    resting_side: str,
    price: Decimal,
    start: datetime,
    end: datetime,
) -> Decimal:
    received = prints.column("received_at")
    inside = pc.and_(pc.greater_equal(received, start), pc.less_equal(received, end))
    quote = pc.and_(
        pc.equal(prints.column("ticker"), ticker),
        pc.equal(prints.column("taker_side"), _AGGRESSOR[resting_side]),
    )
    taken = prints.filter(pc.and_(inside, quote))
    return sum(
        (
            Decimal(count)
            for count, level in zip(
                taken.column("count").to_pylist(),
                taken.column(f"{resting_side}_price").to_pylist(),
                strict=True,
            )
            if Decimal(level) == price
        ),
        Decimal("0"),
    )


class WeightedTally:
    def __init__(self) -> None:
        self._packed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._buffered: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._pending = 0

    def add(self, groups: np.ndarray, values: np.ndarray, weights: np.ndarray) -> None:
        self._buffered.append((groups, values, weights))
        self._pending += values.size
        if self._pending >= _FOLD_PAIRS:
            self._fold()

    # A quantile over several groups at once is not a function of their quantiles, so pooling has
    # to reach the bins themselves; weighted_quantile takes what this returns.
    def pooled(self, groups: Iterable[int]) -> dict[int, int]:
        self._fold()
        merged: dict[int, int] = {}
        for group in groups:
            packed = self._packed.get(group)
            if packed is None:
                continue
            for value, weight in zip(packed[0].tolist(), packed[1].tolist(), strict=True):
                merged[value] = merged.get(value, 0) + weight
        return merged

    def groups(self) -> list[int]:
        self._fold()
        return sorted(self._packed)

    def entries(self) -> int:
        self._fold()
        return sum(values.size for values, _ in self._packed.values())

    def _fold(self) -> None:
        if self._pending == 0:
            return
        chunks = self._buffered
        self._buffered = []
        self._pending = 0
        touched: set[int] = set()
        for chunk in chunks:
            touched.update(np.unique(chunk[0]).tolist())
        # A group already packed rejoins the stream: a value seen again many batches later has to
        # land back in the bin it already holds rather than opening a second one beside it.
        for group in touched & self._packed.keys():
            packed = self._packed.pop(group)
            chunks.append((np.full(packed[0].size, group, dtype=np.int64), packed[0], packed[1]))

        groups = np.concatenate([chunk[0] for chunk in chunks])
        values = np.concatenate([chunk[1] for chunk in chunks])
        weights = np.concatenate([chunk[2] for chunk in chunks])
        order = np.lexsort((values, groups))
        groups, values, weights = groups[order], values[order], weights[order]
        opens = np.ones(groups.size, dtype=bool)
        opens[1:] = (groups[1:] != groups[:-1]) | (values[1:] != values[:-1])
        starts = np.flatnonzero(opens)
        keys, bins, sums = groups[starts], values[starts], np.add.reduceat(weights, starts)
        edges = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        # Copied rather than sliced: a view would pin the whole folded stream behind one group.
        for part_keys, part_values, part_weights in zip(
            np.split(keys, edges), np.split(bins, edges), np.split(sums, edges), strict=True
        ):
            self._packed[int(part_keys[0])] = (part_values.copy(), part_weights.copy())


def weighted_quantile(bins: Mapping[int, int], numerator: int, denominator: int) -> int | None:
    values = sorted(bins)
    if not values:
        return None
    reached = list(accumulate(bins[value] for value in values))
    return values[bisect_left([seen * denominator for seen in reached], numerator * reached[-1])]


def hour_of_day(received_us: np.ndarray) -> np.ndarray:
    return received_us // _MICROS_PER_HOUR % _HOURS_PER_DAY


def hours_to_close_bucket(
    close_us: np.ndarray, received_us: np.ndarray, bucket_hours: int
) -> np.ndarray:
    return (close_us - received_us) // (bucket_hours * _MICROS_PER_HOUR)


# One cell per book state, split again wherever a reporting boundary fell inside a state that
# outlived it. The caller prepends the carried state, so the partition does not move with the
# file the artifact happened to chunk a state into.
def cell_edges(times: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    return np.union1d(times, breaks[(breaks > times[0]) & (breaks < times[-1])])


def screen_cells(
    scope: RunScope, series: str, event_date: date, starts: np.ndarray, ends: np.ndarray
) -> ScreenedCells:
    day = scope.event_days.get((series, event_date))
    if day is None:
        raise ValueError(f"{series} {event_date.isoformat()} is not in the frozen scope")
    inside = (starts >= _micros(day.window_start)) & (ends <= _micros(day.window_end))
    excluded = inside & _intersects(scope.merged, starts, ends)
    return ScreenedCells(
        kept=inside & ~excluded,
        out_of_window=int(np.count_nonzero(~inside)),
        excluded=int(np.count_nonzero(excluded)),
        # Classes overlap in time, so the per-class counts can sum above the dropped count.
        by_class={
            name: int(np.count_nonzero(excluded & _intersects(intervals, starts, ends)))
            for name, intervals in scope.by_class.items()
        },
    )


# The leg picker bot.lag.lead_lag.atm_series carries, on integer arrays rather than on a parquet
# table of strings: that one converts prices in a per-element Decimal loop and materialises every
# column through to_pylist(), which neither finishes nor fits in 3 GB over 239 million rows. The
# caller hands over the rows that already quoted two-sided inside the event-day window.
def atm_leg(legs: Sequence[str], seats: np.ndarray, mid2: np.ndarray) -> int | None:
    distance = np.abs(mid2 - PRICE_TICKS)
    picked = -1
    nearest = 0
    for seat in range(len(legs)):
        rows = np.flatnonzero(seats == seat)
        if rows.size == 0:
            continue
        # An even row count puts the median on a half tick, so double again to keep int() exact.
        median = int(2 * np.median(distance[rows]))
        if picked < 0 or median < nearest:
            picked, nearest = seat, median
    return None if picked < 0 else picked


# Read off the price rather than off taker_side, which leaves the measurement clear of the
# taker_side / taker_outcome_side wire question. Matching both sides at once would need
# yes_bid + no_bid == PRICE_TICKS, a locked book, so the YES arm can take the tie.
def classify_print(trade_price: int, yes_bid: int, no_bid: int) -> int:
    if trade_price == yes_bid:
        return YES_SIDE
    if PRICE_TICKS - trade_price == no_bid:
        return NO_SIDE
    return NO_MATCH


# Index zero is the sentinel for a cell no interval opens at or before, which keeps the class with
# no exclusions on it from needing its own arm.
def _intersects(intervals: Intervals, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    opens = np.fromiter(
        (_micros(stamp) for stamp in intervals.starts), dtype=np.int64, count=len(intervals.starts)
    )
    closes = np.fromiter(
        (_micros(stamp) for stamp in intervals.ends), dtype=np.int64, count=len(intervals.ends)
    )
    reach = np.concatenate(([_BEFORE_EVERYTHING], closes))
    return reach[np.searchsorted(opens, ends, side="right")] >= starts


def _micros(stamp: datetime) -> int:
    return (stamp - _EPOCH) // _MICROSECOND
