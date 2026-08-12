import logging
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bot.lag.depth_map import (
    BUCKET_HOURS,
    CUMULATIVE_TICKS,
    FINAL_HOURS,
    NO_MATCH,
    REPLENISH_FRACTION,
    REPLENISH_WINDOW_S,
    SLIPPAGE_BAR_CENTS,
    TOUCH_DEPTH_BAR,
    YES_SIDE,
    WeightedTally,
    atm_leg,
    cell_edges,
    classify_print,
    cumulative_depth,
    hour_of_day,
    hours_to_close_bucket,
    level_units,
    scaled_units,
    screen_cells,
    walk_capacity,
    weighted_quantile,
)
from bot.lag.fee_floor import MAKER_RATE_SOURCE, PUBLISHED_MAKER_RATE
from bot.lag.ladder_consistency import PRICE_TICKS, SIZE_UNITS
from bot.lag.read_rtt import FloorSource, LatencyFloor
from bot.lag.run_manifest import MANIFEST_NAME, write_manifest
from bot.lag.tape_studies import (
    KIND_SCHEMAS,
    LADDER,
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    TRADES,
    RunScope,
    assemble_run_inputs,
    load_run_scope,
    partition_files,
    window_dates,
)
from bot.markets.parser import parse_ticker
from bot.replay.artifacts import LADDER_DEPTH


logger = logging.getLogger(__name__)

RESULTS_NAME = "results.json"
ALL_LEGS = "all_legs"
ATM = "atm"
UNIVERSES = (ALL_LEGS, ATM)
YES = "yes"
NO = "no"
SIDES = (YES, NO)
CONFIRMED = "confirmed"
REFUTED = "refuted"

MAX_BUCKETS = 8
HOURS_PER_DAY = 24
PRICE_DECIMALS = 4
SIZE_DECIMALS = 2
SLIPPAGE_BAR_UNITS = int(SLIPPAGE_BAR_CENTS * PRICE_TICKS // 100)
TICKS_PER_CENT = PRICE_TICKS // 100
REPLENISH_NUMERATOR, REPLENISH_DENOMINATOR = REPLENISH_FRACTION.as_integer_ratio()
MICROS_PER_S = 1_000_000
MICROS_PER_HOUR = 3_600 * MICROS_PER_S
FINAL_BUCKETS = tuple(range(FINAL_HOURS // BUCKET_HOURS))

LADDER_COLUMNS = (
    "ticker",
    "received_at",
    "yes_prices",
    "yes_sizes",
    "yes_levels",
    "no_prices",
    "no_sizes",
    "no_levels",
)
TOUCH_COLUMNS = ("ticker", "received_at", "yes_bid", "yes_bid_depth", "no_bid", "no_bid_depth")
TRADE_COLUMNS = ("ticker", "received_at", "yes_price")
SIDED_TALLIES = ("touch", "cum5", "capacity", "capacity_censored")
CLOSE_TIMES_QUERY = "SELECT ticker, close_time FROM markets WHERE series = ? AND event_date = ?"

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROSECOND = timedelta(microseconds=1)
_NO_QUOTES = np.empty(0, dtype=np.int32)


@dataclass(frozen=True, slots=True, kw_only=True)
class Book:
    times: np.ndarray
    touch: tuple[np.ndarray, np.ndarray]
    cum5: tuple[np.ndarray, np.ndarray]
    capacity: tuple[np.ndarray, np.ndarray]
    censored: tuple[np.ndarray, np.ndarray]
    two_sided: np.ndarray
    spread: np.ndarray


@dataclass(slots=True)
class CellTally:
    touch: WeightedTally = field(default_factory=WeightedTally)
    cum5: WeightedTally = field(default_factory=WeightedTally)
    capacity: WeightedTally = field(default_factory=WeightedTally)
    capacity_censored: WeightedTally = field(default_factory=WeightedTally)
    spread: WeightedTally = field(default_factory=WeightedTally)


@dataclass(slots=True)
class ScreenTally:
    candidates: int = 0
    out_of_window: int = 0
    past_close: int = 0
    excluded: int = 0
    kept_us: int = 0
    nominal_us: int = 0
    by_class: dict[str, int] = field(default_factory=dict)

    @property
    def excluded_fraction(self) -> Decimal | None:
        if self.candidates == 0:
            return None
        return Decimal(self.excluded) / Decimal(self.candidates)

    @property
    def retained_fraction(self) -> Decimal | None:
        if self.nominal_us == 0:
            return None
        return Decimal(self.kept_us) / Decimal(self.nominal_us)


@dataclass(slots=True)
class PrintTally:
    matched: int = 0
    events: int = 0
    replenished: int = 0
    times_us: list[int] = field(default_factory=list)


@dataclass(slots=True)
class PrintAccounting:
    prints: int = 0
    no_pre_state: int = 0
    no_match: int = 0
    no_post_state: int = 0
    past_close: int = 0
    out_of_window: int = 0
    excluded: int = 0
    by_class: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class Resilience:
    accounting: PrintAccounting = field(default_factory=PrintAccounting)
    tallies: dict[tuple[str, int], PrintTally] = field(default_factory=dict)

    def tally(self, series: str, bucket: int) -> PrintTally:
        return self.tallies.setdefault((series, bucket), PrintTally())


@dataclass(frozen=True, slots=True, kw_only=True)
class ResilienceSweep:
    universes: Mapping[str, Resilience]
    states: int
    trades: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DepthSweep:
    tallies: Mapping[str, CellTally]
    screened: ScreenTally
    picks: Mapping[tuple[str, date], str]
    no_atm_leg: tuple[tuple[str, date], ...]
    cities: tuple[str, ...]
    rows: int
    cells: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Quartiles:
    p25: Decimal | None
    p50: Decimal | None
    p75: Decimal | None


@dataclass(frozen=True, slots=True, kw_only=True)
class CellReadout:
    kept_s: Decimal
    two_sided_fraction: Decimal | None
    spread_p50_cents: Decimal | None
    spread_min_cents: Decimal | None
    touch: Quartiles
    cum5: Quartiles
    capacity: Quartiles
    censored_s: Decimal
    censored_p50: Decimal | None


@dataclass(frozen=True, slots=True, kw_only=True)
class UniverseReadout:
    cube: Mapping[tuple[str, int, int, str], CellReadout]
    by_city: Mapping[tuple[str, str], CellReadout]
    by_bucket: Mapping[tuple[int, str], CellReadout]
    by_hour: Mapping[tuple[int, str], CellReadout]


@dataclass(frozen=True, slots=True, kw_only=True)
class Headline:
    universe: str
    median_touch: Mapping[str, Decimal | None]
    resilience: PrintTally


@dataclass(frozen=True, slots=True, kw_only=True)
class DepthMapRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    floor: LatencyFloor
    scope_start: datetime
    scope_end: datetime
    event_days: tuple[date, ...]
    sweep: DepthSweep
    readouts: Mapping[str, UniverseReadout]
    resilience: ResilienceSweep
    headline: Headline
    verdict: str


def load_close_times(db: Path, scope: RunScope) -> dict[str, datetime]:
    conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    closes: dict[str, datetime] = {}
    try:
        for series, event_date in sorted(scope.event_days):
            for ticker, close_time in conn.execute(
                CLOSE_TIMES_QUERY, (series, event_date.isoformat())
            ):
                if close_time is None:
                    raise ValueError(f"{ticker} carries no close time")
                closes[ticker] = datetime.fromisoformat(close_time + "+00:00")
    finally:
        conn.close()
    return closes


@dataclass(frozen=True, slots=True, kw_only=True)
class _Plan:
    series: str
    event_date: date
    seat: int
    close_us: int
    day_start_us: int
    day_end_us: int
    end_us: int
    breaks: np.ndarray


@dataclass(slots=True)
class _Root:
    series: str
    city: int
    tally: CellTally = field(default_factory=CellTally)
    plans: dict[str, _Plan] = field(default_factory=dict)
    outside: set[str] = field(default_factory=set)
    carries: dict[str, Book] = field(default_factory=dict)
    quotes: dict[str, list[np.ndarray]] = field(default_factory=dict)
    rows: int = 0
    cells: int = 0


def sweep_depth(scope: RunScope, artifacts: Path, closes: Mapping[str, datetime]) -> DepthSweep:
    cities = tuple(sorted({series for series, _ in scope.event_days}))
    dates = window_dates(scope.scope_start, scope.scope_end)
    tallies = {name: CellTally() for name in UNIVERSES}
    screened = ScreenTally()
    picks: dict[tuple[str, date], str] = {}
    silent: list[tuple[str, date]] = []
    rows = 0
    cells = 0

    for city, series in enumerate(cities):
        started = time.monotonic()
        root = _Root(series=series, city=city)
        for path in partition_files(artifacts, LADDER, series, dates):
            _sweep_file(
                root,
                _checked(path, LADDER).read(columns=list(LADDER_COLUMNS)),
                scope,
                closes,
                screened,
            )
        _close_carries(root, scope, screened)
        picked = _picks(root, scope)
        for event_date, ticker in sorted(picked.items()):
            if ticker is None:
                silent.append((series, event_date))
            else:
                picks[(series, event_date)] = ticker
        _fold_root(
            root,
            tallies,
            frozenset(root.plans[ticker].seat for ticker in picked.values() if ticker is not None),
        )
        rows += root.rows
        cells += root.cells
        logger.info(
            "depth_map root=%s rows=%d cells=%d legs=%d entries=%d elapsed_s=%.1f",
            series,
            root.rows,
            root.cells,
            len(root.plans),
            sum(
                getattr(tally, name).entries()
                for tally in tallies.values()
                for name in (*SIDED_TALLIES, "spread")
            ),
            time.monotonic() - started,
        )

    return DepthSweep(
        tallies=tallies,
        screened=screened,
        picks=picks,
        no_atm_leg=tuple(silent),
        cities=cities,
        rows=rows,
        cells=cells,
    )


def _sweep_file(
    root: _Root,
    table: pa.Table,
    scope: RunScope,
    closes: Mapping[str, datetime],
    screened: ScreenTally,
) -> None:
    root.rows += table.num_rows
    if table.num_rows == 0:
        return

    names = table.column("ticker").combine_chunks().dictionary_encode()
    seat_of_row = np.asarray(names.indices)
    times = np.asarray(table.column("received_at").combine_chunks().cast(pa.int64()))
    yes_prices = _levels(table, "yes_prices", PRICE_DECIMALS)
    yes_sizes = _levels(table, "yes_sizes", SIZE_DECIMALS)
    no_prices = _levels(table, "no_prices", PRICE_DECIMALS)
    no_sizes = _levels(table, "no_sizes", SIZE_DECIMALS)
    yes_levels = np.asarray(table.column("yes_levels").combine_chunks())
    no_levels = np.asarray(table.column("no_levels").combine_chunks())

    for index, ticker in enumerate(names.dictionary.to_pylist()):
        if ticker in root.outside:
            continue
        plan = root.plans.get(ticker)
        if plan is None:
            plan = _plan(ticker, scope, closes, len(root.plans))
            if plan is None:
                root.outside.add(ticker)
                continue
            root.plans[ticker] = plan

        rows = np.flatnonzero(seat_of_row == index)
        stamps = times[rows]
        quoted = (
            (yes_sizes[rows, 0] > 0)
            & (no_sizes[rows, 0] > 0)
            & (stamps >= plan.day_start_us)
            & (stamps <= plan.day_end_us)
        )
        if quoted.any():
            mid2 = yes_prices[rows, 0] + PRICE_TICKS - no_prices[rows, 0]
            root.quotes.setdefault(ticker, []).append(mid2[quoted].astype(np.int32))

        held = rows[_last_at_each(stamps)]
        book = _join(
            root.carries.get(ticker),
            _book(
                times[held],
                (yes_prices[held], no_prices[held]),
                (yes_sizes[held], no_sizes[held]),
                (yes_levels[held], no_levels[held]),
            ),
        )
        # The carry is in front, so this covers the file boundary as well as the file itself, and
        # keeping the last row at each timestamp cannot hide an inversion.
        if np.any(np.diff(book.times) < 0):
            raise ValueError(f"{ticker} ladder rows are not in received_at order")
        _measure(root, plan, book, cell_edges(book.times, plan.breaks), scope, screened)
        root.carries[ticker] = _tail(book)


def _measure(
    root: _Root,
    plan: _Plan,
    book: Book,
    edges: np.ndarray,
    scope: RunScope,
    screened: ScreenTally,
) -> None:
    starts, ends = edges[:-1], edges[1:]
    if starts.size == 0:
        return

    durations = ends - starts
    cells = screen_cells(scope, plan.series, plan.event_date, starts, ends)
    past = cells.kept & (ends > plan.close_us)
    keep = cells.kept & ~past
    inside = (starts >= plan.day_start_us) & (ends <= plan.end_us)

    screened.candidates += int(starts.size)
    screened.out_of_window += cells.out_of_window
    screened.excluded += cells.excluded
    screened.past_close += int(np.count_nonzero(past))
    screened.nominal_us += int(durations[inside].sum())
    screened.kept_us += int(durations[keep].sum())
    for name, count in cells.by_class.items():
        screened.by_class[name] = screened.by_class.get(name, 0) + count
    if not keep.any():
        return

    at = (np.searchsorted(book.times, starts, side="right") - 1)[keep]
    starts, durations = starts[keep], durations[keep]
    bucket = _bucket_of(np.full(starts.shape, plan.close_us), starts)
    if int(bucket.max()) >= MAX_BUCKETS:
        raise ValueError(
            f"{plan.series} {plan.event_date.isoformat()} closes more than "
            f"{MAX_BUCKETS * BUCKET_HOURS} hours after its window opens"
        )

    key = _key(plan.seat, bucket, hour_of_day(starts))
    for side in range(len(SIDES)):
        sided = key * 2 + side
        capacity = book.capacity[side][at]
        root.tally.touch.add(sided, book.touch[side][at], durations)
        root.tally.cum5.add(sided, book.cum5[side][at], durations)
        root.tally.capacity.add(sided, capacity, durations)
        censored = book.censored[side][at]
        if censored.any():
            root.tally.capacity_censored.add(
                sided[censored], capacity[censored], durations[censored]
            )
    two_sided = book.two_sided[at]
    if two_sided.any():
        root.tally.spread.add(key[two_sided], book.spread[at][two_sided], durations[two_sided])
    root.cells += int(np.count_nonzero(keep))


def _close_carries(root: _Root, scope: RunScope, screened: ScreenTally) -> None:
    for ticker, carry in root.carries.items():
        plan = root.plans[ticker]
        if int(carry.times[0]) >= plan.end_us:
            continue
        edges = cell_edges(np.append(carry.times, plan.end_us), plan.breaks)
        _measure(root, plan, carry, edges, scope, screened)


def _picks(root: _Root, scope: RunScope) -> dict[date, str | None]:
    by_day: dict[date, list[str]] = {
        event_date: [] for series, event_date in scope.event_days if series == root.series
    }
    for ticker, plan in root.plans.items():
        by_day[plan.event_date].append(ticker)

    picked: dict[date, str | None] = {}
    for event_date, tickers in by_day.items():
        legs = sorted(tickers)
        quotes = [
            np.concatenate(root.quotes[leg]) if leg in root.quotes else _NO_QUOTES for leg in legs
        ]
        mid2 = np.concatenate(quotes) if quotes else _NO_QUOTES
        seats = np.repeat(np.arange(len(legs), dtype=np.int64), [chunk.size for chunk in quotes])
        seat = atm_leg(legs, seats, mid2)
        picked[event_date] = None if seat is None else legs[seat]
    return picked


def _fold_root(root: _Root, tallies: Mapping[str, CellTally], atm: frozenset[int]) -> None:
    for name in SIDED_TALLIES:
        source: WeightedTally = getattr(root.tally, name)
        for group in source.groups():
            key, side = divmod(group, 2)
            seat, bucket, hour = _decode(key)
            targets = [getattr(tallies[ALL_LEGS], name)]
            if seat in atm:
                targets.append(getattr(tallies[ATM], name))
            _transfer(source, group, _key(root.city, bucket, hour) * 2 + side, targets)

    for group in root.tally.spread.groups():
        seat, bucket, hour = _decode(group)
        targets = [tallies[ALL_LEGS].spread]
        if seat in atm:
            targets.append(tallies[ATM].spread)
        _transfer(root.tally.spread, group, _key(root.city, bucket, hour), targets)


def _transfer(
    source: WeightedTally, group: int, target: int, into: Sequence[WeightedTally]
) -> None:
    bins = source.pooled((group,))
    values = np.fromiter(bins, dtype=np.int64, count=len(bins))
    weights = np.fromiter(bins.values(), dtype=np.int64, count=len(bins))
    groups = np.full(values.shape, target, dtype=np.int64)
    for tally in into:
        tally.add(groups, values, weights)


def sweep_resilience(
    scope: RunScope,
    artifacts: Path,
    closes: Mapping[str, datetime],
    picks: Mapping[tuple[str, date], str],
) -> ResilienceSweep:
    universes = {name: Resilience() for name in UNIVERSES}
    states = 0
    trades = 0

    for (series, event_date), day in sorted(scope.event_days.items()):
        dates = window_dates(day.window_start, day.window_end)
        book = _read_legs(artifacts, LADDER, series, dates, event_date, TOUCH_COLUMNS)
        tape = _read_legs(artifacts, TRADES, series, dates, event_date, TRADE_COLUMNS)
        states += book.num_rows
        trades += tape.num_rows
        if book.num_rows == 0 or tape.num_rows == 0:
            continue

        legs = book.column("ticker").combine_chunks().dictionary_encode()
        seat_of_state = np.asarray(legs.indices)
        state_times = np.asarray(book.column("received_at").combine_chunks().cast(pa.int64()))
        yes_bid = _units(book, "yes_bid", PRICE_DECIMALS)
        no_bid = _units(book, "no_bid", PRICE_DECIMALS)
        depths = (
            _units(book, "yes_bid_depth", SIZE_DECIMALS),
            _units(book, "no_bid_depth", SIZE_DECIMALS),
        )

        printed = tape.column("ticker").combine_chunks().dictionary_encode()
        seat_of_trade = np.asarray(printed.indices)
        trade_times = np.asarray(tape.column("received_at").combine_chunks().cast(pa.int64()))
        trade_price = _units(tape, "yes_price", PRICE_DECIMALS)
        seats = {name: seat for seat, name in enumerate(printed.dictionary.to_pylist())}

        for index, ticker in enumerate(legs.dictionary.to_pylist()):
            if ticker not in seats:
                continue
            states_of = np.flatnonzero(seat_of_state == index)
            trades_of = np.flatnonzero(seat_of_trade == seats[ticker])
            stamps = state_times[states_of]
            if np.any(np.diff(stamps) < 0):
                raise ValueError(f"{ticker} ladder rows are not in received_at order")
            if np.any(np.diff(trade_times[trades_of]) < 0):
                raise ValueError(f"{ticker} trade rows are not in received_at order")
            held = states_of[_last_at_each(stamps)]
            targets = [universes[ALL_LEGS]]
            if picks.get((series, event_date)) == ticker:
                targets.append(universes[ATM])
            _replenishment(
                scope,
                series,
                event_date,
                close_us=_close_us(closes, ticker),
                times=state_times[held],
                quotes=(yes_bid[held], no_bid[held]),
                depths=(depths[0][held], depths[1][held]),
                trade_times=trade_times[trades_of],
                trade_price=trade_price[trades_of],
                targets=targets,
            )

    return ResilienceSweep(universes=universes, states=states, trades=trades)


def _replenishment(
    scope: RunScope,
    series: str,
    event_date: date,
    *,
    close_us: int,
    times: np.ndarray,
    quotes: tuple[np.ndarray, np.ndarray],
    depths: tuple[np.ndarray, np.ndarray],
    trade_times: np.ndarray,
    trade_price: np.ndarray,
    targets: Sequence[Resilience],
) -> None:
    pre = np.searchsorted(times, trade_times, side="right") - 1
    opens = np.concatenate(([0], np.flatnonzero(np.diff(pre)) + 1))
    closes = np.append(opens[1:], pre.size)

    for first, stop in zip(opens, closes, strict=True):
        for target in targets:
            target.accounting.prints += 1
        state = int(pre[first])
        if state < 0:
            for target in targets:
                target.accounting.no_pre_state += 1
            continue

        side = NO_MATCH
        matched = -1
        for row in range(first, stop):
            side = classify_print(
                int(trade_price[row]), int(quotes[0][state]), int(quotes[1][state])
            )
            if side != NO_MATCH:
                matched = row
                break
        if matched < 0:
            for target in targets:
                target.accounting.no_match += 1
            continue

        post = state + 1
        if post >= times.size:
            for target in targets:
                target.accounting.no_post_state += 1
            continue

        printed_at = int(trade_times[matched])
        deadline = printed_at + REPLENISH_WINDOW_S * MICROS_PER_S
        if deadline > close_us:
            for target in targets:
                target.accounting.past_close += 1
            continue

        window = screen_cells(
            scope,
            series,
            event_date,
            np.array([times[state]], dtype=np.int64),
            np.array([deadline], dtype=np.int64),
        )
        if not bool(window.kept[0]):
            for target in targets:
                target.accounting.out_of_window += window.out_of_window
                target.accounting.excluded += window.excluded
                for name, count in window.by_class.items():
                    target.accounting.by_class[name] = (
                        target.accounting.by_class.get(name, 0) + count
                    )
            continue

        bucket = int(
            _bucket_of(
                np.array([close_us], dtype=np.int64), np.array([printed_at], dtype=np.int64)
            )[0]
        )
        if bucket >= MAX_BUCKETS:
            raise ValueError(
                f"{series} {event_date.isoformat()} prints more than "
                f"{MAX_BUCKETS * BUCKET_HOURS} hours before it closes"
            )

        depth = depths[0] if side == YES_SIDE else depths[1]
        bar = REPLENISH_NUMERATOR * int(depth[state])
        tallies = [target.tally(series, bucket) for target in targets]
        for tally in tallies:
            tally.matched += 1
        if REPLENISH_DENOMINATOR * int(depth[post]) >= bar:
            continue

        limit = int(np.searchsorted(times, deadline, side="right"))
        refilled = np.flatnonzero(REPLENISH_DENOMINATOR * depth[post:limit] >= bar)
        for tally in tallies:
            tally.events += 1
            if refilled.size:
                tally.replenished += 1
                tally.times_us.append(int(times[post + refilled[0]]) - printed_at)


def universe_readout(sweep: DepthSweep, universe: str) -> UniverseReadout:
    tally = sweep.tallies[universe]
    space = [
        (city, bucket, hour)
        for city in range(len(sweep.cities))
        for bucket in range(MAX_BUCKETS)
        for hour in range(HOURS_PER_DAY)
    ]
    cube = {}
    for city, bucket, hour in space:
        for side, name in enumerate(SIDES):
            readout = _readout(tally, (_key(city, bucket, hour),), side)
            if readout is not None:
                cube[(sweep.cities[city], bucket, hour, name)] = readout
    return UniverseReadout(
        cube=cube,
        by_city={
            (sweep.cities[city], name): cell
            for (city, name), cell in _marginal(tally, space, 0).items()
        },
        by_bucket=_marginal(tally, space, 1),
        by_hour=_marginal(tally, space, 2),
    )


def _marginal(
    tally: CellTally, space: Sequence[tuple[int, int, int]], position: int
) -> dict[tuple[int, str], CellReadout]:
    grouped: dict[int, list[int]] = {}
    for cell in space:
        grouped.setdefault(cell[position], []).append(_key(*cell))
    out = {}
    for value, keys in sorted(grouped.items()):
        for side, name in enumerate(SIDES):
            readout = _readout(tally, keys, side)
            if readout is not None:
                out[(value, name)] = readout
    return out


def _readout(tally: CellTally, keys: Sequence[int], side: int) -> CellReadout | None:
    sided = [key * 2 + side for key in keys]
    touch = tally.touch.pooled(sided)
    kept_us = sum(touch.values())
    if kept_us == 0:
        return None
    two_sided = tally.spread.pooled(keys)
    censored = tally.capacity_censored.pooled(sided)
    return CellReadout(
        kept_s=_seconds(kept_us),
        two_sided_fraction=Decimal(sum(two_sided.values())) / Decimal(kept_us),
        spread_p50_cents=_cents(weighted_quantile(two_sided, 1, 2)),
        spread_min_cents=_cents(min(two_sided) if two_sided else None),
        touch=_quartiles(touch),
        cum5=_quartiles(tally.cum5.pooled(sided)),
        capacity=_quartiles(tally.capacity.pooled(sided)),
        censored_s=_seconds(sum(censored.values())),
        censored_p50=_contracts(weighted_quantile(censored, 1, 2)),
    )


def _quartiles(bins: Mapping[int, int]) -> Quartiles:
    return Quartiles(
        p25=_contracts(weighted_quantile(bins, 1, 4)),
        p50=_contracts(weighted_quantile(bins, 1, 2)),
        p75=_contracts(weighted_quantile(bins, 3, 4)),
    )


def pooled_prints(resilience: Resilience, buckets: Iterable[int] | None = None) -> PrintTally:
    wanted = None if buckets is None else set(buckets)
    out = PrintTally()
    for (_, bucket), tally in resilience.tallies.items():
        if wanted is not None and bucket not in wanted:
            continue
        out.matched += tally.matched
        out.events += tally.events
        out.replenished += tally.replenished
        out.times_us.extend(tally.times_us)
    return out


# Right-censored at the window, so the median is the point the replenished fraction crosses half.
# Over the events that replenished alone it would be the median of a different population.
def replenish_median_s(tally: PrintTally) -> Decimal | None:
    if 2 * tally.replenished <= tally.events:
        return None
    ordered = sorted(tally.times_us)
    return Decimal(ordered[(tally.events + 1) // 2 - 1]) / MICROS_PER_S


def replenished_fraction(tally: PrintTally) -> Decimal | None:
    if tally.events == 0:
        return None
    return Decimal(tally.replenished) / Decimal(tally.events)


def execute(
    *,
    run_id: str,
    preregistration: Path,
    repo: Path,
    run_scope: Path,
    artifacts: Path,
    db: Path,
    rtt_samples: Path,
    floor_source: FloorSource,
    seed: int,
    run_root: Path,
) -> DepthMapRun:
    inputs = assemble_run_inputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        run_scope=run_scope,
        artifacts=artifacts,
        rtt_samples=rtt_samples,
        floor_source=floor_source,
        maker_rate=PUBLISHED_MAKER_RATE,
        maker_rate_source=MAKER_RATE_SOURCE,
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        bootstrap_seed=seed,
    )
    digest = write_manifest(run_root, inputs)

    scope = load_run_scope(run_scope)
    closes = load_close_times(db, scope)
    swept = sweep_depth(scope, artifacts, closes)
    resilience = sweep_resilience(scope, artifacts, closes, swept.picks)

    readouts = {name: universe_readout(swept, name) for name in UNIVERSES}
    final = pooled_prints(resilience.universes[ATM], FINAL_BUCKETS)
    keys = [
        _key(city, bucket, hour)
        for city in range(len(swept.cities))
        for bucket in FINAL_BUCKETS
        for hour in range(HOURS_PER_DAY)
    ]
    depth = {}
    for side, name in enumerate(SIDES):
        cell = _readout(swept.tallies[ATM], keys, side)
        depth[name] = None if cell is None else cell.touch.p50
    headline = Headline(universe=ATM, median_touch=depth, resilience=final)

    deep = all(value is not None and value > TOUCH_DEPTH_BAR for value in depth.values())
    resilient = 2 * final.replenished > final.events > 0
    run = DepthMapRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        floor=inputs.floor,
        scope_start=scope.scope_start,
        scope_end=scope.scope_end,
        event_days=tuple(sorted(scope.discovery_days | scope.holdout_days)),
        sweep=swept,
        readouts=readouts,
        resilience=resilience,
        headline=headline,
        verdict=REFUTED if deep and resilient else CONFIRMED,
    )
    logger.info(
        "depth_map verdict=%s atm_touch_yes=%s atm_touch_no=%s events=%d replenished=%d",
        run.verdict,
        depth[YES],
        depth[NO],
        final.events,
        final.replenished,
    )
    return run


def result_payload(run: DepthMapRun) -> dict:
    screened = run.sweep.screened
    return {
        "run_id": run.run_id,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "latency_floor_source": run.floor.source.value,
        "scope_start": run.scope_start.isoformat(),
        "scope_end": run.scope_end.isoformat(),
        "event_days": [day.isoformat() for day in run.event_days],
        "cities": list(run.sweep.cities),
        "bars": {
            "touch_depth_bar_contracts": str(TOUCH_DEPTH_BAR),
            "slippage_bar_cents": str(SLIPPAGE_BAR_CENTS),
            "replenish_fraction": str(REPLENISH_FRACTION),
            "replenish_window_s": REPLENISH_WINDOW_S,
            "cumulative_ticks": CUMULATIVE_TICKS,
            "final_hours": FINAL_HOURS,
            "bucket_hours": BUCKET_HOURS,
            "max_buckets": MAX_BUCKETS,
            "ladder_depth": LADDER_DEPTH,
        },
        "verdict": run.verdict,
        "headline": {
            "universe": run.headline.universe,
            "final_hours": FINAL_HOURS,
            "median_touch_depth_contracts": {
                name: _text(value) for name, value in run.headline.median_touch.items()
            },
            "resilience": _prints_payload(run.headline.resilience),
        },
        **{
            name: _universe_payload(run.readouts[name], run.resilience.universes[name])
            for name in UNIVERSES
        },
        "exclusions": {
            "candidates": screened.candidates,
            "out_of_window": screened.out_of_window,
            "past_close": screened.past_close,
            "excluded": screened.excluded,
            "excluded_fraction": _text(screened.excluded_fraction),
            "by_class": dict(sorted(screened.by_class.items())),
            "kept_span_s": str(_seconds(screened.kept_us)),
            "nominal_span_s": str(_seconds(screened.nominal_us)),
            "retained_fraction": _text(screened.retained_fraction),
        },
        "atm_ticker_per_city_day": {
            f"{series} {day.isoformat()}": ticker
            for (series, day), ticker in sorted(run.sweep.picks.items())
        },
        "no_atm_leg": [f"{series} {day.isoformat()}" for series, day in run.sweep.no_atm_leg],
        "rows": {
            "ladder": run.sweep.rows,
            "cells": run.sweep.cells,
            "resilience_states": run.resilience.states,
            "trades": run.resilience.trades,
        },
    }


def _universe_payload(readout: UniverseReadout, resilience: Resilience) -> dict:
    accounting = resilience.accounting
    return {
        "cube": {
            f"{series} {bucket} {hour} {side}": _cell_payload(cell)
            for (series, bucket, hour, side), cell in sorted(readout.cube.items())
        },
        "by_city": {
            f"{series} {side}": _cell_payload(cell)
            for (series, side), cell in sorted(readout.by_city.items())
        },
        "by_bucket": {
            f"{bucket} {side}": _cell_payload(cell)
            for (bucket, side), cell in sorted(readout.by_bucket.items())
        },
        "by_hour": {
            f"{hour} {side}": _cell_payload(cell)
            for (hour, side), cell in sorted(readout.by_hour.items())
        },
        "resilience": {
            "by_city_bucket": {
                f"{series} {bucket}": _prints_payload(tally)
                for (series, bucket), tally in sorted(resilience.tallies.items())
            },
            "pooled": _prints_payload(pooled_prints(resilience)),
            "prints": accounting.prints,
            "no_pre_state": accounting.no_pre_state,
            "no_match": accounting.no_match,
            "no_post_state": accounting.no_post_state,
            "past_close": accounting.past_close,
            "out_of_window": accounting.out_of_window,
            "excluded": accounting.excluded,
            "by_class": dict(sorted(accounting.by_class.items())),
        },
    }


def _cell_payload(cell: CellReadout) -> dict:
    return {
        "kept_s": str(cell.kept_s),
        "two_sided_fraction": _text(cell.two_sided_fraction),
        "spread_p50_cents": _text(cell.spread_p50_cents),
        "spread_min_cents": _text(cell.spread_min_cents),
        "touch_contracts": _quartiles_payload(cell.touch),
        "cum5_contracts": _quartiles_payload(cell.cum5),
        "capacity_contracts": _quartiles_payload(cell.capacity),
        "censored_s": str(cell.censored_s),
        "censored_p50_contracts": _text(cell.censored_p50),
    }


def _quartiles_payload(quartiles: Quartiles) -> dict:
    return {
        "p25": _text(quartiles.p25),
        "p50": _text(quartiles.p50),
        "p75": _text(quartiles.p75),
    }


def _prints_payload(tally: PrintTally) -> dict:
    return {
        "matched": tally.matched,
        "events": tally.events,
        "replenished": tally.replenished,
        "replenished_fraction": _text(replenished_fraction(tally)),
        "median_replenish_s": _text(replenish_median_s(tally)),
    }


def _plan(ticker: str, scope: RunScope, closes: Mapping[str, datetime], seat: int) -> _Plan | None:
    parsed = parse_ticker(ticker)
    day = scope.event_days.get((parsed.series, parsed.event_date))
    if day is None:
        return None
    close_us = _close_us(closes, ticker)
    start_us = _micros(day.window_start)
    day_end_us = _micros(day.window_end)
    end_us = min(day_end_us, close_us)
    hours = np.arange(
        start_us - start_us % MICROS_PER_HOUR, end_us + MICROS_PER_HOUR, MICROS_PER_HOUR
    )
    buckets = close_us - np.arange(1, MAX_BUCKETS + 1, dtype=np.int64) * (
        BUCKET_HOURS * MICROS_PER_HOUR
    )
    return _Plan(
        series=parsed.series,
        event_date=parsed.event_date,
        seat=seat,
        close_us=close_us,
        day_start_us=start_us,
        day_end_us=day_end_us,
        end_us=end_us,
        breaks=np.union1d(
            np.array([start_us, end_us], dtype=np.int64), np.concatenate((hours, buckets))
        ),
    )


def _close_us(closes: Mapping[str, datetime], ticker: str) -> int:
    close = closes.get(ticker)
    if close is None:
        raise ValueError(f"{ticker} is in scope but carries no row in the markets table")
    return _micros(close)


def _book(
    times: np.ndarray,
    prices: tuple[np.ndarray, np.ndarray],
    sizes: tuple[np.ndarray, np.ndarray],
    levels: tuple[np.ndarray, np.ndarray],
) -> Book:
    walks = tuple(
        walk_capacity(prices[side], sizes[side], levels[side], SLIPPAGE_BAR_UNITS)
        for side in range(len(SIDES))
    )
    touch = (sizes[0][:, 0], sizes[1][:, 0])
    return Book(
        times=times,
        touch=touch,
        cum5=(
            cumulative_depth(prices[0], sizes[0], CUMULATIVE_TICKS),
            cumulative_depth(prices[1], sizes[1], CUMULATIVE_TICKS),
        ),
        capacity=(walks[0].capacity, walks[1].capacity),
        censored=(walks[0].censored, walks[1].censored),
        two_sided=(touch[0] > 0) & (touch[1] > 0),
        spread=PRICE_TICKS - prices[0][:, 0] - prices[1][:, 0],
    )


def _join(carry: Book | None, book: Book) -> Book:
    if carry is None:
        return book
    return Book(
        times=np.concatenate((carry.times, book.times)),
        touch=_pair(carry.touch, book.touch),
        cum5=_pair(carry.cum5, book.cum5),
        capacity=_pair(carry.capacity, book.capacity),
        censored=_pair(carry.censored, book.censored),
        two_sided=np.concatenate((carry.two_sided, book.two_sided)),
        spread=np.concatenate((carry.spread, book.spread)),
    )


def _pair(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return (np.concatenate((left[0], right[0])), np.concatenate((left[1], right[1])))


# Copied rather than sliced: a view keeps the whole file's arrays alive behind one carried state.
def _tail(book: Book) -> Book:
    return Book(
        times=book.times[-1:].copy(),
        touch=(book.touch[0][-1:].copy(), book.touch[1][-1:].copy()),
        cum5=(book.cum5[0][-1:].copy(), book.cum5[1][-1:].copy()),
        capacity=(book.capacity[0][-1:].copy(), book.capacity[1][-1:].copy()),
        censored=(book.censored[0][-1:].copy(), book.censored[1][-1:].copy()),
        two_sided=book.two_sided[-1:].copy(),
        spread=book.spread[-1:].copy(),
    )


def _read_legs(
    artifacts: Path,
    kind: str,
    series: str,
    dates: Sequence[date],
    event_date: date,
    columns: Sequence[str],
) -> pa.Table:
    tables = []
    for path in partition_files(artifacts, kind, series, dates):
        table = _checked(path, kind).read(columns=list(columns))
        column = table.column("ticker")
        legs = [
            ticker
            for ticker in column.unique().to_pylist()
            if parse_ticker(ticker).event_date == event_date
        ]
        tables.append(table.filter(pc.is_in(column, value_set=pa.array(legs, type=pa.string()))))
    if not tables:
        return KIND_SCHEMAS[kind].empty_table().select(list(columns))
    return pa.concat_tables(tables)


def _checked(path: Path, kind: str) -> pq.ParquetFile:
    stored = pq.ParquetFile(path)
    if not stored.schema_arrow.equals(KIND_SCHEMAS[kind]):
        raise ValueError(f"{path} does not carry the frozen {kind} schema")
    return stored


def _levels(table: pa.Table, name: str, decimals: int) -> np.ndarray:
    return level_units(table.column(name).combine_chunks(), decimals, LADDER_DEPTH)


def _units(table: pa.Table, name: str, decimals: int) -> np.ndarray:
    return scaled_units(table.column(name).combine_chunks(), decimals)


def _last_at_each(times: np.ndarray) -> np.ndarray:
    keep = np.ones(times.size, dtype=bool)
    keep[:-1] = times[:-1] != times[1:]
    return keep


# hours_to_close_bucket is right-closed, so an instant sitting on a bucket edge reads as the older
# bucket. A cell's interior and a print at close - FINAL_HOURS both belong to the newer one, so
# take the bucket a microsecond past the instant handed in.
def _bucket_of(close_us: np.ndarray, at_us: np.ndarray) -> np.ndarray:
    return hours_to_close_bucket(close_us, at_us + 1, BUCKET_HOURS)


def _key(
    seat: int | np.ndarray, bucket: int | np.ndarray, hour: int | np.ndarray
) -> int | np.ndarray:
    return (seat * MAX_BUCKETS + bucket) * HOURS_PER_DAY + hour


def _decode(key: int) -> tuple[int, int, int]:
    rest, hour = divmod(key, HOURS_PER_DAY)
    seat, bucket = divmod(rest, MAX_BUCKETS)
    return seat, bucket, hour


def _seconds(micros: int) -> Decimal:
    return Decimal(micros) / MICROS_PER_S


def _contracts(units: int | None) -> Decimal | None:
    return None if units is None else Decimal(units) / SIZE_UNITS


def _cents(units: int | None) -> Decimal | None:
    return None if units is None else Decimal(units) / TICKS_PER_CENT


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _micros(stamp: datetime) -> int:
    return (stamp - _EPOCH) // _MICROSECOND
