from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from bot.lag.fee_floor import published_taker_fee
from bot.lag.tape_stats import ClusterAggregate
from bot.replay.ladder import _SIZE_EXPONENT, _quantized


HORIZONS_S: tuple[int, ...] = (1, 10, 60, 300)
PRIMARY_HORIZON_S: int = 60
PRINT_MIN_DISCOVERY: int = 5_000
# The exchange quotes in whole cents, so one cent is the smallest move a taker can express.
CENT_BAR: Decimal = Decimal("1.0")
# Stored prices carry _PRICE_EXPONENT, so a stored price is this many units per dollar. A unit is a
# hundredth of the cent CENT_BAR measures, not the increment the exchange trades on.
PRICE_TICKS: int = 10_000

YES = "yes"
NO = "no"

_MICROS_PER_S = 1_000_000
_CENTS_PER_DOLLAR = Decimal(100)
_ROUND_TRIP_LEGS = Decimal(2)
# 200 doubled ticks per contract is one cent: half of it undoes the doubling, and the rest is
# 10,000 ticks to the dollar against 100 cents to the dollar.
_GROSS_DIVISOR = Decimal(200)


# The taker lifted resting size on the side named here, so "yes" is an aggressive buy of YES.
# An empty side is a decode artifact, never a direction: _decode writes `or ""` into a NOT NULL
# column, so the string carries no evidence about which way the aggressor leaned.
# Deliberately not bot.lag.taker_side.taker_direction, which carries the same signs: that one hands
# an unsigned side back as None, and screen_prints has already dropped those, so here one is a bug.
def yes_pressure(taker_side: str) -> int:
    if taker_side == YES:
        return 1
    if taker_side == NO:
        return -1
    raise ValueError(f"taker_side must be {YES!r} or {NO!r}, got {taker_side!r}")


@dataclass(frozen=True, slots=True)
class FlowCounts:
    empty_side: int = 0
    duplicates: int = 0
    unresolved: int = 0
    uncovered: int = 0
    one_sided: int = 0
    host_clock: int = 0

    def __add__(self, other: FlowCounts) -> FlowCounts:
        return FlowCounts(
            empty_side=self.empty_side + other.empty_side,
            duplicates=self.duplicates + other.duplicates,
            unresolved=self.unresolved + other.unresolved,
            uncovered=self.uncovered + other.uncovered,
            one_sided=self.one_sided + other.one_sided,
            host_clock=self.host_clock + other.host_clock,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TickerBook:
    ticker: str
    received_us: np.ndarray
    ts_ms: np.ndarray
    mid2: np.ndarray
    two_sided: np.ndarray
    delta_rows: np.ndarray
    delta_ts: np.ndarray
    snapshots_before: np.ndarray
    ts_violations: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TickerPrints:
    ticker: str
    ts_ms: np.ndarray
    received_us: np.ndarray
    pressure: np.ndarray
    contracts: tuple[Decimal, ...]
    prices: tuple[Decimal, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Anchors:
    index: np.ndarray
    resolved: np.ndarray
    host_clock: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizonWindows:
    horizon_s: int
    start_us: np.ndarray
    end_us: np.ndarray
    end_index: np.ndarray
    usable: np.ndarray
    counts: FlowCounts


@dataclass(frozen=True, slots=True, kw_only=True)
class PrintOutcome:
    ticker: str
    contracts: Decimal
    net_cents: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class PrintHygiene:
    kept: pa.Table
    counts: FlowCounts


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizonResult:
    horizon_s: int
    split: str
    clusters: tuple[ClusterAggregate, ...]
    n_prints: int
    contracts: Decimal
    counts: FlowCounts

    @property
    def mean_net_cents(self) -> Decimal:
        if not self.clusters:
            raise ValueError("a contract-weighted mean needs at least one cluster")
        return sum((item.total for item in self.clusters), Decimal(0)) / sum(
            (item.weight for item in self.clusters), Decimal(0)
        )


def build_ticker_book(ticker: str, table: pa.Table) -> TickerBook:
    stamps = table.column("ts_ms")
    is_delta = stamps.is_valid().to_numpy(zero_copy_only=False)
    ts_ms = stamps.fill_null(0).to_numpy()
    yes_bid = _tick_column(table, "yes_bid")
    no_bid = _tick_column(table, "no_bid")
    delta_rows = np.flatnonzero(is_delta)
    delta_ts = ts_ms[delta_rows]
    return TickerBook(
        ticker=ticker,
        received_us=table.column("received_at").cast(pa.int64()).to_numpy(),
        ts_ms=ts_ms,
        # Doubling keeps every intermediate exact in int64; the caller halves at the end.
        mid2=yes_bid + PRICE_TICKS - no_bid,
        # A zero price is _best reporting an empty side, not a one-tick market, and a book empty
        # on both sides would otherwise read as a mid of exactly 0.5 by construction.
        two_sided=(yes_bid > 0) & (no_bid > 0),
        delta_rows=delta_rows,
        delta_ts=delta_ts,
        snapshots_before=np.concatenate(([0], np.cumsum(~is_delta, dtype=np.int64))),
        ts_violations=int(np.count_nonzero(np.diff(delta_ts) < 0)),
    )


def build_ticker_prints(ticker: str, table: pa.Table) -> TickerPrints:
    sides = table.column("taker_side").to_pylist()
    yes_prices = table.column("yes_price").to_pylist()
    no_prices = table.column("no_price").to_pylist()
    prices = tuple(
        Decimal(yes if side == YES else no)
        for side, yes, no in zip(sides, yes_prices, no_prices, strict=True)
    )
    return TickerPrints(
        ticker=ticker,
        ts_ms=table.column("ts_ms").to_numpy(),
        received_us=table.column("received_at").cast(pa.int64()).to_numpy(),
        pressure=np.array([yes_pressure(side) for side in sides], dtype=np.int64),
        contracts=tuple(
            _quantized(ticker, "count", value, _SIZE_EXPONENT)
            for value in table.column("count").to_pylist()
        ),
        prices=prices,
    )


def screen_prints(table: pa.Table) -> PrintHygiene:
    signed = table.filter(pc.not_equal(table.column("taker_side"), ""))
    ids = signed.column("id").to_numpy()
    trade_ids = signed.column("trade_id").to_pylist()
    # Migration 0010 added trade_id with no unique constraint, so a wire re-send lands as a second
    # row and this is the only place the tape is deduped.
    lowest: dict[str, int] = {}
    for position in np.argsort(ids, kind="stable"):
        lowest.setdefault(trade_ids[position], int(position))
    keep = np.zeros(ids.size, dtype=bool)
    keep[np.fromiter(lowest.values(), dtype=np.int64, count=len(lowest))] = True
    return PrintHygiene(
        kept=signed.filter(pa.array(keep)),
        counts=FlowCounts(
            empty_side=table.num_rows - signed.num_rows,
            duplicates=int(np.count_nonzero(~keep)),
        ),
    )


# The anchor is the book immediately after the first event the exchange stamped later than the
# print, never the last mid before it: a lift that consumes the offer mechanically raises the mid,
# which would make the statistic positive by construction at short horizons.
def resolve_anchors(book: TickerBook, prints: TickerPrints) -> Anchors:
    rows = book.received_us.size
    # Exchange stamps arrive out of order often enough that ts_violations counts them, and a binary
    # search over an unsorted array returns an arbitrary index. The running maximum is sorted and
    # rises only at a row that sets a record, so the first index past the print is the first row in
    # id order stamped after it, whatever the ordering.
    stamped = np.append(book.delta_rows, rows)[
        np.searchsorted(np.maximum.accumulate(book.delta_ts), prints.ts_ms, side="right")
    ]
    arrival = np.searchsorted(book.received_us, prints.received_us, side="right")
    # A snapshot carries no exchange stamp, so one landing between the print and the stamped delta
    # leaves arrival order as the only ordering both rows share.
    fallback = book.snapshots_before[stamped] - book.snapshots_before[arrival] > 0
    index = np.where(fallback, arrival, stamped)
    resolved = index < rows
    return Anchors(index=index, resolved=resolved, host_clock=fallback & resolved)


def resolve_horizon(
    book: TickerBook, prints: TickerPrints, anchors: Anchors, *, horizon_s: int
) -> HorizonWindows:
    at_anchor = np.where(anchors.resolved, anchors.index, 0)
    anchor_us = book.received_us[at_anchor]
    end_us = anchor_us + horizon_s * _MICROS_PER_S
    # Without this a print near the ticker's last row scores a zero move by construction.
    covered = anchors.resolved & (book.received_us[-1] >= end_us)
    end_index = np.searchsorted(book.received_us, end_us, side="right") - 1
    two_sided = book.two_sided[at_anchor] & book.two_sided[np.where(covered, end_index, 0)]
    usable = covered & two_sided
    return HorizonWindows(
        horizon_s=horizon_s,
        # The anchor can precede the print: trades and book deltas arrive on separate channels.
        start_us=np.minimum(prints.received_us, anchor_us),
        end_us=end_us,
        end_index=end_index,
        usable=usable,
        counts=FlowCounts(
            unresolved=int(np.count_nonzero(~anchors.resolved)),
            uncovered=int(np.count_nonzero(anchors.resolved & ~covered)),
            one_sided=int(np.count_nonzero(covered & ~two_sided)),
            host_clock=int(np.count_nonzero(anchors.host_clock & usable)),
        ),
    )


def print_net_cents(*, contracts: Decimal, price: Decimal, signed_move2: int) -> Decimal:
    gross = contracts * Decimal(signed_move2) / _GROSS_DIVISOR
    # Two taker legs at the print's own price on the print's own size, charged once on the
    # aggregate. The fee is symmetric in P(1-P), so the price the taker paid reads true.
    return gross - _ROUND_TRIP_LEGS * _CENTS_PER_DOLLAR * published_taker_fee(contracts, price)


def print_outcomes(
    book: TickerBook, prints: TickerPrints, anchors: Anchors, windows: HorizonWindows
) -> list[PrintOutcome]:
    outcomes = []
    for position in np.flatnonzero(windows.usable):
        contracts = prints.contracts[position]
        move2 = int(book.mid2[windows.end_index[position]] - book.mid2[anchors.index[position]])
        outcomes.append(
            PrintOutcome(
                ticker=prints.ticker,
                contracts=contracts,
                net_cents=print_net_cents(
                    contracts=contracts,
                    price=prints.prices[position],
                    signed_move2=move2 * int(prints.pressure[position]),
                ),
            )
        )
    return outcomes


# The cluster unit is the market-day: one Kalshi market on one event-day is exactly one ticker.
def cluster_aggregates(outcomes: Sequence[PrintOutcome]) -> list[ClusterAggregate]:
    totals: dict[str, Decimal] = {}
    weights: dict[str, Decimal] = {}
    for outcome in outcomes:
        totals[outcome.ticker] = totals.get(outcome.ticker, Decimal(0)) + outcome.net_cents
        weights[outcome.ticker] = weights.get(outcome.ticker, Decimal(0)) + outcome.contracts
    return [
        ClusterAggregate(cluster=ticker, total=totals[ticker], weight=weights[ticker])
        for ticker in sorted(totals)
        if weights[ticker] > 0
    ]


def horizon_result(
    *, horizon_s: int, split: str, outcomes: Sequence[PrintOutcome], counts: FlowCounts
) -> HorizonResult:
    return HorizonResult(
        horizon_s=horizon_s,
        split=split,
        clusters=tuple(cluster_aggregates(outcomes)),
        n_prints=len(outcomes),
        contracts=sum((outcome.contracts for outcome in outcomes), Decimal(0)),
        counts=counts,
    )


def _tick_column(table: pa.Table, name: str) -> np.ndarray:
    prices = table.column(name).to_pylist()
    return np.array([_ticks(Decimal(value)) for value in prices], dtype=np.int64)


def _ticks(price: Decimal) -> int:
    scaled = price * PRICE_TICKS
    ticks = int(scaled)
    if scaled != ticks:
        raise ValueError(f"price {price} is off the {PRICE_TICKS}-per-dollar grid")
    return ticks
