from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from bot.lag.fill_convention import REST_S, Fill
from bot.lag.maker_headroom import HALF_TICK_CAPTURE_CENTS, maker_fee_cents_per_contract
from bot.lag.mid import mid2_array, ticks, two_sided_array
from bot.lag.taker_flow import HORIZONS_S, PRIMARY_HORIZON_S, yes_pressure
from bot.lag.tape_studies import EvidenceWindow, RunScope, Screened, keep_mask, screen_windows


# 200 doubled ticks is one cent on one contract: half of it undoes the doubling, and the rest is
# 10,000 ticks to the dollar against 100 cents to the dollar.
_CENTS_DIVISOR: Decimal = Decimal(200)


@dataclass(frozen=True, slots=True, kw_only=True)
class FillEdge:
    ticker: str
    side: str
    contracts: Decimal
    placement_price: Decimal
    horizon_s: int
    edge_cents_per_contract: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class QuoteBook:
    ticker: str
    stamps: tuple[datetime, ...]
    mid2: np.ndarray
    two_sided: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizonEdges:
    horizon_s: int
    window_cap_s: int
    edges: tuple[FillEdge, ...]
    modelled: int
    dropped: int
    screened: Screened

    @property
    def dropped_fraction(self) -> Decimal:
        return Decimal(self.dropped) / Decimal(self.modelled)


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeCurve:
    by_horizon: Mapping[int, HorizonEdges]

    @property
    def gate(self) -> HorizonEdges:
        return self.by_horizon[PRIMARY_HORIZON_S]


def quote_book(ladder: pa.Table, ticker: str) -> QuoteBook:
    rows = ladder.filter(pc.equal(ladder.column("ticker"), ticker)).sort_by("received_at")
    yes_bid = _tick_column(rows, "yes_bid")
    no_bid = _tick_column(rows, "no_bid")
    return QuoteBook(
        ticker=ticker,
        stamps=tuple(rows.column("received_at").to_pylist()),
        mid2=mid2_array(yes_bid, no_bid),
        two_sided=two_sided_array(yes_bid, no_bid, yes_depth=None, no_depth=None),
    )


# The book steps at arrivals, so the row in force is the last one that landed at or before the
# instant. An instant past the last row has no mid rather than the last one: without that a fill
# near the tape's end scores a zero move by construction.
def mid2_at(book: QuoteBook, at: datetime) -> int | None:
    if not book.stamps[0] <= at <= book.stamps[-1]:
        return None
    index = bisect_right(book.stamps, at) - 1
    return int(book.mid2[index]) if book.two_sided[index] else None


# Signed to the side our quote filled, never to whoever took it: a filled yes quote leaves us long
# yes, so a mid that falls afterwards is a loss on the yes side and a gain on the no side.
def mark_out_cents(
    book: QuoteBook, *, anchor: datetime, horizon_s: int, side: str
) -> Decimal | None:
    opened = mid2_at(book, anchor)
    closed = mid2_at(book, anchor + timedelta(seconds=horizon_s))
    if opened is None or closed is None:
        return None
    return Decimal((closed - opened) * yes_pressure(side)) / _CENTS_DIVISOR


# The fill is marked to the mid, so the capture is one half tick and the mark-out already carries
# the adverse move against it. Adding the mark-out is subtracting that adverse move.
def edge_cents(*, mark_out: Decimal, contracts: Decimal, price: Decimal, rate: Decimal) -> Decimal:
    return HALF_TICK_CAPTURE_CENTS + mark_out - maker_fee_cents_per_contract(contracts, price, rate)


# A quote the convention never filled is retired after REST_S, so placement to fill is at most that
# and no window runs longer than the rest plus its own horizon.
def window_cap_s(horizon_s: int) -> int:
    return REST_S + horizon_s


def evidence_window(fill: Fill, *, series: str, event_date: date, horizon_s: int) -> EvidenceWindow:
    return EvidenceWindow(
        series=series,
        event_date=event_date,
        start=fill.placed_at,
        end=fill.filled_at + timedelta(seconds=horizon_s),
    )


def resolve_edges(
    *,
    book: QuoteBook,
    fills: Sequence[Fill],
    scope: RunScope,
    series: str,
    event_date: date,
    maker_rate: Decimal,
    horizon_s: int,
) -> HorizonEdges:
    windows = [
        evidence_window(fill, series=series, event_date=event_date, horizon_s=horizon_s)
        for fill in fills
    ]
    screened = screen_windows(scope, windows)
    edges = []
    dropped = 0
    for fill, keep in zip(fills, keep_mask(windows, screened.kept), strict=True):
        if not keep:
            continue
        mark_out = mark_out_cents(book, anchor=fill.filled_at, horizon_s=horizon_s, side=fill.side)
        # One-sided books are where the adverse moves concentrate, so this drop biases the effect
        # upward and the count is reported beside it rather than corrected for.
        if mark_out is None:
            dropped += 1
            continue
        edges.append(
            FillEdge(
                ticker=fill.ticker,
                side=fill.side,
                contracts=fill.contracts,
                placement_price=fill.placement_price,
                horizon_s=horizon_s,
                edge_cents_per_contract=edge_cents(
                    mark_out=mark_out,
                    contracts=fill.contracts,
                    price=fill.placement_price,
                    rate=maker_rate,
                ),
            )
        )
    return HorizonEdges(
        horizon_s=horizon_s,
        window_cap_s=window_cap_s(horizon_s),
        edges=tuple(edges),
        modelled=len(fills),
        dropped=dropped,
        screened=screened,
    )


def resolve_curve(
    *,
    book: QuoteBook,
    fills: Sequence[Fill],
    scope: RunScope,
    series: str,
    event_date: date,
    maker_rate: Decimal,
) -> EdgeCurve:
    return EdgeCurve(
        by_horizon={
            horizon_s: resolve_edges(
                book=book,
                fills=fills,
                scope=scope,
                series=series,
                event_date=event_date,
                maker_rate=maker_rate,
                horizon_s=horizon_s,
            )
            for horizon_s in HORIZONS_S
        }
    )


def _tick_column(table: pa.Table, name: str) -> np.ndarray:
    return np.array(
        [ticks(Decimal(value)) for value in table.column(name).to_pylist()], dtype=np.int64
    )
