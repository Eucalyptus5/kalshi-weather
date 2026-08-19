from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import pyarrow as pa
import pyarrow.compute as pc

from bot.lag.depth_map import depth_at_price, volume_at_price
from bot.lag.mid import ticks, two_sided
from bot.lag.placement_grid import CloseSidecar, close_of, placement_grid


YES = "yes"
NO = "no"
SIDES: tuple[str, str] = (YES, NO)
YES_CONTRACTS: Decimal = Decimal("12.36")
NO_CONTRACTS: Decimal = Decimal("26")
REST_S: int = 300

_CONTRACTS = {YES: YES_CONTRACTS, NO: NO_CONTRACTS}
_REST = timedelta(seconds=REST_S)
_SIZE_UNITS = Decimal(100)


@dataclass(frozen=True, slots=True, kw_only=True)
class Fill:
    ticker: str
    side: str
    placement_price: Decimal
    contracts: Decimal
    placed_at: datetime
    filled_at: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketDayFills:
    ticker: str
    close: datetime
    instants: tuple[datetime, ...]
    fills: tuple[Fill, ...]
    offered: int
    yes_fills: int
    no_fills: int
    yes_empty: int
    no_empty: int

    # One sweep is one market-day, so the credited count is already the per-market-day rate.
    @property
    def fill_rate(self) -> Decimal:
        return Decimal(len(self.fills))


@dataclass(frozen=True, slots=True, kw_only=True)
class TickerTape:
    ticker: str
    rows: tuple[Mapping[str, object], ...]
    stamps: tuple[datetime, ...]
    prints: pa.Table
    print_stamps: tuple[datetime, ...]


# The prints stay whole: volume_at_price scopes them to the ticker itself, and two brackets of one
# event rest at the same price often enough that pre-filtering would hide a missing scope. The
# stamps are scoped only to spare a whole-table filter per candidate: our total steps only at our
# own prints, so the earliest crediting stamp is unchanged either way.
def ticker_tape(ladder: pa.Table, prints: pa.Table, ticker: str) -> TickerTape:
    book = ladder.filter(pc.equal(ladder.column("ticker"), ticker)).sort_by("received_at")
    own = prints.filter(pc.equal(prints.column("ticker"), ticker))
    return TickerTape(
        ticker=ticker,
        rows=tuple(book.to_pylist()),
        stamps=tuple(book.column("received_at").to_pylist()),
        prints=prints,
        print_stamps=tuple(sorted(set(own.column("received_at").to_pylist()))),
    )


def credited(volume: Decimal, ahead: Decimal) -> bool:
    return volume > ahead


# two_sided reads its two halves and ands them, so one half handed in twice reads that half alone.
# Depth reaches the artifact quantised to 0.01, so it converts to hundredths first: a plain int()
# would read a side resting under one contract as empty.
def side_is_live(row: Mapping[str, object], side: str) -> bool:
    bid = ticks(Decimal(row[f"{side}_bid"]))
    depth = int(Decimal(row[f"{side}_bid_depth"]) * _SIZE_UNITS)
    return two_sided(bid, bid, yes_depth=depth, no_depth=depth)


def rest_end(tape: TickerTape, side: str, at: int, placed_at: datetime, price: Decimal) -> datetime:
    deadline = placed_at + _REST
    for index in range(at + 1, len(tape.stamps)):
        stamp = tape.stamps[index]
        if stamp > deadline:
            break
        if Decimal(tape.rows[index][f"{side}_bid"]) != price:
            return stamp
    return deadline


def resolve_order(tape: TickerTape, side: str, at: int, placed_at: datetime) -> Fill | None:
    row = tape.rows[at]
    price = Decimal(row[f"{side}_bid"])
    ahead, _ = depth_at_price(row, side, price)
    end = rest_end(tape, side, at, placed_at, price)
    if not credited(volume_at_price(tape.prints, tape.ticker, side, price, placed_at, end), ahead):
        return None
    # Volume rises with the window's end, so the first print whose running total clears the queue
    # ahead is the print the quote traded against, and its stamp is the fill.
    filled_at = next(
        stamp
        for stamp in tape.print_stamps[
            bisect_left(tape.print_stamps, placed_at) : bisect_right(tape.print_stamps, end)
        ]
        if credited(volume_at_price(tape.prints, tape.ticker, side, price, placed_at, stamp), ahead)
    )
    return Fill(
        ticker=tape.ticker,
        side=side,
        placement_price=price,
        contracts=_CONTRACTS[side],
        placed_at=placed_at,
        filled_at=filled_at,
    )


def sweep_market_day(
    *, ladder: pa.Table, prints: pa.Table, sidecar: CloseSidecar, ticker: str
) -> MarketDayFills:
    close = close_of(sidecar, ticker)
    instants = placement_grid(close)
    tape = ticker_tape(ladder, prints, ticker)
    fills: list[Fill] = []
    filled = {YES: 0, NO: 0}
    empty = {YES: 0, NO: 0}
    offered = 0
    for placed_at in instants:
        at = bisect_right(tape.stamps, placed_at) - 1
        for side in SIDES:
            if at < 0 or not side_is_live(tape.rows[at], side):
                empty[side] += 1
                continue
            offered += 1
            fill = resolve_order(tape, side, at, placed_at)
            if fill is not None:
                fills.append(fill)
                filled[side] += 1
    return MarketDayFills(
        ticker=ticker,
        close=close,
        instants=instants,
        fills=tuple(fills),
        offered=offered,
        yes_fills=filled[YES],
        no_fills=filled[NO],
        yes_empty=empty[YES],
        no_empty=empty[NO],
    )
