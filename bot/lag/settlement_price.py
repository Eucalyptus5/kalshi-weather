from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from bot.lag.fee_floor import BAR_CONTEXT, TICK_CENTS, published_taker_fee
from bot.lag.ladder_consistency import PRICE_TICKS
from bot.lag.mid import mid2, ticks, two_sided
from bot.lag.placement_grid import CloseSidecar
from bot.lag.settlement_entry import StraddleEntry
from bot.lag.tape_studies import (
    LADDER,
    EvidenceWindow,
    RunScope,
    partition_files,
    partition_rows,
    screen_windows,
    window_dates,
)
from bot.markets.parser import parse_ticker


SIZE: Decimal = Decimal(26)
ENTRY_WINDOW: timedelta = timedelta(seconds=60)

_CENTS = Decimal(100)


@dataclass(frozen=True, slots=True, kw_only=True)
class PricedStraddle:
    entry_price: Decimal | None
    entry_fee_cents: Decimal | None
    entry_tick_cents: Decimal
    net_profit_cents: Decimal | None
    size: Decimal
    censored: bool
    priced: bool
    no_row: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceCounts:
    n: int
    one_sided_n: int
    no_row_n: int
    censored_n: int


@dataclass(frozen=True, slots=True, kw_only=True)
class EntryScreen:
    kept: tuple[EvidenceWindow, ...]
    candidates: int
    excluded: int
    dropped: int
    city_event_days: int
    city_event_days_kept: int

    @property
    def city_event_days_lost(self) -> Decimal:
        return Decimal(self.city_event_days - self.city_event_days_kept) / Decimal(
            self.city_event_days
        )


# Every class the shared screen drops on is a recorder class, and the observation side is our own
# decoded METAR, which the recorder's uptime does not touch. The one price read is what it touches,
# so the window screened is the minute that read lands in.
def entry_window(entry: StraddleEntry) -> EvidenceWindow:
    parsed = parse_ticker(entry.ticker)
    return EvidenceWindow(
        series=parsed.series,
        event_date=parsed.event_date,
        start=entry.entry_instant,
        end=entry.entry_instant + ENTRY_WINDOW,
    )


def screen_entry_minutes(scope: RunScope, windows: Sequence[EvidenceWindow]) -> EntryScreen:
    screened = screen_windows(scope, windows)
    offered = {(window.series, window.event_date) for window in windows}
    survived = {(window.series, window.event_date) for window in screened.kept}
    return EntryScreen(
        kept=screened.kept,
        candidates=screened.candidates,
        excluded=screened.excluded,
        dropped=screened.candidates - len(screened.kept),
        city_event_days=len(offered),
        city_event_days_kept=len(survived),
    )


def ladder_census(
    artifacts: Path, scope: RunScope, roots: Sequence[str]
) -> dict[tuple[str, date], int]:
    wanted = set(roots)
    census: dict[tuple[str, date], int] = {}
    for key in sorted(key for key in scope.event_days if key[0] in wanted):
        series, event_date = key
        day = scope.event_days[key]
        dates = window_dates(day.window_start, day.window_end)
        if not partition_files(artifacts, LADDER, series, dates):
            raise ValueError(
                f"{series} {event_date.isoformat()} has no ladder partition under {artifacts}"
            )
        rows = partition_rows(artifacts, LADDER, series, dates)
        if not rows:
            raise ValueError(
                f"{series} {event_date.isoformat()} has an empty ladder partition under {artifacts}"
            )
        census[key] = rows
    return census


# The published formula prices a whole order in dollars while the statistic is stated in cents per
# contract, so the size the order was priced at divides back out here.
def fee_cents_per_contract(size: Decimal, price: Decimal) -> Decimal:
    aggregate = BAR_CONTEXT.multiply(_CENTS, published_taker_fee(size, price))
    return BAR_CONTEXT.divide(aggregate, size)


# The stored ladder is bounded at six levels a side, so a size those levels cannot fill is a
# censored read rather than a worse price: the walk past the sixth level is not on the tape.
# Ladder rows land on book updates, so a strike nobody quoted before the instant has no row at all.
# That reads the same as a one-sided book: no price, so it is counted rather than raised on.
def price_of(entry: StraddleEntry, table: pa.Table, sidecar: CloseSidecar) -> PricedStraddle:
    row = _row_at(table, entry.ticker, entry.entry_instant)
    if row is None:
        return PricedStraddle(
            entry_price=None,
            entry_fee_cents=None,
            entry_tick_cents=TICK_CENTS,
            net_profit_cents=None,
            size=SIZE,
            censored=False,
            priced=False,
            no_row=True,
        )
    censored = sum((Decimal(size) for size in row["no_sizes"]), Decimal(0)) < SIZE
    yes_bid = ticks(Decimal(row["yes_bid"]))
    no_bid = ticks(Decimal(row["no_bid"]))
    if not two_sided(
        yes_bid,
        no_bid,
        yes_depth=int(Decimal(row["yes_bid_depth"])),
        no_depth=int(Decimal(row["no_bid_depth"])),
    ):
        return PricedStraddle(
            entry_price=None,
            entry_fee_cents=None,
            entry_tick_cents=TICK_CENTS,
            net_profit_cents=None,
            size=SIZE,
            censored=censored,
            priced=False,
            no_row=False,
        )

    entry_price = Decimal(mid2(yes_bid, no_bid)) / (2 * PRICE_TICKS)
    fee = fee_cents_per_contract(SIZE, entry_price)
    settled = (
        Decimal(1) if sidecar.markets[entry.ticker].result == entry.settlement_side else Decimal(0)
    )
    gross = BAR_CONTEXT.multiply(_CENTS, BAR_CONTEXT.subtract(settled, entry_price))
    return PricedStraddle(
        entry_price=entry_price,
        entry_fee_cents=fee,
        entry_tick_cents=TICK_CENTS,
        net_profit_cents=BAR_CONTEXT.subtract(BAR_CONTEXT.subtract(gross, fee), TICK_CENTS),
        size=SIZE,
        censored=censored,
        priced=True,
        no_row=False,
    )


def price_counts(records: Sequence[PricedStraddle]) -> PriceCounts:
    return PriceCounts(
        n=len(records),
        one_sided_n=sum(1 for record in records if not record.priced and not record.no_row),
        no_row_n=sum(1 for record in records if record.no_row),
        censored_n=sum(1 for record in records if record.censored),
    )


def _row_at(table: pa.Table, ticker: str, instant: datetime) -> Mapping[str, object] | None:
    rows = table.filter(
        pc.and_(
            pc.equal(table.column("ticker"), ticker),
            pc.less_equal(table.column("received_at"), instant),
        )
    )
    if not rows.num_rows:
        return None
    ordered = rows.sort_by([("received_at", "ascending"), ("id", "ascending")])
    return ordered.slice(ordered.num_rows - 1, 1).to_pylist()[0]
