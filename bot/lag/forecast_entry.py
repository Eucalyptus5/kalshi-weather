from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TypeVar

from bot.lag.fee_floor import BAR_CONTEXT, TICK_CENTS, published_taker_fee
from bot.lag.forecast_sample import SampleLeg


# The size is the spent window's no-side touch-depth median, stated from outside: this sample
# carries no book, and a size read off the rows the effect is computed on would not be external.
SIZE: Decimal = Decimal(26)
YES = "yes"
NO = "no"
SCREEN_RULE = "trailing_contracts_ge_26"
TICK_RULE = "one_tick_constant"
WALKED_TICK_RULE = "one_tick_constant_entry_walked_one_tick"
WALKED_TICK_REPORTED_ONLY = (
    "this figure is reported only: it gates nothing, carries no multiplicity correction, and is "
    "not evidence of an edge"
)

_CENTS = Decimal(100)
_TICK_DOLLARS = BAR_CONTEXT.divide(TICK_CENTS, _CENTS)

_Depth = TypeVar("_Depth", int, Decimal)


@dataclass(frozen=True, slots=True, kw_only=True)
class TradedLeg:
    ticker: str
    event_date: date
    lead_hours: int
    split: str
    probability: Decimal
    entry_price: Decimal
    side: str | None
    executed_price: Decimal | None
    entry_fee_cents: Decimal | None
    entry_tick_cents: Decimal
    net_profit_cents: Decimal | None
    walked_executed_price: Decimal | None
    walked_entry_fee_cents: Decimal | None
    walked_net_profit_cents: Decimal | None
    size: Decimal
    traded: bool
    trailing_prints: int
    trailing_contracts: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class EntryCounts:
    n: int
    traded_n: int
    untraded_n: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DepthScreen:
    candidates: int
    kept: int
    dropped: int
    event_days: int
    event_days_kept: int

    @property
    def event_days_lost(self) -> Decimal:
        return BAR_CONTEXT.divide(
            Decimal(self.event_days - self.event_days_kept), Decimal(self.event_days)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DepthDistribution:
    prints_p10: int
    prints_p25: int
    prints_p50: int
    prints_p75: int
    prints_p90: int
    prints_max: int
    prints_at_zero: int
    prints_below_size: int
    contracts_p10: Decimal
    contracts_p25: Decimal
    contracts_p50: Decimal
    contracts_p75: Decimal
    contracts_p90: Decimal
    contracts_max: Decimal
    contracts_at_zero: int
    contracts_below_size: int


def quantile(values: Sequence[_Depth], p: Decimal) -> _Depth:
    return sorted(values)[int(p * (len(values) - 1))]


# The published formula prices a whole order in dollars while the statistic is stated in cents per
# contract, so the size the order was priced at divides back out here.
def fee_cents_per_contract(size: Decimal, price: Decimal) -> Decimal:
    aggregate = BAR_CONTEXT.multiply(_CENTS, published_taker_fee(size, price))
    return BAR_CONTEXT.divide(aggregate, size)


def entry_side(probability: Decimal, price: Decimal) -> str | None:
    if probability > price:
        return YES
    if probability < price:
        return NO
    return None


# The walked price is not clamped: the frozen sample tops out at 0.99, so the walk lands at 1.00 at
# worst, and the published fee at 1.00 is zero rather than undefined.
def entry_of(leg: SampleLeg, probability: Decimal) -> TradedLeg:
    side = entry_side(probability, leg.entry_price)
    if side is None:
        return TradedLeg(
            ticker=leg.ticker,
            event_date=leg.event_date,
            lead_hours=leg.lead_hours,
            split=leg.split,
            probability=probability,
            entry_price=leg.entry_price,
            side=None,
            executed_price=None,
            entry_fee_cents=None,
            entry_tick_cents=TICK_CENTS,
            net_profit_cents=None,
            walked_executed_price=None,
            walked_entry_fee_cents=None,
            walked_net_profit_cents=None,
            size=SIZE,
            traded=False,
            trailing_prints=leg.trailing_prints,
            trailing_contracts=leg.trailing_contracts,
        )
    executed = leg.entry_price if side == YES else BAR_CONTEXT.subtract(Decimal(1), leg.entry_price)
    walked = BAR_CONTEXT.add(executed, _TICK_DOLLARS)
    settled = Decimal(1) if leg.result == side else Decimal(0)
    fee = fee_cents_per_contract(SIZE, executed)
    walked_fee = fee_cents_per_contract(SIZE, walked)
    return TradedLeg(
        ticker=leg.ticker,
        event_date=leg.event_date,
        lead_hours=leg.lead_hours,
        split=leg.split,
        probability=probability,
        entry_price=leg.entry_price,
        side=side,
        executed_price=executed,
        entry_fee_cents=fee,
        entry_tick_cents=TICK_CENTS,
        net_profit_cents=_net_cents(settled, executed, fee),
        walked_executed_price=walked,
        walked_entry_fee_cents=walked_fee,
        walked_net_profit_cents=_net_cents(settled, walked, walked_fee),
        size=SIZE,
        traded=True,
        trailing_prints=leg.trailing_prints,
        trailing_contracts=leg.trailing_contracts,
    )


def entry_counts(rows: Sequence[TradedLeg]) -> EntryCounts:
    traded = sum(1 for row in rows if row.traded)
    return EntryCounts(n=len(rows), traded_n=traded, untraded_n=len(rows) - traded)


def depth_ok(leg: SampleLeg) -> bool:
    return leg.trailing_contracts >= SIZE


def screen_depth(legs: Sequence[SampleLeg]) -> DepthScreen:
    kept = [leg for leg in legs if depth_ok(leg)]
    return DepthScreen(
        candidates=len(legs),
        kept=len(kept),
        dropped=len(legs) - len(kept),
        event_days=len({leg.event_date for leg in legs}),
        event_days_kept=len({leg.event_date for leg in kept}),
    )


def depth_distribution(legs: Sequence[SampleLeg]) -> DepthDistribution:
    prints = [leg.trailing_prints for leg in legs]
    contracts = [leg.trailing_contracts for leg in legs]
    return DepthDistribution(
        prints_p10=quantile(prints, Decimal("0.10")),
        prints_p25=quantile(prints, Decimal("0.25")),
        prints_p50=quantile(prints, Decimal("0.50")),
        prints_p75=quantile(prints, Decimal("0.75")),
        prints_p90=quantile(prints, Decimal("0.90")),
        prints_max=max(prints),
        prints_at_zero=sum(1 for value in prints if value == 0),
        prints_below_size=sum(1 for value in prints if value < SIZE),
        contracts_p10=quantile(contracts, Decimal("0.10")),
        contracts_p25=quantile(contracts, Decimal("0.25")),
        contracts_p50=quantile(contracts, Decimal("0.50")),
        contracts_p75=quantile(contracts, Decimal("0.75")),
        contracts_p90=quantile(contracts, Decimal("0.90")),
        contracts_max=max(contracts),
        contracts_at_zero=sum(1 for value in contracts if value == 0),
        contracts_below_size=sum(1 for value in contracts if value < SIZE),
    )


def _net_cents(settled: Decimal, price: Decimal, fee: Decimal) -> Decimal:
    gross = BAR_CONTEXT.multiply(_CENTS, BAR_CONTEXT.subtract(settled, price))
    return BAR_CONTEXT.subtract(BAR_CONTEXT.subtract(gross, fee), TICK_CENTS)
