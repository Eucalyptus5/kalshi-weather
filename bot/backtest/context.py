from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from bot.backtest.normalize import CanonicalSnapshot
from bot.forecast.cdf import EnsembleCDF
from bot.markets.observation_window import observation_window
from bot.markets.parser import ParsedTicker, parse_ticker
from bot.strategy.edge import DIRECTION_CUSHION, EdgeContext
from bot.strategy.sizing import sigma_t_median_for_lead
from bot.strategy.tails import TailsContext


@dataclass(frozen=True, slots=True)
class BacktestBudgets:
    event_budget_remaining: Decimal
    market_budget_remaining: Decimal


def _fair_yes(parsed: ParsedTicker, cdf: EnsembleCDF) -> Decimal:
    if parsed.kind == "bracket":
        lo = float(parsed.strikes[0])
        hi = float(parsed.strikes[1])
        return Decimal(str(cdf.prob_range(lo, hi)))
    if parsed.kind == "above":
        return Decimal(str(1.0 - cdf.cdf(float(parsed.strikes[0]))))
    return Decimal(str(cdf.cdf(float(parsed.strikes[0]))))


def _is_same_day(parsed: ParsedTicker, station_tz: str, now: datetime) -> bool:
    start_utc, end_utc = observation_window(station_tz, parsed.event_date)
    return start_utc <= now < end_utc


def _lead_hours(close_time: datetime, now: datetime) -> int:
    return int((close_time - now).total_seconds() / 3600)


def build_tails_context(
    snap: CanonicalSnapshot,
    cdf: EnsembleCDF,
    as_of: datetime,
    bankroll: Decimal,
    budgets: BacktestBudgets,
    *,
    ensemble_spread: Decimal,
    depth_at_price: int,
    station_tz: str,
) -> TailsContext:
    if snap.close_time is None:
        raise ValueError(f"close_time required to build tails context: {snap.ticker}")
    parsed = parse_ticker(snap.ticker)
    fair_yes = _fair_yes(parsed, cdf)
    is_same_day = _is_same_day(parsed, station_tz, as_of)
    sigma_T_median = sigma_t_median_for_lead(_lead_hours(snap.close_time, as_of))
    return TailsContext(
        yes_ask=snap.yes_ask,
        yes_bid=snap.yes_bid,
        no_bid=snap.no_bid,
        fair_yes=fair_yes,
        close_time=snap.close_time,
        now=as_of,
        bankroll=bankroll,
        is_same_day=is_same_day,
        ensemble_spread=ensemble_spread,
        sigma_T_median=sigma_T_median,
        event_budget_remaining=budgets.event_budget_remaining,
        market_budget_remaining=budgets.market_budget_remaining,
        depth_at_price=depth_at_price,
        price_per_contract=Decimal("1") - snap.yes_bid,
    )


def build_edge_context(
    snap: CanonicalSnapshot,
    cdf: EnsembleCDF,
    as_of: datetime,
    bankroll: Decimal,
    budgets: BacktestBudgets,
    *,
    ensemble_spread: Decimal,
    buy_yes_depth: int,
    sell_yes_depth: int,
    station_tz: str,
    is_blacklisted: bool = False,
    nbm_divergence: Decimal | None = None,
) -> EdgeContext:
    if snap.close_time is None:
        raise ValueError(f"close_time required to build edge context: {snap.ticker}")
    parsed = parse_ticker(snap.ticker)
    fair_yes = _fair_yes(parsed, cdf)
    is_same_day = _is_same_day(parsed, station_tz, as_of)
    sigma_T_median = sigma_t_median_for_lead(_lead_hours(snap.close_time, as_of))

    if fair_yes > snap.yes_ask + DIRECTION_CUSHION:
        depth = buy_yes_depth
        price_per_contract = snap.yes_ask
    elif fair_yes < snap.yes_bid - DIRECTION_CUSHION:
        depth = sell_yes_depth
        price_per_contract = Decimal("1") - snap.yes_bid
    else:
        depth = buy_yes_depth
        price_per_contract = snap.yes_ask

    return EdgeContext(
        yes_ask=snap.yes_ask,
        yes_bid=snap.yes_bid,
        fair_yes=fair_yes,
        ensemble_spread=ensemble_spread,
        bankroll=bankroll,
        is_same_day=is_same_day,
        is_blacklisted=is_blacklisted,
        nbm_divergence=nbm_divergence,
        sigma_T_median=sigma_T_median,
        event_budget_remaining=budgets.event_budget_remaining,
        market_budget_remaining=budgets.market_budget_remaining,
        depth_at_price=depth,
        price_per_contract=price_per_contract,
    )
