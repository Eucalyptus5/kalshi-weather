from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from bot.execution.fees import taker_fee


logger = logging.getLogger(__name__)


MARKET_REFRESH_INTERVAL_SECONDS = 60.0
STALE_ORDERBOOK_THRESHOLD_SECONDS = MARKET_REFRESH_INTERVAL_SECONDS * 4
STALE_SKIP_RATIO_WARN_THRESHOLD = 0.5
STALE_SKIP_MIN_INTENTS_PER_SERIES = 4


class TradeSide(Enum):
    BUY_YES = "buy_yes"
    SELL_YES = "sell_yes"


@dataclass(frozen=True, slots=True)
class Orderbook:
    yes_ask: Decimal
    yes_bid: Decimal
    yes_ask_depth: int
    yes_bid_depth: int
    snapshot_at: datetime


@dataclass(frozen=True, slots=True)
class TradeIntent:
    market_ticker: str
    side: TradeSide
    contracts: int
    fair_yes: Decimal
    strategy: str
    ensemble_spread_sigma_t: Decimal | None = None
    lead_time_hours: Decimal | None = None
    nbm_divergence: Decimal | None = None


@dataclass(frozen=True, slots=True)
class PaperTrade:
    intended_at: datetime
    market_ticker: str
    side: TradeSide
    contracts: int
    simulated_price: Decimal
    fee_dollars: Decimal
    fair_at_entry: Decimal
    strategy: str
    attempted_contracts: int = 0
    ensemble_spread_sigma_t: Decimal | None = None
    lead_time_hours: Decimal | None = None
    nbm_divergence: Decimal | None = None


def simulate_taker_fill(intent: TradeIntent, book: Orderbook, now: datetime) -> PaperTrade | None:
    if intent.contracts <= 0:
        raise ValueError(f"contracts must be > 0, got {intent.contracts}")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if book.snapshot_at.tzinfo is None:
        raise ValueError("snapshot_at must be timezone-aware")

    age_seconds = (now - book.snapshot_at).total_seconds()
    if age_seconds > STALE_ORDERBOOK_THRESHOLD_SECONDS:
        logger.info(
            "paper_trade_stale_orderbook ticker=%s age_seconds=%.1f",
            intent.market_ticker,
            age_seconds,
        )
        return None

    if intent.side is TradeSide.BUY_YES:
        simulated_price = book.yes_ask
        displayed_depth = book.yes_ask_depth
    else:
        simulated_price = book.yes_bid
        displayed_depth = book.yes_bid_depth

    if displayed_depth <= 0:
        logger.info(
            "paper_trade_zero_depth ticker=%s side=%s",
            intent.market_ticker,
            intent.side.value,
        )
        return None

    filled_contracts = min(intent.contracts, displayed_depth)
    fee_dollars = taker_fee(filled_contracts, simulated_price)

    return PaperTrade(
        intended_at=now,
        market_ticker=intent.market_ticker,
        side=intent.side,
        contracts=filled_contracts,
        simulated_price=simulated_price,
        fee_dollars=fee_dollars,
        fair_at_entry=intent.fair_yes,
        strategy=intent.strategy,
        attempted_contracts=intent.contracts,
        ensemble_spread_sigma_t=intent.ensemble_spread_sigma_t,
        lead_time_hours=intent.lead_time_hours,
        nbm_divergence=intent.nbm_divergence,
    )


def log_stale_skip_ratio(
    stale_skips_by_series: dict[str, int],
    intents_seen_by_series: dict[str, int],
) -> None:
    total_skips = sum(stale_skips_by_series.values())
    total_intents = sum(intents_seen_by_series.values())
    if total_intents > 0:
        ratio = total_skips / total_intents
        if ratio > STALE_SKIP_RATIO_WARN_THRESHOLD:
            logger.warning(
                "stale_skip_ratio_high stale_skips=%d intents=%d ratio=%.2f",
                total_skips,
                total_intents,
                ratio,
            )
    for series, intents in intents_seen_by_series.items():
        if intents < STALE_SKIP_MIN_INTENTS_PER_SERIES:
            continue
        skips = stale_skips_by_series.get(series, 0)
        per_series_ratio = skips / intents
        if per_series_ratio > STALE_SKIP_RATIO_WARN_THRESHOLD:
            logger.warning(
                "stale_skip_ratio_high_series series=%s stale_skips=%d intents=%d ratio=%.2f",
                series,
                skips,
                intents,
                per_series_ratio,
            )
