from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from bot.markets.parser import ParsedTicker, event_id, parse_ticker, series_id
from bot.storage.sqlite import PaperTradeRow, SimulatedPnl

logger = logging.getLogger(__name__)


SETTLEMENT_GRACE_DAYS: int = 7


def _effective_cutoff_date(parsed: ParsedTicker) -> date:
    if not parsed.is_monthly:
        return parsed.event_date
    _, last_day = calendar.monthrange(parsed.event_date.year, parsed.event_date.month)
    return date(parsed.event_date.year, parsed.event_date.month, last_day)


def open_exposures(
    session: Session,
    *,
    now: datetime | None = None,
    grace_days: int = SETTLEMENT_GRACE_DAYS,
) -> tuple[dict[str, Decimal], dict[str, Decimal], dict[str, Decimal]]:
    if now is None:
        now = datetime.now(tz=_timezone.utc)
    cutoff = (now - timedelta(days=grace_days)).date()

    stmt = (
        select(PaperTradeRow)
        .outerjoin(SimulatedPnl, SimulatedPnl.paper_trade_id == PaperTradeRow.id)
        .where(SimulatedPnl.id.is_(None))
    )
    rows = session.scalars(stmt).all()

    by_market: dict[str, Decimal] = {}
    by_event: dict[str, Decimal] = {}
    by_series: dict[str, Decimal] = {}

    for row in rows:
        # paper_trades is persisted state written by prior parser versions (see commit 711f810); historical rows are not guaranteed to parse under the current parser.
        try:
            parsed = parse_ticker(row.market_ticker)
        except ValueError as err:
            logger.warning(
                "open_exposures_skip_unparseable ticker=%s paper_trade_id=%s err=%s",
                row.market_ticker,
                row.id,
                err,
            )
            continue

        if _effective_cutoff_date(parsed) < cutoff:
            continue

        contracts = Decimal(row.contracts)
        price = row.simulated_price
        if row.side == "buy_yes":
            max_loss = price * contracts
        else:
            max_loss = (Decimal("1") - price) * contracts

        market_key = row.market_ticker
        event_key = event_id(row.market_ticker)
        series_key = series_id(row.market_ticker)

        by_market[market_key] = by_market.get(market_key, Decimal("0")) + max_loss
        by_event[event_key] = by_event.get(event_key, Decimal("0")) + max_loss
        by_series[series_key] = by_series.get(series_key, Decimal("0")) + max_loss

    return by_market, by_event, by_series
