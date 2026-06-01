from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Numeric, bindparam, text
from sqlalchemy.orm import Session

from bot.kalshi_client import KalshiDemoClient
from bot.storage.sqlite import UtcDateTime

logger = logging.getLogger(__name__)


_REQUIRED_POSITION_FIELDS: tuple[str, ...] = (
    "market_exposure_dollars",
    "realized_pnl_dollars",
    "fees_paid_dollars",
)


@dataclass(frozen=True, slots=True)
class PositionAggregate:
    per_ticker_realized_pnl: dict[str, Decimal]
    total_exposure_dollars: Decimal
    realized_pnl_dollars: Decimal
    fees_paid_dollars: Decimal
    open_positions_count: int


async def aggregate_positions(client: KalshiDemoClient) -> PositionAggregate:
    params: dict[str, object] = {}
    per_ticker: dict[str, Decimal] = {}
    total_exposure = Decimal("0")
    total_realized = Decimal("0")
    total_fees = Decimal("0")
    open_count = 0
    cursor: str | None = None
    while True:
        if cursor:
            params["cursor"] = cursor
        response = await client.get_signed("/portfolio/positions", params)
        response.raise_for_status()
        payload = response.json()
        for raw in payload.get("market_positions") or []:
            for key in _REQUIRED_POSITION_FIELDS:
                if key not in raw or raw[key] is None:
                    raise KeyError(f"position payload missing required field {key!r}: {raw!r}")
            ticker = raw.get("ticker") or raw.get("market_ticker")
            if ticker is None:
                raise KeyError(f"position payload missing ticker / market_ticker: {raw!r}")
            exposure = Decimal(str(raw["market_exposure_dollars"]))
            realized = Decimal(str(raw["realized_pnl_dollars"]))
            fees = Decimal(str(raw["fees_paid_dollars"]))
            position_fp_raw = raw.get("position_fp")
            if position_fp_raw is None:
                raise KeyError(f"position payload missing position_fp: {raw!r}")
            position_fp = int(Decimal(str(position_fp_raw)))
            total_exposure += exposure
            total_realized += realized
            total_fees += fees
            if position_fp != 0:
                open_count += 1
            per_ticker[str(ticker)] = realized
        cursor = payload.get("cursor") or None
        if not cursor:
            return PositionAggregate(
                per_ticker_realized_pnl=per_ticker,
                total_exposure_dollars=total_exposure,
                realized_pnl_dollars=total_realized,
                fees_paid_dollars=total_fees,
                open_positions_count=open_count,
            )


_UPDATE_DEMO_REALIZED_PNL_SQL = text(
    """
    UPDATE demo_orders
       SET realized_pnl_dollars = :pnl
     WHERE id = (
         SELECT id FROM demo_orders
          WHERE market_ticker = :ticker
            AND status = 'executed'
            AND filled_contracts > 0
            AND client_order_id NOT LIKE 'kw-backfill-%'
            AND placed_at <= :snapshot_at
          ORDER BY placed_at DESC, id DESC
          LIMIT 1
     )
    """
).bindparams(
    bindparam("pnl", type_=Numeric(10, 6)),
    bindparam("snapshot_at", type_=UtcDateTime()),
)


def update_demo_realized_pnl(
    session: Session,
    market_ticker: str,
    realized_pnl_dollars: Decimal,
    snapshot_at: datetime,
) -> int:
    result = session.execute(
        _UPDATE_DEMO_REALIZED_PNL_SQL,
        {"pnl": realized_pnl_dollars, "ticker": market_ticker, "snapshot_at": snapshot_at},
    )
    rowcount = result.rowcount
    if rowcount > 0:
        session.expire_all()
    else:
        logger.info(
            "snapshot_pnl_unattributed ticker=%s pnl=%s reason=no_eligible_row",
            market_ticker,
            realized_pnl_dollars,
        )
    return rowcount
