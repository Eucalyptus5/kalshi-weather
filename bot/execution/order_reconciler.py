from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from bot.execution.demo_row_translator import _paper_trade_row_from_demo_row, paper_trade_row_values
from bot.execution.order_placer import DemoOrder, _parse_order, parse_avg_yes_fill_price
from bot.kalshi_client import KalshiDemoClient
from bot.storage.sqlite import DemoOrder as DemoOrderRow
from bot.storage.sqlite import PaperTradeRow

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"executed", "canceled"})


@dataclass(frozen=True, slots=True)
class DemoFill:
    fill_id: str
    order_id: str
    ticker: str
    outcome_side: str
    book_side: str
    count: int
    yes_price_dollars: Decimal | None
    no_price_dollars: Decimal | None
    is_taker: bool
    created_time: str
    fee_cost: Decimal


@dataclass(frozen=True, slots=True)
class KalshiPosition:
    ticker: str
    position_fp: int


def _now() -> datetime:
    return datetime.now(tz=_timezone.utc)


async def poll_open_orders(
    client: KalshiDemoClient,
    watermark: datetime | None,
    *,
    ticker: str | None = None,
) -> list[DemoOrder]:
    params: dict[str, object] = {"status": "executed,canceled"}
    if watermark is not None:
        params["min_ts"] = int(watermark.timestamp())
    if ticker is not None:
        params["ticker"] = ticker
    out: list[DemoOrder] = []
    cursor: str | None = None
    while True:
        if cursor:
            params["cursor"] = cursor
        response = await client.get_signed("/portfolio/orders", params)
        response.raise_for_status()
        payload = response.json()
        for raw in payload.get("orders") or []:
            out.append(_parse_order(raw, placed_at=_now()))
        cursor = payload.get("cursor") or None
        if not cursor:
            return out


async def poll_positions(client: KalshiDemoClient) -> list[KalshiPosition]:
    params: dict[str, object] = {}
    out: list[KalshiPosition] = []
    cursor: str | None = None
    while True:
        if cursor:
            params["cursor"] = cursor
        response = await client.get_signed("/portfolio/positions", params)
        response.raise_for_status()
        payload = response.json()
        raw_positions = payload.get("market_positions") or payload.get("positions") or []
        for raw in raw_positions:
            ticker = raw.get("ticker") or raw.get("market_ticker")
            if ticker is None:
                raise KeyError(f"position payload missing ticker / market_ticker: {raw!r}")
            raw_pos = raw.get("position_fp")
            if raw_pos is None:
                raw_pos = raw.get("position")
            if raw_pos is None:
                raise KeyError(f"position payload missing position_fp / position: {raw!r}")
            out.append(KalshiPosition(ticker=str(ticker), position_fp=int(Decimal(str(raw_pos)))))
        cursor = payload.get("cursor") or None
        if not cursor:
            return out


def local_position_aggregate(session: Session, ticker: str) -> int:
    rows = session.scalars(select(DemoOrderRow).where(DemoOrderRow.market_ticker == ticker)).all()
    total = 0
    for row in rows:
        if row.side == "yes":
            total += row.filled_contracts
        elif row.side == "no":
            total -= row.filled_contracts
    return total


async def poll_fills(
    client: KalshiDemoClient,
    watermark: datetime | None,
    *,
    ticker: str | None = None,
) -> list[DemoFill]:
    params: dict[str, object] = {}
    if watermark is not None:
        params["min_ts"] = int(watermark.timestamp())
    if ticker is not None:
        params["ticker"] = ticker
    out: list[DemoFill] = []
    cursor: str | None = None
    while True:
        if cursor:
            params["cursor"] = cursor
        response = await client.get_signed("/portfolio/fills", params)
        response.raise_for_status()
        payload = response.json()
        for raw in payload.get("fills") or []:
            out.append(_parse_fill(raw))
        cursor = payload.get("cursor") or None
        if not cursor:
            return out


def _parse_fill(raw: dict[str, object]) -> DemoFill:
    return DemoFill(
        fill_id=str(raw["fill_id"]),
        order_id=str(raw["order_id"]),
        ticker=str(raw["ticker"]),
        outcome_side=str(raw["outcome_side"]),
        book_side=str(raw["book_side"]),
        count=int(Decimal(str(raw.get("count_fp") or raw.get("count") or "0"))),
        yes_price_dollars=parse_avg_yes_fill_price(raw.get("yes_price_dollars")),
        no_price_dollars=parse_avg_yes_fill_price(raw.get("no_price_dollars")),
        is_taker=bool(raw.get("is_taker")),
        created_time=str(raw.get("created_time", "")),
        fee_cost=parse_avg_yes_fill_price(raw.get("fee_cost")) or Decimal("0"),
    )


def upsert_exchange_record(session: Session, record: DemoOrder) -> None:
    now = _now()
    cid = record.client_order_id
    by_cid = (
        session.scalars(
            select(DemoOrderRow).where(DemoOrderRow.client_order_id == cid)
        ).one_or_none()
        if cid
        else None
    )
    if by_cid is not None:
        if (
            record.exchange_order_id is not None
            and by_cid.exchange_order_id is not None
            and by_cid.exchange_order_id != record.exchange_order_id
        ):
            raise ValueError(
                f"exchange_order_id collision for client_order_id {cid!r}: "
                f"row has {by_cid.exchange_order_id!r}, record has {record.exchange_order_id!r}"
            )
        if record.exchange_order_id and not by_cid.exchange_order_id:
            by_cid.exchange_order_id = record.exchange_order_id
        by_cid.status = record.status
        by_cid.filled_contracts = record.filled_contracts
        by_cid.avg_fill_price = record.avg_yes_fill_price_dollars
        by_cid.fee_dollars = record.fee_dollars
        by_cid.last_status_at = now
        return

    by_eid = (
        session.scalars(
            select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == record.exchange_order_id)
        ).one_or_none()
        if record.exchange_order_id
        else None
    )
    if by_eid is not None:
        by_eid.status = record.status
        by_eid.filled_contracts = record.filled_contracts
        by_eid.avg_fill_price = record.avg_yes_fill_price_dollars
        by_eid.fee_dollars = record.fee_dollars
        by_eid.last_status_at = now
        return

    session.add(
        DemoOrderRow(
            client_order_id=f"kw-backfill-{record.exchange_order_id}",
            exchange_order_id=record.exchange_order_id,
            market_ticker=record.ticker,
            strategy=None,
            side=record.side_kalshi,
            requested_contracts=record.requested_contracts,
            filled_contracts=record.filled_contracts,
            requested_yes_price_dollars=None,
            fair_at_entry=None,
            intended_at=None,
            avg_fill_price=record.avg_yes_fill_price_dollars,
            fee_dollars=record.fee_dollars,
            realized_pnl_dollars=None,
            status=record.status,
            placed_at=record.placed_at,
            last_status_at=now,
        )
    )


def idempotent_insert_demo_order_row(
    session: Session,
    *,
    client_order_id: str,
    exchange_order_id: str | None,
    market_ticker: str,
    strategy: str,
    side: str,
    requested_contracts: int,
    filled_contracts: int,
    requested_yes_price_dollars: Decimal | None,
    fair_at_entry: Decimal | None,
    q_raw: Decimal | None,
    intended_at: datetime,
    status: str,
    placed_at: datetime,
) -> None:
    now = _now()
    by_cid = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.client_order_id == client_order_id)
    ).one_or_none()
    if by_cid is not None:
        if exchange_order_id and not by_cid.exchange_order_id:
            by_cid.exchange_order_id = exchange_order_id
        by_cid.status = status
        by_cid.last_status_at = now
        return

    by_eid = (
        session.scalars(
            select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == exchange_order_id)
        ).one_or_none()
        if exchange_order_id
        else None
    )
    if by_eid is not None:
        if not by_eid.client_order_id.startswith("kw-backfill-"):
            raise ValueError(
                f"exchange_order_id {exchange_order_id!r} bound to unrelated "
                f"client_order_id {by_eid.client_order_id!r}"
            )
        by_eid.client_order_id = client_order_id
        by_eid.strategy = strategy
        by_eid.fair_at_entry = fair_at_entry
        by_eid.q_raw = q_raw
        by_eid.intended_at = intended_at
        by_eid.requested_contracts = requested_contracts
        by_eid.requested_yes_price_dollars = requested_yes_price_dollars
        by_eid.side = side
        by_eid.status = status
        by_eid.last_status_at = now
        return

    session.add(
        DemoOrderRow(
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            market_ticker=market_ticker,
            strategy=strategy,
            side=side,
            requested_contracts=requested_contracts,
            filled_contracts=filled_contracts,
            requested_yes_price_dollars=requested_yes_price_dollars,
            fair_at_entry=fair_at_entry,
            q_raw=q_raw,
            intended_at=intended_at,
            avg_fill_price=None,
            fee_dollars=None,
            realized_pnl_dollars=None,
            status=status,
            placed_at=placed_at,
            last_status_at=now,
        )
    )


def stitch_natural_key_order(
    session: Session,
    *,
    exchange_order_id: str,
    client_order_id: str,
    strategy: str,
    side: str,
    fair_at_entry: Decimal,
    q_raw: Decimal,
    intended_at: datetime,
    requested_yes_price_dollars: Decimal,
) -> None:
    matched = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == exchange_order_id)
    ).one_or_none()
    if matched is None:
        raise ValueError(f"no demo_orders row for exchange_order_id {exchange_order_id!r}")

    holder = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.client_order_id == client_order_id)
    ).one_or_none()

    if holder is None or holder.id == matched.id:
        matched.client_order_id = client_order_id
        matched.strategy = strategy
        matched.side = side
        matched.fair_at_entry = fair_at_entry
        matched.q_raw = q_raw
        matched.intended_at = intended_at
        matched.requested_yes_price_dollars = requested_yes_price_dollars
        survivor = matched
    elif holder.exchange_order_id is None:
        captured_eid = matched.exchange_order_id
        captured_filled = matched.filled_contracts
        captured_status = matched.status
        captured_avg = matched.avg_fill_price
        captured_fee = matched.fee_dollars
        captured_last = matched.last_status_at
        session.delete(matched)
        session.flush()
        holder.exchange_order_id = captured_eid
        holder.filled_contracts = captured_filled
        holder.status = captured_status
        holder.avg_fill_price = captured_avg
        holder.fee_dollars = captured_fee
        holder.last_status_at = captured_last
        holder.strategy = strategy
        holder.side = side
        holder.fair_at_entry = fair_at_entry
        holder.q_raw = q_raw
        holder.intended_at = intended_at
        holder.requested_yes_price_dollars = requested_yes_price_dollars
        survivor = holder
    else:
        raise ValueError(
            "natural-key collision: existing row already bound to a different exchange_order_id"
        )

    session.flush()
    row = _paper_trade_row_from_demo_row(survivor, _now())
    if row is not None:
        session.execute(
            sqlite_insert(PaperTradeRow)
            .values(**paper_trade_row_values(row))
            .on_conflict_do_nothing(index_elements=["demo_order_client_id"])
        )


def reconcile_fills_into_demo_orders(
    session: Session, fills: list[DemoFill], orders: list[DemoOrder]
) -> int:
    status_by_eid = {o.exchange_order_id: o.status for o in orders}
    by_order: dict[str, list[DemoFill]] = {}
    for fill in fills:
        by_order.setdefault(fill.order_id, []).append(fill)

    now = _now()
    processed = 0
    for eid, order_fills in by_order.items():
        row = session.scalars(
            select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == eid)
        ).one_or_none()
        if row is None:
            continue
        filled = sum(f.count for f in order_fills)
        fee = sum((f.fee_cost for f in order_fills), Decimal("0"))
        last = order_fills[-1]
        row.filled_contracts = filled
        row.avg_fill_price = last.yes_price_dollars if row.side == "yes" else last.no_price_dollars
        row.fee_dollars = fee
        row.last_status_at = now
        if eid in status_by_eid:
            row.status = status_by_eid[eid]
        elif filled >= row.requested_contracts:
            row.status = "executed"
        processed += 1

        pt = _paper_trade_row_from_demo_row(row, now)
        if pt is not None:
            session.execute(
                sqlite_insert(PaperTradeRow)
                .values(**paper_trade_row_values(pt))
                .on_conflict_do_nothing(index_elements=["demo_order_client_id"])
            )
    return processed
