from __future__ import annotations

import asyncio
import decimal
import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

from bot.execution.paper import TradeIntent, TradeSide
from bot.execution.price_format import format_price_dollars
from bot.kalshi_client import KalshiDemoClient, KalshiOrderbook
from bot.markets.parser import parse_ticker

logger = logging.getLogger(__name__)

_MAX_CID_LEN = 64
_CID_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_MAX_429_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0
_BREAKER_WINDOW_SECONDS = 30.0
_BREAKER_PAUSE_SECONDS = 60.0
_BREAKER_THRESHOLD = 3

_breaker_state: dict[str, object] = {"recent_429": [], "paused_until": 0.0}


@dataclass(frozen=True, slots=True)
class DemoOrder:
    client_order_id: str
    exchange_order_id: str
    ticker: str
    side_kalshi: str
    requested_contracts: int
    filled_contracts: int
    requested_yes_price_dollars: Decimal
    avg_yes_fill_price_dollars: Decimal | None
    fee_dollars: Decimal
    status: str
    placed_at: datetime


@dataclass(frozen=True, slots=True)
class DemoOrderIdempotent:
    client_order_id: str
    exchange_order_id: str
    status: str


def parse_avg_yes_fill_price(raw: str | Decimal | None) -> Decimal | None:
    if raw is None:
        return None
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, str) and raw.strip() == "":
        return None
    try:
        return Decimal(raw)
    except decimal.InvalidOperation as err:
        raise ValueError(f"malformed avg_yes_fill_price: {raw!r}") from err


def _client_order_id(strategy: str, side: TradeSide, market_ticker: str, event_date: date) -> str:
    side_literal = "yes" if side is TradeSide.BUY_YES else "no"
    raw = f"kw-{strategy}-{side_literal}-{market_ticker}-{event_date.isoformat()}"
    sanitized = _CID_INVALID_CHARS.sub("-", raw)
    if len(sanitized) <= _MAX_CID_LEN:
        return sanitized
    digest = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()[:32]
    return f"kw-{digest}"


def _build_body(
    intent: TradeIntent,
    book: KalshiOrderbook,
    client_order_id: str,
) -> dict[str, object]:
    if intent.side is TradeSide.BUY_YES:
        side_kalshi = "yes"
        price = book.yes_ask
        price_key = "yes_price_dollars"
    else:
        side_kalshi = "no"
        price = book.no_ask
        price_key = "no_price_dollars"
    return {
        "ticker": intent.market_ticker,
        "side": side_kalshi,
        "action": "buy",
        "count": intent.contracts,
        "time_in_force": "immediate_or_cancel",
        price_key: format_price_dollars(price),
        "client_order_id": client_order_id,
    }


def _record_429(now_monotonic: float) -> None:
    recent: list[float] = _breaker_state["recent_429"]  # type: ignore[assignment]
    recent.append(now_monotonic)
    cutoff = now_monotonic - _BREAKER_WINDOW_SECONDS
    while recent and recent[0] < cutoff:
        recent.pop(0)
    if len(recent) >= _BREAKER_THRESHOLD:
        _breaker_state["paused_until"] = now_monotonic + _BREAKER_PAUSE_SECONDS


def _breaker_paused(now_monotonic: float) -> bool:
    return now_monotonic < _breaker_state["paused_until"]  # type: ignore[operator]


def _backoff_seconds(attempt: int) -> float:
    upper = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2**attempt))
    return random.uniform(0, upper)


def _parse_order(payload: dict[str, object], placed_at: datetime) -> DemoOrder:
    yes_price = parse_avg_yes_fill_price(payload.get("yes_price_dollars"))
    if yes_price is None:
        no_price = parse_avg_yes_fill_price(payload.get("no_price_dollars"))
        requested_yes_price = Decimal("1") - no_price if no_price is not None else Decimal("0")
    else:
        requested_yes_price = yes_price
    cid = str(payload["client_order_id"])
    side = str(payload["side"])
    fill_count = int(Decimal(str(payload["fill_count_fp"])))
    initial_count = int(Decimal(str(payload["initial_count_fp"])))
    taker_fee = parse_avg_yes_fill_price(payload.get("taker_fees_dollars")) or Decimal("0")
    maker_fee = parse_avg_yes_fill_price(payload.get("maker_fees_dollars")) or Decimal("0")
    fee_dollars = taker_fee + maker_fee
    taker_cost = parse_avg_yes_fill_price(payload.get("taker_fill_cost_dollars")) or Decimal("0")
    maker_cost = parse_avg_yes_fill_price(payload.get("maker_fill_cost_dollars")) or Decimal("0")
    fill_cost = taker_cost + maker_cost
    avg_yes_fill: Decimal | None
    if fill_count == 0:
        avg_yes_fill = None
    elif fill_cost == 0:
        avg_yes_fill = None
        logger.warning("demo_order_zero_cost_fill cid=%s fill_count=%d", cid, fill_count)
    else:
        per_contract = fill_cost / Decimal(fill_count)
        avg_yes_fill = Decimal("1") - per_contract if side == "no" else per_contract
    return DemoOrder(
        client_order_id=cid,
        exchange_order_id=str(payload.get("order_id") or payload.get("exchange_order_id") or ""),
        ticker=str(payload["ticker"]),
        side_kalshi=side,
        requested_contracts=initial_count,
        filled_contracts=fill_count,
        requested_yes_price_dollars=requested_yes_price,
        avg_yes_fill_price_dollars=avg_yes_fill,
        fee_dollars=fee_dollars,
        status=str(payload["status"]),
        placed_at=placed_at,
    )


_MALFORMED_CREATED_TIME_SENTINEL = datetime.max.replace(tzinfo=timezone.utc)


def _parse_created_time(raw: object) -> datetime:
    if not isinstance(raw, str) or not raw:
        return _MALFORMED_CREATED_TIME_SENTINEL
    text = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return _MALFORMED_CREATED_TIME_SENTINEL


async def _fetch_existing_by_cid(
    client: KalshiDemoClient, cid: str
) -> tuple[dict[str, object] | None, int]:
    response = await client.get_signed("/portfolio/orders", {"client_order_id": cid})
    if response.status_code != 200:
        return None, 0
    payload = response.json()
    orders = payload.get("orders") or []
    count = len(orders)
    if not orders:
        return None, 0
    fallback_count = sum(
        1
        for order in orders
        if _parse_created_time(order.get("created_time")) is _MALFORMED_CREATED_TIME_SENTINEL
    )
    if fallback_count >= 1:
        logger.warning(
            "demo_order_malformed_created_time cid=%s orders_count=%d fallback_count=%d",
            cid,
            count,
            fallback_count,
        )
    if fallback_count >= 2:
        return None, count
    newest = max(orders, key=lambda o: _parse_created_time(o.get("created_time")))
    return newest, count


async def place_order_demo(
    intent: TradeIntent,
    book: KalshiOrderbook,
    client: KalshiDemoClient,
    *,
    now: datetime,
) -> DemoOrder | DemoOrderIdempotent | None:
    parsed = parse_ticker(intent.market_ticker)
    cid = _client_order_id(intent.strategy, intent.side, intent.market_ticker, parsed.event_date)
    body = _build_body(intent, book, cid)

    if _breaker_paused(time.monotonic()):
        logger.warning("demo_order_breaker_paused cid=%s", cid)
        return None

    attempt = 0
    while True:
        response = await client.post_signed("/portfolio/orders", body)
        status = response.status_code
        if status == 201:
            payload = response.json()
            order_payload = payload.get("order") or payload
            return _parse_order(order_payload, placed_at=now)
        if status == 409:
            existing, existing_count = await _fetch_existing_by_cid(client, cid)
            if existing is None:
                logger.error("demo_order_409_without_existing cid=%s", cid)
                return None
            existing_status = str(existing.get("status", ""))
            existing_contracts = int(Decimal(str(existing.get("initial_count_fp") or "0")))
            if existing_status == "canceled":
                logger.info(
                    "demo_order_idempotent_canceled cid=%s prior_initial_count=%d",
                    cid,
                    existing_contracts,
                )
                return None
            if existing_contracts != intent.contracts:
                logger.info(
                    "demo_order_intent_drift cid=%s prior=%d new=%d status=%s count=%d",
                    cid,
                    existing_contracts,
                    intent.contracts,
                    existing_status,
                    existing_count,
                )
                return None
            logger.info("demo_order_idempotent_duplicate cid=%s", cid)
            existing_eid = str(existing.get("order_id") or existing.get("exchange_order_id") or "")
            return DemoOrderIdempotent(
                client_order_id=cid,
                exchange_order_id=existing_eid,
                status=existing_status,
            )
        if status == 429:
            _record_429(time.monotonic())
            attempt += 1
            if attempt >= _MAX_429_RETRIES:
                logger.info("demo_order_rate_limited cid=%s attempts=%d", cid, attempt)
                return None
            await asyncio.sleep(_backoff_seconds(attempt))
            continue
        body_text = response.text
        logger.warning("demo_order_rejected status=%d reason=%s", status, body_text)
        return None


async def cancel_order(client: KalshiDemoClient, order_id: str) -> bool:
    response = await client.delete_signed(f"/portfolio/orders/{order_id}")
    if response.status_code == 404:
        logger.info("demo_order_cancel_not_found order_id=%s", order_id)
        return False
    if 200 <= response.status_code < 300:
        return True
    logger.warning(
        "demo_order_cancel_rejected order_id=%s status=%d body=%s",
        order_id,
        response.status_code,
        response.text,
    )
    return False
