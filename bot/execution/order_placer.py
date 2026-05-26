from __future__ import annotations

import asyncio
import decimal
import hashlib
import logging
import random
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from bot.execution.paper import TradeIntent, TradeSide
from bot.execution.price_format import format_price_dollars
from bot.kalshi_client import KalshiDemoClient, KalshiOrderbook
from bot.markets.parser import parse_ticker

logger = logging.getLogger(__name__)

_MAX_CID_LEN = 64
_MAX_429_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0
_BREAKER_WINDOW_SECONDS = 30.0
_BREAKER_PAUSE_SECONDS = 60.0
_BREAKER_THRESHOLD = 3

_breaker_state: dict[str, object] = {"recent_429": [], "paused_until": 0.0}

_capture_next_400 = True
_capture_next_201 = True


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
    if len(raw) <= _MAX_CID_LEN:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
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
        "type": "limit",
        "time_in_force": "immediate_or_cancel",
        price_key: format_price_dollars(price),
        "client_order_id": client_order_id,
        "post_only": False,
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
    fee = parse_avg_yes_fill_price(payload.get("fee_dollars"))
    return DemoOrder(
        client_order_id=str(payload["client_order_id"]),
        exchange_order_id=str(payload.get("order_id") or payload.get("exchange_order_id") or ""),
        ticker=str(payload["ticker"]),
        side_kalshi=str(payload["side"]),
        requested_contracts=int(payload.get("requested_contracts", 0)),
        filled_contracts=int(payload.get("filled_contracts", 0)),
        requested_yes_price_dollars=requested_yes_price,
        avg_yes_fill_price_dollars=parse_avg_yes_fill_price(
            payload.get("avg_yes_fill_price_dollars")
        ),
        fee_dollars=fee if fee is not None else Decimal("0"),
        status=str(payload["status"]),
        placed_at=placed_at,
    )


async def _fetch_existing_by_cid(client: KalshiDemoClient, cid: str) -> dict[str, object] | None:
    response = await client.get_signed("/portfolio/orders", {"client_order_id": cid})
    if response.status_code != 200:
        return None
    payload = response.json()
    orders = payload.get("orders") or []
    if not orders:
        return None
    return orders[0]


async def place_order_demo(
    intent: TradeIntent,
    book: KalshiOrderbook,
    client: KalshiDemoClient,
    *,
    now: datetime,
) -> DemoOrder | None:
    global _capture_next_201, _capture_next_400
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
            if _capture_next_201:
                logger.warning(
                    "demo_order_request_payload body=%s yes_ask=%s no_ask=%s",
                    body,
                    book.yes_ask,
                    book.no_ask,
                )
                logger.warning("demo_order_response_body status=%d body=%s", status, response.text)
                _capture_next_201 = False
            order_payload = payload.get("order") or payload
            return _parse_order(order_payload, placed_at=now)
        if status == 409:
            existing = await _fetch_existing_by_cid(client, cid)
            if existing is None:
                logger.error("demo_order_409_without_existing cid=%s", cid)
                return None
            existing_status = str(existing.get("status", ""))
            existing_contracts = int(existing.get("requested_contracts", 0))
            if existing_status == "canceled" and existing_contracts != intent.contracts:
                logger.info(
                    "demo_order_intent_drift cid=%s prior=%d new=%d status=canceled",
                    cid,
                    existing_contracts,
                    intent.contracts,
                )
                return None
            logger.info("demo_order_idempotent_duplicate cid=%s", cid)
            return _parse_order(existing, placed_at=now)
        if status == 429:
            _record_429(time.monotonic())
            attempt += 1
            if attempt >= _MAX_429_RETRIES:
                logger.info("demo_order_rate_limited cid=%s attempts=%d", cid, attempt)
                return None
            await asyncio.sleep(_backoff_seconds(attempt))
            continue
        body_text = response.text
        if status == 400 and _capture_next_400:
            logger.warning(
                "demo_order_request_payload body=%s yes_ask=%s no_ask=%s",
                body,
                book.yes_ask,
                book.no_ask,
            )
            logger.warning("demo_order_response_body status=%d body=%s", status, body_text)
            _capture_next_400 = False
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
