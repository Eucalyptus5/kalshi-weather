from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol

import websockets
from kalshi_python_async import KalshiAuth
from websockets.exceptions import ConnectionClosed

from bot.config import Settings

logger = logging.getLogger(__name__)

_WS_PATH = "/trade-api/ws/v2"
_ALLOWED_CHANNELS = frozenset({"orderbook_delta", "trade"})
_TERMINAL_ERROR_CODES = frozenset({10, 17, 25})


@dataclass(frozen=True)
class BookLevel:
    price: Decimal
    size: Decimal


@dataclass(frozen=True)
class BookSnapshot:
    ticker: str
    sid: int
    seq: int
    yes_levels: tuple[BookLevel, ...]
    no_levels: tuple[BookLevel, ...]
    received_at: datetime


@dataclass(frozen=True)
class BookDelta:
    ticker: str
    sid: int
    seq: int
    side: str
    price: Decimal
    delta: Decimal
    ts_ms: int
    received_at: datetime


@dataclass(frozen=True)
class TradePrint:
    ticker: str
    sid: int
    trade_id: str
    yes_price: Decimal
    no_price: Decimal
    count: Decimal
    taker_side: str
    ts_ms: int
    received_at: datetime


@dataclass(frozen=True)
class GapDetected:
    ticker: str
    sid: int
    last_seq: int
    next_seq: int
    reason: str


class WSConnection(Protocol):
    async def send(self, data: str) -> None: ...
    async def recv(self) -> str: ...
    async def close(self) -> None: ...


_ConnectFn = Callable[[str, dict[str, str]], AbstractAsyncContextManager[WSConnection]]


def _default_connect(
    url: str, headers: dict[str, str]
) -> AbstractAsyncContextManager[WSConnection]:
    return websockets.connect(url, additional_headers=headers)


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _decode_levels(raw: object) -> tuple[BookLevel, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"orderbook levels not a list: {type(raw).__name__}")
    return tuple(BookLevel(price=Decimal(str(p)), size=Decimal(str(s))) for p, s in raw)


def _error_code(payload: dict[str, object]) -> int | None:
    top = payload.get("code")
    if isinstance(top, int):
        return top
    msg = payload.get("msg")
    if isinstance(msg, dict):
        nested = msg.get("code")
        if isinstance(nested, int):
            return nested
    return None


class KalshiWSClient:
    def __init__(
        self,
        settings: Settings,
        url: str,
        *,
        connect_fn: _ConnectFn | None = None,
        clock: Callable[[], datetime] | None = None,
        backoff_seconds: tuple[float, ...] = (1.0, 2.0, 5.0, 15.0, 30.0),
        frame_sink: Callable[[str, datetime], None] | None = None,
    ) -> None:
        if not backoff_seconds:
            raise ValueError("backoff_seconds must have at least one entry")
        self._settings = settings
        self._url = url
        self._connect_fn: _ConnectFn = connect_fn or _default_connect
        self._clock = clock or _utc_now
        self._backoff_seconds = backoff_seconds
        self._frame_sink = frame_sink

        self._auth: KalshiAuth | None = None
        self._conn: WSConnection | None = None
        self._conn_ctx: AbstractAsyncContextManager[WSConnection] | None = None
        self._last_subscribe: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self._next_sub_id = 1
        self._seq_by_sid: dict[int, int] = {}
        self._pending_gap: GapDetected | None = None

    async def aopen(self) -> None:
        if not self._settings.kalshi_prod_key_id:
            raise RuntimeError("kalshi_prod_key_id is not configured")
        key_path = self._settings.kalshi_prod_private_key_path
        if key_path is None or not key_path.exists():
            raise RuntimeError(f"prod private key file not found at {key_path}")
        private_key_pem = key_path.read_text()
        self._auth = KalshiAuth(
            key_id=self._settings.kalshi_prod_key_id,
            private_key_pem=private_key_pem,
        )

    async def aclose(self) -> None:
        if self._conn_ctx is not None:
            try:
                await self._conn_ctx.__aexit__(None, None, None)
            except (ConnectionClosed, OSError):
                pass
            self._conn_ctx = None
            self._conn = None
        self._auth = None
        self._seq_by_sid.clear()
        self._pending_gap = None

    async def subscribe(self, channels: Sequence[str], tickers: Sequence[str]) -> None:
        for channel in channels:
            if channel not in _ALLOWED_CHANNELS:
                raise ValueError(f"unsupported ws channel: {channel}")
        self._last_subscribe = (tuple(channels), tuple(tickers))
        await self._ensure_connected()
        await self._send_subscribe()

    async def _ensure_connected(self) -> None:
        if self._conn is not None:
            return
        if self._auth is None:
            raise RuntimeError("aopen() must be called before subscribe/events")
        headers = self._auth.create_auth_headers("GET", _WS_PATH)
        host = urllib.parse.urlparse(self._url).hostname or "?"
        logger.info("kalshi_ws_connect host=%s", host)
        ctx = self._connect_fn(self._url, headers)
        self._conn = await ctx.__aenter__()
        self._conn_ctx = ctx

    async def _send_subscribe(self) -> None:
        if self._conn is None or self._last_subscribe is None:
            raise RuntimeError("cannot send subscribe without an active connection")
        channels, tickers = self._last_subscribe
        payload = {
            "id": self._next_sub_id,
            "cmd": "subscribe",
            "params": {"channels": list(channels), "market_tickers": list(tickers)},
        }
        self._next_sub_id += 1
        logger.info(
            "kalshi_ws_subscribe channels=%s tickers=%d",
            ",".join(channels),
            len(tickers),
        )
        await self._conn.send(json.dumps(payload))

    async def _trigger_reconnect(self, *, reason: str, sid: int, last_seq: int) -> None:
        if self._conn_ctx is not None:
            try:
                await self._conn_ctx.__aexit__(None, None, None)
            except (ConnectionClosed, OSError):
                pass
        self._conn_ctx = None
        self._conn = None
        self._seq_by_sid.clear()
        self._pending_gap = GapDetected(
            ticker="",
            sid=sid,
            last_seq=last_seq,
            next_seq=0,
            reason=reason,
        )

        attempt = 0
        while True:
            delay = self._backoff_seconds[min(attempt, len(self._backoff_seconds) - 1)]
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                await self._ensure_connected()
                await self._send_subscribe()
                return
            except (ConnectionClosed, OSError) as exc:
                logger.warning("kalshi_ws_reconnect_failed attempt=%d err=%s", attempt, exc)
                attempt += 1

    def _track_seq(self, sid: int, seq: int, ticker: str) -> GapDetected | None:
        prev = self._seq_by_sid.get(sid)
        self._seq_by_sid[sid] = seq
        if prev is None or seq == prev + 1:
            return None
        return GapDetected(
            ticker=ticker,
            sid=sid,
            last_seq=prev,
            next_seq=seq,
            reason="seq_skip",
        )

    def _decode(self, payload: dict[str, object]) -> BookSnapshot | BookDelta | TradePrint | None:
        mtype = payload.get("type")
        sid = payload.get("sid")
        raw_msg = payload.get("msg")
        if not isinstance(sid, int) or not isinstance(raw_msg, dict):
            return None
        ticker = str(raw_msg.get("market_ticker", ""))
        now = self._clock()

        if mtype == "orderbook_snapshot":
            seq = payload.get("seq")
            if not isinstance(seq, int):
                return None
            return BookSnapshot(
                ticker=ticker,
                sid=sid,
                seq=seq,
                yes_levels=_decode_levels(raw_msg.get("yes_dollars_fp")),
                no_levels=_decode_levels(raw_msg.get("no_dollars_fp")),
                received_at=now,
            )
        if mtype == "orderbook_delta":
            seq = payload.get("seq")
            if not isinstance(seq, int):
                return None
            return BookDelta(
                ticker=ticker,
                sid=sid,
                seq=seq,
                side=str(raw_msg["side"]),
                price=Decimal(str(raw_msg["price_dollars"])),
                delta=Decimal(str(raw_msg["delta_fp"])),
                ts_ms=int(raw_msg["ts_ms"]),
                received_at=now,
            )
        if mtype == "trade":
            taker = raw_msg.get("taker_outcome_side") or raw_msg.get("taker_side") or ""
            return TradePrint(
                ticker=ticker,
                sid=sid,
                trade_id=str(raw_msg["trade_id"]),
                yes_price=Decimal(str(raw_msg["yes_price_dollars"])),
                no_price=Decimal(str(raw_msg["no_price_dollars"])),
                count=Decimal(str(raw_msg["count_fp"])),
                taker_side=str(taker),
                ts_ms=int(raw_msg["ts_ms"]),
                received_at=now,
            )
        return None

    async def events(
        self,
    ) -> AsyncIterator[BookSnapshot | BookDelta | TradePrint | GapDetected]:
        while True:
            if self._pending_gap is not None:
                gap = self._pending_gap
                self._pending_gap = None
                yield gap
                continue

            if self._conn is None:
                await self._trigger_reconnect(reason="connection_reset", sid=0, last_seq=0)
                continue

            try:
                raw = await self._conn.recv()
            except (ConnectionClosed, OSError):
                await self._trigger_reconnect(reason="connection_reset", sid=0, last_seq=0)
                continue

            if self._frame_sink is not None:
                self._frame_sink(raw, self._clock())

            payload = json.loads(raw)
            if not isinstance(payload, dict):
                continue

            mtype = payload.get("type")
            sid = payload.get("sid")

            if mtype == "error":
                code = _error_code(payload)
                if code in _TERMINAL_ERROR_CODES:
                    err_sid = sid if isinstance(sid, int) else 0
                    last_seq = self._seq_by_sid.get(err_sid, 0)
                    await self._trigger_reconnect(
                        reason=f"terminal_error_{code}",
                        sid=err_sid,
                        last_seq=last_seq,
                    )
                else:
                    logger.warning("kalshi_ws_error code=%s payload=%s", code, payload)
                continue

            if mtype in ("orderbook_snapshot", "orderbook_delta") and isinstance(sid, int):
                seq = payload.get("seq")
                raw_msg = payload.get("msg")
                ticker = ""
                if isinstance(raw_msg, dict):
                    ticker = str(raw_msg.get("market_ticker", ""))
                if isinstance(seq, int):
                    gap = self._track_seq(sid, seq, ticker)
                    if gap is not None:
                        yield gap

            decoded = self._decode(payload)
            if decoded is not None:
                yield decoded
