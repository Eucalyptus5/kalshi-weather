from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal

import httpx
from kalshi_python_async import KalshiAuth

from bot.config import Settings

logger = logging.getLogger(__name__)

_API_PREFIX = "/trade-api/v2"


@dataclass(frozen=True, slots=True)
class KalshiMarket:
    ticker: str
    event_ticker: str
    series: str
    status: str
    close_time: datetime | None
    yes_ask: Decimal
    yes_bid: Decimal


@dataclass(frozen=True, slots=True)
class KalshiOrderbook:
    ticker: str
    yes_ask: Decimal
    yes_bid: Decimal
    no_ask: Decimal
    no_bid: Decimal
    snapshot_at: datetime


class KalshiDemoClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_http = http_client is None
        self._http: httpx.AsyncClient | None = http_client
        self._auth: KalshiAuth | None = None

    async def aopen(self) -> None:
        if not self._settings.kalshi_demo_key_id:
            raise RuntimeError("kalshi_demo_key_id is not configured")
        key_path = self._settings.kalshi_demo_private_key_path
        if key_path is None or not key_path.exists():
            raise RuntimeError(f"private key file not found at {key_path}")

        private_key_pem = key_path.read_text()
        self._auth = KalshiAuth(
            key_id=self._settings.kalshi_demo_key_id,
            private_key_pem=private_key_pem,
        )
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._settings.kalshi_demo_api_base,
                timeout=30.0,
            )
            logger.info("kalshi_client_open host=%s", self._settings.kalshi_demo_api_base)

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            await self._http.aclose()
        self._http = None
        self._auth = None

    async def list_open_markets_for_series(self, series_prefix: str) -> list[KalshiMarket]:
        assert self._http is not None and self._auth is not None
        path = "/markets"
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        response = await self._http.get(
            path, params={"status": "open", "limit": 1000}, headers=headers
        )
        response.raise_for_status()
        payload = response.json()

        raw_markets = payload.get("markets") or []
        out: list[KalshiMarket] = []
        skipped_null = 0
        for m in raw_markets:
            if m.get("yes_ask_dollars") is None or m.get("yes_bid_dollars") is None:
                skipped_null += 1
                continue
            ticker = m["ticker"]
            if not ticker.startswith(series_prefix):
                continue
            series = ticker.split("-", 1)[0]
            close_time = _parse_close_time(m.get("close_time"))
            out.append(
                KalshiMarket(
                    ticker=ticker,
                    event_ticker=m["event_ticker"],
                    series=series,
                    status=m["status"],
                    close_time=close_time,
                    yes_ask=Decimal(str(m["yes_ask_dollars"])),
                    yes_bid=Decimal(str(m["yes_bid_dollars"])),
                )
            )

        if payload.get("cursor"):
            logger.warning(
                "kalshi_list_markets_pagination_unimplemented series=%s",
                series_prefix,
            )

        logger.info(
            "kalshi_list_markets series=%s total=%d matched=%d skipped_null=%d",
            series_prefix,
            len(raw_markets),
            len(out),
            skipped_null,
        )
        return out

    async def get_orderbook(self, ticker: str) -> KalshiOrderbook:
        assert self._http is not None and self._auth is not None
        path = f"/markets/{ticker}/orderbook"
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        response = await self._http.get(path, headers=headers)
        response.raise_for_status()
        payload = response.json()
        ob = payload["orderbook"]

        yes_bid = _best_price(ob.get("yes_dollars"))
        no_bid = _best_price(ob.get("no_dollars"))
        yes_ask = Decimal("1") - no_bid
        no_ask = Decimal("1") - yes_bid

        return KalshiOrderbook(
            ticker=ticker,
            yes_ask=yes_ask,
            yes_bid=yes_bid,
            no_ask=no_ask,
            no_bid=no_bid,
            snapshot_at=datetime.now(tz=_timezone.utc),
        )


def _parse_close_time(raw: object) -> datetime | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"unexpected close_time type: {type(raw).__name__}")
    text = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    return datetime.fromisoformat(text)


def _best_price(levels: list[list[str]] | None) -> Decimal:
    if not levels:
        return Decimal("0")
    return max(Decimal(str(level[0])) for level in levels)
