from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal

from kalshi_python_async import ApiClient, Configuration, KalshiAuth, MarketApi

from bot.config import Settings

logger = logging.getLogger(__name__)


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
        _market_api: MarketApi | object | None = None,
    ) -> None:
        self._settings = settings
        self._api_client: ApiClient | None = None
        self._market_api = _market_api
        self._injected = _market_api is not None

    async def aopen(self) -> None:
        if self._injected:
            return
        if not self._settings.kalshi_demo_key_id:
            raise RuntimeError("kalshi_demo_key_id is not configured")
        key_path = self._settings.kalshi_demo_private_key_path
        if key_path is None or not key_path.exists():
            raise RuntimeError(f"private key file not found at {key_path}")

        private_key_pem = key_path.read_text()
        config = Configuration(host=self._settings.kalshi_demo_api_base)
        api_client = ApiClient(configuration=config)
        api_client.kalshi_auth = KalshiAuth(
            key_id=self._settings.kalshi_demo_key_id,
            private_key_pem=private_key_pem,
        )
        self._api_client = api_client
        self._market_api = MarketApi(api_client)
        logger.info("kalshi_client_open host=%s", self._settings.kalshi_demo_api_base)

    async def aclose(self) -> None:
        if self._injected:
            return
        if self._api_client is not None:
            await self._api_client.close()
            self._api_client = None
            self._market_api = None

    async def list_open_markets_for_series(self, series_prefix: str) -> list[KalshiMarket]:
        if self._market_api is None:
            raise RuntimeError("kalshi client not opened")
        response = await self._market_api.get_markets(status="open", limit=1000)
        out: list[KalshiMarket] = []
        for m in response.markets:
            ticker = m.ticker
            if not ticker.startswith(series_prefix):
                continue
            series = ticker.split("-", 1)[0]
            out.append(
                KalshiMarket(
                    ticker=ticker,
                    event_ticker=m.event_ticker,
                    series=series,
                    status=m.status,
                    close_time=m.close_time,
                    yes_ask=Decimal(str(m.yes_ask_dollars)),
                    yes_bid=Decimal(str(m.yes_bid_dollars)),
                )
            )
        logger.info(
            "kalshi_list_markets series=%s total=%d matched=%d",
            series_prefix,
            len(response.markets),
            len(out),
        )
        return out

    async def get_orderbook(self, ticker: str) -> KalshiOrderbook:
        if self._market_api is None:
            raise RuntimeError("kalshi client not opened")
        response = await self._market_api.get_market_orderbook(ticker=ticker)
        ob = response.orderbook

        yes_bid = _best_price(ob.yes_dollars)
        no_bid = _best_price(ob.no_dollars)
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


def _best_price(levels: list[list[str]] | None) -> Decimal:
    if not levels:
        return Decimal("0")
    return max(Decimal(str(level[0])) for level in levels)
