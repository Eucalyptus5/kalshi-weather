from __future__ import annotations

# Out-of-loop probe scripts that POST to /portfolio/orders must live under
# scripts/probes/ and instantiate KalshiDemoClient directly. Probe POSTs that
# bypass _place / _commit_phase2 orphan demo_orders rows. Convention is
# enforced by review and by tests/test_probes_directory.py, not by code here.

import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal

import httpx
from kalshi_python_async import KalshiAuth
from pydantic import BaseModel, ConfigDict

from bot.config import Settings
from bot.execution.token_bucket import TokenBucket
from bot.markets.parser import series_id

logger = logging.getLogger(__name__)

_API_PREFIX = "/trade-api/v2"


def _assert_demo_host(source: str, url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname != "demo-api.kalshi.co":
        raise RuntimeError(f"kalshi {source} host must equal 'demo-api.kalshi.co'; refusing: {url}")


def _assert_not_demo_host(source: str, url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname is None or parsed.hostname == "demo-api.kalshi.co":
        raise RuntimeError(
            f"kalshi read {source} host must not be 'demo-api.kalshi.co'; refusing: {url}"
        )


def _resolved_request_url(client: httpx.AsyncClient, method: str, path: str) -> str:
    return str(client.build_request(method, path).url)


@dataclass(frozen=True, slots=True)
class KalshiMarket:
    ticker: str
    event_ticker: str
    series: str
    status: str
    close_time: datetime | None
    yes_ask: Decimal
    yes_bid: Decimal


class BalanceBreakdownEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    balance: str
    exchange_index: int


class BalancePayload(BaseModel):
    # forward-compat for unannounced demo wire fields; the snapshot loop reads a
    # known subset and silently drops the rest rather than crashing on additions.
    model_config = ConfigDict(extra="ignore")

    balance: int
    # absent on accounts with no trade activity (demo wire omits the breakdown
    # block until the first fill lands).
    balance_breakdown: list[BalanceBreakdownEntry] | None = None
    balance_dollars: Decimal
    portfolio_value: int
    updated_ts: int


@dataclass(frozen=True, slots=True)
class KalshiOrderbook:
    ticker: str
    yes_ask: Decimal
    yes_bid: Decimal
    no_ask: Decimal
    no_bid: Decimal
    yes_ask_depth: int
    yes_bid_depth: int
    no_ask_depth: int
    no_bid_depth: int
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
        self._read_bucket = TokenBucket(capacity=200, refill_per_second=200)
        self._write_bucket = TokenBucket(capacity=100, refill_per_second=100)

    async def aopen(self) -> None:
        if not self._settings.kalshi_demo_key_id:
            raise RuntimeError("kalshi_demo_key_id is not configured")
        key_path = self._settings.kalshi_demo_private_key_path
        if key_path is None or not key_path.exists():
            raise RuntimeError(f"private key file not found at {key_path}")

        if self._http is None:
            _assert_demo_host("settings", self._settings.kalshi_demo_api_base)
        else:
            _assert_demo_host("injected", str(self._http.base_url))

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

        await self._read_bucket.aopen()
        await self._write_bucket.aopen()

    async def aclose(self) -> None:
        await self._read_bucket.aclose()
        await self._write_bucket.aclose()
        if self._http is not None and self._owns_http:
            await self._http.aclose()
        self._http = None
        self._auth = None

    async def list_open_markets_for_series(self, series_ticker: str) -> list[KalshiMarket]:
        assert self._http is not None and self._auth is not None
        await self._read_bucket.acquire(cost=1)
        path = "/markets"
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        response = await self._http.get(
            path,
            params={"status": "open", "series_ticker": series_ticker, "limit": 1000},
            headers=headers,
        )
        response.raise_for_status()
        return _markets_from_payload(response.json(), series_ticker)

    async def get_orderbook(self, ticker: str) -> KalshiOrderbook:
        assert self._http is not None and self._auth is not None
        await self._read_bucket.acquire(cost=1)
        path = f"/markets/{ticker}/orderbook"
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        response = await self._http.get(path, headers=headers)
        response.raise_for_status()
        return _orderbook_from_payload(response.json(), ticker)

    async def get_signed(
        self, path: str, params: dict[str, object] | None = None
    ) -> httpx.Response:
        assert self._http is not None and self._auth is not None
        if "://" in path:
            raise RuntimeError("absolute URL forbidden on signed call")
        _assert_demo_host("read", _resolved_request_url(self._http, "GET", path))
        await self._read_bucket.acquire(cost=1)
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        return await self._http.get(path, params=params, headers=headers)

    async def post_signed(self, path: str, body: dict[str, object]) -> httpx.Response:
        assert self._http is not None and self._auth is not None
        if "://" in path:
            raise RuntimeError("absolute URL forbidden on signed call")
        _assert_demo_host("write", _resolved_request_url(self._http, "POST", path))
        await self._write_bucket.acquire()
        headers = self._auth.create_auth_headers("POST", f"{_API_PREFIX}{path}")
        return await self._http.post(path, json=body, headers=headers)

    async def delete_signed(self, path: str) -> httpx.Response:
        assert self._http is not None and self._auth is not None
        if "://" in path:
            raise RuntimeError("absolute URL forbidden on signed call")
        _assert_demo_host("write", _resolved_request_url(self._http, "DELETE", path))
        await self._write_bucket.acquire()
        headers = self._auth.create_auth_headers("DELETE", f"{_API_PREFIX}{path}")
        return await self._http.delete(path, headers=headers)

    async def get_balance(self) -> Decimal:
        assert self._http is not None and self._auth is not None
        path = "/portfolio/balance"
        if "://" in path:
            raise RuntimeError("absolute URL forbidden on signed call")
        _assert_demo_host("write", _resolved_request_url(self._http, "GET", path))
        await self._read_bucket.acquire(cost=1)
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        response = await self._http.get(path, headers=headers)
        response.raise_for_status()
        payload = response.json()
        return Decimal(str(payload["balance_dollars"]))

    async def get_balance_full(self) -> BalancePayload:
        assert self._http is not None and self._auth is not None
        path = "/portfolio/balance"
        _assert_demo_host("read", _resolved_request_url(self._http, "GET", path))
        await self._read_bucket.acquire(cost=1)
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        response = await self._http.get(path, headers=headers)
        response.raise_for_status()
        return BalancePayload.model_validate(response.json())


class KalshiReadClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_http = http_client is None
        self._http: httpx.AsyncClient | None = http_client
        self._auth: KalshiAuth | None = None
        self._read_bucket = TokenBucket(capacity=200, refill_per_second=200)

    async def aopen(self) -> None:
        if not self._settings.kalshi_prod_key_id:
            raise RuntimeError("kalshi_prod_key_id is not configured")
        key_path = self._settings.kalshi_prod_private_key_path
        if key_path is None or not key_path.exists():
            raise RuntimeError(f"prod private key file not found at {key_path}")

        if self._http is None:
            _assert_not_demo_host("settings", self._settings.kalshi_prod_api_base)
        else:
            _assert_not_demo_host("injected", str(self._http.base_url))

        private_key_pem = key_path.read_text()
        self._auth = KalshiAuth(
            key_id=self._settings.kalshi_prod_key_id,
            private_key_pem=private_key_pem,
        )
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._settings.kalshi_prod_api_base,
                timeout=30.0,
            )
            logger.info("kalshi_read_client_open host=%s", self._settings.kalshi_prod_api_base)

        await self._read_bucket.aopen()

    async def aclose(self) -> None:
        await self._read_bucket.aclose()
        if self._http is not None and self._owns_http:
            await self._http.aclose()
        self._http = None
        self._auth = None

    async def list_open_markets_for_series(self, series_ticker: str) -> list[KalshiMarket]:
        assert self._http is not None and self._auth is not None
        await self._read_bucket.acquire(cost=1)
        path = "/markets"
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        response = await self._http.get(
            path,
            params={"status": "open", "series_ticker": series_ticker, "limit": 1000},
            headers=headers,
        )
        response.raise_for_status()
        return _markets_from_payload(response.json(), series_ticker)

    async def get_orderbook(self, ticker: str) -> KalshiOrderbook:
        assert self._http is not None and self._auth is not None
        await self._read_bucket.acquire(cost=1)
        path = f"/markets/{ticker}/orderbook"
        headers = self._auth.create_auth_headers("GET", f"{_API_PREFIX}{path}")
        response = await self._http.get(path, headers=headers)
        response.raise_for_status()
        return _orderbook_from_payload(response.json(), ticker)


def _markets_from_payload(payload: dict[str, object], series_ticker: str) -> list[KalshiMarket]:
    raw_markets = payload.get("markets") or []
    out: list[KalshiMarket] = []
    skipped_null = 0
    for m in raw_markets:
        if m.get("yes_ask_dollars") is None or m.get("yes_bid_dollars") is None:
            skipped_null += 1
            continue
        ticker = m["ticker"]
        series = series_id(ticker)
        if series != series_ticker:
            logger.warning(
                "kalshi_list_markets_series_mismatch ticker=%s expected=%s",
                ticker,
                series_ticker,
            )
            continue
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

    logger.info(
        "kalshi_list_markets series=%s total=%d matched=%d skipped_null=%d",
        series_ticker,
        len(raw_markets),
        len(out),
        skipped_null,
    )
    return out


def _orderbook_from_payload(payload: dict[str, object], ticker: str) -> KalshiOrderbook:
    ob = payload.get("orderbook_fp") or payload.get("orderbook")
    if ob is None:
        raise KeyError(f"orderbook response missing orderbook_fp / orderbook key: {ticker}")

    yes_bid, yes_bid_depth = _best_level(ob.get("yes_dollars"))
    no_bid, no_bid_depth = _best_level(ob.get("no_dollars"))
    yes_ask = Decimal("1") - no_bid
    no_ask = Decimal("1") - yes_bid
    yes_ask_depth = no_bid_depth
    no_ask_depth = yes_bid_depth

    return KalshiOrderbook(
        ticker=ticker,
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=no_ask,
        no_bid=no_bid,
        yes_ask_depth=yes_ask_depth,
        yes_bid_depth=yes_bid_depth,
        no_ask_depth=no_ask_depth,
        no_bid_depth=no_bid_depth,
        snapshot_at=datetime.now(tz=_timezone.utc),
    )


def _parse_close_time(raw: object) -> datetime | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"unexpected close_time type: {type(raw).__name__}")
    text = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    return datetime.fromisoformat(text)


def _best_level(levels: list[list[str]] | None) -> tuple[Decimal, int]:
    if not levels:
        return Decimal("0"), 0
    best = max(levels, key=lambda lvl: Decimal(str(lvl[0])))
    return Decimal(str(best[0])), int(Decimal(str(best[1])))
