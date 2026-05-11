from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from bot.config import Settings
from bot.kalshi_client import KalshiDemoClient, KalshiMarket, KalshiOrderbook


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("kalshi") / "demo.pem"
    path.write_bytes(pem)
    return path


def _settings_with_pem(pem_path: Path) -> Settings:
    return Settings(
        paper_mode=True,
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=pem_path,
    )


def _settings_no_key() -> Settings:
    return Settings(
        paper_mode=True,
        kalshi_demo_key_id=None,
        kalshi_demo_private_key_path=None,
    )


def _settings_missing_pem(tmp_path: Path) -> Settings:
    return Settings(
        paper_mode=True,
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=tmp_path / "does-not-exist.pem",
    )


def _market_dict(
    ticker: str,
    yes_ask: str | None,
    yes_bid: str | None,
    close_time: str | None = "2026-05-06T23:00:00Z",
    event_ticker: str | None = None,
    status: str = "open",
) -> dict[str, object]:
    if event_ticker is None:
        parts = ticker.split("-")
        event_ticker = "-".join(parts[:2])
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "status": status,
        "close_time": close_time,
        "yes_ask_dollars": yes_ask,
        "yes_bid_dollars": yes_bid,
    }


async def test_aopen_raises_when_key_id_missing() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    async with httpx.AsyncClient(transport=transport) as http:
        client = KalshiDemoClient(_settings_no_key(), http_client=http)
        with pytest.raises(RuntimeError, match="kalshi_demo_key_id"):
            await client.aopen()


async def test_aopen_raises_when_pem_file_missing(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    async with httpx.AsyncClient(transport=transport) as http:
        client = KalshiDemoClient(_settings_missing_pem(tmp_path), http_client=http)
        with pytest.raises(RuntimeError, match="private key"):
            await client.aopen()


async def test_aopen_succeeds_with_valid_pem(rsa_pem: Path) -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    async with httpx.AsyncClient(transport=transport) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.aclose()


async def test_list_open_markets_sends_series_filter_and_skips_null_priced(
    rsa_pem: Path,
) -> None:
    captured: dict[str, httpx.Request] = {}
    payload = {
        "markets": [
            _market_dict("KXHIGHDEN-26MAY06-T70-75", "0.4500", "0.4300"),
            _market_dict("KXHIGHDEN-26MAY06-T75-80", None, "0.1000"),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        markets = await client.list_open_markets_for_series("KXHIGHDEN")

    req = captured["req"]
    assert req.url.params.get("series_ticker") == "KXHIGHDEN"
    assert req.url.params.get("status") == "open"
    assert req.url.params.get("limit") == "1000"

    assert len(markets) == 1
    market = markets[0]
    assert isinstance(market, KalshiMarket)
    assert market.ticker == "KXHIGHDEN-26MAY06-T70-75"
    assert market.series == "KXHIGHDEN"
    assert market.event_ticker == "KXHIGHDEN-26MAY06"
    assert market.status == "open"
    assert market.yes_ask == Decimal("0.4500")
    assert market.yes_bid == Decimal("0.4300")
    assert isinstance(market.yes_ask, Decimal)
    assert isinstance(market.yes_bid, Decimal)


async def test_list_open_markets_warns_on_series_mismatch(
    rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    payload = {
        "markets": [
            _market_dict("KXHIGHDEN-26MAY06-T70-75", "0.4500", "0.4300"),
            _market_dict("KXHIGHAUS-26MAY06-T80-85", "0.5000", "0.4900"),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        caplog.set_level(logging.WARNING, logger="bot.kalshi_client")
        markets = await client.list_open_markets_for_series("KXHIGHDEN")

    assert len(markets) == 1
    assert markets[0].ticker == "KXHIGHDEN-26MAY06-T70-75"
    mismatch_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "series_mismatch" in r.getMessage()
    ]
    assert len(mismatch_warnings) == 1
    assert "KXHIGHAUS-26MAY06-T80-85" in mismatch_warnings[0].getMessage()


async def test_list_open_markets_parses_close_time_with_z(rsa_pem: Path) -> None:
    payload = {
        "markets": [
            _market_dict(
                "KXHIGHDEN-26MAY06-T70-75",
                "0.4500",
                "0.4300",
                close_time="2026-05-06T23:00:00Z",
            ),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        markets = await client.list_open_markets_for_series("KXHIGHDEN")

    assert markets[0].close_time == datetime(2026, 5, 6, 23, 0, tzinfo=timezone.utc)


async def test_list_open_markets_parses_close_time_with_offset(rsa_pem: Path) -> None:
    payload = {
        "markets": [
            _market_dict(
                "KXHIGHDEN-26MAY06-T70-75",
                "0.4500",
                "0.4300",
                close_time="2026-05-06T23:00:00+00:00",
            ),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        markets = await client.list_open_markets_for_series("KXHIGHDEN")

    assert markets[0].close_time == datetime(2026, 5, 6, 23, 0, tzinfo=timezone.utc)


async def test_list_open_markets_handles_null_close_time(rsa_pem: Path) -> None:
    payload = {
        "markets": [
            _market_dict(
                "KXHIGHDEN-26MAY06-T70-75",
                "0.4500",
                "0.4300",
                close_time=None,
            ),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        markets = await client.list_open_markets_for_series("KXHIGHDEN")

    assert markets[0].close_time is None


async def test_list_open_markets_ignores_non_empty_cursor(
    rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    payload = {
        "markets": [_market_dict("KXHIGHDEN-26MAY06-T70-75", "0.4500", "0.4300")],
        "cursor": "abc123",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        caplog.set_level(logging.WARNING, logger="bot.kalshi_client")
        markets = await client.list_open_markets_for_series("KXHIGHDEN")

    assert len(markets) == 1
    pagination_warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and "pagination" in r.getMessage()
    ]
    assert pagination_warnings == []


async def test_list_open_markets_sends_signing_headers(rsa_pem: Path) -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, json={"markets": [], "cursor": ""})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.list_open_markets_for_series("KXHIGHDEN")

    req = captured["req"]
    assert req.headers.get("KALSHI-ACCESS-KEY") == "demo-key-id"
    assert req.headers.get("KALSHI-ACCESS-SIGNATURE")
    assert req.headers.get("KALSHI-ACCESS-TIMESTAMP")


async def test_list_open_markets_hits_absolute_path(rsa_pem: Path) -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, json={"markets": [], "cursor": ""})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.list_open_markets_for_series("KXHIGHDEN")

    assert captured["req"].url.path == "/trade-api/v2/markets"


async def test_list_open_markets_raises_on_500(rsa_pem: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        with pytest.raises(httpx.HTTPStatusError):
            await client.list_open_markets_for_series("KXHIGHDEN")


async def test_get_orderbook_reconstructs_asks(rsa_pem: Path) -> None:
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.30", "100"]],
            "no_dollars": [["0.55", "50"]],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert isinstance(book, KalshiOrderbook)
    assert book.ticker == "KXHIGHDEN-26MAY06-T70-75"
    assert book.yes_bid == Decimal("0.30")
    assert book.no_bid == Decimal("0.55")
    assert book.yes_ask == Decimal("0.45")
    assert book.no_ask == Decimal("0.70")
    assert book.snapshot_at.tzinfo is not None


async def test_get_orderbook_picks_best_bid_across_levels(rsa_pem: Path) -> None:
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.20", "100"], ["0.30", "50"], ["0.25", "10"]],
            "no_dollars": [["0.55", "50"], ["0.50", "10"]],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert book.yes_bid == Decimal("0.30")
    assert book.no_bid == Decimal("0.55")


async def test_get_orderbook_round_trips_six_decimals(rsa_pem: Path) -> None:
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.987654", "1"]],
            "no_dollars": [["0.001000", "1"]],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert book.yes_bid == Decimal("0.987654")
    assert book.no_bid == Decimal("0.001000")


async def test_get_orderbook_empty_book_returns_zero_bids(rsa_pem: Path) -> None:
    payload = {"orderbook_fp": {"yes_dollars": None, "no_dollars": None}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert book.yes_bid == Decimal("0")
    assert book.no_bid == Decimal("0")
    assert book.yes_ask == Decimal("1")
    assert book.no_ask == Decimal("1")


async def test_get_orderbook_hits_absolute_path(rsa_pem: Path) -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(
            200,
            json={"orderbook_fp": {"yes_dollars": None, "no_dollars": None}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert captured["req"].url.path == "/trade-api/v2/markets/KXHIGHDEN-26MAY06-T70-75/orderbook"


async def test_get_orderbook_falls_back_to_legacy_orderbook_key(rsa_pem: Path) -> None:
    payload = {
        "orderbook": {
            "yes_dollars": [["0.30", "100"]],
            "no_dollars": [["0.55", "50"]],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert isinstance(book, KalshiOrderbook)
    assert book.yes_bid == Decimal("0.30")
    assert book.no_bid == Decimal("0.55")
    assert book.yes_ask == Decimal("0.45")
    assert book.no_ask == Decimal("0.70")


async def test_get_orderbook_raises_when_both_keys_missing(rsa_pem: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": "bar"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        with pytest.raises(KeyError) as excinfo:
            await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    msg = str(excinfo.value)
    assert "orderbook" in msg
    assert "KXHIGHDEN-26MAY06-T70-75" in msg


async def test_default_client_is_closed_by_aclose(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"markets": [], "cursor": ""})

    transport = httpx.MockTransport(handler)
    real_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    client = KalshiDemoClient(_settings_with_pem(rsa_pem))
    await client.aopen()
    await client.list_open_markets_for_series("KXHIGHDEN")
    http = client._http
    await client.aclose()
    assert http is not None
    assert http.is_closed


async def test_caller_owned_client_not_closed_by_aclose(rsa_pem: Path) -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"markets": [], "cursor": ""})
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.list_open_markets_for_series("KXHIGHDEN")
        await client.aclose()
        assert not http.is_closed


async def test_aclose_is_idempotent(rsa_pem: Path) -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"markets": [], "cursor": ""})
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.aclose()
        await client.aclose()
