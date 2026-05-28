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
from bot.kalshi_client import (
    KalshiDemoClient,
    KalshiMarket,
    KalshiOrderbook,
    _assert_demo_host,
    _resolved_request_url,
)
from bot.markets.parser import event_id


_EVENT_TICKER_GOLDEN: tuple[tuple[str, str], ...] = (
    ("KXHIGHTNOLA-26MAY22-B86.5", "KXHIGHTNOLA-26MAY22"),
    ("KXHIGHDEN-26MAY08-T96.5", "KXHIGHDEN-26MAY08"),
    ("KXHIGHDEN-26MAY06-T43", "KXHIGHDEN-26MAY06"),
    ("KXHIGHDEN-26APR28-T70.5-72.5", "KXHIGHDEN-26APR28"),
)


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("kalshi") / "demo.pem"
    path.write_bytes(pem)
    return path


def _settings_with_pem(pem_path: Path) -> Settings:
    return Settings(
        mode="paper",
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=pem_path,
    )


def _settings_no_key() -> Settings:
    return Settings(
        mode="paper",
        kalshi_demo_key_id=None,
        kalshi_demo_private_key_path=None,
    )


def _settings_missing_pem(tmp_path: Path) -> Settings:
    return Settings(
        mode="paper",
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
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
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


@pytest.mark.parametrize("ticker,expected_event_ticker", _EVENT_TICKER_GOLDEN)
async def test_event_ticker_matches_event_id_helper(
    rsa_pem: Path, ticker: str, expected_event_ticker: str
) -> None:
    payload = {
        "markets": [
            {
                "ticker": ticker,
                "event_ticker": expected_event_ticker,
                "status": "open",
                "close_time": "2026-05-06T23:00:00Z",
                "yes_ask_dollars": "0.45",
                "yes_bid_dollars": "0.43",
            }
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    series = ticker.split("-", 1)[0]
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        markets = await client.list_open_markets_for_series(series)

    assert len(markets) == 1
    market = markets[0]
    assert market.event_ticker == expected_event_ticker
    assert market.event_ticker == event_id(market.ticker)


async def test_event_ticker_negative_control_helper_mutation_breaks_equivalence(
    rsa_pem: Path,
) -> None:
    def broken_event_id(t: str) -> str:
        return t.split("-", 1)[0]

    mismatches = 0
    for ticker, expected_event_ticker in _EVENT_TICKER_GOLDEN:
        payload = {
            "markets": [
                {
                    "ticker": ticker,
                    "event_ticker": expected_event_ticker,
                    "status": "open",
                    "close_time": "2026-05-06T23:00:00Z",
                    "yes_ask_dollars": "0.45",
                    "yes_bid_dollars": "0.43",
                }
            ],
            "cursor": "",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        series = ticker.split("-", 1)[0]
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
        ) as http:
            client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
            await client.aopen()
            markets = await client.list_open_markets_for_series(series)

        market = markets[0]
        if market.event_ticker != broken_event_id(market.ticker):
            mismatches += 1

    assert mismatches >= 1


async def test_get_orderbook_parses_depth_from_levels(rsa_pem: Path) -> None:
    payload_str = {
        "orderbook_fp": {
            "yes_dollars": [["0.42", "13"]],
            "no_dollars": [["0.55", "7"]],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload_str)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert book.yes_bid == Decimal("0.42")
    assert book.yes_bid_depth == 13
    assert book.no_bid_depth == 7
    assert book.yes_ask_depth == 7
    assert book.no_ask_depth == 13

    payload_int = {
        "orderbook_fp": {
            "yes_dollars": [["0.42", 13]],
            "no_dollars": [["0.55", 7]],
        }
    }

    def handler_int(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload_int)

    transport_int = httpx.MockTransport(handler_int)
    async with httpx.AsyncClient(
        transport=transport_int, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert book.yes_bid_depth == 13
    assert book.no_bid_depth == 7


async def test_get_orderbook_parses_depth_from_float_string_levels(rsa_pem: Path) -> None:
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.42", "13.00"]],
            "no_dollars": [["0.55", "1.00"]],
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

    assert book.yes_bid_depth == 13
    assert book.no_bid_depth == 1
    assert book.yes_ask_depth == 1
    assert book.no_ask_depth == 13


async def test_get_orderbook_zero_depth_for_empty_book(rsa_pem: Path) -> None:
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
    assert book.yes_bid_depth == 0
    assert book.no_bid_depth == 0
    assert book.yes_ask_depth == 0
    assert book.no_ask_depth == 0


def _settings_production_base(pem_path: Path, base: str) -> Settings:
    return Settings(
        mode="paper",
        kalshi_demo_api_base=base,
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=pem_path,
    )


async def test_aopen_refuses_production_host_in_settings(rsa_pem: Path) -> None:
    settings = _settings_production_base(rsa_pem, "https://api.elections.kalshi.com/trade-api/v2")
    client = KalshiDemoClient(settings)
    with pytest.raises(RuntimeError, match="demo-api.kalshi.co"):
        await client.aopen()


async def test_aopen_refuses_query_string_demo_bypass(rsa_pem: Path) -> None:
    settings = _settings_production_base(rsa_pem, "https://api.elections.kalshi.com/?env=demo")
    client = KalshiDemoClient(settings)
    with pytest.raises(RuntimeError, match="demo-api.kalshi.co"):
        await client.aopen()


async def test_aopen_refuses_injected_client_with_production_base_url(rsa_pem: Path) -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://api.elections.kalshi.com/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        with pytest.raises(RuntimeError, match="demo-api.kalshi.co"):
            await client.aopen()


async def test_aopen_accepts_injected_client_with_demo_base_url(rsa_pem: Path) -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.aclose()


async def test_post_signed_refuses_absolute_url_to_production(rsa_pem: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        with pytest.raises(RuntimeError, match="absolute URL forbidden"):
            await client.post_signed("https://api.elections.kalshi.com/foo", {})
        await client.aclose()

    assert calls == []


async def test_get_signed_refuses_absolute_url_to_production(rsa_pem: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        with pytest.raises(RuntimeError, match="absolute URL forbidden"):
            await client.get_signed("https://api.elections.kalshi.com/portfolio/fills")
        await client.aclose()

    assert calls == []


async def test_get_signed_sends_params_and_signed_headers(rsa_pem: Path) -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, json={"orders": []})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.get_signed("/portfolio/orders", {"client_order_id": "kw-edge-yes-x"})
        await client.aclose()

    req = captured["req"]
    assert req.method == "GET"
    assert req.url.params.get("client_order_id") == "kw-edge-yes-x"
    assert req.headers.get("KALSHI-ACCESS-KEY") == "demo-key-id"
    assert req.headers.get("KALSHI-ACCESS-SIGNATURE")


def test_url_resolution_keeps_trade_api_v2_prefix() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200))
    client = httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    )
    assert (
        _resolved_request_url(client, "POST", "/portfolio/orders")
        == "https://demo-api.kalshi.co/trade-api/v2/portfolio/orders"
    )


def test_resolved_url_keeps_demo_host_for_absolute_production_url() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200))
    client = httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    )
    resolved = _resolved_request_url(client, "POST", "https://api.elections.kalshi.com/foo")
    assert resolved == "https://api.elections.kalshi.com/foo"
    with pytest.raises(RuntimeError, match="demo-api.kalshi.co"):
        _assert_demo_host("write", resolved)


async def test_post_signed_includes_kalshi_headers(rsa_pem: Path) -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(
            201,
            json={
                "order": {
                    "order_id": "ex-1",
                    "client_order_id": "cid-1",
                    "ticker": "KXHIGHDEN-26MAY06-T70-75",
                    "side": "yes",
                    "status": "executed",
                    "fill_count_fp": "1.00",
                    "initial_count_fp": "1.00",
                    "remaining_count_fp": "0.00",
                    "yes_price_dollars": "0.5000",
                    "taker_fees_dollars": "0.010000",
                    "maker_fees_dollars": "0.000000",
                    "taker_fill_cost_dollars": "0.500000",
                    "maker_fill_cost_dollars": "0.000000",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        await client.post_signed("/portfolio/orders", {"ticker": "x"})
        await client.aclose()

    req = captured["req"]
    assert req.method == "POST"
    assert req.headers.get("KALSHI-ACCESS-KEY") == "demo-key-id"
    assert req.headers.get("KALSHI-ACCESS-TIMESTAMP")
    assert req.headers.get("KALSHI-ACCESS-SIGNATURE")


async def test_get_balance_parses_balance_dollars(rsa_pem: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "balance": 53725,
                "balance_dollars": "537.250000",
                "portfolio_value": 53725,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        client = KalshiDemoClient(_settings_with_pem(rsa_pem), http_client=http)
        await client.aopen()
        balance = await client.get_balance()
        await client.aclose()

    assert balance == Decimal("537.250000")


def test_signing_golden_pss_verifies(rsa_pem: Path) -> None:
    import base64

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from kalshi_python_async import KalshiAuth

    private_key_pem = rsa_pem.read_text()
    auth = KalshiAuth(key_id="demo-key-id", private_key_pem=private_key_pem)
    headers = auth.create_auth_headers("POST", "/trade-api/v2/portfolio/orders")

    timestamp_ms = headers["KALSHI-ACCESS-TIMESTAMP"]
    signature_b64 = headers["KALSHI-ACCESS-SIGNATURE"]
    message = f"{timestamp_ms}POST/trade-api/v2/portfolio/orders".encode()

    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    public_key = private_key.public_key()
    signature = base64.b64decode(signature_b64)
    public_key.verify(
        signature,
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
        hashes.SHA256(),
    )
