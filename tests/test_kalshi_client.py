from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bot.config import Settings
from bot.kalshi_client import KalshiDemoClient, KalshiMarket, KalshiOrderbook


def _settings(tmp_path: Path, with_key: bool = True, key_exists: bool = True) -> Settings:
    if not with_key:
        return Settings(
            paper_mode=True,
            kalshi_demo_key_id=None,
            kalshi_demo_private_key_path=None,
        )
    key_path = tmp_path / "demo.pem"
    if key_exists:
        key_path.write_text("-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----\n")
    return Settings(
        paper_mode=True,
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=key_path,
    )


async def test_aopen_raises_when_key_id_missing(tmp_path: Path) -> None:
    client = KalshiDemoClient(_settings(tmp_path, with_key=False))
    with pytest.raises(RuntimeError, match="kalshi_demo_key_id"):
        await client.aopen()


async def test_aopen_raises_when_key_file_missing(tmp_path: Path) -> None:
    client = KalshiDemoClient(_settings(tmp_path, with_key=True, key_exists=False))
    with pytest.raises(RuntimeError, match="private key"):
        await client.aopen()


async def test_get_orderbook_reconstructs_asks_from_no_bid_and_yes_bid(tmp_path: Path) -> None:
    sdk_response = type(
        "Resp",
        (),
        {
            "orderbook": type(
                "OB",
                (),
                {
                    "yes_dollars": [["0.3000", "100"]],
                    "no_dollars": [["0.5500", "50"]],
                },
            )()
        },
    )()
    market_api = AsyncMock()
    market_api.get_market_orderbook = AsyncMock(return_value=sdk_response)

    client = KalshiDemoClient(_settings(tmp_path), _market_api=market_api)
    book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert isinstance(book, KalshiOrderbook)
    assert book.ticker == "KXHIGHDEN-26MAY06-T70-75"
    assert book.yes_bid == Decimal("0.3000")
    assert book.no_bid == Decimal("0.5500")
    assert book.yes_ask == Decimal("1") - Decimal("0.5500")
    assert book.no_ask == Decimal("1") - Decimal("0.3000")
    assert book.snapshot_at.tzinfo is not None


async def test_get_orderbook_picks_best_bid_when_multiple_levels(tmp_path: Path) -> None:
    sdk_response = type(
        "Resp",
        (),
        {
            "orderbook": type(
                "OB",
                (),
                {
                    "yes_dollars": [["0.2000", "100"], ["0.3000", "50"], ["0.2500", "10"]],
                    "no_dollars": [["0.5500", "50"], ["0.5000", "10"]],
                },
            )()
        },
    )()
    market_api = AsyncMock()
    market_api.get_market_orderbook = AsyncMock(return_value=sdk_response)

    client = KalshiDemoClient(_settings(tmp_path), _market_api=market_api)
    book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert book.yes_bid == Decimal("0.3000")
    assert book.no_bid == Decimal("0.5500")


async def test_get_orderbook_decimal_string_round_trip(tmp_path: Path) -> None:
    sdk_response = type(
        "Resp",
        (),
        {
            "orderbook": type(
                "OB",
                (),
                {
                    "yes_dollars": [["0.987654", "1"]],
                    "no_dollars": [["0.001000", "1"]],
                },
            )()
        },
    )()
    market_api = AsyncMock()
    market_api.get_market_orderbook = AsyncMock(return_value=sdk_response)

    client = KalshiDemoClient(_settings(tmp_path), _market_api=market_api)
    book = await client.get_orderbook("KXHIGHDEN-26MAY06-T70-75")

    assert book.yes_bid == Decimal("0.987654")
    assert book.no_bid == Decimal("0.001000")


async def test_list_open_markets_filters_by_series_prefix(tmp_path: Path) -> None:
    close_at = datetime(2026, 5, 6, 23, 0, tzinfo=timezone.utc)
    sdk_market_a = type(
        "M",
        (),
        {
            "ticker": "KXHIGHDEN-26MAY06-T70-75",
            "event_ticker": "KXHIGHDEN-26MAY06",
            "status": "open",
            "close_time": close_at,
            "yes_ask_dollars": "0.4500",
            "yes_bid_dollars": "0.4300",
        },
    )()
    sdk_market_b = type(
        "M",
        (),
        {
            "ticker": "KXHIGHAUS-26MAY06-T80-85",
            "event_ticker": "KXHIGHAUS-26MAY06",
            "status": "open",
            "close_time": close_at,
            "yes_ask_dollars": "0.5000",
            "yes_bid_dollars": "0.4900",
        },
    )()
    sdk_response = type(
        "R",
        (),
        {"markets": [sdk_market_a, sdk_market_b], "cursor": ""},
    )()
    market_api = AsyncMock()
    market_api.get_markets = AsyncMock(return_value=sdk_response)

    client = KalshiDemoClient(_settings(tmp_path), _market_api=market_api)
    markets = await client.list_open_markets_for_series("KXHIGHDEN")

    assert len(markets) == 1
    market = markets[0]
    assert isinstance(market, KalshiMarket)
    assert market.ticker == "KXHIGHDEN-26MAY06-T70-75"
    assert market.series == "KXHIGHDEN"
    assert market.event_ticker == "KXHIGHDEN-26MAY06"
    assert market.status == "open"
    assert market.yes_ask == Decimal("0.4500")
    assert market.yes_bid == Decimal("0.4300")
    assert market.close_time == close_at
