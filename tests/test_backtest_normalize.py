from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from bot.backtest.normalize import from_kalshi_api, from_trevorjs


_FIXTURE = Path(__file__).parent / "data" / "kalshi_settled_market.json"


def _trevorjs_row(**overrides: object) -> dict:
    row: dict = {
        "ticker": "KXHIGHDEN-26APR03-T58",
        "event_ticker": "KXHIGHDEN-26APR03",
        "market_type": "binary",
        "title": "Will the high temp in Denver be 58 or above on Apr 3, 2026?",
        "yes_sub_title": "58 or above",
        "no_sub_title": "below 58",
        "status": "finalized",
        "yes_bid": 85,
        "yes_ask": 87,
        "no_bid": 13,
        "no_ask": 15,
        "last_price": 86,
        "volume": 12345,
        "volume_24h": 678,
        "open_interest": 234,
        "result": "no",
        "created_time": datetime(2026, 4, 2, 14, 0, tzinfo=timezone.utc),
        "open_time": datetime(2026, 4, 3, 14, 0, tzinfo=timezone.utc),
        "close_time": datetime(2026, 4, 4, 6, 59, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


def _kalshi_market(**overrides: object) -> dict:
    market: dict = {
        "ticker": "KXHIGHDEN-26APR03-T58",
        "event_ticker": "KXHIGHDEN-26APR03",
        "market_type": "binary",
        "status": "finalized",
        "result": "no",
        "yes_ask_dollars": "0.8700",
        "yes_bid_dollars": "0.8500",
        "no_ask_dollars": "0.1500",
        "no_bid_dollars": "0.1300",
        "last_price_dollars": "0.8600",
        "volume_fp": "12345.00",
        "volume_24h_fp": "678.00",
        "open_interest_fp": "234.00",
        "floor_strike": 58,
        "strike_type": "greater",
        "expiration_value": "54.00",
        "yes_bid_size_fp": "12.50",
        "no_bid_size_fp": "9.00",
        "created_time": "2026-04-02T14:00:00Z",
        "open_time": "2026-04-03T14:00:00Z",
        "close_time": "2026-04-04T06:59:00Z",
    }
    market.update(overrides)
    return market


def test_from_trevorjs_normalizes_cent_prices() -> None:
    snap = from_trevorjs(_trevorjs_row())

    assert snap.yes_ask == Decimal("0.87")
    assert snap.yes_bid == Decimal("0.85")
    assert snap.no_ask == Decimal("0.15")
    assert snap.no_bid == Decimal("0.13")
    assert snap.last_price == Decimal("0.86")
    assert snap.ticker == "KXHIGHDEN-26APR03-T58"
    assert snap.series_ticker == "KXHIGHDEN"


def test_from_kalshi_api_normalizes_dollar_strings() -> None:
    snap = from_kalshi_api(_kalshi_market())

    assert snap.yes_ask == Decimal("0.87")
    assert snap.yes_bid == Decimal("0.85")
    assert snap.no_ask == Decimal("0.15")
    assert snap.no_bid == Decimal("0.13")
    assert snap.last_price == Decimal("0.86")
    assert snap.ticker == "KXHIGHDEN-26APR03-T58"
    assert snap.series_ticker == "KXHIGHDEN"


def test_both_sources_produce_equal_price_fields() -> None:
    snap_trev = from_trevorjs(_trevorjs_row())
    snap_api = from_kalshi_api(_kalshi_market())

    assert snap_trev.yes_ask == snap_api.yes_ask
    assert snap_trev.yes_bid == snap_api.yes_bid
    assert snap_trev.no_ask == snap_api.no_ask
    assert snap_trev.no_bid == snap_api.no_bid
    assert snap_trev.last_price == snap_api.last_price
    assert snap_trev.result == snap_api.result == "no"


def test_result_carries_through_open_and_settled_trevorjs() -> None:
    open_snap = from_trevorjs(_trevorjs_row(result=""))
    settled_snap = from_trevorjs(_trevorjs_row(result="no"))

    assert open_snap.result == ""
    assert settled_snap.result == "no"


def test_kalshi_api_settled_fixture_carries_strike_and_observed_value() -> None:
    market = json.loads(_FIXTURE.read_text())

    snap = from_kalshi_api(market)

    assert snap.result == "no"
    assert snap.floor_strike == 58
    assert snap.strike_type == "greater"
    assert snap.observed_value == Decimal("54.00")
    assert snap.yes_ask == Decimal("1.00")
    assert snap.yes_bid == Decimal("0.99")
    assert snap.status == "finalized"
    assert snap.ticker == "KXHIGHDEN-26APR03-T58"
    assert snap.event_ticker == "KXHIGHDEN-26APR03"
    assert snap.close_time == datetime(2026, 4, 4, 6, 59, tzinfo=timezone.utc)


def test_trevorjs_lacks_floor_strike_and_observed_value() -> None:
    snap = from_trevorjs(_trevorjs_row())

    assert snap.floor_strike is None
    assert snap.strike_type is None
    assert snap.observed_value is None


def test_canonical_snapshot_is_frozen() -> None:
    snap = from_trevorjs(_trevorjs_row())
    import pytest

    with pytest.raises(Exception):
        snap.yes_ask = Decimal("0.50")  # type: ignore[misc]
