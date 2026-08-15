from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import httpx

from bot.backtest.backfill import fetch_settled
from bot.backtest.normalize import CanonicalSnapshot


_DATA = Path(__file__).parent / "data"
_PAGE1 = _DATA / "kalshi_settled_page1.json"
_PAGE2_EMPTY = _DATA / "kalshi_settled_page2_empty.json"
_BETWEEN = _DATA / "kalshi_settled_between.json"


def _scripted_handler(
    pages: list[str],
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = pages[len(seen)]
        seen.append(request)
        return httpx.Response(200, content=body)

    return handler, seen


async def test_fetch_settled_parses_recorded_b3_market() -> None:
    pages = [_PAGE1.read_text(), _PAGE2_EMPTY.read_text()]
    handler, _ = _scripted_handler(pages)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snaps = await fetch_settled(
            "KXHIGHDEN", min_ts=1700000000, max_ts=1800000000, client=client
        )

    assert len(snaps) == 1
    snap = snaps[0]
    assert isinstance(snap, CanonicalSnapshot)
    assert snap.ticker == "KXHIGHDEN-26APR03-T58"
    assert snap.result == "no"
    assert snap.yes_ask == Decimal("1.00")
    assert snap.floor_strike == 58
    assert snap.observed_value == Decimal("54.00")


async def test_fetch_settled_pagination_stops_on_empty_cursor() -> None:
    page1_payload = json.loads(_PAGE1.read_text())
    pages = [_PAGE1.read_text(), _PAGE2_EMPTY.read_text()]
    handler, seen = _scripted_handler(pages)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snaps = await fetch_settled(
            "KXHIGHDEN", min_ts=1700000000, max_ts=1800000000, client=client
        )

    assert len(seen) == 2
    assert "cursor" not in seen[0].url.params
    assert seen[1].url.params["cursor"] == page1_payload["cursor"]
    assert len(snaps) == 1


async def test_fetch_settled_sends_settled_status_and_window_params() -> None:
    handler, seen = _scripted_handler([_PAGE2_EMPTY.read_text()])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fetch_settled("KXHIGHDEN", min_ts=1700000000, max_ts=1800000000, client=client)

    params = seen[0].url.params
    assert params["status"] == "settled"
    assert params["series_ticker"] == "KXHIGHDEN"
    assert params["min_close_ts"] == "1700000000"
    assert params["max_close_ts"] == "1800000000"
    assert params["limit"] == "200"
    assert str(seen[0].url).startswith("https://api.elections.kalshi.com/trade-api/v2/markets")


async def test_fetch_settled_carries_the_cap_strike_of_a_between_bracket() -> None:
    handler, _ = _scripted_handler([_BETWEEN.read_text()])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snaps = await fetch_settled(
            "KXHIGHDEN", min_ts=1754000000, max_ts=1756000000, client=client
        )

    assert len(snaps) == 1
    snap = snaps[0]
    assert snap.ticker == "KXHIGHDEN-26AUG13-B8889"
    assert snap.strike_type == "between"
    assert snap.floor_strike == 88
    assert snap.cap_strike == 89


async def test_fetch_settled_leaves_cap_strike_unset_on_a_one_sided_bracket() -> None:
    handler, _ = _scripted_handler([_PAGE1.read_text(), _PAGE2_EMPTY.read_text()])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snaps = await fetch_settled(
            "KXHIGHDEN", min_ts=1700000000, max_ts=1800000000, client=client
        )

    assert snaps[0].floor_strike == 58
    assert snaps[0].cap_strike is None
