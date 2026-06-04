from __future__ import annotations

import httpx

from bot.backtest.normalize import CanonicalSnapshot, from_kalshi_api

_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
_PAGE_LIMIT = 200


async def fetch_settled(
    series: str,
    min_ts: int,
    max_ts: int,
    client: httpx.AsyncClient,
) -> list[CanonicalSnapshot]:
    snapshots: list[CanonicalSnapshot] = []
    cursor: str | None = None
    while True:
        params: dict[str, object] = {
            "status": "settled",
            "series_ticker": series,
            "min_close_ts": min_ts,
            "max_close_ts": max_ts,
            "limit": _PAGE_LIMIT,
        }
        if cursor:
            params["cursor"] = cursor
        response = await client.get(_MARKETS_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        for market in payload.get("markets") or []:
            snapshots.append(from_kalshi_api(market))
        cursor = payload.get("cursor") or None
        if not cursor:
            return snapshots
