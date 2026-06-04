from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa


_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string()),
        pa.field("event_ticker", pa.string()),
        pa.field("market_type", pa.string()),
        pa.field("title", pa.string()),
        pa.field("yes_sub_title", pa.string()),
        pa.field("no_sub_title", pa.string()),
        pa.field("status", pa.string()),
        pa.field("yes_bid", pa.int64()),
        pa.field("yes_ask", pa.int64()),
        pa.field("no_bid", pa.int64()),
        pa.field("no_ask", pa.int64()),
        pa.field("last_price", pa.int64()),
        pa.field("volume", pa.int64()),
        pa.field("volume_24h", pa.int64()),
        pa.field("open_interest", pa.int64()),
        pa.field("result", pa.string()),
        pa.field("created_time", pa.timestamp("us", tz="UTC")),
        pa.field("open_time", pa.timestamp("us", tz="UTC")),
        pa.field("close_time", pa.timestamp("us", tz="UTC")),
    ]
)


def _row(
    ticker: str,
    *,
    event_ticker: str,
    status: str = "finalized",
    yes_bid: int = 50,
    yes_ask: int = 55,
    no_bid: int = 45,
    no_ask: int = 50,
    last_price: int = 52,
    volume: int = 100,
    volume_24h: int = 20,
    open_interest: int = 30,
    result: str = "yes",
    created_time: datetime | None = None,
    open_time: datetime | None = None,
    close_time: datetime | None = None,
) -> dict:
    base = datetime(2024, 11, 20, 12, 0, tzinfo=timezone.utc)
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "market_type": "binary",
        "title": f"{ticker} synthetic fixture",
        "yes_sub_title": "yes",
        "no_sub_title": "no",
        "status": status,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": last_price,
        "volume": volume,
        "volume_24h": volume_24h,
        "open_interest": open_interest,
        "result": result,
        "created_time": created_time or base,
        "open_time": open_time or base,
        "close_time": close_time or base,
    }


def make_market_state_table() -> pa.Table:
    den = [
        _row(
            "KXHIGHDEN-26APR03-T58",
            event_ticker="KXHIGHDEN-26APR03",
            yes_bid=85,
            yes_ask=87,
            no_bid=13,
            no_ask=15,
            last_price=86,
            result="no",
        ),
        _row(
            "KXHIGHDEN-26APR03-T60",
            event_ticker="KXHIGHDEN-26APR03",
            yes_bid=60,
            yes_ask=63,
            no_bid=37,
            no_ask=40,
            last_price=61,
            result="yes",
        ),
        _row(
            "KXHIGHDEN-26APR04-T58",
            event_ticker="KXHIGHDEN-26APR04",
            yes_bid=70,
            yes_ask=72,
            no_bid=28,
            no_ask=30,
            last_price=71,
            result="yes",
        ),
        _row(
            "KXHIGHDEN-26APR05-T62",
            event_ticker="KXHIGHDEN-26APR05",
            status="active",
            yes_bid=40,
            yes_ask=45,
            no_bid=55,
            no_ask=60,
            last_price=42,
            result="",
        ),
    ]
    non_weather = [
        _row("KXPRES-26-DEM", event_ticker="KXPRES-26"),
        _row("KXNFL-26W14-SF", event_ticker="KXNFL-26W14"),
        _row("KXBTCD-26JUN09-T70000", event_ticker="KXBTCD-26JUN09"),
        _row("KXOIL-26Q2-T75", event_ticker="KXOIL-26Q2"),
        _row("KXFED-26JUN-T25", event_ticker="KXFED-26JUN"),
        _row("KXCPI-26MAY-T3", event_ticker="KXCPI-26MAY"),
    ]
    return pa.Table.from_pylist(den + non_weather, schema=_SCHEMA)
