from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict


class CanonicalSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    event_ticker: str
    series_ticker: str
    status: str
    result: str

    yes_ask: Decimal
    yes_bid: Decimal
    no_ask: Decimal
    no_bid: Decimal
    last_price: Decimal

    volume: Decimal
    volume_24h: Decimal
    open_interest: Decimal

    open_time: datetime | None = None
    close_time: datetime | None = None
    created_time: datetime | None = None

    floor_strike: int | None = None
    strike_type: str | None = None
    observed_value: Decimal | None = None

    yes_bid_size: Decimal | None = None
    no_bid_size: Decimal | None = None


_CENTS = Decimal(100)


def _cents_to_dollars(cents: int) -> Decimal:
    return Decimal(int(cents)) / _CENTS


def _api_decimal(market: dict, key: str) -> Decimal | None:
    value = market.get(key)
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        # some settled binaries carry the literal "No" in expiration_value
        return None


def _api_price(market: dict, key: str) -> Decimal:
    value = market.get(key)
    if value is None or value == "":
        return Decimal(0)
    return Decimal(str(value))


def _iso_to_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def from_trevorjs(row: dict) -> CanonicalSnapshot:
    ticker = row["ticker"]
    return CanonicalSnapshot(
        ticker=ticker,
        event_ticker=row["event_ticker"],
        series_ticker=ticker.split("-", 1)[0],
        status=row["status"],
        result=row.get("result", "") or "",
        yes_ask=_cents_to_dollars(row["yes_ask"]),
        yes_bid=_cents_to_dollars(row["yes_bid"]),
        no_ask=_cents_to_dollars(row["no_ask"]),
        no_bid=_cents_to_dollars(row["no_bid"]),
        last_price=_cents_to_dollars(row["last_price"]),
        volume=Decimal(int(row.get("volume", 0))),
        volume_24h=Decimal(int(row.get("volume_24h", 0))),
        open_interest=Decimal(int(row.get("open_interest", 0))),
        open_time=_iso_to_datetime(row.get("open_time")),
        close_time=_iso_to_datetime(row.get("close_time")),
        created_time=_iso_to_datetime(row.get("created_time")),
    )


def from_kalshi_api(market: dict) -> CanonicalSnapshot:
    ticker = market["ticker"]
    return CanonicalSnapshot(
        ticker=ticker,
        event_ticker=market["event_ticker"],
        series_ticker=ticker.split("-", 1)[0],
        status=market["status"],
        result=market.get("result", "") or "",
        yes_ask=_api_price(market, "yes_ask_dollars"),
        yes_bid=_api_price(market, "yes_bid_dollars"),
        no_ask=_api_price(market, "no_ask_dollars"),
        no_bid=_api_price(market, "no_bid_dollars"),
        last_price=_api_price(market, "last_price_dollars"),
        volume=_api_price(market, "volume_fp"),
        volume_24h=_api_price(market, "volume_24h_fp"),
        open_interest=_api_price(market, "open_interest_fp"),
        open_time=_iso_to_datetime(market.get("open_time")),
        close_time=_iso_to_datetime(market.get("close_time")),
        created_time=_iso_to_datetime(market.get("created_time")),
        floor_strike=market.get("floor_strike"),
        strike_type=market.get("strike_type"),
        observed_value=_api_decimal(market, "expiration_value"),
        yes_bid_size=_api_decimal(market, "yes_bid_size_fp"),
        no_bid_size=_api_decimal(market, "no_bid_size_fp"),
    )
