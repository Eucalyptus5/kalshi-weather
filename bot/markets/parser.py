from __future__ import annotations

import logging
import re
from datetime import date
from decimal import Decimal
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

TickerKind = Literal["above", "below", "bracket"]

_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

_DAILY_DATE = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})$")
_MONTHLY_DATE = re.compile(r"^(\d{2})([A-Z]{3})$")
_ABOVE_STRIKE = re.compile(r"^T(\d+(?:\.\d+)?)$")
_BELOW_STRIKE = re.compile(r"^B(\d+(?:\.\d+)?)$")
_HIGH_STRIKE = re.compile(r"^\d+(?:\.\d+)?$")


class ParsedTicker(BaseModel):
    model_config = ConfigDict(frozen=True)

    series: str
    event_date: date
    is_monthly: bool
    strikes: tuple[Decimal, ...]
    kind: TickerKind
    raw: str

    @property
    def is_tail(self) -> bool:
        return len(self.strikes) == 1

    @property
    def is_bracket(self) -> bool:
        return len(self.strikes) == 2


def parse_ticker(ticker: str) -> ParsedTicker:
    """Parse a Kalshi KX-prefixed weather ticker. Raises ValueError on anything malformed."""
    if not ticker:
        raise ValueError("ticker is empty")
    if not ticker.startswith("KX"):
        raise ValueError(f"ticker does not start with KX: {ticker!r}")

    parts = ticker.split("-")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"ticker has {len(parts)} dash-separated parts, expected 3 or 4: {ticker!r}"
        )

    series = parts[0]
    event_date, is_monthly = _parse_date(parts[1], ticker)

    below_match = _BELOW_STRIKE.match(parts[2])
    above_match = _ABOVE_STRIKE.match(parts[2])

    if below_match:
        if len(parts) == 4:
            raise ValueError(f"B-form must be single-strike: got {ticker!r}")
        raw_strike = below_match.group(1)
        if "." not in raw_strike or raw_strike.rsplit(".", 1)[1] != "5":
            raise ValueError(
                f"B-form strike must be half-integer (<n>.5): got {raw_strike!r} in {ticker!r}"
            )
        low = Decimal(raw_strike.split(".", 1)[0])
        strikes: tuple[Decimal, ...] = (low, low + 1)
        kind: TickerKind = "bracket"
    elif above_match:
        low = Decimal(above_match.group(1))
        if len(parts) == 3:
            strikes = (low,)
            kind = "above"
        else:
            if not _HIGH_STRIKE.match(parts[3]):
                raise ValueError(f"high strike must match <number>: got {parts[3]!r} in {ticker!r}")
            high = Decimal(parts[3])
            if high <= low:
                raise ValueError(f"high strike {high} must exceed low {low} in {ticker!r}")
            strikes = (low, high)
            kind = "bracket"
    else:
        raise ValueError(
            f"strike component must match T<number> or B<number>: got {parts[2]!r} in {ticker!r}"
        )

    return ParsedTicker(
        series=series,
        event_date=event_date,
        is_monthly=is_monthly,
        strikes=strikes,
        kind=kind,
        raw=ticker,
    )


def event_yes_sum_ok(
    yes_prices: Sequence[Decimal],
    tolerance: Decimal = Decimal("0.05"),
) -> bool:
    """True if YES prices across an event's full bracket ladder sum to ~1.0; logs a warning when False."""
    total = sum(yes_prices, Decimal("0"))
    if abs(total - Decimal("1")) <= tolerance:
        return True
    logger.warning(
        "event yes sum off: total=%s tolerance=%s n=%d",
        total,
        tolerance,
        len(yes_prices),
    )
    return False


def _parse_date(component: str, ticker: str) -> tuple[date, bool]:
    daily = _DAILY_DATE.match(component)
    if daily:
        yy, mon, dd = daily.groups()
        month = _MONTHS.get(mon)
        if month is None:
            raise ValueError(f"unknown month {mon!r} in {ticker!r}")
        return date(2000 + int(yy), month, int(dd)), False

    monthly = _MONTHLY_DATE.match(component)
    if monthly:
        yy, mon = monthly.groups()
        month = _MONTHS.get(mon)
        if month is None:
            raise ValueError(f"unknown month {mon!r} in {ticker!r}")
        return date(2000 + int(yy), month, 1), True

    raise ValueError(f"date component {component!r} not in YYMMMDD or YYMMM form: {ticker!r}")
