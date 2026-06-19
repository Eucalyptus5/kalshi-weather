from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import httpx

logger = logging.getLogger(__name__)

ENDPOINT = "https://aviationweather.gov/api/data/metar"

_T_GROUP = re.compile(r"\bT([01]\d{3})([01]\d{3})\b")


@dataclass(frozen=True, slots=True)
class StationObservation:
    station: str
    valid_time: datetime
    publication_time: datetime
    temp_f: Decimal
    is_special: bool
    raw: str
    source: str


class MetarClient:
    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient(timeout=30.0)

    async def fetch_observations(
        self,
        stations: list[str],
        hours: float = 2.0,
    ) -> list[StationObservation]:
        params = {
            "ids": ",".join(stations),
            "format": "json",
            "hours": hours,
        }
        response = await self._http.get(ENDPOINT, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("aviationweather metar response is not a JSON array")

        out: list[StationObservation] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            parsed = _parse_entry(entry)
            if parsed is not None:
                out.append(parsed)
        logger.info(
            "metar_fetch stations=%s hours=%s returned=%d",
            ",".join(stations),
            hours,
            len(out),
        )
        return out

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


def _parse_entry(entry: dict[str, object]) -> StationObservation | None:
    station = entry.get("icaoId")
    obs_time = entry.get("obsTime")
    receipt_time = entry.get("receiptTime")
    raw_ob = entry.get("rawOb")

    if not isinstance(station, str):
        return None
    if not isinstance(obs_time, int):
        return None
    if not isinstance(receipt_time, str):
        return None
    if not isinstance(raw_ob, str):
        return None

    temp_c = _parse_temp_c(raw_ob, entry.get("temp"))
    if temp_c is None:
        return None

    return StationObservation(
        station=station,
        valid_time=datetime.fromtimestamp(obs_time, tz=timezone.utc),
        publication_time=_parse_iso_utc(receipt_time),
        temp_f=_celsius_to_fahrenheit(temp_c),
        is_special=_is_speci(raw_ob),
        raw=raw_ob,
        source="metar",
    )


def _parse_temp_c(raw_ob: str | None, fallback_c: object | None) -> Decimal | None:
    if raw_ob is not None:
        match = _T_GROUP.search(raw_ob)
        if match is not None:
            sign_digit = match.group(1)[0]
            tenths = int(match.group(1)[1:])
            value = Decimal(tenths) / Decimal(10)
            if sign_digit == "1":
                value = -value
            return value
    if fallback_c is None:
        return None
    if not isinstance(fallback_c, (int, float, str, Decimal)):
        return None
    return Decimal(str(fallback_c))


def _celsius_to_fahrenheit(temp_c: Decimal) -> Decimal:
    return temp_c * Decimal("9") / Decimal("5") + Decimal("32")


def _is_speci(raw_ob: str) -> bool:
    return raw_ob.startswith("SPECI ")


def _parse_iso_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    return dt.astimezone(timezone.utc)
