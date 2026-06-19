from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal

import httpx

from bot.markets.observation_window import observation_window
from bot.observations.metar import MetarClient, StationObservation
from bot.validation.reconcile import ACISClient


logger = logging.getLogger(__name__)

IOWA_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_STATION_TIMEZONES: dict[str, str] = {
    "KDEN": "America/Denver",
    "KAUS": "America/Chicago",
    "KMDW": "America/Chicago",
    "KNYC": "America/New_York",
    "KPHL": "America/New_York",
    "KATL": "America/New_York",
    "KBOS": "America/New_York",
    "KDFW": "America/Chicago",
    "KDCA": "America/New_York",
    "KMIA": "America/New_York",
    "KLAX": "America/Los_Angeles",
}


@dataclass(frozen=True, slots=True)
class BasisCompareRow:
    station: str
    observation_day: date
    metar_max_f: Decimal
    acis_high_f: Decimal
    delta_f: Decimal
    basis_valid: bool


@dataclass(frozen=True, slots=True)
class BasisSummary:
    station: str
    n_days: int
    median_delta_f: Decimal
    p10_delta_f: Decimal
    p90_delta_f: Decimal
    fraction_invalid: Decimal


async def compare_basis(
    station: str,
    start_date: date,
    end_date: date,
    acis_client: ACISClient,
    metar_client: MetarClient | None,
    *,
    source: Literal["live_metar", "iowa_asos_archive"] = "live_metar",
    http_client: httpx.AsyncClient | None = None,
) -> list[BasisCompareRow]:
    tz_name = _STATION_TIMEZONES.get(station)
    if tz_name is None:
        raise ValueError(f"no timezone mapping for station {station}")
    if not station.startswith("K"):
        raise ValueError(f"expected ICAO station (leading K), got {station}")
    acis_sid = station[1:]

    if source == "iowa_asos_archive":
        if http_client is None:
            raise ValueError("iowa_asos_archive source requires http_client")
        observations = await _fetch_iowa_asos_archive(station, start_date, end_date, http_client)
    else:
        if metar_client is None:
            raise ValueError("live_metar source requires metar_client")
        span_hours = max(
            1.0,
            ((end_date - start_date).total_seconds() / 3600.0) + 48.0,
        )
        observations = await metar_client.fetch_observations([station], hours=span_hours)

    by_day: dict[date, list[StationObservation]] = {}
    for day_offset in range((end_date - start_date).days + 1):
        day = start_date + timedelta(days=day_offset)
        start_utc, end_utc = observation_window(tz_name, day)
        bucket = [o for o in observations if start_utc <= o.valid_time < end_utc]
        if bucket:
            by_day[day] = bucket

    rows: list[BasisCompareRow] = []
    for day in sorted(by_day):
        bucket = by_day[day]
        metar_max_f = max(o.temp_f for o in bucket)
        acis_high_f = await acis_client.fetch_daily_high(acis_sid, day)
        if acis_high_f is None:
            continue
        delta_f = metar_max_f - acis_high_f
        rows.append(
            BasisCompareRow(
                station=station,
                observation_day=day,
                metar_max_f=metar_max_f,
                acis_high_f=acis_high_f,
                delta_f=delta_f,
                basis_valid=abs(delta_f) < Decimal("1.0"),
            )
        )

    logger.info(
        "basis_compare station=%s start=%s end=%s source=%s rows=%d",
        station,
        start_date.isoformat(),
        end_date.isoformat(),
        source,
        len(rows),
    )
    return rows


def summarize_basis(rows: list[BasisCompareRow]) -> list[BasisSummary]:
    if not rows:
        return []
    by_station: dict[str, list[BasisCompareRow]] = {}
    for row in rows:
        by_station.setdefault(row.station, []).append(row)

    out: list[BasisSummary] = []
    for station in sorted(by_station):
        group = by_station[station]
        deltas = sorted(r.delta_f for r in group)
        n = len(group)
        invalid_count = sum(1 for r in group if not r.basis_valid)
        out.append(
            BasisSummary(
                station=station,
                n_days=n,
                median_delta_f=_quantile(deltas, Decimal("0.5")),
                p10_delta_f=_quantile(deltas, Decimal("0.1")),
                p90_delta_f=_quantile(deltas, Decimal("0.9")),
                fraction_invalid=Decimal(invalid_count) / Decimal(n),
            )
        )
    return out


def _quantile(sorted_values: list[Decimal], q: Decimal) -> Decimal:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = q * Decimal(n - 1)
    idx = int(rank.to_integral_value(rounding="ROUND_HALF_UP"))
    if idx < 0:
        idx = 0
    if idx > n - 1:
        idx = n - 1
    return sorted_values[idx]


async def _fetch_iowa_asos_archive(
    station: str,
    start_date: date,
    end_date: date,
    http_client: httpx.AsyncClient,
) -> list[StationObservation]:
    params = [
        ("station", station),
        ("data", "tmpc"),
        ("report_type", "3"),
        ("report_type", "4"),
        ("year1", str(start_date.year)),
        ("month1", str(start_date.month)),
        ("day1", str(start_date.day)),
        ("year2", str(end_date.year)),
        ("month2", str(end_date.month)),
        ("day2", str(end_date.day)),
        ("format", "onlycomma"),
    ]
    response = await http_client.get(IOWA_ASOS_URL, params=params)
    response.raise_for_status()
    body = response.text

    out: list[StationObservation] = []
    lines = body.splitlines()
    if not lines:
        return out
    header = lines[0].split(",")
    if header[:3] != ["station", "valid", "tmpc"]:
        raise ValueError(f"unexpected iowa asos header: {lines[0]}")

    for raw_line in lines[1:]:
        if not raw_line.strip():
            continue
        parts = raw_line.split(",")
        if len(parts) < 3:
            continue
        sid, valid_s, tmpc_s = parts[0], parts[1], parts[2]
        if tmpc_s == "M":
            continue
        valid_time = datetime.strptime(valid_s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        temp_c = Decimal(tmpc_s)
        temp_f = temp_c * Decimal("9") / Decimal("5") + Decimal("32")
        out.append(
            StationObservation(
                station=sid,
                valid_time=valid_time,
                publication_time=valid_time,
                temp_f=temp_f,
                is_special=False,
                raw="",
                source="iowa_asos_archive",
            )
        )
    return out
