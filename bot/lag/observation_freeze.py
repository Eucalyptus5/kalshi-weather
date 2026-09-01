from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import httpx

from bot.lag.r0_universe import freeze_digest
from bot.main import STATIONS
from bot.markets.observation_window import observation_window
from bot.observations.basis_check import IEM_1MIN_URL, fetch_iem_1min_asos_archive
from bot.observations.metar import StationObservation
from bot.validation.reconcile import ACIS_URL, ACISClient


logger = logging.getLogger(__name__)

MAX = "max"
READING_SOURCE = "iem_1min_asos_archive"
INDEX_NAME = "index.json"
_FETCH_TIMEOUT_SECONDS = 120.0

F2_STATIONS: Mapping[str, str] = {config.station: config.timezone for config in STATIONS.values()}


@dataclass(frozen=True, slots=True, kw_only=True)
class StationDay:
    station: str
    event_date: date
    extreme: str
    window_start: datetime
    window_end: datetime
    readings: tuple[StationObservation, ...]
    acis_f: Decimal | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationSidecar:
    station: str
    timezone: str
    days: Mapping[date, StationDay]
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StationIndexRow:
    station: str
    timezone: str
    sha256: str
    event_days: int
    decoded_minutes: Mapping[date, int]
    missing_acis: tuple[date, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationIndex:
    observed_at: datetime
    start_date: date
    end_date: date
    extreme: str
    source: str
    iem_url: str
    acis_url: str
    stations: Mapping[str, StationIndexRow]
    sha256: str


def sidecar_payload(station: str, tz_name: str, days: Sequence[StationDay]) -> dict:
    return {
        "station": station,
        "timezone": tz_name,
        "extreme": MAX,
        "source": READING_SOURCE,
        "days": [
            {
                "event_date": day.event_date.isoformat(),
                "window_start": day.window_start.isoformat(),
                "window_end": day.window_end.isoformat(),
                "acis_f": None if day.acis_f is None else str(day.acis_f),
                "readings": [
                    {"valid_time": row.valid_time.isoformat(), "temp_f": str(row.temp_f)}
                    for row in day.readings
                ],
            }
            for day in sorted(days, key=lambda item: item.event_date)
        ],
    }


def write_observation_sidecar(
    path: Path, station: str, tz_name: str, days: Sequence[StationDay]
) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = sidecar_payload(station, tz_name, days)
    digest = freeze_digest(payload)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=1))
    return digest


def read_observation_sidecar(path: Path) -> ObservationSidecar:
    payload = json.loads(path.read_text())
    if "sha256" not in payload:
        raise ValueError(f"{path} carries no sha256")
    stored = payload.pop("sha256")
    digest = freeze_digest(payload)
    if digest != stored:
        raise ValueError(f"{path} does not match the sha256 it carries")
    station = payload["station"]
    source = payload["source"]
    days = {}
    for row in payload["days"]:
        event_date = date.fromisoformat(row["event_date"])
        days[event_date] = StationDay(
            station=station,
            event_date=event_date,
            extreme=payload["extreme"],
            window_start=datetime.fromisoformat(row["window_start"]),
            window_end=datetime.fromisoformat(row["window_end"]),
            readings=tuple(
                _reading(station, source, reading["valid_time"], reading["temp_f"])
                for reading in row["readings"]
            ),
            acis_f=None if row["acis_f"] is None else Decimal(row["acis_f"]),
        )
    return ObservationSidecar(
        station=station,
        timezone=payload["timezone"],
        days=days,
        sha256=digest,
    )


def _reading(station: str, source: str, valid_time: str, temp_f: str) -> StationObservation:
    stamp = datetime.fromisoformat(valid_time)
    return StationObservation(
        station=station,
        valid_time=stamp,
        publication_time=stamp,
        temp_f=Decimal(temp_f),
        is_special=False,
        raw="",
        source=source,
    )


async def _fetch_station(
    station: str,
    event_dates: Sequence[date],
    client: httpx.AsyncClient,
    acis: ACISClient,
) -> list[StationDay]:
    tz_name = F2_STATIONS[station]
    observations = await fetch_iem_1min_asos_archive(
        station, event_dates[0], event_dates[-1], client
    )
    days = []
    for event_date in event_dates:
        start, end = observation_window(tz_name, event_date)
        readings = tuple(
            sorted(
                (row for row in observations if start <= row.valid_time < end),
                key=lambda row: row.valid_time,
            )
        )
        days.append(
            StationDay(
                station=station,
                event_date=event_date,
                extreme=MAX,
                window_start=start,
                window_end=end,
                readings=readings,
                acis_f=await acis.fetch_daily_high(station[1:], event_date),
            )
        )
    logger.info(
        "f2 freeze station=%s days=%d decoded_minutes=%d missing_acis=%d",
        station,
        len(days),
        sum(len(day.readings) for day in days),
        sum(1 for day in days if day.acis_f is None),
    )
    return days


async def _fetch_stations(
    stations: Sequence[str],
    event_dates: Sequence[date],
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, list[StationDay]]:
    span = sorted(set(event_dates))
    async with httpx.AsyncClient(transport=transport, timeout=_FETCH_TIMEOUT_SECONDS) as client:
        acis = ACISClient(http_client=client)
        return {
            station: await _fetch_station(station, span, client, acis)
            for station in sorted(set(stations))
        }


def pull_observations(
    stations: Sequence[str],
    event_dates: Sequence[date],
    directory: Path,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, str]:
    pulled = asyncio.run(_fetch_stations(stations, event_dates, transport))
    return {
        station: write_observation_sidecar(
            directory / f"{station}.json", station, F2_STATIONS[station], days
        )
        for station, days in pulled.items()
    }


def index_payload(observed_at: datetime, sidecars: Mapping[str, ObservationSidecar]) -> dict:
    span = sorted({event_date for sidecar in sidecars.values() for event_date in sidecar.days})
    stations = []
    for station in sorted(sidecars):
        sidecar = sidecars[station]
        days = sorted(sidecar.days)
        stations.append(
            {
                "station": station,
                "timezone": sidecar.timezone,
                "sha256": sidecar.sha256,
                "event_days": len(days),
                "decoded_minutes": {
                    day.isoformat(): len(sidecar.days[day].readings) for day in days
                },
                "missing_acis": [
                    day.isoformat() for day in days if sidecar.days[day].acis_f is None
                ],
            }
        )
    return {
        "observed_at": observed_at.isoformat(),
        "start_date": span[0].isoformat(),
        "end_date": span[-1].isoformat(),
        "extreme": MAX,
        "source": READING_SOURCE,
        "iem_url": IEM_1MIN_URL,
        "acis_url": ACIS_URL,
        "stations": stations,
    }


def write_observation_index(directory: Path, observed_at: datetime) -> str:
    path = directory / INDEX_NAME
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    sidecars = {
        sidecar_path.stem: read_observation_sidecar(sidecar_path)
        for sidecar_path in sorted(directory.glob("*.json"))
    }
    payload = index_payload(observed_at, sidecars)
    digest = freeze_digest(payload)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=1))
    return digest


def read_observation_index(directory: Path) -> ObservationIndex:
    path = directory / INDEX_NAME
    payload = json.loads(path.read_text())
    if "sha256" not in payload:
        raise ValueError(f"{path} carries no sha256")
    stored = payload.pop("sha256")
    digest = freeze_digest(payload)
    if digest != stored:
        raise ValueError(f"{path} does not match the sha256 it carries")
    return ObservationIndex(
        observed_at=datetime.fromisoformat(payload["observed_at"]),
        start_date=date.fromisoformat(payload["start_date"]),
        end_date=date.fromisoformat(payload["end_date"]),
        extreme=payload["extreme"],
        source=payload["source"],
        iem_url=payload["iem_url"],
        acis_url=payload["acis_url"],
        stations={
            row["station"]: StationIndexRow(
                station=row["station"],
                timezone=row["timezone"],
                sha256=row["sha256"],
                event_days=row["event_days"],
                decoded_minutes={
                    date.fromisoformat(day): count for day, count in row["decoded_minutes"].items()
                },
                missing_acis=tuple(date.fromisoformat(day) for day in row["missing_acis"]),
            )
            for row in payload["stations"]
        },
        sha256=digest,
    )
