from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import numpy as np

from bot.backtest.gefs_grib import kelvin_to_fahrenheit, tmp2m_byte_range
from bot.lag.forecast_classes import LST_FULL, lst_window_hours
from bot.lag.forecast_sample import SampleLeg


HRRR_BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
# Only these four cycles run past f18; the others stop there and cannot cover a 24h or 36h window.
EXTENDED_CYCLES: tuple[int, ...] = (0, 6, 12, 18)
PUBLICATION_ALLOWANCE = timedelta(hours=4)
FETCH_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 0.25
POINTS_KEY = "points"

FieldKey = tuple[datetime, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class GridPoint:
    y: int
    x: int
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True, eq=False)
class HrrrGrid:
    latitudes: np.ndarray
    longitudes: np.ndarray

    @classmethod
    def from_message(cls, grib_bytes: bytes) -> HrrrGrid:
        return cls(*_grid_arrays(grib_bytes))

    def nearest(self, latitude: float, longitude: float) -> GridPoint:
        squared = (self.latitudes - latitude) ** 2 + (self.longitudes - longitude) ** 2
        y, x = np.unravel_index(int(squared.argmin()), squared.shape)
        return GridPoint(
            y=int(y),
            x=int(x),
            latitude=float(self.latitudes[y, x]),
            longitude=float(self.longitudes[y, x]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LegFields:
    run: datetime
    hours: tuple[datetime, ...]
    fxx: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class HrrrPullReport:
    values: Mapping[FieldKey, Mapping[str, Decimal]]
    points: Mapping[str, GridPoint]
    downloaded: int
    cache_hits: int
    transferred_bytes: int
    missing: tuple[str, ...]


class DecodedFieldCache:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._values: dict[FieldKey, dict[str, Decimal]] = {}
        self.points: dict[str, GridPoint] = {}
        if path is not None and path.exists():
            for line in path.read_text().splitlines():
                payload = json.loads(line)
                if POINTS_KEY in payload:
                    self.points = {
                        station: GridPoint(**cell) for station, cell in payload[POINTS_KEY].items()
                    }
                    continue
                self._values[field_key(payload["field"])] = {
                    station: Decimal(value) for station, value in payload["values"].items()
                }

    def get(self, key: FieldKey) -> dict[str, Decimal] | None:
        return self._values.get(key)

    def put(self, key: FieldKey, values: Mapping[str, Decimal]) -> None:
        self._values[key] = dict(values)
        self._append(
            {
                "field": field_name(key),
                "values": {station: str(value) for station, value in values.items()},
            }
        )

    def put_points(self, points: Mapping[str, GridPoint]) -> None:
        self.points = dict(points)
        self._append({POINTS_KEY: {station: asdict(cell) for station, cell in points.items()}})

    def _append(self, payload: dict) -> None:
        if self._path is None:
            return
        with self._path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")


def build_hrrr_url(run_date: date, cycle: int, fxx: int) -> str:
    return f"{HRRR_BUCKET}/hrrr.{run_date:%Y%m%d}/conus/hrrr.t{cycle:02d}z.wrfsfcf{fxx:02d}.grib2"


def run_directory_url(run: datetime) -> str:
    return f"{HRRR_BUCKET}/hrrr.{run:%Y%m%d}/conus"


def choose_run(as_of: datetime) -> datetime:
    limit = as_of - PUBLICATION_ALLOWANCE
    midnight = limit.replace(hour=0, minute=0, second=0, microsecond=0)
    behind = [cycle for cycle in EXTENDED_CYCLES if cycle <= limit.hour]
    if behind:
        return midnight + timedelta(hours=max(behind))
    return midnight - timedelta(days=1) + timedelta(hours=EXTENDED_CYCLES[-1])


def leg_fields(leg: SampleLeg) -> LegFields:
    run = choose_run(leg.as_of)
    hours = lst_window_hours(leg.timezone, leg.event_date, LST_FULL)
    return LegFields(
        run=run,
        hours=hours,
        fxx=tuple(int((hour - run).total_seconds() // 3600) for hour in hours),
    )


def field_name(key: FieldKey) -> str:
    run, fxx = key
    return f"{run:%Y-%m-%dT%H}Z/f{fxx:03d}"


def field_key(name: str) -> FieldKey:
    stamp, fxx = name.split("/f")
    return datetime.strptime(stamp, "%Y-%m-%dT%HZ").replace(tzinfo=timezone.utc), int(fxx)


def fold_longitudes(longitudes: np.ndarray) -> np.ndarray:
    return ((longitudes + 180.0) % 360.0) - 180.0


def decode_points(grib_bytes: bytes, points: Sequence[GridPoint]) -> list[Decimal]:
    field = _field_array(grib_bytes)
    return [
        Decimal(f"{kelvin_to_fahrenheit(float(field[point.y, point.x])):.2f}") for point in points
    ]


async def fetch_tmp2m_field(run: datetime, fxx: int, client: httpx.AsyncClient) -> bytes:
    url = build_hrrr_url(run.date(), run.hour, fxx)
    idx = await _get(f"{url}.idx", {}, client)
    start, end = tmp2m_byte_range(idx.text, fxx)
    record = await _get(url, {"Range": f"bytes={start}-{end}"}, client)
    return record.content


async def pull_fields(
    requests: Sequence[FieldKey],
    stations: Mapping[str, tuple[float, float]],
    cache: DecodedFieldCache,
    client: httpx.AsyncClient,
    concurrency: int,
) -> HrrrPullReport:
    values: dict[FieldKey, Mapping[str, Decimal]] = {}
    pending: list[FieldKey] = []
    for key in requests:
        cached = cache.get(key)
        if cached is None:
            pending.append(key)
        else:
            values[key] = cached

    points: dict[str, GridPoint] = dict(cache.points)
    missing: list[str] = []
    downloaded = 0
    transferred = 0
    gate = asyncio.Semaphore(concurrency)
    decoding = asyncio.Lock()

    async def pull(key: FieldKey) -> None:
        nonlocal downloaded, transferred
        run, fxx = key
        async with gate:
            try:
                body = await fetch_tmp2m_field(run, fxx, client)
            except (httpx.HTTPStatusError, httpx.TransportError, ValueError):
                missing.append(field_name(key))
                return
        async with decoding:
            if not points:
                grid = HrrrGrid.from_message(body)
                points.update({name: grid.nearest(*coords) for name, coords in stations.items()})
                cache.put_points(points)
            names = list(points)
            decoded = decode_points(body, [points[name] for name in names])
            values[key] = dict(zip(names, decoded))
            cache.put(key, values[key])
            downloaded += 1
            transferred += len(body)

    await asyncio.gather(*(pull(key) for key in pending))
    return HrrrPullReport(
        values=values,
        points=points,
        downloaded=downloaded,
        cache_hits=len(requests) - len(pending),
        transferred_bytes=transferred,
        missing=tuple(sorted(missing)),
    )


async def _get(url: str, headers: Mapping[str, str], client: httpx.AsyncClient) -> httpx.Response:
    failure: httpx.TransportError | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            response = await client.get(url, headers=dict(headers))
        except httpx.TransportError as error:
            failure = error
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * 2**attempt)
            continue
        response.raise_for_status()
        return response
    raise failure


def _grid_arrays(grib_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    import cfgrib

    with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
        handle.write(grib_bytes)
        handle.flush()
        dataset = cfgrib.open_file(Path(handle.name), indexpath="")
        return (
            np.asarray(dataset.variables["latitude"].data),
            fold_longitudes(np.asarray(dataset.variables["longitude"].data)),
        )


def _field_array(grib_bytes: bytes) -> np.ndarray:
    import cfgrib

    with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
        handle.write(grib_bytes)
        handle.flush()
        dataset = cfgrib.open_file(Path(handle.name), indexpath="")
        return np.asarray(dataset.variables["t2m"].data[:, :])
