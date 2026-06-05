import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytz
from pydantic import BaseModel, ConfigDict

from bot.backtest import gefs_grib
from bot.forecast.cdf import EnsembleCDF

logger = logging.getLogger(__name__)

GEFS_MEMBERS: tuple[str, ...] = ("gec00",) + tuple(f"gep{i:02d}" for i in range(1, 31))
GEFS_CYCLE_HOURS: tuple[int, ...] = (0, 6, 12, 18)
_DEFAULT_PUBLICATION_LAG = timedelta(hours=4, minutes=30)


class StationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    latitude: float
    longitude: float
    timezone: str


class ForecastReplay(Protocol):
    async def replay(
        self,
        station: StationSpec,
        valid_date: date,
        as_of: datetime,
    ) -> EnsembleCDF: ...


class GefsGribForecastReplay:
    def __init__(
        self,
        client,
        cache_dir: Path,
        publication_lag: timedelta = _DEFAULT_PUBLICATION_LAG,
    ) -> None:
        self._client = client
        self._cache_dir = Path(cache_dir)
        self._publication_lag = publication_lag

    async def replay(
        self,
        station: StationSpec,
        valid_date: date,
        as_of: datetime,
    ) -> EnsembleCDF:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        init_time = self._pick_cycle(as_of)
        assert init_time + self._publication_lag <= as_of, (
            f"init {init_time.isoformat()} would use look-ahead data vs as_of {as_of.isoformat()}"
        )

        cache_path = self._cache_path(station, init_time, valid_date)
        highs = self._load_cache(cache_path)
        if highs is None:
            highs = await self._build_member_highs(station, init_time, valid_date)
            self._save_cache(cache_path, highs)
        return EnsembleCDF.from_members(highs, smoothing=1.0)

    def _pick_cycle(self, as_of: datetime) -> datetime:
        as_of_utc = as_of.astimezone(timezone.utc)
        floor_hour = (as_of_utc.hour // 6) * 6
        candidate = as_of_utc.replace(hour=floor_hour, minute=0, second=0, microsecond=0)
        while candidate + self._publication_lag > as_of_utc:
            candidate = candidate - timedelta(hours=6)
        return candidate

    def _window_hours(
        self, init_time: datetime, valid_date: date, station_tz: str
    ) -> tuple[int, int]:
        tz = pytz.timezone(station_tz)
        local_start = tz.localize(datetime.combine(valid_date, time(0, 0)))
        local_end = tz.localize(datetime.combine(valid_date + timedelta(days=1), time(0, 0)))
        start_h = int((local_start.astimezone(timezone.utc) - init_time).total_seconds() // 3600)
        end_h = int((local_end.astimezone(timezone.utc) - init_time).total_seconds() // 3600)
        if end_h <= 0:
            raise ValueError(
                f"valid_date {valid_date} ({station_tz}) is before init {init_time.isoformat()}"
            )
        return start_h, end_h

    def _tmp_fxx(self, start_h: int, end_h: int) -> list[int]:
        first = max(0, -(-start_h // 3) * 3)
        return list(range(first, end_h, 3))

    def _tmax_fxx(self, start_h: int, end_h: int) -> list[int]:
        first = max(6, -(-start_h // 6) * 6 + 6)
        return list(range(first, end_h + 1, 6))

    async def _build_member_highs(
        self,
        station: StationSpec,
        init_time: datetime,
        valid_date: date,
    ) -> np.ndarray:
        start_h, end_h = self._window_hours(init_time, valid_date, station.timezone)
        # TMAX windows are UTC-aligned, so they cannot tile a local civil day; the in-window
        # 3-hourly TMP point samples cover the boundary hours the contained windows miss
        tmp_fxx = self._tmp_fxx(start_h, end_h)
        tmax_fxx = self._tmax_fxx(start_h, end_h)

        highs: list[float] = []
        for member in GEFS_MEMBERS:
            samples: list[float] = []
            for fxx in tmp_fxx:
                grib_bytes = await gefs_grib.fetch_member_field(
                    init_time.date(),
                    cycle=init_time.hour,
                    member=member,
                    fxx=fxx,
                    client=self._client,
                )
                samples.append(
                    gefs_grib.decode_point(
                        grib_bytes,
                        latitude=station.latitude,
                        longitude=station.longitude,
                    )
                )
            for fxx in tmax_fxx:
                grib_bytes = await gefs_grib.fetch_member_tmax(
                    init_time.date(),
                    cycle=init_time.hour,
                    member=member,
                    fxx=fxx,
                    client=self._client,
                )
                samples.append(
                    gefs_grib.decode_point(
                        grib_bytes,
                        latitude=station.latitude,
                        longitude=station.longitude,
                    )
                )
            highs.append(max(samples))
        return np.asarray(highs, dtype=np.float64)

    def _cache_path(self, station: StationSpec, init_time: datetime, valid_date: date) -> Path:
        stamp = init_time.strftime("%Y%m%dT%HZ")
        return self._cache_dir / f"{station.name}_{stamp}_{valid_date.isoformat()}.parquet"

    def _load_cache(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        table = pq.read_table(path)
        return table.column("high").to_numpy().astype(np.float64, copy=False)

    def _save_cache(self, path: Path, highs: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "member": list(GEFS_MEMBERS),
                "high": highs.tolist(),
            }
        )
        pq.write_table(table, path)
