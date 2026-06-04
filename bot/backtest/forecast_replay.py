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
from bot.backtest.daily_high import daily_high_members
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

    def _fxx_list_for(self, init_time: datetime, valid_date: date, station_tz: str) -> list[int]:
        tz = pytz.timezone(station_tz)
        local_start = tz.localize(datetime.combine(valid_date, time(0, 0)))
        local_end = tz.localize(datetime.combine(valid_date + timedelta(days=1), time(0, 0)))
        utc_start = local_start.astimezone(timezone.utc)
        utc_end = local_end.astimezone(timezone.utc)

        delta_start_h = (utc_start - init_time).total_seconds() / 3600.0
        delta_end_h = (utc_end - init_time).total_seconds() / 3600.0
        fxx_start = max(0, int(delta_start_h))
        fxx_end = int(delta_end_h) + 1
        if fxx_end <= fxx_start:
            raise ValueError(
                f"valid_date {valid_date} ({station_tz}) is before init {init_time.isoformat()}"
            )
        return list(range(fxx_start, fxx_end))

    async def _build_member_highs(
        self,
        station: StationSpec,
        init_time: datetime,
        valid_date: date,
    ) -> np.ndarray:
        fxx_list = self._fxx_list_for(init_time, valid_date, station.timezone)
        times = [init_time + timedelta(hours=f) for f in fxx_list]

        rows: list[list[float]] = []
        for member in GEFS_MEMBERS:
            series: list[float] = []
            for fxx in fxx_list:
                grib_bytes = await gefs_grib.fetch_member_field(
                    init_time.date(),
                    cycle=init_time.hour,
                    member=member,
                    fxx=fxx,
                    client=self._client,
                )
                series.append(
                    gefs_grib.decode_point(
                        grib_bytes,
                        latitude=station.latitude,
                        longitude=station.longitude,
                    )
                )
            rows.append(series)
        matrix = np.asarray(rows, dtype=np.float64)
        return daily_high_members(times, matrix, station.timezone, valid_date)

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
