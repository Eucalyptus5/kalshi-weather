from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pytz
from scipy.stats import norm

from bot.backtest.daily_high import daily_high_members
from bot.backtest.depth_table import lead_bucket_for
from bot.backtest.forecast_replay import StationSpec
from bot.forecast.cdf import EnsembleCDF

ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "data" / "gfs_seamless_spread_calibration.json"
)

_MEMBER_Z = norm.ppf(np.arange(1, 32) / 32)
_MEMBER_Z_STD = float(np.std(_MEMBER_Z, ddof=1))


class SpreadCalibration:
    def __init__(self, sigmas: Mapping[str, float]) -> None:
        self._sigmas = dict(sigmas)

    @classmethod
    def load(cls, path: Path) -> SpreadCalibration:
        payload = json.loads(Path(path).read_text())
        return cls({row["bucket"]: float(row["sigma"]) for row in payload["buckets"]})

    def sigma_for(self, lead: timedelta) -> float:
        bucket = lead_bucket_for(lead)
        if bucket not in self._sigmas:
            raise ValueError(f"no spread calibration for lead bucket {bucket!r}")
        return self._sigmas[bucket]


def day_end_utc(valid_date: date, station_tz: str) -> datetime:
    tz = pytz.timezone(station_tz)
    local_end = tz.localize(datetime.combine(valid_date + timedelta(days=1), time(0, 0)))
    return local_end.astimezone(timezone.utc)


class HistoricalOpenMeteoForecastReplay:
    def __init__(self, client: httpx.AsyncClient, calibration: SpreadCalibration) -> None:
        self._client = client
        self._calibration = calibration

    async def replay(
        self,
        station: StationSpec,
        valid_date: date,
        as_of: datetime,
    ) -> EnsembleCDF:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        sigma = self._calibration.sigma_for(day_end_utc(valid_date, station.timezone) - as_of)
        params = {
            "latitude": station.latitude,
            "longitude": station.longitude,
            "hourly": "temperature_2m",
            "models": "gfs_seamless",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
            "start_date": valid_date.isoformat(),
            "end_date": (valid_date + timedelta(days=1)).isoformat(),
        }
        response = await self._client.get(ENDPOINT, params=params)
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise ValueError("open-meteo response missing 'hourly' object")
        times_raw = hourly.get("time")
        temps = hourly.get("temperature_2m")
        if not isinstance(times_raw, list) or not times_raw:
            raise ValueError("open-meteo response missing 'hourly.time'")
        if not isinstance(temps, list) or len(temps) != len(times_raw):
            raise ValueError("open-meteo response 'temperature_2m' length mismatch")
        times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times_raw]
        matrix = np.asarray([temps], dtype=np.float64)
        det_high = float(daily_high_members(times, matrix, station.timezone, valid_date)[0])
        members = det_high + sigma * _MEMBER_Z / _MEMBER_Z_STD
        return EnsembleCDF.from_members(members, smoothing=1.0)


def cli_source() -> HistoricalOpenMeteoForecastReplay:
    return HistoricalOpenMeteoForecastReplay(
        httpx.AsyncClient(timeout=30.0),
        SpreadCalibration.load(CALIBRATION_PATH),
    )
