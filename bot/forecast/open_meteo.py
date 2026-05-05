from __future__ import annotations

import logging
import re
from datetime import date, datetime
from datetime import timezone as _timezone

import httpx
import numpy as np
import pytz
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

ENDPOINT = "https://ensemble-api.open-meteo.com/v1/ensemble"
MIN_HOURS_PER_DAY = 12

_MEMBER_KEY = re.compile(r"^temperature_2m(?:_member(\d+))?$")


class StationForecast(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    station: str
    latitude: float
    longitude: float
    timezone: str
    run_time: datetime
    daily_highs: dict[date, np.ndarray]


class OpenMeteoClient:
    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient(timeout=30.0)

    async def fetch_station(
        self,
        station: str,
        latitude: float,
        longitude: float,
        timezone: str,
        forecast_days: int = 7,
    ) -> StationForecast:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "models": "gfs_seamless",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
            "forecast_days": forecast_days,
        }
        run_time = datetime.now(tz=_timezone.utc)
        response = await self._http.get(ENDPOINT, params=params)
        response.raise_for_status()
        payload = response.json()
        daily_highs = _daily_max_per_member(payload, timezone)
        logger.info(
            "open_meteo_fetch station=%s lat=%s lon=%s days=%d members_per_day=%s",
            station,
            latitude,
            longitude,
            forecast_days,
            {str(d): int(a.size) for d, a in daily_highs.items()},
        )
        return StationForecast(
            station=station,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
            run_time=run_time,
            daily_highs=daily_highs,
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


def _daily_max_per_member(
    payload: dict[str, object],
    station_tz: str,
) -> dict[date, np.ndarray]:
    hourly = payload["hourly"]
    if not isinstance(hourly, dict):
        raise ValueError("open-meteo response missing 'hourly' object")

    times_raw = hourly.get("time")
    if not isinstance(times_raw, list) or not times_raw:
        raise ValueError("open-meteo response missing 'hourly.time'")

    member_keys = sorted(k for k in hourly.keys() if _MEMBER_KEY.match(k))
    if not member_keys:
        raise ValueError("open-meteo response has no temperature_2m series")

    series = []
    for key in member_keys:
        values = hourly[key]
        if not isinstance(values, list) or len(values) != len(times_raw):
            raise ValueError(f"open-meteo response series '{key}' length mismatch")
        series.append(values)
    arr = np.asarray(series, dtype=np.float64)

    tz = pytz.timezone(station_tz)
    utc_times = [datetime.fromisoformat(t).replace(tzinfo=_timezone.utc) for t in times_raw]
    local_dates = [t.astimezone(tz).date() for t in utc_times]

    by_day: dict[date, list[int]] = {}
    for idx, d in enumerate(local_dates):
        by_day.setdefault(d, []).append(idx)

    out: dict[date, np.ndarray] = {}
    for d, idxs in by_day.items():
        if len(idxs) < MIN_HOURS_PER_DAY:
            continue
        out[d] = arr[:, idxs].max(axis=1)
    return out
