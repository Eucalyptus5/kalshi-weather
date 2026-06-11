from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.backtest.depth_table import LEAD_BUCKETS, lead_bucket_for  # noqa: E402
from bot.backtest.historical_open_meteo import day_end_utc  # noqa: E402
from bot.forecast.open_meteo import ENDPOINT as ENSEMBLE_ENDPOINT  # noqa: E402
from bot.forecast.open_meteo import _daily_max_per_member  # noqa: E402
from bot.main import STATIONS  # noqa: E402

DETERMINISTIC_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 16
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "gfs_seamless_spread_calibration.json"
)
BUCKET_BOUNDS: dict[str, tuple[int, int | None]] = {
    "<=2h": (0, 2),
    "2-8h": (2, 8),
    "8-24h": (8, 24),
    "24-72h": (24, 72),
    ">72h": (72, None),
}


def collect_points(
    ensemble_payload: dict[str, object],
    deterministic_payload: dict[str, object],
    station_tz: str,
    issuance: datetime,
) -> list[tuple[str, np.ndarray]]:
    member_highs = _daily_max_per_member(ensemble_payload, station_tz)
    det_highs = _daily_max_per_member(deterministic_payload, station_tz)
    samples: list[tuple[str, np.ndarray]] = []
    for valid_date, highs in sorted(member_highs.items()):
        det = det_highs.get(valid_date)
        if det is None:
            continue
        lead = day_end_utc(valid_date, station_tz) - issuance
        if lead <= timedelta(0):
            continue
        samples.append((lead_bucket_for(lead), (highs - float(det[0])) ** 2))
    return samples


def sigma_table(samples: list[tuple[str, np.ndarray]]) -> list[dict[str, object]]:
    grouped: dict[str, list[np.ndarray]] = {}
    for bucket, squared in samples:
        grouped.setdefault(bucket, []).append(squared)
    rows: list[dict[str, object]] = []
    for bucket in LEAD_BUCKETS:
        if bucket not in grouped:
            continue
        pooled = np.concatenate(grouped[bucket])
        min_hours, max_hours = BUCKET_BOUNDS[bucket]
        rows.append(
            {
                "bucket": bucket,
                "min_hours": min_hours,
                "max_hours": max_hours,
                "sigma": float(np.sqrt(pooled.mean())),
                "pairs": len(grouped[bucket]),
                "points": int(pooled.size),
            }
        )
    return rows


async def fetch_samples() -> list[tuple[str, np.ndarray]]:
    base_params = {
        "hourly": "temperature_2m",
        "models": "gfs_seamless",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "forecast_days": FORECAST_DAYS,
    }
    samples: list[tuple[str, np.ndarray]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for cfg in STATIONS.values():
            params = {"latitude": cfg.latitude, "longitude": cfg.longitude, **base_params}
            issuance = datetime.now(tz=timezone.utc)
            ensemble = await client.get(ENSEMBLE_ENDPOINT, params=params)
            ensemble.raise_for_status()
            deterministic = await client.get(DETERMINISTIC_ENDPOINT, params=params)
            deterministic.raise_for_status()
            samples.extend(
                collect_points(ensemble.json(), deterministic.json(), cfg.timezone, issuance)
            )
    return samples


def main() -> None:
    rows = sigma_table(asyncio.run(fetch_samples()))
    if not rows:
        raise SystemExit("no overlapping forecast days; nothing to calibrate")
    table = {
        "model": "gfs_seamless",
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "buckets": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH} buckets={len(rows)}")


if __name__ == "__main__":
    main()
