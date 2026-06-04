from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytz

from bot.forecast.open_meteo import MIN_HOURS_PER_DAY

__all__ = ["MIN_HOURS_PER_DAY", "daily_high_members"]


def daily_high_members(
    times: list[datetime],
    member_matrix: np.ndarray,
    station_tz: str,
    valid_date: date,
) -> np.ndarray:
    if member_matrix.ndim != 2:
        raise ValueError("member_matrix must be 2-D (n_members, n_times)")
    if member_matrix.shape[1] != len(times):
        raise ValueError(
            f"member_matrix has {member_matrix.shape[1]} time columns, got {len(times)} times"
        )

    tz = pytz.timezone(station_tz)
    idxs = [i for i, t in enumerate(times) if t.astimezone(tz).date() == valid_date]
    if len(idxs) < MIN_HOURS_PER_DAY:
        raise ValueError(
            f"valid_date {valid_date} has {len(idxs)} in-window hours, "
            f"need >= MIN_HOURS_PER_DAY ({MIN_HOURS_PER_DAY})"
        )
    return member_matrix[:, idxs].max(axis=1)
