from __future__ import annotations

from datetime import date, datetime, time, timedelta
from datetime import timezone as _timezone

import pytz


def observation_window(timezone: str, observation_date: date) -> tuple[datetime, datetime]:
    """Return (start_utc, end_utc) for the LST observation day matching observation_date.

    NWS CLI uses local standard time, so the window is exactly 24 hours regardless of DST.
    """
    tz = pytz.timezone(timezone)
    standard_offset = tz.localize(datetime(observation_date.year, 1, 15), is_dst=False).utcoffset()
    naive_midnight = datetime.combine(observation_date, time())
    start_utc = (naive_midnight - standard_offset).replace(tzinfo=_timezone.utc)
    return start_utc, start_utc + timedelta(hours=24)
