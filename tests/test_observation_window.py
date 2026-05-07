from __future__ import annotations

from datetime import date, datetime, timedelta
from datetime import timezone as _timezone

import pytest
import pytz

from bot.markets.observation_window import observation_window


UTC = _timezone.utc


@pytest.mark.parametrize(
    "tz_name, obs_date, expected_start, expected_end",
    [
        (
            "America/Denver",
            date(2026, 1, 15),
            datetime(2026, 1, 15, 7, 0, tzinfo=UTC),
            datetime(2026, 1, 16, 7, 0, tzinfo=UTC),
        ),
        (
            "America/Denver",
            date(2026, 7, 15),
            datetime(2026, 7, 15, 7, 0, tzinfo=UTC),
            datetime(2026, 7, 16, 7, 0, tzinfo=UTC),
        ),
        (
            "America/Denver",
            date(2026, 3, 8),
            datetime(2026, 3, 8, 7, 0, tzinfo=UTC),
            datetime(2026, 3, 9, 7, 0, tzinfo=UTC),
        ),
        (
            "America/Denver",
            date(2026, 11, 1),
            datetime(2026, 11, 1, 7, 0, tzinfo=UTC),
            datetime(2026, 11, 2, 7, 0, tzinfo=UTC),
        ),
        (
            "America/Chicago",
            date(2026, 1, 15),
            datetime(2026, 1, 15, 6, 0, tzinfo=UTC),
            datetime(2026, 1, 16, 6, 0, tzinfo=UTC),
        ),
        (
            "America/Chicago",
            date(2026, 7, 15),
            datetime(2026, 7, 15, 6, 0, tzinfo=UTC),
            datetime(2026, 7, 16, 6, 0, tzinfo=UTC),
        ),
        (
            "America/Chicago",
            date(2026, 3, 8),
            datetime(2026, 3, 8, 6, 0, tzinfo=UTC),
            datetime(2026, 3, 9, 6, 0, tzinfo=UTC),
        ),
        (
            "America/New_York",
            date(2026, 2, 10),
            datetime(2026, 2, 10, 5, 0, tzinfo=UTC),
            datetime(2026, 2, 11, 5, 0, tzinfo=UTC),
        ),
        (
            "America/New_York",
            date(2026, 11, 1),
            datetime(2026, 11, 1, 5, 0, tzinfo=UTC),
            datetime(2026, 11, 2, 5, 0, tzinfo=UTC),
        ),
        (
            "America/Los_Angeles",
            date(2026, 6, 21),
            datetime(2026, 6, 21, 8, 0, tzinfo=UTC),
            datetime(2026, 6, 22, 8, 0, tzinfo=UTC),
        ),
    ],
)
def test_observation_window_golden(
    tz_name: str,
    obs_date: date,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    start, end = observation_window(tz_name, obs_date)
    assert start == expected_start
    assert end == expected_end


@pytest.mark.parametrize(
    "tz_name, obs_date",
    [
        ("America/Denver", date(2026, 3, 8)),
        ("America/Denver", date(2026, 11, 1)),
        ("America/Chicago", date(2026, 3, 8)),
        ("America/New_York", date(2026, 11, 1)),
        ("America/Los_Angeles", date(2026, 1, 15)),
    ],
)
def test_window_length_is_exactly_24h(tz_name: str, obs_date: date) -> None:
    start, end = observation_window(tz_name, obs_date)
    assert end - start == timedelta(hours=24)


@pytest.mark.parametrize(
    "tz_name, obs_date",
    [
        ("America/Denver", date(2026, 1, 15)),
        ("America/Chicago", date(2026, 7, 15)),
        ("America/New_York", date(2026, 11, 1)),
        ("America/Los_Angeles", date(2026, 6, 21)),
    ],
)
def test_outputs_are_tz_aware_utc(tz_name: str, obs_date: date) -> None:
    start, end = observation_window(tz_name, obs_date)
    assert start.tzinfo is not None
    assert end.tzinfo is not None
    assert start.utcoffset() == timedelta(0)
    assert end.utcoffset() == timedelta(0)


def test_bad_timezone_raises() -> None:
    with pytest.raises(pytz.exceptions.UnknownTimeZoneError):
        observation_window("Mars/Olympus_Mons", date(2026, 1, 15))
