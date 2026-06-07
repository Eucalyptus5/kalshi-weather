from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import pytz

from bot.backtest.daily_high import MIN_HOURS_PER_DAY, daily_high_members
from bot.forecast.open_meteo import _MEMBER_KEY, _daily_max_per_member

_FIXTURE_PATH = Path(__file__).parent / "data" / "open_meteo_ensemble_payload.json"


def _matrix_from_payload(payload: dict) -> tuple[list[datetime], np.ndarray]:
    hourly = payload["hourly"]
    times_raw = hourly["time"]
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times_raw]
    member_keys = sorted(k for k in hourly.keys() if _MEMBER_KEY.match(k))
    series = [hourly[k] for k in member_keys]
    matrix = np.asarray(series, dtype=np.float64)
    return times, matrix


def test_known_per_member_maxima_over_calendar_day() -> None:
    tz = "America/Denver"
    valid = date(2024, 6, 15)
    times = [datetime(2024, 6, 15, h, tzinfo=timezone.utc) for h in range(24)] + [
        datetime(2024, 6, 16, h, tzinfo=timezone.utc) for h in range(24)
    ]
    n = len(times)
    m0 = [60.0] * n
    m0[15] = 95.0
    m1 = [70.0] * n
    m1[20] = 88.0
    m2 = [50.0] * n
    m2[25] = 77.0
    matrix = np.asarray([m0, m1, m2], dtype=np.float64)

    out = daily_high_members(times, matrix, tz, valid)

    assert out.shape == (3,)
    assert out.dtype == np.float64
    assert out[0] == pytest.approx(95.0)
    assert out[1] == pytest.approx(88.0)
    assert out[2] == pytest.approx(77.0)


def test_constant_value_propagates_when_no_outlier_in_window() -> None:
    tz = "America/Denver"
    valid = date(2024, 6, 15)
    times = [datetime(2024, 6, 15, h, tzinfo=timezone.utc) for h in range(24)] + [
        datetime(2024, 6, 16, h, tzinfo=timezone.utc) for h in range(24)
    ]
    n = len(times)
    m0 = [60.0] * n
    m0[0] = 99.0
    m0[47] = 99.0
    matrix = np.asarray([m0], dtype=np.float64)

    out = daily_high_members(times, matrix, tz, valid)

    assert out[0] == pytest.approx(60.0)


def test_insufficient_in_window_hours_raises() -> None:
    tz = "America/Denver"
    valid = date(2024, 6, 15)
    times = [datetime(2024, 6, 15, h, tzinfo=timezone.utc) for h in range(6, 17)]
    matrix = np.asarray([[1.0] * len(times)], dtype=np.float64)

    with pytest.raises(ValueError, match="MIN_HOURS_PER_DAY"):
        daily_high_members(times, matrix, tz, valid)


def test_min_hours_per_day_constant_matches_live() -> None:
    from bot.forecast.open_meteo import MIN_HOURS_PER_DAY as LIVE_MIN

    assert MIN_HOURS_PER_DAY == LIVE_MIN == 12


@pytest.mark.parametrize(
    "case_key",
    ["spring_forward_kden", "fall_back_kchi", "control_kden"],
)
def test_parity_with_live_daily_max_per_member(case_key: str) -> None:
    fixture = json.loads(_FIXTURE_PATH.read_text())
    case = fixture[case_key]
    tz = case["tz"]
    valid = date.fromisoformat(case["valid_date"])
    payload = case["payload"]

    live_out = _daily_max_per_member(payload, tz)
    assert valid in live_out, f"live function dropped {valid} for {case_key}"
    live_arr = live_out[valid]

    times, matrix = _matrix_from_payload(payload)
    bt05_arr = daily_high_members(times, matrix, tz, valid)

    assert np.array_equal(live_arr, bt05_arr), (
        f"divergence on {case_key}: live={live_arr.tolist()}, bt05={bt05_arr.tolist()}"
    )


def test_spring_forward_window_is_23_hours() -> None:
    fixture = json.loads(_FIXTURE_PATH.read_text())
    case = fixture["spring_forward_kden"]
    tz = case["tz"]
    valid = date.fromisoformat(case["valid_date"])
    times, _ = _matrix_from_payload(case["payload"])

    local_tz = pytz.timezone(tz)
    count = sum(1 for t in times if t.astimezone(local_tz).date() == valid)
    assert count == 23


def test_fall_back_window_is_25_hours() -> None:
    fixture = json.loads(_FIXTURE_PATH.read_text())
    case = fixture["fall_back_kchi"]
    tz = case["tz"]
    valid = date.fromisoformat(case["valid_date"])
    times, _ = _matrix_from_payload(case["payload"])

    local_tz = pytz.timezone(tz)
    count = sum(1 for t in times if t.astimezone(local_tz).date() == valid)
    assert count == 25
