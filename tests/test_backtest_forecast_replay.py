from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from bot.backtest import gefs_grib
from bot.backtest.forecast_replay import (
    GEFS_MEMBERS,
    ForecastReplay,
    GefsGribForecastReplay,
    StationSpec,
)
from bot.forecast.cdf import EnsembleCDF


def _make_station() -> StationSpec:
    return StationSpec(
        name="KDEN",
        latitude=39.86,
        longitude=-104.67,
        timezone="America/Denver",
    )


def test_gefs_members_constant_has_31_unique_names() -> None:
    assert len(GEFS_MEMBERS) == 31
    assert len(set(GEFS_MEMBERS)) == 31
    assert GEFS_MEMBERS[0] == "gec00"
    assert "gep01" in GEFS_MEMBERS
    assert "gep30" in GEFS_MEMBERS


def test_forecast_replay_protocol_is_satisfied_by_gefs_replay(tmp_path) -> None:
    replay: ForecastReplay = GefsGribForecastReplay(client=None, cache_dir=tmp_path)
    assert hasattr(replay, "replay")


def test_pick_cycle_returns_latest_cycle_satisfying_publication_lag(tmp_path) -> None:
    replay = GefsGribForecastReplay(client=None, cache_dir=tmp_path)
    as_of = datetime(2024, 11, 21, 10, 30, tzinfo=timezone.utc)

    init = replay._pick_cycle(as_of)

    assert init == datetime(2024, 11, 21, 6, 0, tzinfo=timezone.utc)
    assert init + replay._publication_lag <= as_of


def test_pick_cycle_backs_off_when_publication_lag_not_yet_elapsed(tmp_path) -> None:
    replay = GefsGribForecastReplay(client=None, cache_dir=tmp_path)
    as_of = datetime(2024, 11, 21, 4, 0, tzinfo=timezone.utc)

    init = replay._pick_cycle(as_of)

    assert init == datetime(2024, 11, 20, 18, 0, tzinfo=timezone.utc)
    assert init + replay._publication_lag <= as_of


def test_pick_cycle_accepts_exact_publication_lag_boundary(tmp_path) -> None:
    replay = GefsGribForecastReplay(
        client=None,
        cache_dir=tmp_path,
        publication_lag=timedelta(hours=4, minutes=30),
    )
    as_of = datetime(2024, 11, 21, 4, 30, tzinfo=timezone.utc)

    init = replay._pick_cycle(as_of)

    assert init == datetime(2024, 11, 21, 0, 0, tzinfo=timezone.utc)


async def test_replay_returns_cdf_with_31_members_and_matching_prob_range(
    monkeypatch, tmp_path
) -> None:
    expected_highs = np.linspace(50.0, 80.0, 31)

    async def fake_fetch(d, cycle, member, fxx, client):
        return member.encode()

    def fake_decode(grib_bytes, latitude, longitude):
        member = grib_bytes.decode()
        idx = GEFS_MEMBERS.index(member)
        return float(expected_highs[idx])

    monkeypatch.setattr(gefs_grib, "fetch_member_field", fake_fetch)
    monkeypatch.setattr(gefs_grib, "decode_point", fake_decode)

    replay = GefsGribForecastReplay(client=None, cache_dir=tmp_path)
    cdf = await replay.replay(
        _make_station(),
        valid_date=date(2024, 11, 21),
        as_of=datetime(2024, 11, 21, 23, 0, tzinfo=timezone.utc),
    )

    reference = EnsembleCDF.from_members(expected_highs, smoothing=1.0)
    assert cdf.prob_range(60.0, 65.0) == pytest.approx(reference.prob_range(60.0, 65.0))
    assert cdf.prob_range(70.0, 80.0) == pytest.approx(reference.prob_range(70.0, 80.0))
    assert cdf.cdf(65.0) == pytest.approx(reference.cdf(65.0))


def test_pick_cycle_backs_off_multiple_cycles_for_large_publication_lag(tmp_path) -> None:
    replay = GefsGribForecastReplay(
        client=None,
        cache_dir=tmp_path,
        publication_lag=timedelta(hours=20),
    )
    as_of = datetime(2024, 11, 21, 12, 0, tzinfo=timezone.utc)

    init = replay._pick_cycle(as_of)

    assert init == datetime(2024, 11, 20, 12, 0, tzinfo=timezone.utc)
    assert init + replay._publication_lag <= as_of


async def test_replay_caches_member_highs_across_repeated_calls(monkeypatch, tmp_path) -> None:
    call_count = {"fetch": 0}

    async def fake_fetch(d, cycle, member, fxx, client):
        call_count["fetch"] += 1
        return member.encode()

    def fake_decode(grib_bytes, latitude, longitude):
        member = grib_bytes.decode()
        return float(60 + GEFS_MEMBERS.index(member) * 0.1)

    monkeypatch.setattr(gefs_grib, "fetch_member_field", fake_fetch)
    monkeypatch.setattr(gefs_grib, "decode_point", fake_decode)

    replay = GefsGribForecastReplay(client=None, cache_dir=tmp_path)
    station = _make_station()
    valid = date(2024, 11, 21)
    as_of = datetime(2024, 11, 21, 23, 0, tzinfo=timezone.utc)

    cdf1 = await replay.replay(station, valid_date=valid, as_of=as_of)
    first = call_count["fetch"]
    cdf2 = await replay.replay(station, valid_date=valid, as_of=as_of)

    assert first > 0
    assert call_count["fetch"] == first
    assert cdf1.cdf(62.0) == pytest.approx(cdf2.cdf(62.0))
