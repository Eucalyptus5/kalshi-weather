from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np
import pytest

from bot.backtest import gefs_grib
from bot.backtest.forecast_replay import (
    GEFS_MEMBERS,
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


def _offline_client() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _gefs_handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        name = request.url.path.rsplit("/", 1)[-1]
        if name.endswith(".idx"):
            fxx = int(name.removesuffix(".idx").rsplit("f", 1)[-1])
            tmp_fcst = ":anl:" if fxx == 0 else f":{fxx} hour fcst:"
            tmax_fcst = f":{6 * ((fxx - 1) // 6)}-{fxx} hour max fcst:"
            idx_text = (
                f"1:0:d=2024112118:TMAX:2 m above ground{tmax_fcst}ENS=+1\n"
                f"2:100:d=2024112118:TMP:2 m above ground{tmp_fcst}ENS=+1\n"
                f"3:200:d=2024112118:DPT:2 m above ground{tmp_fcst}ENS=+1\n"
            )
            return httpx.Response(200, content=idx_text.encode())
        return httpx.Response(206, content=name.split(".")[0].encode())

    return handler


def test_gefs_members_constant_has_31_unique_names() -> None:
    assert len(GEFS_MEMBERS) == 31
    assert len(set(GEFS_MEMBERS)) == 31
    assert GEFS_MEMBERS[0] == "gec00"
    assert "gep01" in GEFS_MEMBERS
    assert "gep30" in GEFS_MEMBERS


def test_pick_cycle_returns_latest_cycle_satisfying_publication_lag(tmp_path) -> None:
    replay = GefsGribForecastReplay(client=_offline_client(), cache_dir=tmp_path)
    as_of = datetime(2024, 11, 21, 10, 30, tzinfo=timezone.utc)

    init = replay._pick_cycle(as_of)

    assert init == datetime(2024, 11, 21, 6, 0, tzinfo=timezone.utc)
    assert init + replay._publication_lag <= as_of


def test_pick_cycle_backs_off_when_publication_lag_not_yet_elapsed(tmp_path) -> None:
    replay = GefsGribForecastReplay(client=_offline_client(), cache_dir=tmp_path)
    as_of = datetime(2024, 11, 21, 4, 0, tzinfo=timezone.utc)

    init = replay._pick_cycle(as_of)

    assert init == datetime(2024, 11, 20, 18, 0, tzinfo=timezone.utc)
    assert init + replay._publication_lag <= as_of


def test_pick_cycle_accepts_exact_publication_lag_boundary(tmp_path) -> None:
    replay = GefsGribForecastReplay(
        client=_offline_client(),
        cache_dir=tmp_path,
        publication_lag=timedelta(hours=4, minutes=30),
    )
    as_of = datetime(2024, 11, 21, 4, 30, tzinfo=timezone.utc)

    init = replay._pick_cycle(as_of)

    assert init == datetime(2024, 11, 21, 0, 0, tzinfo=timezone.utc)


def test_pick_cycle_backs_off_multiple_cycles_for_large_publication_lag(tmp_path) -> None:
    replay = GefsGribForecastReplay(
        client=_offline_client(),
        cache_dir=tmp_path,
        publication_lag=timedelta(hours=20),
    )
    as_of = datetime(2024, 11, 21, 12, 0, tzinfo=timezone.utc)

    init = replay._pick_cycle(as_of)

    assert init == datetime(2024, 11, 20, 12, 0, tzinfo=timezone.utc)
    assert init + replay._publication_lag <= as_of


def test_window_hours_and_fxx_steps_follow_3hourly_archive_layout(tmp_path) -> None:
    replay = GefsGribForecastReplay(client=_offline_client(), cache_dir=tmp_path)
    init = datetime(2024, 11, 20, 12, 0, tzinfo=timezone.utc)

    start_h, end_h = replay._window_hours(init, date(2024, 11, 21), "America/Denver")

    assert (start_h, end_h) == (19, 43)
    assert replay._tmp_fxx(start_h, end_h) == [21, 24, 27, 30, 33, 36, 39, 42]
    assert replay._tmax_fxx(start_h, end_h) == [30, 36, 42]


def test_window_hours_clamps_tmp_steps_to_archive_start(tmp_path) -> None:
    replay = GefsGribForecastReplay(client=_offline_client(), cache_dir=tmp_path)
    init = datetime(2024, 11, 21, 12, 0, tzinfo=timezone.utc)

    start_h, end_h = replay._window_hours(init, date(2024, 11, 21), "America/Denver")

    assert (start_h, end_h) == (-5, 19)
    assert replay._tmp_fxx(start_h, end_h) == [0, 3, 6, 9, 12, 15, 18]
    assert replay._tmax_fxx(start_h, end_h) == [6, 12, 18]


def test_window_hours_handles_dst_spring_forward_short_day(tmp_path) -> None:
    replay = GefsGribForecastReplay(client=_offline_client(), cache_dir=tmp_path)
    init = datetime(2025, 3, 9, 0, 0, tzinfo=timezone.utc)

    start_h, end_h = replay._window_hours(init, date(2025, 3, 9), "America/Denver")

    assert (start_h, end_h) == (7, 30)
    assert replay._tmp_fxx(start_h, end_h) == [9, 12, 15, 18, 21, 24, 27]
    assert replay._tmax_fxx(start_h, end_h) == [18, 24, 30]


def test_window_hours_raises_for_day_entirely_before_init(tmp_path) -> None:
    replay = GefsGribForecastReplay(client=_offline_client(), cache_dir=tmp_path)
    init = datetime(2024, 11, 22, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        replay._window_hours(init, date(2024, 11, 20), "America/Denver")


async def test_replay_returns_cdf_with_31_members_and_matching_prob_range(
    monkeypatch, tmp_path
) -> None:
    expected_highs = np.linspace(50.0, 80.0, 31)

    def fake_decode(grib_bytes: bytes, latitude: float, longitude: float) -> float:
        return float(expected_highs[GEFS_MEMBERS.index(grib_bytes.decode())])

    monkeypatch.setattr(gefs_grib, "decode_point", fake_decode)

    transport = httpx.MockTransport(_gefs_handler([]))
    async with httpx.AsyncClient(transport=transport) as client:
        replay = GefsGribForecastReplay(client=client, cache_dir=tmp_path)
        cdf = await replay.replay(
            _make_station(),
            valid_date=date(2024, 11, 21),
            as_of=datetime(2024, 11, 21, 23, 0, tzinfo=timezone.utc),
        )

    assert cdf._members.shape == (31,)
    reference = EnsembleCDF.from_members(expected_highs, smoothing=1.0)
    assert cdf.prob_range(60.0, 65.0) == pytest.approx(reference.prob_range(60.0, 65.0))
    assert cdf.prob_range(70.0, 80.0) == pytest.approx(reference.prob_range(70.0, 80.0))
    assert cdf.cdf(65.0) == pytest.approx(reference.cdf(65.0))


async def test_replay_caches_member_highs_across_repeated_calls(monkeypatch, tmp_path) -> None:
    def fake_decode(grib_bytes: bytes, latitude: float, longitude: float) -> float:
        return float(60 + GEFS_MEMBERS.index(grib_bytes.decode()) * 0.1)

    monkeypatch.setattr(gefs_grib, "decode_point", fake_decode)

    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_gefs_handler(seen))) as client:
        replay = GefsGribForecastReplay(client=client, cache_dir=tmp_path)
        station = _make_station()
        valid = date(2024, 11, 21)
        as_of = datetime(2024, 11, 21, 23, 0, tzinfo=timezone.utc)

        cdf1 = await replay.replay(station, valid_date=valid, as_of=as_of)
        first = len(seen)
        cdf2 = await replay.replay(station, valid_date=valid, as_of=as_of)

    assert first > 0
    assert len(seen) == first
    assert cdf1.cdf(62.0) == pytest.approx(cdf2.cdf(62.0))
