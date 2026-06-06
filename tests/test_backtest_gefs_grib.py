import importlib
import math
from datetime import date
from pathlib import Path

import httpx
import pytest

from bot.backtest.gefs_grib import (
    build_grib_url,
    decode_point,
    fetch_member_field,
    fetch_member_tmax,
    kelvin_to_fahrenheit,
    tmax2m_byte_range,
    tmp2m_byte_range,
)

_IDX_FIXTURE = Path(__file__).parent / "data" / "gefs_gep01_f024.idx"
_GRIB_FIXTURE = Path(__file__).parent / "data" / "gefs_tmp2m_msg.grib2"


def _eccodes_available() -> bool:
    try:
        importlib.import_module("cfgrib")
    except RuntimeError:
        return False
    return True


def test_tmp2m_byte_range_returns_record_to_next_record_offsets() -> None:
    idx_text = _IDX_FIXTURE.read_text()

    start, end = tmp2m_byte_range(idx_text, fxx=24)

    assert start == 4066078
    assert end == 4820322


def test_tmp2m_byte_range_raises_when_fxx_mismatch() -> None:
    idx_text = _IDX_FIXTURE.read_text()

    with pytest.raises(ValueError):
        tmp2m_byte_range(idx_text, fxx=48)


def test_tmp2m_byte_range_raises_when_record_missing() -> None:
    idx_text = "1:0:d=2024112000:PRES:surface:24 hour fcst:ENS=+1\n"

    with pytest.raises(ValueError):
        tmp2m_byte_range(idx_text, fxx=24)


def test_tmax2m_byte_range_returns_record_to_next_record_offsets() -> None:
    idx_text = _IDX_FIXTURE.read_text()

    start, end = tmax2m_byte_range(idx_text, fxx=24)

    assert start == 412345
    assert end == 1234566


def test_tmax2m_byte_range_uses_six_hour_reset_window_marker() -> None:
    idx_text = (
        "1:0:d=2024112000:TMAX:2 m above ground:18-21 hour max fcst:ENS=+1\n"
        "2:700000:d=2024112000:TMIN:2 m above ground:18-21 hour min fcst:ENS=+1\n"
    )

    assert tmax2m_byte_range(idx_text, fxx=21) == (0, 699999)


def test_tmax2m_byte_range_raises_when_window_mismatch() -> None:
    idx_text = _IDX_FIXTURE.read_text()

    with pytest.raises(ValueError):
        tmax2m_byte_range(idx_text, fxx=30)


def test_build_grib_url_matches_noaa_layout() -> None:
    url = build_grib_url(date(2024, 11, 20), cycle=0, member="gep01", fxx=24)
    assert url == (
        "https://noaa-gefs-pds.s3.amazonaws.com/"
        "gefs.20241120/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f024"
    )


async def test_fetch_member_field_issues_range_header_for_tmp_record() -> None:
    idx_body = _IDX_FIXTURE.read_text().encode()
    grib_body = b"\x00" * 754245
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if url.endswith(".idx"):
            return httpx.Response(200, content=idx_body)
        return httpx.Response(206, content=grib_body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out = await fetch_member_field(
            date(2024, 11, 20), cycle=0, member="gep01", fxx=24, client=client
        )

    assert out == grib_body
    assert len(seen) == 2
    assert str(seen[0].url).endswith("gep01.t00z.pgrb2s.0p25.f024.idx")
    assert str(seen[1].url).endswith("gep01.t00z.pgrb2s.0p25.f024")
    assert seen[1].headers["Range"] == "bytes=4066078-4820322"


async def test_fetch_member_tmax_issues_range_header_for_tmax_record() -> None:
    idx_body = _IDX_FIXTURE.read_text().encode()
    grib_body = b"\x00" * 822222
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if str(request.url).endswith(".idx"):
            return httpx.Response(200, content=idx_body)
        return httpx.Response(206, content=grib_body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out = await fetch_member_tmax(
            date(2024, 11, 20), cycle=0, member="gep01", fxx=24, client=client
        )

    assert out == grib_body
    assert len(seen) == 2
    assert seen[1].headers["Range"] == "bytes=412345-1234566"


@pytest.mark.parametrize(
    ("kelvin", "fahrenheit"),
    [
        (273.15, 32.0),
        (276.96, 38.858),
        (310.15, 98.6),
        (233.15, -40.0),
    ],
)
def test_kelvin_to_fahrenheit_golden_values(kelvin: float, fahrenheit: float) -> None:
    assert kelvin_to_fahrenheit(kelvin) == pytest.approx(fahrenheit, abs=1e-9)


@pytest.mark.skipif(not _eccodes_available(), reason="eccodes library not installed")
def test_decode_point_returns_fahrenheit_for_recorded_grib2() -> None:
    grib_bytes = _GRIB_FIXTURE.read_bytes()

    value = decode_point(grib_bytes, latitude=39.86, longitude=-104.67)

    assert isinstance(value, float)
    assert math.isfinite(value)
    assert value == pytest.approx(38.861, abs=0.01)
