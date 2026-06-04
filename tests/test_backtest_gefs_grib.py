import math
from datetime import date
from pathlib import Path

import httpx
import pytest

from bot.backtest.gefs_grib import (
    build_grib_url,
    decode_point,
    fetch_member_field,
    tmp2m_byte_range,
)

_IDX_FIXTURE = Path(__file__).parent / "data" / "gefs_gep01_f024.idx"
_GRIB_FIXTURE = Path(__file__).parent / "data" / "gefs_tmp2m_msg.grib2"


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


@pytest.mark.skipif(
    not _GRIB_FIXTURE.exists(),
    reason="recorded GRIB2 fixture not bundled; populate tests/data/gefs_tmp2m_msg.grib2",
)
def test_decode_point_returns_finite_float_for_recorded_grib2() -> None:
    cfgrib = pytest.importorskip("cfgrib")
    assert cfgrib is not None
    grib_bytes = _GRIB_FIXTURE.read_bytes()

    value = decode_point(grib_bytes, latitude=39.86, longitude=-104.67)

    assert isinstance(value, float)
    assert math.isfinite(value)
