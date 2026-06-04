import logging
import tempfile
from datetime import date
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
_TMP_2M_MARKER = ":TMP:2 m above ground:"


def build_grib_url(d: date, cycle: int, member: str, fxx: int) -> str:
    cycle_str = f"{cycle:02d}"
    return (
        f"{_BUCKET}/gefs.{d:%Y%m%d}/{cycle_str}/atmos/pgrb2sp25/"
        f"{member}.t{cycle_str}z.pgrb2s.0p25.f{fxx:03d}"
    )


def tmp2m_byte_range(idx_text: str, fxx: int) -> tuple[int, int]:
    if fxx == 0:
        fcst_marker = ":anl:"
    else:
        fcst_marker = f":{fxx} hour fcst:"

    lines = [line for line in idx_text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if _TMP_2M_MARKER in line and fcst_marker in line:
            start = int(line.split(":")[1])
            if i + 1 >= len(lines):
                raise ValueError("TMP record is the last entry in idx; cannot infer end byte")
            next_start = int(lines[i + 1].split(":")[1])
            return start, next_start - 1
    raise ValueError(f"no TMP 2m record found for fxx={fxx} in idx ({len(lines)} lines)")


async def fetch_member_field(
    d: date,
    cycle: int,
    member: str,
    fxx: int,
    client: httpx.AsyncClient,
) -> bytes:
    grib_url = build_grib_url(d, cycle, member, fxx)
    idx_url = f"{grib_url}.idx"

    idx_response = await client.get(idx_url)
    idx_response.raise_for_status()
    start, end = tmp2m_byte_range(idx_response.text, fxx)

    grib_response = await client.get(grib_url, headers={"Range": f"bytes={start}-{end}"})
    grib_response.raise_for_status()
    return grib_response.content


def decode_point(grib_bytes: bytes, latitude: float, longitude: float) -> float:
    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as fh:
        fh.write(grib_bytes)
        tmp_path = Path(fh.name)
    try:
        ds = xr.open_dataset(
            tmp_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
        try:
            lon_wrapped = longitude % 360.0
            point = ds["t2m"].sel(
                latitude=latitude,
                longitude=lon_wrapped,
                method="nearest",
            )
            return float(point.values)
        finally:
            ds.close()
    finally:
        tmp_path.unlink(missing_ok=True)
