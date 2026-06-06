import logging
import tempfile
from datetime import date
from pathlib import Path

import httpx
import numpy as np

logger = logging.getLogger(__name__)

_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
_TMP_2M_MARKER = ":TMP:2 m above ground:"
_TMAX_2M_MARKER = ":TMAX:2 m above ground:"


def build_grib_url(d: date, cycle: int, member: str, fxx: int) -> str:
    cycle_str = f"{cycle:02d}"
    return (
        f"{_BUCKET}/gefs.{d:%Y%m%d}/{cycle_str}/atmos/pgrb2sp25/"
        f"{member}.t{cycle_str}z.pgrb2s.0p25.f{fxx:03d}"
    )


def _record_byte_range(idx_text: str, marker: str, fcst_marker: str) -> tuple[int, int]:
    lines = [line for line in idx_text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if marker in line and fcst_marker in line:
            start = int(line.split(":")[1])
            if i + 1 >= len(lines):
                raise ValueError(f"{marker} record is the last entry in idx; cannot infer end byte")
            next_start = int(lines[i + 1].split(":")[1])
            return start, next_start - 1
    raise ValueError(f"no {marker} record matching {fcst_marker} in idx ({len(lines)} lines)")


def tmp2m_byte_range(idx_text: str, fxx: int) -> tuple[int, int]:
    fcst_marker = ":anl:" if fxx == 0 else f":{fxx} hour fcst:"
    return _record_byte_range(idx_text, _TMP_2M_MARKER, fcst_marker)


def tmax2m_byte_range(idx_text: str, fxx: int) -> tuple[int, int]:
    # TMAX windows reset at 6-hour boundaries: f021 covers 18-21, f024 covers 18-24
    window_start = 6 * ((fxx - 1) // 6)
    fcst_marker = f":{window_start}-{fxx} hour max fcst:"
    return _record_byte_range(idx_text, _TMAX_2M_MARKER, fcst_marker)


async def fetch_member_field(
    d: date,
    cycle: int,
    member: str,
    fxx: int,
    client: httpx.AsyncClient,
) -> bytes:
    grib_url = build_grib_url(d, cycle, member, fxx)

    idx_response = await client.get(f"{grib_url}.idx")
    idx_response.raise_for_status()
    start, end = tmp2m_byte_range(idx_response.text, fxx)

    grib_response = await client.get(grib_url, headers={"Range": f"bytes={start}-{end}"})
    grib_response.raise_for_status()
    return grib_response.content


async def fetch_member_tmax(
    d: date,
    cycle: int,
    member: str,
    fxx: int,
    client: httpx.AsyncClient,
) -> bytes:
    grib_url = build_grib_url(d, cycle, member, fxx)

    idx_response = await client.get(f"{grib_url}.idx")
    idx_response.raise_for_status()
    start, end = tmax2m_byte_range(idx_response.text, fxx)

    grib_response = await client.get(grib_url, headers={"Range": f"bytes={start}-{end}"})
    grib_response.raise_for_status()
    return grib_response.content


def kelvin_to_fahrenheit(kelvin: float) -> float:
    return (kelvin - 273.15) * 9.0 / 5.0 + 32.0


def decode_point(grib_bytes: bytes, latitude: float, longitude: float) -> float:
    import cfgrib

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as fh:
        fh.write(grib_bytes)
        tmp_path = Path(fh.name)
    try:
        ds = cfgrib.open_file(tmp_path, indexpath="")
        field = next(
            var for var in ds.variables.values() if var.dimensions == ("latitude", "longitude")
        )
        lats = np.asarray(ds.variables["latitude"].data)
        lons = np.asarray(ds.variables["longitude"].data)
        lat_i = int(np.abs(lats - latitude).argmin())
        lon_i = int(np.abs(lons - (longitude % 360.0)).argmin())
        return kelvin_to_fahrenheit(float(field.data[lat_i, lon_i]))
    finally:
        tmp_path.unlink(missing_ok=True)
