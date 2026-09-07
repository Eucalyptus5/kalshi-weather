from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import httpx


IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
NBS_MODEL = "NBS"


@dataclass(frozen=True, slots=True, kw_only=True)
class MosRow:
    runtime: datetime
    ftime: datetime
    tmp: Decimal | None
    txn: Decimal | None
    xnd: Decimal | None
    tsd: Decimal | None


@dataclass(frozen=True, slots=True, kw_only=True)
class MosArchive:
    station: str
    rows: tuple[MosRow, ...]
    source_url: str


def parse_mos_csv(text: str) -> tuple[MosRow, ...]:
    rows: dict[tuple[str, str], MosRow] = {}
    for row in csv.DictReader(io.StringIO(text)):
        key = (row["runtime"], row["ftime"])
        # The archive serves the 2025-10-14 07Z run twice; the copies agree on tmp, txn, xnd and
        # tsd and differ only in the wind columns, so the first copy wins and the key stays unique.
        if key in rows:
            continue
        rows[key] = MosRow(
            runtime=_stamp(row["runtime"]),
            ftime=_stamp(row["ftime"]),
            tmp=_number(row["tmp"]),
            txn=_number(row["txn"]),
            xnd=_number(row["xnd"]),
            tsd=_number(row["tsd"]),
        )
    return tuple(rows.values())


def daily_high_row(rows: Sequence[MosRow], event_date: date, as_of: datetime) -> MosRow | None:
    ftime = datetime.combine(event_date + timedelta(days=1), time(), tzinfo=timezone.utc)
    behind = [
        row for row in rows if row.ftime == ftime and row.txn is not None and row.runtime <= as_of
    ]
    if not behind:
        return None
    return max(behind, key=lambda row: row.runtime)


async def fetch_mos_archive(
    *,
    station: str,
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient,
) -> MosArchive:
    response = await client.get(
        IEM_MOS_URL,
        params={
            "station": station,
            "model": NBS_MODEL,
            "sts": f"{start_date.isoformat()}T00:00Z",
            "ets": f"{end_date.isoformat()}T00:00Z",
            "format": "csv",
        },
    )
    response.raise_for_status()
    return MosArchive(
        station=station, rows=parse_mos_csv(response.text), source_url=str(response.url)
    )


def _stamp(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _number(value: str) -> Decimal | None:
    return Decimal(value) if value else None
