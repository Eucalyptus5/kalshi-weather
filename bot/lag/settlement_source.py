from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from bot.lag.r0_universe import freeze_digest


SERIES_URL = "https://api.elections.kalshi.com/trade-api/v2/series"

# The settlement family compares the venue's number against ACIS, which is neither of the sources
# the venue names, and nothing here reconciles the two. Freezing it alongside keeps a later reader
# from reading a verdict about ACIS as a verdict about the named settlement source.
OBSERVATION_SOURCE = "ACIS"

BOUNDARY_SOURCE = "product_metadata.important_info.markdown"

NOTICE = re.compile(r"Effective\s+([A-Za-z]+),\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,")
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class SettlementSourceUnreadable(RuntimeError):
    """A series response names no settlement source or no stamp, so the run cannot record it."""

    def __init__(self, root: str, key: str) -> None:
        super().__init__(f"{root} carries no readable {key}")
        self.root = root
        self.key = key


class SettlementScopeShort(RuntimeError):
    """The frozen sidecar names fewer roots than the run's scope, so the run aborts."""

    def __init__(self, offenders: Sequence[str]) -> None:
        super().__init__(
            "run aborted, the frozen settlement sidecar is short: " + "; ".join(offenders)
        )
        self.offenders = tuple(offenders)


class SettlementNoticeUnreadable(RuntimeError):
    """No frozen notice names an effective date, so the run has no boundary to derive."""

    def __init__(self, roots: Sequence[str]) -> None:
        super().__init__(
            "run aborted, no frozen notice names an effective date: " + ", ".join(roots)
        )
        self.roots = tuple(roots)


class SettlementNoticeMisdated(RuntimeError):
    """A notice names a weekday its own date misses, so the sidecar's year is not the notice's."""

    def __init__(self, root: str, named: str, moment: date) -> None:
        super().__init__(
            f"run aborted, {root} names {named} but {moment.isoformat()} is a "
            f"{WEEKDAYS[moment.weekday()]}"
        )
        self.root = root
        self.named = named
        self.moment = moment


class BoundaryOutsideWindow(RuntimeError):
    """The boundary misses the window, so one side of the split is a zero nobody checked."""

    def __init__(self, boundary: date, first: date, last: date) -> None:
        super().__init__(
            f"run aborted, the boundary {boundary.isoformat()} does not fall inside the window "
            f"{first.isoformat()}..{last.isoformat()}"
        )
        self.boundary = boundary
        self.first = first
        self.last = last


@dataclass(frozen=True, slots=True)
class SeriesSettlementSource:
    root: str
    settlement_source: str
    settlement_source_url: str
    last_updated_ts: datetime
    important_info: str
    important_info_id: str


@dataclass(frozen=True, slots=True)
class SettlementProvenance:
    observed_at: datetime
    observation_source: str
    series: Mapping[str, SeriesSettlementSource]
    sha256: str


@dataclass(frozen=True, slots=True)
class BoundarySplit:
    boundary_date: date
    days_before_boundary: int
    days_on_or_after_boundary: int
    boundary_source: str


def _stamp(root: str, raw: object) -> datetime:
    if not isinstance(raw, str) or not raw:
        raise SettlementSourceUnreadable(root, "last_updated_ts")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SettlementSourceUnreadable(root, "last_updated_ts")
    return parsed.astimezone(timezone.utc)


async def fetch_series_settlement_source(
    root: str, client: httpx.AsyncClient
) -> SeriesSettlementSource:
    response = await client.get(f"{SERIES_URL}/{root}")
    response.raise_for_status()
    series = response.json()["series"]
    sources = series.get("settlement_sources")
    if not sources:
        raise SettlementSourceUnreadable(root, "settlement_sources")
    named = sources[0]
    metadata = series.get("product_metadata") or {}
    info = metadata.get("important_info") or {}
    return SeriesSettlementSource(
        root=series["ticker"],
        settlement_source=named["name"],
        settlement_source_url=named["url"],
        last_updated_ts=_stamp(root, series.get("last_updated_ts")),
        important_info=info.get("markdown") or "",
        important_info_id=info.get("id") or "",
    )


async def _fetch_roots(
    roots: Sequence[str], transport: httpx.AsyncBaseTransport | None
) -> list[SeriesSettlementSource]:
    async with httpx.AsyncClient(transport=transport) as client:
        return [await fetch_series_settlement_source(root, client) for root in roots]


def settlement_payload(observed_at: datetime, series: Sequence[SeriesSettlementSource]) -> dict:
    return {
        "observed_at": observed_at.isoformat(),
        "observation_source": OBSERVATION_SOURCE,
        "series": [
            {
                "root": item.root,
                "settlement_source": item.settlement_source,
                "settlement_source_url": item.settlement_source_url,
                "last_updated_ts": item.last_updated_ts.isoformat(),
                "important_info": item.important_info,
                "important_info_id": item.important_info_id,
            }
            for item in sorted(series, key=lambda item: item.root)
        ],
    }


def pull_settlement_sources(
    roots: Sequence[str],
    observed_at: datetime,
    path: Path,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = settlement_payload(observed_at, asyncio.run(_fetch_roots(roots, transport)))
    digest = freeze_digest(payload)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=1))
    return digest


def read_settlement_sources(path: Path) -> SettlementProvenance:
    payload = json.loads(path.read_text())
    if "sha256" not in payload:
        raise ValueError(f"{path} carries no sha256")
    stored = payload.pop("sha256")
    digest = freeze_digest(payload)
    if digest != stored:
        raise ValueError(f"{path} does not match the sha256 it carries")
    return SettlementProvenance(
        observed_at=datetime.fromisoformat(payload["observed_at"]),
        observation_source=payload["observation_source"],
        series={
            row["root"]: SeriesSettlementSource(
                root=row["root"],
                settlement_source=row["settlement_source"],
                settlement_source_url=row["settlement_source_url"],
                last_updated_ts=_stamp(row["root"], row["last_updated_ts"]),
                important_info=row["important_info"],
                important_info_id=row["important_info_id"],
            )
            for row in payload["series"]
        },
        sha256=digest,
    )


def _notice_date(root: str, body: str, year: int) -> date | None:
    found = NOTICE.search(body)
    if found is None:
        return None
    named, month, day = found.groups()
    if named not in WEEKDAYS or month not in MONTHS:
        return None
    moment = date(year, MONTHS.index(month) + 1, int(day))
    if WEEKDAYS[moment.weekday()] != named:
        raise SettlementNoticeMisdated(root, named, moment)
    return moment


def boundary_split(provenance: SettlementProvenance, event_dates: Iterable[date]) -> BoundarySplit:
    # The earliest notice across the roots sets the boundary: the first root to move is the first
    # day any of the run's evidence sits on a source the rest of the window was not read under.
    # The frozen stamp ratchets forward on any metadata edit, so it dates no change and is not read.
    year = provenance.observed_at.year
    notices = [
        moment
        for root, item in sorted(provenance.series.items())
        if (moment := _notice_date(root, item.important_info, year)) is not None
    ]
    if not notices:
        raise SettlementNoticeUnreadable(sorted(provenance.series))
    boundary = min(notices)
    days = tuple(event_dates)
    first, last = min(days), max(days)
    if not first < boundary <= last:
        raise BoundaryOutsideWindow(boundary, first, last)
    before = sum(1 for day in days if day < boundary)
    return BoundarySplit(
        boundary_date=boundary,
        days_before_boundary=before,
        days_on_or_after_boundary=len(days) - before,
        boundary_source=BOUNDARY_SOURCE,
    )


def check_settlement_scope(provenance: SettlementProvenance, roots: Iterable[str]) -> None:
    offenders = [
        f"{root} is not named in the frozen sidecar"
        for root in sorted(set(roots))
        if root not in provenance.series
    ]
    if offenders:
        raise SettlementScopeShort(offenders)


def source_root_counts(provenance: SettlementProvenance) -> dict[str, int]:
    return dict(Counter(item.settlement_source for item in provenance.series.values()))


def distinct_notice_bodies(provenance: SettlementProvenance) -> int:
    return len({item.important_info for item in provenance.series.values()})
