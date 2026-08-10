from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time as clock, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.mechanism_rates import (  # noqa: E402
    DECODE_DEFECT,
    MECHANISM_ORDER,
    MISSING_OBSERVATION,
    ROUNDING_DIFFERENCE,
    THRESHOLDS,
    WINDOW_DIFFERENCE,
    MechanismRate,
    first_matching,
    mechanism_rate,
    separating_strikes,
)
from bot.markets.observation_window import observation_window  # noqa: E402

# The event-day token parser and the METAR archive reader are the repo's only ones; a second copy
# of either here would drift away from what the rest of the pipeline reads.
from bot.markets.parser import _parse_date  # noqa: E402
from bot.observations.basis_check import (  # noqa: E402
    _fetch_iowa_asos_archive,
    fetch_iem_1min_asos_archive,
)
from bot.observations.metar import StationObservation  # noqa: E402
from bot.replay.analysis_stations import ANALYSIS_STATIONS, HIGH, ladder_of  # noqa: E402
from bot.validation.reconcile import ACISClient  # noqa: E402


logger = logging.getLogger(__name__)

UTC = timezone.utc
MAX = "max"
MIN = "min"
MISSING_TOKENS = ("", "M", "null", "None")
NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
# The METAR product encodes whole degrees Celsius, which is 0.9F of rounding before the two
# products disagree about anything, and they average over slightly different instants. The band is
# wide enough that only a structural fault - a C-for-F swap, an hour of timestamp drift - trips it.
METAR_TOLERANCE_F: Decimal = Decimal("2.0")
MISSING_OBSERVATION_REASON = (
    "clean lock crossings live on the recorder tape, which is not on this machine, so the row is "
    "unmeasured here rather than zero"
)


@dataclass(frozen=True, slots=True)
class LadderDay:
    root: str
    station: str
    timezone: str
    ladder: str
    event_date: date
    listed_strikes: tuple[int, ...]

    @property
    def extreme(self) -> str:
        return MAX if self.ladder == HIGH else MIN


@dataclass(frozen=True, slots=True)
class StationArchive:
    station: str
    decoded: tuple[StationObservation, ...]
    published: Mapping[datetime, Decimal]
    unparsable: int
    metar: tuple[StationObservation, ...]


@dataclass(frozen=True, slots=True)
class Row:
    mechanism: str
    detail: dict = field(default_factory=dict)
    rate: MechanismRate | None = None
    reason: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read which pre-registered mechanism explains the settlement disagreement"
    )
    parser.add_argument(
        "--start", type=date.fromisoformat, required=True, help="first day of the accrual window"
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, required=True, help="last day of the accrual window"
    )
    parser.add_argument(
        "--universe", type=Path, required=True, help="the frozen r0 universe naming the roots"
    )
    parser.add_argument(
        "--ladders", type=Path, required=True, help="settled ladders keyed by root and event day"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="where the raw archive and settle bodies are kept so a rerun refetches neither",
    )
    parser.add_argument("--out", type=Path, required=True, help="where the report json is written")
    return parser


def cache_name(request: httpx.Request) -> str:
    query = parse_qs(request.url.query.decode())
    label = (query.get("station") or query.get("sid") or ["query"])[0]
    digest = hashlib.sha256(str(request.url).encode()).hexdigest()[:16]
    return f"{request.url.path.rsplit('/', 1)[-1]}-{label}-{digest}.txt"


# Caching under the transport rather than around the fetch keeps the production readers on the
# exact query they build for themselves, and puts the raw body on disk for the second reader.
class CachingTransport(httpx.AsyncBaseTransport):
    def __init__(self, cache: Path, inner: httpx.AsyncBaseTransport) -> None:
        self._cache = cache
        self._inner = inner
        self.served: list[Path] = []
        self.fetched = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = self._cache / cache_name(request)
        source = "cache"
        if not path.exists():
            response = await self._inner.handle_async_request(request)
            body = await response.aread()
            await response.aclose()
            if response.status_code != 200:
                return httpx.Response(response.status_code, content=body, request=request)
            path.write_bytes(body)
            self.fetched += 1
            source = "network"
        self.served.append(path)
        # httpx logs every request at INFO whether or not it left the process, so the source is
        # stated here rather than read off a line that says 200 OK for a file on disk.
        logger.info("k3 payload source=%s file=%s", source, path.name)
        return httpx.Response(200, content=path.read_bytes(), request=request)

    async def aclose(self) -> None:
        await self._inner.aclose()


# Not observation_window's job: that helper is the NWS local-standard-time day, and row 3 exists
# to ask what the local wall-clock day would have implied instead.
def wall_clock_window(zone_name: str, day: date) -> tuple[datetime, datetime]:
    zone = pytz.timezone(zone_name)
    start = zone.localize(datetime.combine(day, clock())).astimezone(UTC)
    end = zone.localize(datetime.combine(day + timedelta(days=1), clock())).astimezone(UTC)
    return start, end


def load_universe(path: Path) -> tuple[str, ...]:
    return tuple(json.loads(path.read_text())["passing"])


def load_ladders(
    path: Path, roots: Sequence[str], start: date, end: date
) -> tuple[tuple[LadderDay, ...], dict]:
    unmapped = sorted(root for root in roots if root not in ANALYSIS_STATIONS)
    if unmapped:
        raise ValueError("no station mapping for universe roots: " + ", ".join(unmapped))

    payload = json.loads(path.read_text())
    days: list[LadderDay] = []
    outside_window = 0
    for root in roots:
        for token, entry in payload.get(root, {}).items():
            event_date, _ = _parse_date(token, f"{root}-{token}")
            if not start <= event_date <= end:
                outside_window += 1
                continue
            cfg = ANALYSIS_STATIONS[root]
            days.append(
                LadderDay(
                    root=root,
                    station=cfg.station,
                    timezone=cfg.timezone,
                    ladder=ladder_of(root),
                    event_date=event_date,
                    listed_strikes=tuple(int(strike) for strike in entry["listed_strikes"]),
                )
            )
    days.sort(key=lambda day: (day.root, day.event_date))
    covered = {day.root for day in days}
    notes = {
        "roots": len(roots),
        "ladder_rows": len(days),
        "roots_without_ladders": sorted(set(roots) - covered),
        "ladder_roots_outside_universe": sorted(set(payload) - set(roots)),
        "event_days_outside_window": outside_window,
    }
    return tuple(days), notes


# Deliberately shares nothing with fetch_iem_1min_asos_archive: row 1 asks whether the production
# decode reproduces the archive's own published value, and a shared reader would agree by
# construction. Proper CSV quoting is the difference that matters.
def read_published_minutes(body: str) -> tuple[dict[datetime, Decimal], int]:
    published: dict[datetime, Decimal] = {}
    unparsable = 0
    for row in csv.DictReader(io.StringIO(body)):
        raw = (row["tmpf"] or "").strip()
        if raw in MISSING_TOKENS:
            continue
        if not NUMBER.match(raw):
            unparsable += 1
            continue
        stamp = datetime.fromisoformat(row["valid(UTC)"].strip()).replace(tzinfo=UTC)
        published[stamp] = Decimal(raw)
    return published, unparsable


async def gather_archives(
    stations: Sequence[str],
    start: date,
    end: date,
    transport: CachingTransport,
    http: httpx.AsyncClient,
) -> dict[str, StationArchive]:
    out: dict[str, StationArchive] = {}
    for index, station in enumerate(stations, start=1):
        transport.served.clear()
        decoded = await fetch_iem_1min_asos_archive(station, start, end, http)
        (one_minute,) = transport.served
        published, unparsable = read_published_minutes(one_minute.read_text())
        transport.served.clear()
        metar = await _fetch_iowa_asos_archive(station, start, end, http)
        transport.served.clear()
        out[station] = StationArchive(
            station=station,
            decoded=tuple(decoded),
            published=published,
            unparsable=unparsable,
            metar=tuple(metar),
        )
        logger.info(
            "k3 archive station=%s decoded=%d published=%d unparsable=%d metar=%d (%d/%d)",
            station,
            len(decoded),
            len(published),
            unparsable,
            len(metar),
            index,
            len(stations),
        )
    return out


async def gather_settles(
    ladders: Sequence[LadderDay], acis: ACISClient
) -> dict[tuple[str, date, str], Decimal]:
    wanted = sorted({(day.station, day.event_date, day.extreme) for day in ladders})
    out: dict[tuple[str, date, str], Decimal] = {}
    for index, (station, event_date, extreme) in enumerate(wanted, start=1):
        sid = station[1:]
        if extreme == MAX:
            value = await acis.fetch_daily_high(sid, event_date)
        else:
            value = await acis.fetch_daily_low(sid, event_date)
        if value is not None:
            out[(station, event_date, extreme)] = value
        logger.info(
            "k3 settle station=%s event_date=%s extreme=%s value=%s (%d/%d)",
            station,
            event_date.isoformat(),
            extreme,
            value,
            index,
            len(wanted),
        )
    return out


def extreme_of(values: Sequence[Decimal], extreme: str) -> Decimal:
    return max(values) if extreme == MAX else min(values)


def readings_in(archive: StationArchive, window: tuple[datetime, datetime]) -> tuple[Decimal, ...]:
    start, end = window
    return tuple(row.temp_f for row in archive.decoded if start <= row.valid_time < end)


def decode_row(archives: Mapping[str, StationArchive]) -> Row:
    stations = []
    denominator = 0
    numerator = 0
    for station in sorted(archives):
        archive = archives[station]
        decoded = {row.valid_time: row.temp_f for row in archive.decoded}
        absent = sum(1 for stamp in archive.published if stamp not in decoded)
        differing = sum(
            1
            for stamp, value in archive.published.items()
            if stamp in decoded and decoded[stamp] != value
        )
        denominator += len(archive.published)
        numerator += absent + differing
        stations.append(
            {
                "station": station,
                "published_minutes": len(archive.published),
                "decoded_minutes": len(decoded),
                "absent": absent,
                "differing": differing,
                "unparsable_published": archive.unparsable,
                "decoded_not_published": sum(
                    1 for stamp in decoded if stamp not in archive.published
                ),
            }
        )
    detail = {"stations": stations, "metar_agreement": metar_agreement(archives)}
    return measured(DECODE_DEFECT, numerator, denominator, detail)


def metar_agreement(archives: Mapping[str, StationArchive]) -> dict:
    deltas: list[Decimal] = []
    for station in sorted(archives):
        archive = archives[station]
        by_minute = {row.valid_time: row.temp_f for row in archive.metar}
        for stamp in sorted(set(archive.published) & set(by_minute)):
            deltas.append(abs(archive.published[stamp] - by_minute[stamp]))
    agree = sum(1 for delta in deltas if delta <= METAR_TOLERANCE_F)
    ordered = sorted(deltas)
    return {
        "compared_minutes": len(deltas),
        "agree": agree,
        "disagree": len(deltas) - agree,
        "tolerance_f": str(METAR_TOLERANCE_F),
        "max_abs_delta_f": str(ordered[-1]) if ordered else None,
        "median_abs_delta_f": str(ordered[len(ordered) // 2]) if ordered else None,
    }


def rounding_row(
    ladders: Sequence[LadderDay],
    archives: Mapping[str, StationArchive],
    settles: Mapping[tuple[str, date, str], Decimal],
) -> Row:
    denominator = 0
    separations = []
    no_observations = 0
    no_settle = 0
    for day in ladders:
        readings = readings_in(
            archives[day.station], observation_window(day.timezone, day.event_date)
        )
        if not readings:
            no_observations += 1
            continue
        published = settles.get((day.station, day.event_date, day.extreme))
        if published is None:
            no_settle += 1
            continue
        denominator += 1
        observed = extreme_of(readings, day.extreme)
        strikes = separating_strikes(observed, published, day.listed_strikes)
        if strikes:
            separations.append(
                {
                    "root": day.root,
                    "station": day.station,
                    "event_date": day.event_date.isoformat(),
                    "extreme": day.extreme,
                    "observed_f": str(observed),
                    "acis_f": str(published),
                    "separating_strikes": list(strikes),
                }
            )
    detail = {
        "separations": separations,
        "no_observations": no_observations,
        "no_settle": no_settle,
    }
    return measured(ROUNDING_DIFFERENCE, len(separations), denominator, detail)


def window_row(ladders: Sequence[LadderDay], archives: Mapping[str, StationArchive]) -> Row:
    separated: set[tuple[str, date]] = set()
    usable: set[tuple[str, date]] = set()
    on_dst: set[str] = set()
    no_observations = 0
    separations = []
    for day in ladders:
        standard = observation_window(day.timezone, day.event_date)
        wall = wall_clock_window(day.timezone, day.event_date)
        if standard != wall:
            on_dst.add(day.station)
        standard_readings = readings_in(archives[day.station], standard)
        wall_readings = readings_in(archives[day.station], wall)
        if not standard_readings or not wall_readings:
            no_observations += 1
            continue
        usable.add((day.station, day.event_date))
        standard_extreme = extreme_of(standard_readings, day.extreme)
        wall_extreme = extreme_of(wall_readings, day.extreme)
        strikes = separating_strikes(standard_extreme, wall_extreme, day.listed_strikes)
        if strikes:
            separated.add((day.station, day.event_date))
            separations.append(
                {
                    "root": day.root,
                    "station": day.station,
                    "event_date": day.event_date.isoformat(),
                    "extreme": day.extreme,
                    "standard_f": str(standard_extreme),
                    "wall_clock_f": str(wall_extreme),
                    "separating_strikes": list(strikes),
                }
            )
    stations = sorted({day.station for day in ladders})
    detail = {
        "separations": separations,
        "no_observations": no_observations,
        # Phoenix keeps standard time all summer, so its two windows are the same window and it
        # cannot contribute to this numerator however the weather behaves.
        "cities_on_dst": sorted(on_dst),
        "cities_off_dst": [station for station in stations if station not in on_dst],
        "by_city": [
            {
                "station": station,
                "station_days": sum(1 for key in usable if key[0] == station),
                "separated": sum(1 for key in separated if key[0] == station),
                "on_dst": station in on_dst,
            }
            for station in stations
        ],
    }
    return measured(WINDOW_DIFFERENCE, len(separated), len(usable), detail)


def measured(mechanism: str, numerator: int, denominator: int, detail: dict) -> Row:
    if denominator == 0:
        return Row(
            mechanism=mechanism,
            detail=detail,
            reason=f"{mechanism} compared nothing over the window, so it has no denominator",
        )
    return Row(
        mechanism=mechanism, detail=detail, rate=mechanism_rate(mechanism, numerator, denominator)
    )


def row_payload(row: Row) -> dict:
    if row.rate is None:
        return {
            "mechanism": row.mechanism,
            "measurable": False,
            "reason": row.reason,
            "threshold": str(THRESHOLDS[row.mechanism]),
            "matched": False,
            "detail": row.detail,
        }
    return {
        "mechanism": row.mechanism,
        "measurable": True,
        "numerator": row.rate.numerator,
        "denominator": row.rate.denominator,
        "rate": str(row.rate.rate),
        "threshold": str(row.rate.threshold),
        "matched": row.rate.matched,
        "detail": row.detail,
    }


def selection_payload(rows: Sequence[Row]) -> dict:
    hit = first_matching([row.rate for row in rows if row.rate is not None])
    unmeasured = [row.mechanism for row in rows if row.rate is None]
    if hit is None:
        ahead = unmeasured
        selected = None
        if unmeasured:
            note = (
                "no measured row reached its threshold and "
                + ", ".join(unmeasured)
                + " went unmeasured, so the selection is undetermined pending those rows"
            )
        else:
            note = "every row was measured and none reached its threshold"
    else:
        selected = hit.mechanism
        ahead = [
            mechanism
            for mechanism in unmeasured
            if MECHANISM_ORDER.index(mechanism) < MECHANISM_ORDER.index(selected)
        ]
        if ahead:
            note = (
                ", ".join(ahead)
                + f" sits before {selected} in the order and went unmeasured, so the read is"
                " provisional"
            )
        else:
            note = f"every unmeasured row sits after {selected} in the order, so it is unaffected"
    return {
        "selected": selected,
        "undetermined": bool(ahead),
        "unmeasured": unmeasured,
        "unmeasured_ahead": ahead,
        "note": note,
    }


def report(
    start: date,
    end: date,
    universe: Path,
    roots: Sequence[str],
    ladder_notes: dict,
    stations: Sequence[str],
    rows: Sequence[Row],
) -> dict:
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "universe": {"path": str(universe), "roots": list(roots)},
        "ladders": ladder_notes,
        "stations": list(stations),
        "rows": [row_payload(row) for row in rows],
        "selection": selection_payload(rows),
    }


def format_report(payload: dict) -> str:
    lines = [
        f"== K3 MECHANISM RATES  {payload['window']['start']}..{payload['window']['end']}",
        f"roots={len(payload['universe']['roots'])} stations={len(payload['stations'])} "
        f"ladder_rows={payload['ladders']['ladder_rows']}",
        "",
    ]
    for row in payload["rows"]:
        if row["measurable"]:
            lines.append(
                f"  {row['mechanism']:<21} {row['numerator']}/{row['denominator']} = {row['rate']} "
                f"vs {row['threshold']}  matched={row['matched']}"
            )
        else:
            lines.append(f"  {row['mechanism']:<21} not measurable: {row['reason']}")
    selection = payload["selection"]
    lines += [
        "",
        f"selected={selection['selected']} undetermined={selection['undetermined']}",
        f"  {selection['note']}",
    ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    args.cache.mkdir(parents=True, exist_ok=True)
    roots = load_universe(args.universe)
    ladders, ladder_notes = load_ladders(args.ladders, roots, args.start, args.end)
    stations = sorted({ANALYSIS_STATIONS[root].station for root in roots})

    transport = CachingTransport(args.cache, httpx.AsyncHTTPTransport())
    async with httpx.AsyncClient(transport=transport, timeout=120.0) as http:
        archives = await gather_archives(stations, args.start, args.end, transport, http)
        settles = await gather_settles(ladders, ACISClient(http_client=http))

    rows = [
        decode_row(archives),
        rounding_row(ladders, archives, settles),
        window_row(ladders, archives),
        Row(mechanism=MISSING_OBSERVATION, reason=MISSING_OBSERVATION_REASON),
    ]
    payload = report(args.start, args.end, args.universe, roots, ladder_notes, stations, rows)
    payload["fetched_payloads"] = transport.fetched
    payload["elapsed_s"] = round(time.monotonic() - started, 1)
    args.out.write_text(json.dumps(payload, indent=1))
    print(format_report(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
