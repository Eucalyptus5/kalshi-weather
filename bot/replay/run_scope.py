import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bot.main import StationConfig
from bot.markets.observation_window import observation_window
from bot.markets.parser import parse_ticker


logger = logging.getLogger(__name__)

D_EVAL = 14

QUIET_BAND_OPEN_HOUR = 7
QUIET_BAND_CLOSE_HOUR = 9
INCIDENT_PAD = timedelta(minutes=5)

RECORDED_GAP = "recorded_gap"
SUBSCRIPTION_WIDE = "subscription_wide"
RESUBSCRIBE_BLIND = "resubscribe_blind"
QUIET_BAND = "quiet_band"
SUBSCRIPTION_WIDE_REASONS = frozenset(
    {"seq_skip", "terminal_error_10", "terminal_error_17", "terminal_error_25"}
)
# The quiet band is a standing decision to ignore the small hours, not a hole in the tape, so it
# never counts against an event day's evaluability.
OUTAGE_CLASSES = frozenset({RECORDED_GAP, SUBSCRIPTION_WIDE, RESUBSCRIBE_BLIND})

DISCOVERY = "discovery"
HOLDOUT = "holdout"

_MICROSECOND = timedelta(microseconds=1)
_DAY_US = 86_400_000_000
_ONE_DAY = timedelta(days=1)
_ZERO = timedelta(0)

OUTAGE_TOLERANCE_US = _DAY_US * 5 // 100

EXCLUSIONS_SCHEMA = pa.schema(
    [
        ("exclusion_id", pa.int64()),
        ("exclusion_class", pa.string()),
        ("start", pa.timestamp("us", tz="UTC")),
        ("end", pa.timestamp("us", tz="UTC")),
        ("duration_us", pa.int64()),
        ("boundary_id", pa.int64()),
        ("gap_id", pa.int64()),
        ("gap_reason", pa.string()),
        ("padded", pa.bool_()),
    ]
)

EVENT_DAYS_SCHEMA = pa.schema(
    [
        ("series", pa.string()),
        ("station", pa.string()),
        ("timezone", pa.string()),
        ("event_date", pa.date32()),
        ("window_start", pa.timestamp("us", tz="UTC")),
        ("window_end", pa.timestamp("us", tz="UTC")),
        ("tickers", pa.int64()),
        ("ladder_rows", pa.int64()),
        ("first_event_at", pa.timestamp("us", tz="UTC")),
        ("last_event_at", pa.timestamp("us", tz="UTC")),
        ("covered", pa.bool_()),
        ("evaluable", pa.bool_()),
        ("in_scope", pa.bool_()),
        ("day_index", pa.int64()),
        ("split", pa.string()),
        ("excluded_us", pa.int64()),
        ("span_us", pa.int64()),
    ]
)


# A boundary tears down the whole connection, so every class here applies to every ticker and
# names none of them.
@dataclass(frozen=True, slots=True)
class Exclusion:
    exclusion_class: str
    start: datetime
    end: datetime
    boundary_id: int | None
    gap_id: int | None
    gap_reason: str | None
    padded: bool

    @property
    def duration_us(self) -> int:
        return (self.end - self.start) // _MICROSECOND


@dataclass(frozen=True, slots=True)
class EventDay:
    series: str
    station: str
    timezone: str
    event_date: date
    window_start: datetime
    window_end: datetime
    tickers: int
    ladder_rows: int
    first_event_at: datetime
    last_event_at: datetime
    covered: bool
    evaluable: bool
    in_scope: bool
    day_index: int
    split: str
    excluded_us: int
    span_us: int


@dataclass(frozen=True, slots=True)
class DayInventory:
    days: tuple[EventDay, ...]
    outage_us: Mapping[tuple[str, date], int]
    over_tolerance: int


@dataclass(frozen=True, slots=True)
class Split:
    cities: tuple[str, ...]
    discovery_days: tuple[date, ...]
    holdout_days: tuple[date, ...]
    boundary_event_day: date
    scope_start: datetime
    scope_end: datetime


def split_lengths(d_eval: int) -> tuple[int, int]:
    discovery = 2 * d_eval // 3
    return discovery, d_eval - discovery


def classify_window(has_gap_row: bool, gap_reason: str | None) -> str:
    if not has_gap_row:
        return RESUBSCRIBE_BLIND
    if gap_reason in SUBSCRIPTION_WIDE_REASONS:
        return SUBSCRIPTION_WIDE
    return RECORDED_GAP


def read_blind_windows(path: Path) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    return sorted(rows, key=lambda row: row["start"])


def read_coverage(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def blind_exclusions(
    rows: Sequence[Mapping[str, object]],
    *,
    scope_start: datetime,
    scope_end: datetime,
    incident: datetime,
) -> list[Exclusion]:
    out = []
    for row in rows:
        detected = row["gap_detected_at"]
        padded = detected is not None and detected.replace(microsecond=0) == incident
        start = row["start"] - INCIDENT_PAD if padded else row["start"]
        end = row["end"] + INCIDENT_PAD if padded else row["end"]
        if end < scope_start or start > scope_end:
            continue
        out.append(
            Exclusion(
                exclusion_class=classify_window(row["has_gap_row"], row["gap_reason"]),
                start=max(start, scope_start),
                end=min(end, scope_end),
                boundary_id=row["boundary_id"],
                gap_id=row["gap_id"],
                gap_reason=row["gap_reason"],
                padded=padded,
            )
        )
    return out


def quiet_band_exclusions(scope_start: datetime, scope_end: datetime) -> list[Exclusion]:
    out = []
    day = scope_start.date()
    while day <= scope_end.date():
        opens = datetime(day.year, day.month, day.day, QUIET_BAND_OPEN_HOUR, tzinfo=timezone.utc)
        closes = datetime(day.year, day.month, day.day, QUIET_BAND_CLOSE_HOUR, tzinfo=timezone.utc)
        day += timedelta(days=1)
        if closes <= scope_start or opens >= scope_end:
            continue
        out.append(
            Exclusion(
                exclusion_class=QUIET_BAND,
                start=max(opens, scope_start),
                end=min(closes, scope_end),
                boundary_id=None,
                gap_id=None,
                gap_reason=None,
                padded=False,
            )
        )
    return out


def union_overlap_us(exclusions: Sequence[Exclusion], start: datetime, end: datetime) -> int:
    merged: list[list[datetime]] = []
    for exclusion in sorted(exclusions, key=lambda item: (item.start, item.end)):
        if merged and exclusion.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], exclusion.end)
        else:
            merged.append([exclusion.start, exclusion.end])
    return sum(
        max(_ZERO, min(right, end) - max(left, start)) // _MICROSECOND for left, right in merged
    )


def event_day_inventory(
    coverage: Sequence[Mapping[str, object]],
    *,
    stations: Mapping[str, StationConfig],
    exclusions: Sequence[Exclusion],
    tape_first: datetime,
    tape_last: datetime,
    scope_open: datetime,
    d_eval: int,
) -> DayInventory:
    grouped: dict[tuple[str, date], list[Mapping[str, object]]] = {}
    for row in coverage:
        ticker = row["ticker"]
        if ticker.split("-")[0] not in stations:
            continue
        try:
            parsed = parse_ticker(ticker)
        except ValueError:
            continue
        grouped.setdefault((parsed.series, parsed.event_date), []).append(row)

    outages = [item for item in exclusions if item.exclusion_class in OUTAGE_CLASSES]
    days = []
    outage_us: dict[tuple[str, date], int] = {}
    over_tolerance = 0
    for (series, event_date), rows in grouped.items():
        cfg = stations[series]
        window_start, window_end = observation_window(cfg.timezone, event_date)
        outage = union_overlap_us(outages, window_start, window_end)
        outage_us[(series, event_date)] = outage
        covered = tape_first <= window_start and window_end <= tape_last
        candidate = covered and scope_open <= window_start
        over_tolerance += int(candidate and outage > OUTAGE_TOLERANCE_US)
        days.append(
            EventDay(
                series=series,
                station=cfg.station,
                timezone=cfg.timezone,
                event_date=event_date,
                window_start=window_start,
                window_end=window_end,
                tickers=len(rows),
                ladder_rows=sum(row["rows"] for row in rows),
                first_event_at=min(row["first_received_at"] for row in rows),
                last_event_at=max(row["last_received_at"] for row in rows),
                covered=covered,
                evaluable=candidate and outage <= OUTAGE_TOLERANCE_US,
                in_scope=False,
                day_index=0,
                split="",
                excluded_us=0,
                span_us=0,
            )
        )

    days.sort(key=lambda day: (day.series, day.event_date))
    cities = {day.series for day in days}
    ready: dict[date, set[str]] = {}
    for day in days:
        if day.evaluable:
            ready.setdefault(day.event_date, set()).add(day.series)
    shared = sorted(event_date for event_date, seen in ready.items() if seen == cities)
    sequence = ", ".join(event_date.isoformat() for event_date in shared) or "none"

    if d_eval <= 0 or d_eval % D_EVAL:
        raise ValueError(
            f"an accrual of {d_eval} event-days is not a whole multiple of {D_EVAL}, "
            f"evaluable in every city: {sequence}"
        )

    run: list[date] = []
    for event_date in shared:
        if run and event_date - run[-1] != _ONE_DAY:
            run = []
        run.append(event_date)
        if len(run) == d_eval:
            break
    else:
        evaluable = dict.fromkeys(sorted(cities), 0)
        for day in days:
            evaluable[day.series] += int(day.evaluable)
        raise ValueError(
            f"no {d_eval} contiguous event-days are evaluable in every city, only {sequence}"
            + "; evaluable days per city: "
            + ", ".join(f"{city}={count}" for city, count in evaluable.items())
        )

    d_disc, _ = split_lengths(d_eval)
    index_of = {event_date: position + 1 for position, event_date in enumerate(run)}
    logger.info(
        "run_scope event_days=%d covered=%d evaluable=%d over_tolerance=%d first=%s last=%s",
        len(days),
        sum(1 for day in days if day.covered),
        sum(1 for day in days if day.evaluable),
        over_tolerance,
        run[0].isoformat(),
        run[-1].isoformat(),
    )
    return DayInventory(
        days=tuple(
            replace(
                day,
                in_scope=True,
                day_index=index_of[day.event_date],
                split=DISCOVERY if index_of[day.event_date] <= d_disc else HOLDOUT,
            )
            if day.event_date in index_of
            else day
            for day in days
        ),
        outage_us=outage_us,
        over_tolerance=over_tolerance,
    )


def apply_spans(days: Sequence[EventDay], exclusions: Sequence[Exclusion]) -> list[EventDay]:
    out = []
    for day in days:
        if not day.in_scope:
            out.append(day)
            continue
        excluded = union_overlap_us(exclusions, day.window_start, day.window_end)
        out.append(replace(day, excluded_us=excluded, span_us=_DAY_US - excluded))
    return out


def freeze_split(days: Sequence[EventDay], *, d_eval: int) -> Split:
    scoped = [day for day in days if day.in_scope]
    per_city = {}
    for day in scoped:
        per_city.setdefault(day.series, set()).add(day.event_date)
    distinct = {tuple(sorted(dates)) for dates in per_city.values()}
    if len(distinct) != 1 or len(next(iter(distinct))) != d_eval:
        raise ValueError(
            "cities disagree on the frozen event-days: "
            + ", ".join(f"{city}={len(dates)}" for city, dates in sorted(per_city.items()))
        )
    discovery = sorted({day.event_date for day in scoped if day.split == DISCOVERY})
    holdout = sorted({day.event_date for day in scoped if day.split == HOLDOUT})
    return Split(
        cities=tuple(sorted(per_city)),
        discovery_days=tuple(discovery),
        holdout_days=tuple(holdout),
        boundary_event_day=holdout[0],
        scope_start=min(day.window_start for day in scoped),
        scope_end=max(day.window_end for day in scoped),
    )


def split_payload(split: Split) -> dict:
    return {
        "d_eval": len(split.discovery_days) + len(split.holdout_days),
        "d_disc": len(split.discovery_days),
        "d_hold": len(split.holdout_days),
        "cities": list(split.cities),
        "first_evaluable_event_day": split.discovery_days[0].isoformat(),
        "last_evaluable_event_day": split.holdout_days[-1].isoformat(),
        "boundary_event_day": split.boundary_event_day.isoformat(),
        "discovery_days": [day.isoformat() for day in split.discovery_days],
        "holdout_days": [day.isoformat() for day in split.holdout_days],
        "scope_start": split.scope_start.isoformat(),
        "scope_end": split.scope_end.isoformat(),
    }


def freeze_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_exclusions(path: Path, exclusions: Sequence[Exclusion]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    ordered = sorted(exclusions, key=lambda item: (item.start, item.exclusion_class, item.end))
    rows = [
        {
            "exclusion_id": index,
            "exclusion_class": exclusion.exclusion_class,
            "start": exclusion.start,
            "end": exclusion.end,
            "duration_us": exclusion.duration_us,
            "boundary_id": exclusion.boundary_id,
            "gap_id": exclusion.gap_id,
            "gap_reason": exclusion.gap_reason,
            "padded": exclusion.padded,
        }
        for index, exclusion in enumerate(ordered)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=EXCLUSIONS_SCHEMA), path)


def write_event_days(path: Path, days: Sequence[EventDay]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    rows = [
        {
            "series": day.series,
            "station": day.station,
            "timezone": day.timezone,
            "event_date": day.event_date,
            "window_start": day.window_start,
            "window_end": day.window_end,
            "tickers": day.tickers,
            "ladder_rows": day.ladder_rows,
            "first_event_at": day.first_event_at,
            "last_event_at": day.last_event_at,
            "covered": day.covered,
            "evaluable": day.evaluable,
            "in_scope": day.in_scope,
            "day_index": day.day_index,
            "split": day.split,
            "excluded_us": day.excluded_us,
            "span_us": day.span_us,
        }
        for day in days
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA), path)


def write_split(path: Path, split: Split) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = split_payload(split)
    digest = freeze_digest(payload)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=1))
    return digest
