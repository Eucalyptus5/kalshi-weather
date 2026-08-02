import json
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bot.lag.fee_floor import fee_source
from bot.lag.r0_universe import R0Universe
from bot.lag.r0_universe import freeze_digest as universe_digest
from bot.lag.read_rtt import FloorSource, load_samples
from bot.lag.run_manifest import RunInputs, resolve_latency_floor
from bot.replay.artifacts import (
    BOUNDARIES_SCHEMA,
    COVERAGE_SCHEMA,
    LADDER_SCHEMA,
    SCALARS_SCHEMA,
    TOUCH_SCHEMA,
    TRADES_SCHEMA,
    WINDOWS_SCHEMA,
)
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    QUIET_BAND,
    RECORDED_GAP,
    RESUBSCRIBE_BLIND,
    SUBSCRIPTION_WIDE,
    EventDay,
    Exclusion,
)
from bot.replay.run_scope import freeze_digest as split_digest


TOUCH = "touch"
LADDER = "ladder"
TRADES = "trades"
KIND_SCHEMAS = {TOUCH: TOUCH_SCHEMA, LADDER: LADDER_SCHEMA, TRADES: TRADES_SCHEMA}
EXCLUSION_CLASSES = (QUIET_BAND, RECORDED_GAP, RESUBSCRIBE_BLIND, SUBSCRIPTION_WIDE)

_DAY = timedelta(days=1)


def window_dates(window_start: datetime, window_end: datetime) -> list[date]:
    day = window_start.astimezone(timezone.utc).date()
    last = window_end.astimezone(timezone.utc).date()
    out = []
    while day <= last:
        out.append(day)
        day += _DAY
    return out


# The partition date is the UTC arrival date of the rows, and one partition holds one file per
# barrier the pass closed, so the id order the caller reads is (date, barrier) order.
def partition_files(root: Path, kind: str, series_root: str, dates: Iterable[date]) -> list[Path]:
    directory = root / kind
    return [
        path
        for day in sorted(dates)
        for path in sorted(directory.glob(f"{series_root}-{day.isoformat()}-b*.parquet"))
    ]


def read_window(
    root: Path, kind: str, series_root: str, window_start: datetime, window_end: datetime
) -> pa.Table:
    schema = KIND_SCHEMAS[kind]
    paths = partition_files(root, kind, series_root, window_dates(window_start, window_end))
    tables = [_read_checked(path, kind, schema) for path in paths]
    if not tables:
        return schema.empty_table()
    table = pa.concat_tables(tables)
    received = table.column("received_at")
    return table.filter(
        pc.and_(
            pc.greater_equal(received, window_start),
            pc.less_equal(received, window_end),
        )
    )


def partition_rows(root: Path, kind: str, series_root: str, dates: Iterable[date]) -> int:
    return sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in partition_files(root, kind, series_root, dates)
    )


@dataclass(frozen=True, slots=True)
class Inventory:
    windows: pa.Table
    boundaries: pa.Table
    coverage: pa.Table
    scalars: pa.Table
    values: Mapping[str, str]


def read_inventory(root: Path) -> Inventory:
    directory = root / "inventory"
    scalars = _read_checked(directory / "scalars-b000000.parquet", "scalars", SCALARS_SCHEMA)
    return Inventory(
        windows=_read_checked(directory / "windows-b000000.parquet", "windows", WINDOWS_SCHEMA),
        boundaries=_read_checked(
            directory / "boundaries-b000000.parquet", "boundaries", BOUNDARIES_SCHEMA
        ),
        coverage=_read_checked(directory / "coverage-b000000.parquet", "coverage", COVERAGE_SCHEMA),
        scalars=scalars,
        values={row["name"]: row["value"] for row in scalars.to_pylist()},
    )


@dataclass(frozen=True, slots=True)
class Intervals:
    starts: tuple[datetime, ...]
    ends: tuple[datetime, ...]


def merge_intervals(spans: Iterable[tuple[datetime, datetime]]) -> Intervals:
    merged: list[list[datetime]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return Intervals(
        starts=tuple(span[0] for span in merged), ends=tuple(span[1] for span in merged)
    )


# Endpoint-inclusive, and merged ends rise with merged starts, so the last interval opening at or
# before the window's end is the only one that can reach back into it.
def intersects(intervals: Intervals, start: datetime, end: datetime) -> bool:
    index = bisect_right(intervals.starts, end)
    return index > 0 and intervals.ends[index - 1] >= start


@dataclass(frozen=True, slots=True)
class RunScope:
    exclusions: tuple[Exclusion, ...]
    merged: Intervals
    by_class: Mapping[str, Intervals]
    event_days: Mapping[tuple[str, date], EventDay]
    discovery_days: frozenset[date]
    holdout_days: frozenset[date]
    scope_start: datetime
    scope_end: datetime
    universe: R0Universe


def load_run_scope(directory: Path) -> RunScope:
    exclusions_path = directory / "exclusions.parquet"
    event_days_path = directory / "event_days.parquet"
    split_path = directory / "split.json"
    universe_path = directory / "r0_universe.json"

    exclusion_rows = _read_checked(exclusions_path, "exclusions", EXCLUSIONS_SCHEMA).to_pylist()
    day_rows = _read_checked(event_days_path, "event_days", EVENT_DAYS_SCHEMA).to_pylist()
    split = _read_frozen_json(split_path, split_digest)
    universe = _read_frozen_json(universe_path, universe_digest)

    discovery = frozenset(date.fromisoformat(day) for day in split["discovery_days"])
    holdout = frozenset(date.fromisoformat(day) for day in split["holdout_days"])
    frozen = {day: DISCOVERY for day in discovery} | {day: HOLDOUT for day in holdout}

    days = {}
    for row in day_rows:
        if not row["in_scope"]:
            continue
        expected = frozen.get(row["event_date"])
        if row["split"] != expected:
            raise ValueError(
                f"{event_days_path} splits {row['series']} {row['event_date'].isoformat()} as "
                f"{row['split']!r}, {split_path.name} says {expected!r}"
            )
        days[(row["series"], row["event_date"])] = EventDay(**row)

    exclusions = tuple(
        Exclusion(
            exclusion_class=row["exclusion_class"],
            start=row["start"],
            end=row["end"],
            boundary_id=row["boundary_id"],
            gap_id=row["gap_id"],
            gap_reason=row["gap_reason"],
            padded=row["padded"],
        )
        for row in exclusion_rows
    )
    unknown = sorted({item.exclusion_class for item in exclusions}.difference(EXCLUSION_CLASSES))
    if unknown:
        raise ValueError(
            f"{exclusions_path} carries exclusion classes outside the frozen set: "
            + ", ".join(unknown)
        )
    return RunScope(
        exclusions=exclusions,
        merged=merge_intervals((item.start, item.end) for item in exclusions),
        by_class={
            name: merge_intervals(
                (item.start, item.end) for item in exclusions if item.exclusion_class == name
            )
            for name in EXCLUSION_CLASSES
        },
        event_days=days,
        discovery_days=discovery,
        holdout_days=holdout,
        scope_start=datetime.fromisoformat(split["scope_start"]),
        scope_end=datetime.fromisoformat(split["scope_end"]),
        universe=R0Universe(
            fraction_invalid_max=Decimal(universe["fraction_invalid_max"]),
            passing=tuple(universe["passing"]),
            lock_dependent=tuple(universe["lock_dependent"]),
            recorded=tuple(universe["recorded"]),
            ladder_widths=tuple(universe["ladder_widths"]),
            in_scope_city_days=universe["in_scope_city_days"],
            reconciliation=universe["reconciliation"],
            recorded_not_passing=tuple(universe["recorded_not_passing"]),
            passing_not_recorded=tuple(universe["passing_not_recorded"]),
        ),
    )


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    series: str
    event_date: date
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class Screened:
    kept: tuple[EvidenceWindow, ...]
    candidates: int
    excluded: int
    out_of_scope: int
    out_of_window: int
    by_class: Mapping[str, int]

    @property
    def excluded_fraction(self) -> Decimal:
        return Decimal(self.excluded) / Decimal(self.candidates)


def intersects_exclusion(scope: RunScope, start: datetime, end: datetime) -> bool:
    return intersects(scope.merged, start, end)


# A market lists and prints about 41 hours before it closes, so its own rows reach the tape well
# before its observation window opens, and those rows are not evidence about the day it settles on.
def within_event_day(scope: RunScope, window: EvidenceWindow) -> bool:
    day = scope.event_days.get((window.series, window.event_date))
    if day is None:
        raise ValueError(
            f"{window.series} {window.event_date.isoformat()} is not in the frozen scope"
        )
    return day.window_start <= window.start and window.end <= day.window_end


def split_of(scope: RunScope, series: str, event_date: date) -> str:
    if (series, event_date) not in scope.event_days:
        raise ValueError(f"{series} {event_date.isoformat()} is not in the frozen scope")
    return DISCOVERY if event_date in scope.discovery_days else HOLDOUT


# Classes overlap in time, so the per-class counts can sum above the dropped count.
def screen_windows(scope: RunScope, windows: Sequence[EvidenceWindow]) -> Screened:
    kept = []
    excluded = 0
    out_of_scope = 0
    out_of_window = 0
    by_class = dict.fromkeys(scope.by_class, 0)
    for window in windows:
        if (window.series, window.event_date) not in scope.event_days:
            out_of_scope += 1
            continue
        if not within_event_day(scope, window):
            out_of_window += 1
            continue
        if not intersects(scope.merged, window.start, window.end):
            kept.append(window)
            continue
        excluded += 1
        for name, intervals in scope.by_class.items():
            if intersects(intervals, window.start, window.end):
                by_class[name] += 1
    return Screened(
        kept=tuple(kept),
        candidates=len(windows),
        excluded=excluded,
        out_of_scope=out_of_scope,
        out_of_window=out_of_window,
        by_class=by_class,
    )


# screen_windows keeps its input order, so walking the offer against what came back in one pass
# recovers the mask; two equal windows screen alike, so a greedy match cannot misalign.
def keep_mask(offered: Sequence[EvidenceWindow], kept: Sequence[EvidenceWindow]) -> np.ndarray:
    mask = np.zeros(len(offered), dtype=bool)
    cursor = 0
    for index, window in enumerate(offered):
        if cursor < len(kept) and kept[cursor] == window:
            mask[index] = True
            cursor += 1
    if cursor != len(kept):
        raise ValueError("the screened windows are not a subsequence of the ones offered")
    return mask


SELF_CHARGED_BAR: Decimal = Decimal("0")
SELF_CHARGED_BAR_SOURCE = "statistic_charges_its_own_fee"


def assemble_run_inputs(
    *,
    run_id: str,
    preregistration: Path,
    repo: Path,
    run_scope: Path,
    artifacts: Path,
    rtt_samples: Path,
    floor_source: FloorSource,
    economic_bar_size: Decimal,
    economic_bar_price: Decimal,
    economic_bar_price_source: str,
    bootstrap_seed: int,
) -> RunInputs:
    scope = load_run_scope(run_scope)
    arrivals: dict[str, set[date]] = {}
    for (series, _), day in scope.event_days.items():
        arrivals.setdefault(series, set()).update(window_dates(day.window_start, day.window_end))

    row_counts = {
        kind: sum(
            partition_rows(artifacts, kind, series, dates) for series, dates in arrivals.items()
        )
        for kind in KIND_SCHEMAS
    }
    row_counts["exclusions"] = pq.ParquetFile(run_scope / "exclusions.parquet").metadata.num_rows
    row_counts["event_days"] = pq.ParquetFile(run_scope / "event_days.parquet").metadata.num_rows

    return RunInputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        accrual_start=scope.scope_start,
        accrual_end=scope.scope_end,
        row_counts=row_counts,
        universe=scope.universe,
        fee=fee_source(),
        floor=resolve_latency_floor(load_samples(rtt_samples), floor_source),
        economic_bar_size=economic_bar_size,
        economic_bar_price=economic_bar_price,
        economic_bar_price_source=economic_bar_price_source,
        bootstrap_seed=bootstrap_seed,
    )


def _read_checked(path: Path, name: str, schema: pa.Schema) -> pa.Table:
    table = pq.read_table(path)
    if not table.schema.equals(schema):
        raise ValueError(f"{path} does not carry the frozen {name} schema")
    return table


def _read_frozen_json(path: Path, digest: Callable[[Mapping[str, object]], str]) -> dict:
    payload = json.loads(path.read_text())
    if "sha256" not in payload:
        raise ValueError(f"{path} carries no sha256")
    stored = payload.pop("sha256")
    if digest(payload) != stored:
        raise ValueError(f"{path} does not match the sha256 it carries")
    return payload
