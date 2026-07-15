from __future__ import annotations

import gzip
import json
import sqlite3
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from bot.markets.parser import parse_ticker


TRADE_TYPE = "trade"
OUTCOME_KEY = "taker_outcome_side"
LEGACY_KEY = "taker_side"
HIGH_PREFIX = "KXHIGH"

PRESENT = "present"
EMPTY = "empty"
ABSENT = "absent"

DISCOVERY = "discovery"
HOLDOUT = "holdout"

# Both the trade frame and the subscribe ack that opens the trade channel carry the word, so the
# screen only narrows the lines worth parsing; the frame's own type field decides.
_FRAME_SCREEN = rb"\"trade\""
_DB_TS = "%Y-%m-%d %H:%M:%S.%f"

TRADES_QUERY = (
    "SELECT trade_id, received_at, taker_side FROM ws_trades "
    "WHERE ticker = ? AND received_at >= ? AND received_at < ?"
)


# The taker lifted resting size on the side named here, so taker_side = "yes" is an aggressive
# buy of YES and evidence the market is moving toward YES settling true.
@dataclass(frozen=True, slots=True)
class TakerDirection:
    outcome_bought: str
    yes_pressure: int


DIRECTIONS: dict[str, TakerDirection] = {
    "yes": TakerDirection(outcome_bought="yes", yes_pressure=1),
    "no": TakerDirection(outcome_bought="no", yes_pressure=-1),
}


def taker_direction(taker_side: str) -> TakerDirection | None:
    return DIRECTIONS.get(taker_side)


def is_trade_frame(frame: Mapping[str, object]) -> bool:
    return frame.get("type") == TRADE_TYPE


def key_state(msg: Mapping[str, object], key: str) -> str:
    if key not in msg:
        return ABSENT
    value = msg[key]
    return EMPTY if value is None or value == "" else PRESENT


@dataclass(slots=True)
class DecodeTally:
    frames: int = 0
    high_frames: int = 0
    both_present: int = 0
    disagreed: int = 0
    outcome_state: Counter[str] = field(default_factory=Counter)
    legacy_state: Counter[str] = field(default_factory=Counter)
    outcome_values: Counter[str] = field(default_factory=Counter)
    legacy_values: Counter[str] = field(default_factory=Counter)
    key_sets: Counter[str] = field(default_factory=Counter)

    def __add__(self, other: DecodeTally) -> DecodeTally:
        return DecodeTally(
            frames=self.frames + other.frames,
            high_frames=self.high_frames + other.high_frames,
            both_present=self.both_present + other.both_present,
            disagreed=self.disagreed + other.disagreed,
            outcome_state=self.outcome_state + other.outcome_state,
            legacy_state=self.legacy_state + other.legacy_state,
            outcome_values=self.outcome_values + other.outcome_values,
            legacy_values=self.legacy_values + other.legacy_values,
            key_sets=self.key_sets + other.key_sets,
        )


def read_trade_frames(path: Path, *, start: datetime, end: datetime) -> Iterator[dict]:
    with gzip.open(path, "rb") as handle:
        for line in handle:
            if _FRAME_SCREEN not in line:
                continue
            record = json.loads(line)
            frame = json.loads(record["raw"])
            if not is_trade_frame(frame):
                continue
            received_at = datetime.fromisoformat(record["received_at"])
            if start <= received_at < end:
                yield frame


def tally_frames(frames: Iterable[Mapping[str, object]]) -> DecodeTally:
    tally = DecodeTally()
    for frame in frames:
        msg = frame["msg"]
        tally.frames += 1
        if msg["market_ticker"].startswith(HIGH_PREFIX):
            tally.high_frames += 1
        tally.key_sets[",".join(sorted(msg))] += 1
        outcome = key_state(msg, OUTCOME_KEY)
        legacy = key_state(msg, LEGACY_KEY)
        tally.outcome_state[outcome] += 1
        tally.legacy_state[legacy] += 1
        if outcome == PRESENT:
            tally.outcome_values[msg[OUTCOME_KEY]] += 1
        if legacy == PRESENT:
            tally.legacy_values[msg[LEGACY_KEY]] += 1
        if outcome == PRESENT and legacy == PRESENT:
            tally.both_present += 1
            if msg[OUTCOME_KEY] != msg[LEGACY_KEY]:
                tally.disagreed += 1
    return tally


@dataclass(frozen=True, slots=True)
class Interval:
    start: datetime
    end: datetime


class ExcludedTime:
    intervals: tuple[Interval, ...]

    def __init__(self, intervals: Iterable[Interval]) -> None:
        merged: list[list[datetime]] = []
        for interval in sorted(intervals, key=lambda item: (item.start, item.end)):
            if merged and interval.start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], interval.end)
            else:
                merged.append([interval.start, interval.end])
        self.intervals = tuple(Interval(start=left, end=right) for left, right in merged)
        self._starts = [interval.start for interval in self.intervals]

    def covers(self, at: datetime) -> bool:
        index = bisect_right(self._starts, at) - 1
        return index >= 0 and at < self.intervals[index].end


# Deliberately not imported from bot.replay.run_scope: that module pulls bot.main for STATIONS,
# and bot/replay already depends on bot/lag, so importing it back would invert the layering.
@dataclass(frozen=True, slots=True)
class Split:
    cities: tuple[str, ...]
    discovery_days: tuple[date, ...]
    holdout_days: tuple[date, ...]
    boundary_event_day: date
    scope_start: datetime
    scope_end: datetime


@dataclass(frozen=True, slots=True)
class ScopeDay:
    series: str
    event_date: date
    window_start: datetime
    window_end: datetime


@dataclass(frozen=True, slots=True)
class ScopedTicker:
    ticker: str
    series: str
    event_date: date
    window_start: datetime
    window_end: datetime
    split: str


@dataclass(frozen=True, slots=True)
class TradeRow:
    ticker: str
    trade_id: str
    received_at: datetime
    taker_side: str


@dataclass(frozen=True, slots=True)
class TradeCounts:
    rows: int
    dropped_excluded: int
    empty_side: int
    side_values: dict[str, int]
    distinct_trade_ids: int
    duplicated_values: int
    surplus_rows: int


def read_split(path: Path) -> Split:
    raw = json.loads(path.read_text())
    return Split(
        cities=tuple(raw["cities"]),
        discovery_days=tuple(date.fromisoformat(day) for day in raw["discovery_days"]),
        holdout_days=tuple(date.fromisoformat(day) for day in raw["holdout_days"]),
        boundary_event_day=date.fromisoformat(raw["boundary_event_day"]),
        scope_start=datetime.fromisoformat(raw["scope_start"]),
        scope_end=datetime.fromisoformat(raw["scope_end"]),
    )


def split_of(event_date: date, split: Split) -> str:
    if event_date in split.discovery_days:
        return DISCOVERY
    if event_date in split.holdout_days:
        return HOLDOUT
    return ""


def read_scope_days(path: Path) -> tuple[ScopeDay, ...]:
    rows = pq.read_table(
        path, columns=["series", "event_date", "window_start", "window_end", "in_scope"]
    ).to_pylist()
    days = [
        ScopeDay(
            series=row["series"],
            event_date=row["event_date"],
            window_start=row["window_start"],
            window_end=row["window_end"],
        )
        for row in rows
        if row["in_scope"]
    ]
    return tuple(sorted(days, key=lambda day: (day.series, day.event_date)))


def read_exclusions(path: Path) -> ExcludedTime:
    rows = pq.read_table(path, columns=["start", "end"]).to_pylist()
    return ExcludedTime(Interval(start=row["start"], end=row["end"]) for row in rows)


def scoped_tickers(
    tickers: Iterable[str], days: Sequence[ScopeDay], split: Split
) -> tuple[ScopedTicker, ...]:
    by_event = {(day.series, day.event_date): day for day in days}
    scoped = []
    for ticker in tickers:
        parsed = parse_ticker(ticker)
        day = by_event.get((parsed.series, parsed.event_date))
        if day is None:
            continue
        scoped.append(
            ScopedTicker(
                ticker=ticker,
                series=day.series,
                event_date=day.event_date,
                window_start=day.window_start,
                window_end=day.window_end,
                split=split_of(day.event_date, split),
            )
        )
    return tuple(sorted(scoped, key=lambda item: (item.series, item.event_date, item.ticker)))


def read_trades(conn: sqlite3.Connection, scoped: ScopedTicker) -> Iterator[TradeRow]:
    bounds = (
        scoped.ticker,
        scoped.window_start.astimezone(timezone.utc).strftime(_DB_TS),
        scoped.window_end.astimezone(timezone.utc).strftime(_DB_TS),
    )
    for trade_id, received_at, taker_side in conn.execute(TRADES_QUERY, bounds):
        yield TradeRow(
            ticker=scoped.ticker,
            trade_id=trade_id,
            received_at=datetime.strptime(received_at, _DB_TS).replace(tzinfo=timezone.utc),
            taker_side=taker_side,
        )


def count_trades(rows: Iterable[TradeRow], excluded: ExcludedTime) -> TradeCounts:
    kept = 0
    dropped = 0
    empty = 0
    values: Counter[str] = Counter()
    seen: set[str] = set()
    repeated: set[str] = set()
    for row in rows:
        if excluded.covers(row.received_at):
            dropped += 1
            continue
        kept += 1
        values[row.taker_side] += 1
        if not row.taker_side:
            empty += 1
        if row.trade_id in seen:
            repeated.add(row.trade_id)
        else:
            seen.add(row.trade_id)
    return TradeCounts(
        rows=kept,
        dropped_excluded=dropped,
        empty_side=empty,
        side_values=dict(values),
        distinct_trade_ids=len(seen),
        duplicated_values=len(repeated),
        surplus_rows=kept - len(seen),
    )
