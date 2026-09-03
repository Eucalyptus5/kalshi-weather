import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from statistics import median

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bot.lag.fee_floor import MAKER_RATE_SOURCE, PUBLISHED_MAKER_RATE
from bot.lag.lead_lag import (
    CITY_SERIES,
    CORRIDORS,
    MOVE_BAR_CENTS,
    PAIRS,
    WINDOW_S,
    AtmSeries,
    Episode,
    Pair,
    atm_series,
    pair_episodes,
    round_trip_fee_cents,
)
from bot.lag.read_rtt import FloorSource, LatencyFloor
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME, write_manifest
from bot.lag.tape_stats import (
    BLOCK_DAYS,
    CorridorDayAggregate,
    Direction,
    day_blocks,
    wild_cluster_bootstrap,
)
from bot.lag.tape_studies import (
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    TOUCH,
    EvidenceWindow,
    RunScope,
    Screened,
    assemble_run_inputs,
    keep_mask,
    load_run_scope,
    partition_files,
    screen_windows,
    window_dates,
)
from bot.markets.parser import parse_ticker
from bot.replay.artifacts import TOUCH_SCHEMA
from bot.replay.run_scope import DISCOVERY, EventDay


logger = logging.getLogger(__name__)

RESULTS_NAME = "results.json"
ATM_COLUMNS = ("id", "ticker", "received_at", "yes_bid", "no_bid")
CI_LEVEL = 0.95
CORRIDOR_DAY_MIN_REPORT: int = 12
FORWARD = "forward"
REVERSE = "reverse"
CORRIDOR_DAYS = "corridor-days"
NULL_VALUE = Decimal(0)

MICROS_PER_S = 1_000_000


@dataclass(slots=True)
class ScreenTally:
    candidates: int = 0
    excluded: int = 0
    out_of_window: int = 0
    out_of_scope: int = 0
    by_class: dict[str, int] = field(default_factory=dict)

    def add(self, screened: Screened) -> None:
        self.candidates += screened.candidates
        self.excluded += screened.excluded
        self.out_of_window += screened.out_of_window
        self.out_of_scope += screened.out_of_scope
        for name, count in screened.by_class.items():
            self.by_class[name] = self.by_class.get(name, 0) + count

    @property
    def excluded_fraction(self) -> Decimal | None:
        if self.candidates == 0:
            return None
        return Decimal(self.excluded) / Decimal(self.candidates)


@dataclass(frozen=True, slots=True, kw_only=True)
class Sweep:
    episodes: Mapping[str, tuple[Episode, ...]]
    screened: ScreenTally
    offered: int
    kept: int
    rows: int
    in_scope: int
    silent: tuple[tuple[str, date], ...]
    tickers: Mapping[tuple[str, date], str]


@dataclass(frozen=True, slots=True, kw_only=True)
class MedianInterval:
    low: Decimal | None
    high: Decimal | None
    ci_level: float
    tail: float
    tested: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PairReadout:
    corridor: str
    episodes: int
    median_lead_s: Decimal | None


@dataclass(frozen=True, slots=True, kw_only=True)
class CorridorReadout:
    episodes: int
    corridor_days: int
    median_lead_s: Decimal | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionReadout:
    reading: str
    episodes: int
    corridor_days: int
    n_blocks: int
    median_lead_s: Decimal | None
    interval: MedianInterval | None
    per_pair: Mapping[str, PairReadout]
    per_corridor: Mapping[str, CorridorReadout]
    episodes_per_corridor_day: Mapping[str, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadLagRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    seed: int
    floor: LatencyFloor
    universe: tuple[str, ...]
    discovery_days: tuple[date, ...]
    sweep: Sweep
    forward: DirectionReadout
    reverse: DirectionReadout


def _checked(path: Path) -> pq.ParquetFile:
    stored = pq.ParquetFile(path)
    if not stored.schema_arrow.equals(TOUCH_SCHEMA):
        raise ValueError(f"{path} does not carry the frozen {TOUCH} schema")
    return stored


# A partition is keyed by the rows' UTC arrival date and a market lists about 41 hours before it
# closes, so one file carries legs of several event dates; without the filter the leg nearest half
# would be picked across a mixture of them. Each file is pruned before it joins the buffer: a whole
# arrival date of a busy root does not fit beside the rest on a host with under 2 GB free.
def read_city_day(artifacts: Path, series: str, event_date: date, day: EventDay) -> pa.Table:
    columns = list(ATM_COLUMNS)
    tables = []
    for path in partition_files(
        artifacts, TOUCH, series, window_dates(day.window_start, day.window_end)
    ):
        table = _checked(path).read(columns=columns)
        column = table.column("ticker")
        legs = [
            ticker
            for ticker in column.unique().to_pylist()
            if parse_ticker(ticker).event_date == event_date
        ]
        tables.append(table.filter(pc.is_in(column, value_set=pa.array(legs, type=pa.string()))))
    if not tables:
        return TOUCH_SCHEMA.empty_table().select(columns)
    return pa.concat_tables(tables)


def sweep_pairs(scope: RunScope, artifacts: Path) -> Sweep:
    swept = {series for series, _ in scope.event_days}
    recorded = set(scope.universe.recorded)
    if swept != recorded:
        raise ValueError(
            "the frozen event days and the recorded universe disagree on: "
            + ", ".join(sorted(swept ^ recorded))
        )

    episodes: dict[str, list[Episode]] = {FORWARD: [], REVERSE: []}
    screened = ScreenTally()
    tickers: dict[tuple[str, date], str] = {}
    silent: list[tuple[str, date]] = []
    offered = 0
    kept = 0
    rows = 0
    in_scope = 0

    # Event dates outer and cities inner, so only one day of one city's series is held at a time.
    for event_date in sorted(scope.discovery_days):
        started = time.monotonic()
        before = rows
        found_today = 0
        built: dict[str, AtmSeries] = {}
        for city, series in CITY_SERIES.items():
            day = scope.event_days.get((series, event_date))
            if day is None:
                continue
            in_scope += 1
            table = read_city_day(artifacts, series, event_date, day)
            rows += table.num_rows
            picked = atm_series(
                series,
                event_date,
                table,
                window_start=day.window_start,
                window_end=day.window_end,
            )
            if picked is None:
                silent.append((series, event_date))
                continue
            tickers[(series, event_date)] = picked.ticker
            built[city] = picked

        for pair in PAIRS:
            upstream = built.get(pair.upstream)
            downstream = built.get(pair.downstream)
            if upstream is None or downstream is None:
                continue
            for reading, leader, follower in (
                (FORWARD, upstream, downstream),
                (REVERSE, downstream, upstream),
            ):
                found = pair_episodes(pair, leader, follower, reverse=reading == REVERSE)
                offered += len(found)
                found_today += len(found)
                spans = [item.evidence_span() for item in found]
                # The two cities sit in different time zones, so one span can fall inside one
                # event-day window and outside the other.
                leader_windows = [
                    EvidenceWindow(
                        series=leader.series, event_date=event_date, start=start, end=end
                    )
                    for start, end in spans
                ]
                follower_windows = [
                    EvidenceWindow(
                        series=follower.series, event_date=event_date, start=start, end=end
                    )
                    for start, end in spans
                ]
                on_leader = screen_windows(scope, leader_windows)
                on_follower = screen_windows(scope, follower_windows)
                screened.add(on_leader)
                screened.add(on_follower)
                both = keep_mask(leader_windows, on_leader.kept) & keep_mask(
                    follower_windows, on_follower.kept
                )
                admitted = [item for item, keep in zip(found, both, strict=True) if keep]
                episodes[reading].extend(admitted)
                kept += len(admitted)

        logger.info(
            "lead_lag event_date=%s cities=%d silent=%d episodes=%d rows=%d elapsed_s=%.1f",
            event_date.isoformat(),
            len(built),
            sum(1 for _, day_key in silent if day_key == event_date),
            found_today,
            rows - before,
            time.monotonic() - started,
        )

    return Sweep(
        episodes={reading: tuple(items) for reading, items in episodes.items()},
        screened=screened,
        offered=offered,
        kept=kept,
        rows=rows,
        in_scope=in_scope,
        silent=tuple(silent),
        tickers=tickers,
    )


def pooled_median(episodes: Sequence[Episode]) -> Decimal:
    return median(item.lead_s for item in episodes)


# The wild bootstrap estimates a ratio of sums, so a median reaches it as a sign statistic: the
# ratio below is the mean of sign(lead - theta) and a null of zero on it is the hypothesis that
# the median is theta, which is why the interval and the point estimate are one quantity.
def sign_observations(
    leads: np.ndarray, groups: np.ndarray, keys: Sequence[tuple[str, date]], theta: Decimal
) -> list[CorridorDayAggregate]:
    signs = np.sign(leads - int(theta * MICROS_PER_S))
    counted = np.bincount(groups, minlength=len(keys))
    above = np.bincount(groups[signs > 0], minlength=len(keys))
    below = np.bincount(groups[signs < 0], minlength=len(keys))
    return [
        CorridorDayAggregate(
            corridor=corridor,
            day=day,
            total=Decimal(int(above[seat]) - int(below[seat])),
            weight=Decimal(int(counted[seat])),
        )
        for seat, (corridor, day) in enumerate(keys)
    ]


def _sign_inputs(
    episodes: Sequence[Episode],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, date], ...]]:
    seats: dict[tuple[str, date], int] = {}
    groups = []
    for item in episodes:
        key = (item.corridor, item.event_date)
        if key not in seats:
            seats[key] = len(seats)
        groups.append(seats[key])
    return (
        np.array([int(item.lead_s * MICROS_PER_S) for item in episodes], dtype=np.int64),
        np.array(groups, dtype=np.int64),
        tuple(seats),
    )


# A theta outside the whole sample gives every episode the same sign, which flattens the panel and
# leaves no studentized statistic; the raise is that documented signal, and it reads as a rejection.
def _admits(
    observations: Sequence[CorridorDayAggregate],
    direction: Direction,
    *,
    ci_level: float,
    resamples: int,
    seed: int,
    block_days: int,
) -> bool:
    try:
        result = wild_cluster_bootstrap(
            observations,
            null_value=NULL_VALUE,
            direction=direction,
            block_days=block_days,
            resamples=resamples,
            seed=seed,
            ci_level=ci_level,
        )
    except ValueError:
        return False
    return result.p_value > (1 - ci_level) / 2


# Outward from the candidate nearest the estimate rather than inward from the ends, so the walk
# costs one resample run per candidate the interval covers rather than one per candidate observed.
def median_interval(
    episodes: Sequence[Episode], *, ci_level: float, resamples: int, seed: int, block_days: int
) -> MedianInterval:
    leads, groups, keys = _sign_inputs(episodes)
    candidates = sorted({item.lead_s for item in episodes})
    estimate = pooled_median(episodes)
    start = min(range(len(candidates)), key=lambda seat: (abs(candidates[seat] - estimate), seat))
    tested = 0

    # A bound is the last candidate admitted, so a start refused on its first probe leaves none.
    low: Decimal | None = None
    high: Decimal | None = None
    seat = start
    while seat >= 0:
        tested += 1
        if not _admits(
            sign_observations(leads, groups, keys, candidates[seat]),
            "greater",
            ci_level=ci_level,
            resamples=resamples,
            seed=seed,
            block_days=block_days,
        ):
            break
        low = candidates[seat]
        seat -= 1

    seat = start
    while seat < len(candidates):
        tested += 1
        if not _admits(
            sign_observations(leads, groups, keys, candidates[seat]),
            "less",
            ci_level=ci_level,
            resamples=resamples,
            seed=seed,
            block_days=block_days,
        ):
            break
        high = candidates[seat]
        seat += 1

    return MedianInterval(
        low=low, high=high, ci_level=ci_level, tail=(1 - ci_level) / 2, tested=tested
    )


def readout(episodes: Sequence[Episode], *, reading: str, seed: int) -> DirectionReadout:
    leads, groups, keys = _sign_inputs(episodes)
    # Blocks partition the corridor-days themselves, so any theta names the same panel.
    clusters = sign_observations(leads, groups, keys, NULL_VALUE)
    reported = len(keys) >= CORRIDOR_DAY_MIN_REPORT

    by_pair: dict[tuple[str, str], list[Episode]] = {}
    by_corridor: dict[str, list[Episode]] = {}
    per_day: dict[tuple[str, date], int] = {}
    for item in episodes:
        by_pair.setdefault((item.upstream, item.downstream), []).append(item)
        by_corridor.setdefault(item.corridor, []).append(item)
        key = (item.corridor, item.event_date)
        per_day[key] = per_day.get(key, 0) + 1

    per_pair = {}
    for pair in PAIRS:
        found = by_pair.get((pair.upstream, pair.downstream), [])
        per_pair[_pair_key(pair)] = PairReadout(
            corridor=pair.corridor,
            episodes=len(found),
            median_lead_s=pooled_median(found) if reported and found else None,
        )

    per_corridor = {}
    for corridor in CORRIDORS:
        found = by_corridor.get(corridor, [])
        per_corridor[corridor] = CorridorReadout(
            episodes=len(found),
            corridor_days=len({item.event_date for item in found}),
            median_lead_s=pooled_median(found) if reported and found else None,
        )

    return DirectionReadout(
        reading=reading,
        episodes=len(episodes),
        corridor_days=len(keys),
        n_blocks=max(day_blocks(clusters, BLOCK_DAYS)) + 1 if clusters else 0,
        median_lead_s=pooled_median(episodes) if reported else None,
        interval=median_interval(
            episodes,
            ci_level=CI_LEVEL,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed,
            block_days=BLOCK_DAYS,
        )
        if reported
        else None,
        per_pair=per_pair,
        per_corridor=per_corridor,
        episodes_per_corridor_day={
            f"{corridor} {day.isoformat()}": count
            for (corridor, day), count in sorted(per_day.items())
        },
    )


def execute(
    *,
    run_id: str,
    preregistration: Path,
    repo: Path,
    run_scope: Path,
    artifacts: Path,
    rtt_samples: Path,
    floor_source: FloorSource,
    seed: int,
    run_root: Path,
) -> LeadLagRun:
    inputs = assemble_run_inputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        run_scope=run_scope,
        artifacts=artifacts,
        kinds=(TOUCH,),
        rtt_samples=rtt_samples,
        floor_source=floor_source,
        maker_rate=PUBLISHED_MAKER_RATE,
        maker_rate_source=MAKER_RATE_SOURCE,
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        bootstrap_seed=seed,
    )
    digest = write_manifest(run_root, inputs)

    scope = load_run_scope(run_scope)
    swept = sweep_pairs(scope, artifacts)
    run = LeadLagRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        seed=seed,
        floor=inputs.floor,
        universe=scope.universe.recorded,
        discovery_days=tuple(sorted(scope.discovery_days)),
        sweep=swept,
        forward=readout(swept.episodes[FORWARD], reading=FORWARD, seed=seed),
        reverse=readout(swept.episodes[REVERSE], reading=REVERSE, seed=seed),
    )
    logger.info(
        "lead_lag forward_median_s=%s forward_n=%d reverse_median_s=%s reverse_n=%d",
        run.forward.median_lead_s,
        run.forward.corridor_days,
        run.reverse.median_lead_s,
        run.reverse.corridor_days,
    )
    return run


def result_payload(run: LeadLagRun) -> dict:
    screened = run.sweep.screened
    fraction = screened.excluded_fraction
    paired = set(CITY_SERIES.values())
    return {
        "run_id": run.run_id,
        "split": DISCOVERY,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "bootstrap_seed": run.seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "block_days": BLOCK_DAYS,
        "ci_level": CI_LEVEL,
        "move_bar_cents": str(MOVE_BAR_CENTS),
        "round_trip_fee_cents_at_mid": str(round_trip_fee_cents(Decimal("1"), Decimal("0.50"))),
        "window_s": WINDOW_S,
        "latency_floor_source": run.floor.source.value,
        "corridor_day_unit": CORRIDOR_DAYS,
        "corridor_day_ceiling": len(CORRIDORS) * len(run.discovery_days),
        "corridor_day_min_report": CORRIDOR_DAY_MIN_REPORT,
        "discovery_days": [day.isoformat() for day in run.discovery_days],
        FORWARD: _reading_payload(run.forward),
        REVERSE: _reading_payload(run.reverse),
        "episodes": {
            "offered": run.sweep.offered,
            "kept": run.sweep.kept,
            FORWARD: len(run.sweep.episodes[FORWARD]),
            REVERSE: len(run.sweep.episodes[REVERSE]),
        },
        "exclusions": {
            "candidates": screened.candidates,
            "excluded": screened.excluded,
            "out_of_window": screened.out_of_window,
            "out_of_scope": screened.out_of_scope,
            "excluded_fraction": None if fraction is None else str(fraction),
            "by_class": dict(sorted(screened.by_class.items())),
        },
        "rows": run.sweep.rows,
        "city_event_days": run.sweep.in_scope,
        "no_atm_series": [_key(series, day) for series, day in run.sweep.silent],
        "atm_ticker_per_city_day": {
            _key(series, day): ticker for (series, day), ticker in sorted(run.sweep.tickers.items())
        },
        "pairs": {
            _pair_key(pair): {
                "corridor": pair.corridor,
                "upstream_series": CITY_SERIES[pair.upstream],
                "downstream_series": CITY_SERIES[pair.downstream],
            }
            for pair in PAIRS
        },
        "unpaired_roots": [root for root in run.universe if root not in paired],
    }


def _key(series: str, event_date: date) -> str:
    return f"{series} {event_date.isoformat()}"


def _pair_key(pair: Pair) -> str:
    return f"{pair.upstream}->{pair.downstream}"


def _reading_payload(item: DirectionReadout) -> dict:
    interval = item.interval
    return {
        "reading": item.reading,
        "episodes": item.episodes,
        "corridor_days": item.corridor_days,
        "n_blocks": item.n_blocks,
        "median_lead_s": None if item.median_lead_s is None else str(item.median_lead_s),
        "interval": None
        if interval is None
        else {
            "low": None if interval.low is None else str(interval.low),
            "high": None if interval.high is None else str(interval.high),
            "ci_level": interval.ci_level,
            "tail": interval.tail,
            "tested": interval.tested,
        },
        "per_pair": {
            name: {
                "corridor": block.corridor,
                "episodes": block.episodes,
                "median_lead_s": None if block.median_lead_s is None else str(block.median_lead_s),
            }
            for name, block in item.per_pair.items()
        },
        "per_corridor": {
            name: {
                "episodes": block.episodes,
                "corridor_days": block.corridor_days,
                "median_lead_s": None if block.median_lead_s is None else str(block.median_lead_s),
            }
            for name, block in item.per_corridor.items()
        },
        "episodes_per_corridor_day": dict(item.episodes_per_corridor_day),
    }
