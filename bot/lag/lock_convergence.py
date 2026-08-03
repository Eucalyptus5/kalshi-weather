import logging
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median

import pyarrow as pa
import pyarrow.compute as pc

from bot.lag.fee_floor import published_taker_fee
from bot.lag.ladder_run import census
from bot.lag.lock_events import LockEvent, detect_lock_events, is_low_ladder
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME, write_manifest
from bot.lag.tape_stats import (
    ALPHA,
    HOLDOUT_ALPHA,
    BootstrapResult,
    GateVerdict,
    HoldoutVerdict,
    ValueCluster,
    cluster_median_bootstrap,
    evaluate_gate,
    evaluate_holdout,
)
from bot.lag.tape_studies import (
    TOUCH,
    TRADES,
    EvidenceWindow,
    RunScope,
    Screened,
    assemble_run_inputs,
    keep_mask,
    load_run_scope,
    read_window,
    screen_windows,
    split_of,
)
from bot.markets.parser import parse_ticker, resolve_event_kinds
from bot.observations.metar import StationObservation
from bot.replay.analysis_stations import in_cohort
from bot.replay.run_scope import DISCOVERY, HOLDOUT, EventDay


logger = logging.getLogger(__name__)

LOCK_BAND = Decimal("0.95")
PERSIST_S = 60
HALF_LIFE_THRESHOLD_S = Decimal(120)
STATION_DAY_MIN = 20
CI_LEVEL = 0.95
ROUNDING_MARGIN_F = Decimal("1.0")
ECONOMIC_BAR_SIZE: Decimal = Decimal("26")
ECONOMIC_BAR_PRICE: Decimal = Decimal("0.50")
ECONOMIC_BAR_PRICE_SOURCE = "preregistration"

DIRECTION = "greater"
STATION_DAYS = "station event-days"
SPLITS = (DISCOVERY, HOLDOUT)
YES = "yes"
NO = "no"
TAKING = "invalidated_side_taking"
PROVIDING = "invalidated_side_providing"
BUCKETS = (TAKING, PROVIDING)

PASS = "PASS"
CLOSED = "CLOSED"
UNDERPOWERED = "UNDERPOWERED"
UNDECIDABLE = "UNDECIDABLE"

NOT_POWERED = "discovery carries too few station event-days for any statistic to be read"
NO_ESTIMATE = "a split with no surviving half-life carries no estimate to replicate"
ZERO_ESTIMATE = "a discovery estimate of exactly zero fixes no direction to replicate"
DESCRIPTIVE = "reported only; no gate ran on this reading"
CENSORING = (
    "a censored half-life is a lower bound and the gate reads greater, so censoring can only "
    "make a pass harder to reach"
)
LENGTH_BIAS = (
    "a fast leg holds a short evidence window and meets fewer exclusions than a slow one; the "
    "counts are surfaced and no correction is applied"
)

_ONE = Decimal(1)
_PERSIST = timedelta(seconds=PERSIST_S)
_MICROSECOND = timedelta(microseconds=1)
_MICROS_PER_S = Decimal(1_000_000)


@dataclass(frozen=True, slots=True, kw_only=True)
class LockScan:
    cities: tuple[str, ...]
    markets: int
    clean: tuple[tuple[LockEvent, EventDay], ...]
    ambiguous: int
    no_lock: int
    no_observations: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Convergence:
    ticker: str
    series: str
    event_date: date
    station: str
    side_locked: str
    t_lock: datetime
    t_end: datetime
    half_life_s: Decimal
    censored: bool


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


@dataclass(slots=True)
class ArrivalTally:
    deltas: list[Decimal] = field(default_factory=list)
    no_arrival_rows: int = 0
    never_clears: int = 0
    arrival_precedes_lock: int = 0


@dataclass(slots=True)
class FillTally:
    fills: int = 0
    contracts: Decimal = Decimal(0)
    notional: Decimal = Decimal(0)
    fees: Decimal = Decimal(0)
    prices: list[Decimal] = field(default_factory=list)
    by_cent: dict[str, int] = field(default_factory=dict)

    def add(self, contracts: Decimal, price: str) -> None:
        value = Decimal(price)
        self.fills += 1
        self.contracts += contracts
        self.notional += contracts * value
        self.fees += published_taker_fee(contracts, value)
        self.prices.append(value)
        self.by_cent[price] = self.by_cent.get(price, 0) + 1


@dataclass(frozen=True, slots=True, kw_only=True)
class Sweep:
    scan: LockScan
    kept: tuple[Convergence, ...]
    values: Mapping[str, Mapping[str, list[Decimal]]]
    clean_station_days: Mapping[str, int]
    ceiling: int
    screened: ScreenTally
    one_sided_rows: int
    no_book_at_lock: int
    censored: int
    unsettled_station_days: int
    settle_contradicts: int
    usable_event_days: int
    last_settled_event_day: date | None
    arrivals: ArrivalTally
    fills: Mapping[str, FillTally]
    fills_dropped: int
    fills_unclassified: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitReadout:
    split: str
    clusters: tuple[ValueCluster, ...]
    events: int
    bootstrap: BootstrapResult | None

    @property
    def n_station_days(self) -> int:
        return len(self.clusters)


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    gate: GateVerdict | None
    replication: HoldoutVerdict | None
    skipped: str
    verdict: str


@dataclass(frozen=True, slots=True, kw_only=True)
class LockConvergenceRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    seed: int
    sweep: Sweep
    discovery: SplitReadout
    holdout: SplitReadout
    decision: Decision


def scan_locks(
    scope: RunScope,
    artifacts: Path,
    observations: Mapping[tuple[str, date], list[StationObservation]],
    *,
    cohort: str | None = None,
) -> LockScan:
    scoped = set(in_cohort({series for series, _ in scope.event_days}, cohort))
    lock_dependent = set(scope.universe.lock_dependent)
    cities = sorted(scoped & lock_dependent)
    # Zero locks would otherwise read as an underpowered run rather than a freeze paired with the
    # wrong ladder.
    if not cities:
        raise ValueError(
            f"the scope's series {sorted(scoped)} share nothing with the universe's "
            f"lock-dependent series {sorted(lock_dependent)}"
        )
    markets = 0
    ambiguous = 0
    no_lock = 0
    no_observations = 0
    clean: list[tuple[LockEvent, EventDay]] = []

    for series_root in cities:
        groups = census(artifacts, scope, series_root)
        for event_date in sorted(groups):
            day = scope.event_days.get((series_root, event_date))
            if day is None:
                continue
            legs = [parse_ticker(ticker) for ticker in groups[event_date]]
            markets += len(legs)
            recorded = observations.get((day.station, event_date))
            if recorded is None:
                no_observations += len(legs)
                continue
            # A tail read on its own parses as a bracket, so the kinds are settled across the whole
            # event ladder before the detector reads any leg's strike.
            for market in resolve_event_kinds(legs):
                # clears_strike reads this margin for the settle check and the arrival anchor, so
                # the detector is handed it rather than left to carry its own default.
                found = detect_lock_events(
                    market, recorded, tz_name=day.timezone, rounding_margin_f=ROUNDING_MARGIN_F
                )
                if not found:
                    no_lock += 1
                    continue
                if found[0].lock_ambiguous:
                    ambiguous += 1
                    continue
                clean.append((found[0], day))

    logger.info(
        "lock_convergence markets=%d clean=%d ambiguous=%d no_lock=%d no_observations=%d",
        markets,
        len(clean),
        ambiguous,
        no_lock,
        no_observations,
    )
    return LockScan(
        cities=tuple(cities),
        markets=markets,
        clean=tuple(clean),
        ambiguous=ambiguous,
        no_lock=no_lock,
        no_observations=no_observations,
    )


# yes_ask is stored as one minus the NO bid, so an empty NO book prices the ask at 1.00 against no
# size at all. A mid taken there reads as locked the instant the losing side's book empties, which
# is exactly what a lock causes, so both sides must carry depth before the mid means anything.
def book_states(table: pa.Table, ticker: str) -> tuple[list[datetime], list[Decimal | None], int]:
    rows = table.filter(pc.equal(table.column("ticker"), ticker))
    stamps = rows.column("received_at").to_pylist()
    mids: list[Decimal | None] = []
    one_sided = 0
    for bid, bid_depth, ask, ask_depth in zip(
        rows.column("yes_bid").to_pylist(),
        rows.column("yes_bid_depth").to_pylist(),
        rows.column("yes_ask").to_pylist(),
        rows.column("yes_ask_depth").to_pylist(),
        strict=True,
    ):
        if Decimal(bid_depth) <= 0 or Decimal(ask_depth) <= 0:
            mids.append(None)
            one_sided += 1
            continue
        mids.append((Decimal(bid) + Decimal(ask)) / 2)
    return stamps, mids, one_sided


def in_band(mid: Decimal | None, side_locked: str) -> bool:
    if mid is None:
        return False
    if side_locked == YES:
        return mid >= LOCK_BAND
    return mid <= _ONE - LOCK_BAND


# The book is a step function, so it can only enter the band at the lock itself or at a row after
# it, and an entry counts only when every state it takes through the persistence window holds.
def converged(
    stamps: Sequence[datetime],
    mids: Sequence[Decimal | None],
    *,
    side_locked: str,
    t_lock: datetime,
    window_end: datetime,
) -> datetime | None:
    band = [in_band(mid, side_locked) for mid in mids]
    seen: datetime | None = None
    for instant in [t_lock, *(stamp for stamp in stamps if stamp > t_lock)]:
        if instant == seen:
            continue
        seen = instant
        if instant + _PERSIST > window_end:
            return None
        state = bisect_right(stamps, instant) - 1
        if state < 0 or not band[state]:
            continue
        closes = bisect_right(stamps, instant + _PERSIST)
        if all(band[index] for index in range(state + 1, closes)):
            return instant
    return None


def clears_strike(temp_f: Decimal, event: LockEvent) -> bool:
    if is_low_ladder(event.series):
        bar = event.strike - ROUNDING_MARGIN_F
        return temp_f <= bar if event.side_locked == YES else temp_f < bar
    bar = event.strike + ROUNDING_MARGIN_F
    return temp_f >= bar if event.side_locked == YES else temp_f > bar


def arrival_anchor(
    rows: Sequence[StationObservation], day: EventDay, event: LockEvent, tally: ArrivalTally
) -> None:
    in_window = sorted(
        (row for row in rows if day.window_start <= row.valid_time < day.window_end),
        key=lambda row: row.valid_time,
    )
    if not in_window:
        tally.no_arrival_rows += 1
        return
    low = is_low_ladder(event.series)
    running_f = Decimal("Infinity") if low else Decimal("-Infinity")
    for row in in_window:
        running_f = min(running_f, row.temp_f) if low else max(running_f, row.temp_f)
        if not clears_strike(running_f, event):
            continue
        if row.publication_time < event.t0:
            tally.arrival_precedes_lock += 1
            return
        tally.deltas.append(seconds(row.publication_time - event.t0))
        return
    tally.never_clears += 1


def post_lock_fills(
    prints: pa.Table,
    scope: RunScope,
    day: EventDay,
    event: LockEvent,
    tallies: Mapping[str, FillTally],
) -> tuple[int, int]:
    rows = prints.filter(
        pc.and_(
            pc.equal(prints.column("ticker"), event.ticker),
            pc.greater(prints.column("received_at"), event.t0),
        )
    )
    invalidated = NO if event.side_locked == YES else YES
    offered = [
        EvidenceWindow(series=day.series, event_date=day.event_date, start=stamp, end=stamp)
        for stamp in rows.column("received_at").to_pylist()
    ]
    screened = screen_windows(scope, offered)
    unclassified = 0
    for keep, contracts, price, taker_side in zip(
        keep_mask(offered, screened.kept),
        rows.column("count").to_pylist(),
        rows.column(f"{invalidated}_price").to_pylist(),
        rows.column("taker_side").to_pylist(),
        strict=True,
    ):
        if not keep:
            continue
        if taker_side == invalidated:
            tallies[TAKING].add(Decimal(contracts), price)
        elif taker_side == event.side_locked:
            tallies[PROVIDING].add(Decimal(contracts), price)
        else:
            unclassified += 1
    return len(offered) - len(screened.kept), unclassified


def sweep_convergence(
    scope: RunScope,
    artifacts: Path,
    *,
    observations: Mapping[tuple[str, date], list[StationObservation]],
    arrivals: Mapping[str, list[StationObservation]],
    settles: Mapping[tuple[str, date], Decimal],
) -> Sweep:
    scan = scan_locks(scope, artifacts, observations)
    grouped: dict[tuple[str, date], list[LockEvent]] = {}
    for event, day in scan.clean:
        grouped.setdefault((day.series, day.event_date), []).append(event)

    values: dict[str, dict[str, list[Decimal]]] = {split: {} for split in SPLITS}
    station_days = dict.fromkeys(SPLITS, 0)
    tallies = {bucket: FillTally() for bucket in BUCKETS}
    arrival = ArrivalTally()
    screened = ScreenTally()
    kept: list[Convergence] = []
    one_sided_rows = 0
    no_book_at_lock = 0
    censored = 0
    unsettled = 0
    contradicts = 0
    dropped_fills = 0
    unclassified = 0

    for (series_root, event_date), events in sorted(grouped.items()):
        day = scope.event_days[(series_root, event_date)]
        split = split_of(scope, series_root, event_date)
        station_days[split] += 1
        settle = settles.get((day.station, event_date))
        if settle is None:
            unsettled += 1
            continue

        book = read_window(artifacts, TOUCH, series_root, day.window_start, day.window_end)
        prints = read_window(artifacts, TRADES, series_root, day.window_start, day.window_end)
        offered: list[EvidenceWindow] = []
        measured: list[Convergence] = []
        for event in events:
            contradicts += int(not clears_strike(settle, event))
            arrival_anchor(arrivals.get(day.station, ()), day, event, arrival)
            dropped, loose = post_lock_fills(prints, scope, day, event, tallies)
            dropped_fills += dropped
            unclassified += loose

            stamps, mids, one_sided = book_states(book, event.ticker)
            one_sided_rows += one_sided
            no_book_at_lock += int(bisect_right(stamps, event.t0) == 0)
            t_conv = converged(
                stamps,
                mids,
                side_locked=event.side_locked,
                t_lock=event.t0,
                window_end=day.window_end,
            )
            t_end = day.window_end if t_conv is None else t_conv + _PERSIST
            reached = day.window_end if t_conv is None else t_conv
            measured.append(
                Convergence(
                    ticker=event.ticker,
                    series=series_root,
                    event_date=event_date,
                    station=day.station,
                    side_locked=event.side_locked,
                    t_lock=event.t0,
                    t_end=t_end,
                    half_life_s=seconds(reached - event.t0),
                    censored=t_conv is None,
                )
            )
            offered.append(
                EvidenceWindow(series=series_root, event_date=event_date, start=event.t0, end=t_end)
            )

        passed = screen_windows(scope, offered)
        screened.add(passed)
        cluster = station_day_key(day.station, event_date)
        for item, keep in zip(measured, keep_mask(offered, passed.kept), strict=True):
            if not keep:
                continue
            kept.append(item)
            censored += int(item.censored)
            values[split].setdefault(cluster, []).append(item.half_life_s)
        # The next event-day is read at the top of the loop and the recorder host cannot hold two
        # days of a busy root at once.
        del book, prints

    logger.info(
        "lock_convergence kept=%d censored=%d station_days=%d unsettled=%d one_sided=%d",
        len(kept),
        censored,
        sum(station_days.values()),
        unsettled,
        one_sided_rows,
    )
    return Sweep(
        scan=scan,
        kept=tuple(kept),
        values=values,
        clean_station_days=station_days,
        ceiling=len(scope.universe.lock_dependent) * len(scope.discovery_days),
        screened=screened,
        one_sided_rows=one_sided_rows,
        no_book_at_lock=no_book_at_lock,
        censored=censored,
        unsettled_station_days=unsettled,
        settle_contradicts=contradicts,
        usable_event_days=sum(
            1
            for (_, event_date), day in scope.event_days.items()
            if (day.station, event_date) in settles
        ),
        last_settled_event_day=max((event_date for _, event_date in settles), default=None),
        arrivals=arrival,
        fills=tallies,
        fills_dropped=dropped_fills,
        fills_unclassified=unclassified,
    )


def bootstrap_of(clusters: Sequence[ValueCluster], seed: int) -> BootstrapResult:
    return cluster_median_bootstrap(
        clusters,
        null_value=HALF_LIFE_THRESHOLD_S,
        direction=DIRECTION,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=seed,
        ci_level=CI_LEVEL,
    )


def readout(
    values: Mapping[str, Sequence[Decimal]], *, split: str, seed: int, powered: bool
) -> SplitReadout:
    clusters = tuple(
        ValueCluster(cluster=name, values=tuple(values[name])) for name in sorted(values)
    )
    return SplitReadout(
        split=split,
        clusters=clusters,
        events=sum(len(item.values) for item in clusters),
        bootstrap=bootstrap_of(clusters, seed) if powered and clusters else None,
    )


def decide(discovery: SplitReadout, holdout: SplitReadout) -> Decision:
    if discovery.n_station_days < STATION_DAY_MIN:
        return Decision(gate=None, replication=None, skipped=NOT_POWERED, verdict=UNDERPOWERED)

    gate = evaluate_gate(
        estimate=discovery.bootstrap.estimate,
        p_value=discovery.bootstrap.p_value,
        result=discovery.bootstrap,
        threshold=HALF_LIFE_THRESHOLD_S,
        direction=DIRECTION,
        alpha=ALPHA,
        n_min=STATION_DAY_MIN,
        n_unit=STATION_DAYS,
        undecidable=discovery.bootstrap.degenerate,
    )
    replication = None
    if holdout.bootstrap is None:
        skipped = NO_ESTIMATE
    elif gate.estimate == 0:
        skipped = ZERO_ESTIMATE
    else:
        skipped = ""
        replication = evaluate_holdout(
            discovery_estimate=gate.estimate,
            holdout_estimate=holdout.bootstrap.estimate,
            holdout_p_value=holdout.bootstrap.p_value,
            holdout_result=holdout.bootstrap,
            discovery_n_min=STATION_DAY_MIN,
            alpha=HOLDOUT_ALPHA,
            n_unit=STATION_DAYS,
            undecidable=holdout.bootstrap.degenerate,
        )

    # A half-life the threshold already rejects is closed on its economics whatever the resamples
    # did, so degeneracy is only read once the estimate has cleared it.
    if not gate.economic:
        verdict = CLOSED
    elif replication is not None and (gate.undecidable or replication.undecidable):
        verdict = UNDECIDABLE
    elif replication is not None and gate.passed and replication.replicated:
        verdict = PASS
    else:
        verdict = CLOSED
    return Decision(gate=gate, replication=replication, skipped=skipped, verdict=verdict)


def execute(
    *,
    run_id: str,
    preregistration: Path,
    repo: Path,
    run_scope: Path,
    artifacts: Path,
    observations: Mapping[tuple[str, date], list[StationObservation]],
    arrivals: Mapping[str, list[StationObservation]],
    settles: Mapping[tuple[str, date], Decimal],
    rtt_samples: Path,
    floor_source: FloorSource,
    economic_bar_size: Decimal,
    economic_bar_price: Decimal,
    economic_bar_price_source: str,
    seed: int,
    run_root: Path,
) -> LockConvergenceRun:
    inputs = assemble_run_inputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        run_scope=run_scope,
        artifacts=artifacts,
        rtt_samples=rtt_samples,
        floor_source=floor_source,
        economic_bar_size=economic_bar_size,
        economic_bar_price=economic_bar_price,
        economic_bar_price_source=economic_bar_price_source,
        bootstrap_seed=seed,
    )
    digest = write_manifest(run_root, inputs)

    scope = load_run_scope(run_scope)
    swept = sweep_convergence(
        scope, artifacts, observations=observations, arrivals=arrivals, settles=settles
    )
    powered = len(swept.values[DISCOVERY]) >= STATION_DAY_MIN
    discovery = readout(swept.values[DISCOVERY], split=DISCOVERY, seed=seed, powered=powered)
    holdout = readout(swept.values[HOLDOUT], split=HOLDOUT, seed=seed, powered=powered)
    run = LockConvergenceRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        seed=seed,
        sweep=swept,
        discovery=discovery,
        holdout=holdout,
        decision=decide(discovery, holdout),
    )
    logger.info(
        "lock_convergence verdict=%s discovery_median_s=%s discovery_p=%s n=%d",
        run.decision.verdict,
        None if discovery.bootstrap is None else discovery.bootstrap.estimate,
        None if discovery.bootstrap is None else discovery.bootstrap.p_value,
        discovery.n_station_days,
    )
    return run


def station_day_key(station: str, event_date: date) -> str:
    return f"{station} {event_date.isoformat()}"


def seconds(span: timedelta) -> Decimal:
    return Decimal(span // _MICROSECOND) / _MICROS_PER_S


def summarise(values: Sequence[Decimal]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": str(ordered[0]),
        "median": str(median(ordered)),
        "max": str(ordered[-1]),
    }


def result_payload(run: LockConvergenceRun) -> dict:
    sweep = run.sweep
    screened = sweep.screened
    fraction = screened.excluded_fraction
    gate_ran = run.decision.gate is not None
    station_days = sum(sweep.clean_station_days.values())
    pooled = [value for split in SPLITS for value in _split_values(sweep, split)]
    return {
        "run_id": run.run_id,
        "verdict": run.decision.verdict,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "bootstrap_seed": run.seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "lock_band": str(LOCK_BAND),
        "persist_s": PERSIST_S,
        "half_life_threshold_s": str(HALF_LIFE_THRESHOLD_S),
        "station_day_min": STATION_DAY_MIN,
        "censoring": CENSORING,
        "length_bias": LENGTH_BIAS,
        "discovery": _split_payload(run.discovery),
        "holdout": _split_payload(run.holdout),
        "gate": _gate_payload(run.decision.gate),
        "replication": _replication_payload(run.decision.replication),
        "replication_skipped": run.decision.skipped,
        "locks": {
            "cities": list(sweep.scan.cities),
            "markets": sweep.scan.markets,
            "clean_events": len(sweep.scan.clean),
            "clean_station_days": station_days,
            "clean_station_days_discovery": sweep.clean_station_days[DISCOVERY],
            "clean_station_days_holdout": sweep.clean_station_days[HOLDOUT],
            "lock_rate": (
                None
                if station_days == 0
                else str(Decimal(len(sweep.scan.clean)) / Decimal(station_days))
            ),
            "ambiguous": sweep.scan.ambiguous,
            "no_lock": sweep.scan.no_lock,
            "no_observations": sweep.scan.no_observations,
            "population_ceiling": sweep.ceiling,
            "realised_discovery": sweep.clean_station_days[DISCOVERY],
        },
        "half_life": {
            "gate_ran": gate_ran,
            "note": None if gate_ran else DESCRIPTIVE,
            "censored": sweep.censored,
            "pooled": summarise(pooled),
            "discovery": summarise(_split_values(sweep, DISCOVERY)),
            "holdout": summarise(_split_values(sweep, HOLDOUT)),
        },
        "settles": {
            "last_settled_event_day": (
                None
                if sweep.last_settled_event_day is None
                else sweep.last_settled_event_day.isoformat()
            ),
            "usable_event_days": sweep.usable_event_days,
            "unsettled_station_days": sweep.unsettled_station_days,
            "settle_contradicts": sweep.settle_contradicts,
        },
        "arrival_anchor": {
            "gating": False,
            "no_arrival_rows": sweep.arrivals.no_arrival_rows,
            "never_clears": sweep.arrivals.never_clears,
            "arrival_precedes_lock": sweep.arrivals.arrival_precedes_lock,
            **summarise(sweep.arrivals.deltas),
        },
        "post_lock_fills": {
            "gating": False,
            "dropped": sweep.fills_dropped,
            "unclassified_taker_side": sweep.fills_unclassified,
            TAKING: _fill_payload(sweep.fills[TAKING]),
            PROVIDING: _fill_payload(sweep.fills[PROVIDING]),
        },
        "exclusions": {
            "candidates": screened.candidates,
            "excluded": screened.excluded,
            "out_of_window": screened.out_of_window,
            "out_of_scope": screened.out_of_scope,
            "excluded_fraction": None if fraction is None else str(fraction),
            "by_class": dict(sorted(screened.by_class.items())),
        },
        "book": {
            "one_sided_rows": sweep.one_sided_rows,
            "no_book_at_lock": sweep.no_book_at_lock,
        },
    }


def _split_values(sweep: Sweep, split: str) -> list[Decimal]:
    return [value for values in sweep.values[split].values() for value in values]


def _split_payload(item: SplitReadout) -> dict:
    bootstrap = item.bootstrap
    return {
        "split": item.split,
        "median_half_life_s": None if bootstrap is None else str(bootstrap.estimate),
        "ci_low": None if bootstrap is None else bootstrap.ci_low,
        "ci_high": None if bootstrap is None else bootstrap.ci_high,
        "ci_level": None if bootstrap is None else bootstrap.ci_level,
        "p_value": None if bootstrap is None else bootstrap.p_value,
        "n_station_days": item.n_station_days,
        "n_unit": STATION_DAYS,
        "events": item.events,
    }


def _fill_payload(tally: FillTally) -> dict:
    return {
        "fills": tally.fills,
        "contracts": str(tally.contracts),
        "notional": str(tally.notional),
        "fees": str(tally.fees),
        "invalidated_price": summarise(tally.prices)
        | {"by_cent": dict(sorted(tally.by_cent.items()))},
    }


def _gate_payload(gate: GateVerdict | None) -> dict | None:
    if gate is None:
        return None
    return {
        "estimate": str(gate.estimate),
        "threshold": str(gate.threshold),
        "direction": gate.direction,
        "p_value": gate.p_value,
        "alpha": gate.alpha,
        "n": gate.n,
        "n_min": gate.n_min,
        "n_unit": gate.n_unit,
        "economic": gate.economic,
        "significant": gate.significant,
        "powered": gate.powered,
        "undecidable": gate.undecidable,
        "passed": gate.passed,
    }


def _replication_payload(replication: HoldoutVerdict | None) -> dict | None:
    if replication is None:
        return None
    return {
        "discovery_estimate": str(replication.discovery_estimate),
        "holdout_estimate": str(replication.holdout_estimate),
        "holdout_p_value": replication.holdout_p_value,
        "alpha": replication.alpha,
        "holdout_n": replication.holdout_n,
        "holdout_n_min": replication.holdout_n_min,
        "n_unit": replication.n_unit,
        "same_sign": replication.same_sign,
        "magnitude": replication.magnitude,
        "significant": replication.significant,
        "powered": replication.powered,
        "undecidable": replication.undecidable,
        "replicated": replication.replicated,
    }
