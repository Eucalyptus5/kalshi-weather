import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bot.lag.fee_floor import (
    MAKER_RATE_SOURCE as PUBLISHED_MAKER_RATE_SOURCE,
    PUBLISHED_MAKER_RATE,
)
from bot.lag.ladder_consistency import (
    CITY_DAY_MIN_DISCOVERY,
    DEPTH_MIN,
    EXCESS_BAR,
    PRICE_TICKS,
    SIZE_UNITS,
    STREAMS,
    Episode,
    Ladder,
    build_ladder,
    build_ladder_tape,
    ladder_episodes,
)
from bot.lag.read_rtt import FloorSource, LatencyFloor
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
    EvidenceWindow,
    RunScope,
    Screened,
    assemble_run_inputs,
    keep_mask,
    load_run_scope,
    partition_files,
    screen_windows,
    split_of,
    window_dates,
)
from bot.markets.parser import parse_ticker
from bot.replay.analysis_stations import in_cohort
from bot.replay.artifacts import TOUCH_SCHEMA
from bot.replay.run_scope import DISCOVERY, HOLDOUT


logger = logging.getLogger(__name__)

RESULTS_NAME = "results.json"
TOUCH_COLUMNS = (
    "id",
    "ticker",
    "received_at",
    "yes_bid",
    "yes_bid_depth",
    "yes_ask",
    "yes_ask_depth",
)
SPLITS = (DISCOVERY, HOLDOUT)
CI_LEVEL = 0.95
NULL_VALUE = Decimal("0")
ECONOMIC_BAR_SIZE: Decimal = Decimal("26")
ECONOMIC_BAR_PRICE: Decimal = Decimal("0.50")
ECONOMIC_BAR_PRICE_SOURCE = "preregistration"
MAKER_RATE: Decimal = PUBLISHED_MAKER_RATE
MAKER_RATE_SOURCE = PUBLISHED_MAKER_RATE_SOURCE
DIRECTION = "greater"
CITY_DAYS = "city event-days"
RECORDED = "recorded"
POOLED = "pooled"
EVERY = "all"
ADMITTED = "kept_tradeable"

PASS = "PASS"
CLOSED = "CLOSED"
UNDECIDABLE = "UNDECIDABLE"
ZERO_ESTIMATE = "a discovery estimate of exactly zero fixes no direction to replicate"
NO_ESTIMATE = "a split with no surviving tradeable episode carries no estimate to replicate"

TICKS_PER_CENT = PRICE_TICKS // 100
MICROS_PER_S = 1_000_000
METRICS: Mapping[str, tuple[str, int]] = {
    "magnitude_cents": ("magnitude_cents", TICKS_PER_CENT),
    "depth_contracts": ("depth", SIZE_UNITS),
    "duration_s": ("duration_s", MICROS_PER_S),
}
COUNTS = ("found", "tradeable", "kept", "censored", "incomplete_states")
QUANTILES = (("p25", 1, 4), ("median", 1, 2), ("p75", 3, 4), ("p90", 9, 10))
SUMMARY = ("min", *(name for name, _, _ in QUANTILES), "max")

_EMPTY = np.empty(0, dtype=np.int64)


@dataclass(slots=True)
class StreamTally:
    found: int = 0
    tradeable: int = 0
    kept: int = 0
    censored: int = 0
    incomplete_states: int = 0
    all_episodes: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {name: [] for name in METRICS}
    )
    kept_episodes: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {name: [] for name in METRICS}
    )

    def add(self, episodes: Sequence[Episode], admitted: Sequence[Episode]) -> None:
        self.found += len(episodes)
        self.tradeable += sum(1 for item in episodes if item.tradeable)
        self.kept += len(admitted)
        for name in METRICS:
            self.all_episodes[name].append(measure(episodes, name))
            self.kept_episodes[name].append(measure(admitted, name))


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
    tallies: Mapping[str, StreamTally]
    values: Mapping[str, Mapping[str, list[Decimal]]]
    admitted: tuple[Episode, ...]
    screened: ScreenTally
    rows: int
    in_scope: int
    incomplete: tuple[tuple[str, date], ...]
    tickers: Mapping[tuple[str, date], int]

    @property
    def complete(self) -> int:
        return self.in_scope - len(self.incomplete)


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitReadout:
    split: str
    clusters: tuple[ValueCluster, ...]
    episodes: int
    population: int
    bootstrap: BootstrapResult | None

    @property
    def n_city_days(self) -> int:
        return len(self.clusters)


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    gate: GateVerdict | None
    replication: HoldoutVerdict | None
    skipped: str
    verdict: str


@dataclass(frozen=True, slots=True, kw_only=True)
class LadderRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    seed: int
    floor: LatencyFloor
    t_persist_s: Decimal
    universe: tuple[str, ...]
    sweep: Sweep
    discovery: SplitReadout
    holdout: SplitReadout
    decision: Decision


def measure(episodes: Sequence[Episode], metric: str) -> np.ndarray:
    attribute, scale = METRICS[metric]
    return np.array([int(getattr(item, attribute) * scale) for item in episodes], dtype=np.int64)


# Nearest rank rather than an interpolated quantile, so every figure reported is a magnitude, a
# depth or a duration the tape actually carried.
def summarise(chunks: Sequence[np.ndarray], scale: int) -> dict:
    values = np.sort(np.concatenate(chunks)) if chunks else _EMPTY
    if values.size == 0:
        return {"count": 0} | dict.fromkeys(SUMMARY)
    ranked = {
        "min": values[0],
        **{
            name: values[_rank(numerator, denominator, values.size)]
            for name, numerator, denominator in QUANTILES
        },
        "max": values[-1],
    }
    return {"count": int(values.size)} | {
        name: str(Decimal(int(value)) / scale) for name, value in ranked.items()
    }


def _rank(numerator: int, denominator: int, size: int) -> int:
    return -(-numerator * size // denominator) - 1


def _checked(path: Path) -> pq.ParquetFile:
    stored = pq.ParquetFile(path)
    if not stored.schema_arrow.equals(TOUCH_SCHEMA):
        raise ValueError(f"{path} does not carry the frozen {TOUCH} schema")
    return stored


def census(artifacts: Path, scope: RunScope, series_root: str) -> dict[date, tuple[str, ...]]:
    dates = {
        day
        for (series, _), item in scope.event_days.items()
        if series == series_root
        for day in window_dates(item.window_start, item.window_end)
    }
    groups: dict[date, set[str]] = {}
    for path in partition_files(artifacts, TOUCH, series_root, dates):
        column = _checked(path).read(columns=["ticker"]).column("ticker")
        for ticker in column.unique().to_pylist():
            groups.setdefault(parse_ticker(ticker).event_date, set()).add(ticker)
    return {event_date: tuple(sorted(names)) for event_date, names in groups.items()}


# Each file is pruned to the ladder's own legs before it joins the buffer: a whole arrival date of
# a busy root does not fit beside the rest on a recorder host with under 2 GB free.
def read_legs(artifacts: Path, ladder: Ladder, dates: Sequence[date]) -> pa.Table:
    columns = list(TOUCH_COLUMNS)
    legs = pa.array(ladder.legs, type=pa.string())
    tables = []
    for path in partition_files(artifacts, TOUCH, ladder.series, dates):
        table = _checked(path).read(columns=columns)
        tables.append(table.filter(pc.is_in(table.column("ticker"), value_set=legs)))
    if not tables:
        return TOUCH_SCHEMA.empty_table().select(columns)
    return pa.concat_tables(tables)


def sweep_ladders(
    scope: RunScope, artifacts: Path, *, t_persist_s: Decimal, cohort: str | None = None
) -> Sweep:
    swept = set(in_cohort({series for series, _ in scope.event_days}, cohort))
    recorded = set(in_cohort(scope.universe.recorded, cohort))
    if swept != recorded:
        raise ValueError(
            "the frozen event days and the recorded universe disagree on: "
            + ", ".join(sorted(swept ^ recorded))
        )

    tallies = {stream: StreamTally() for stream in STREAMS}
    values: dict[str, dict[str, list[Decimal]]] = {split: {} for split in SPLITS}
    admitted: list[Episode] = []
    screened = ScreenTally()
    tickers: dict[tuple[str, date], int] = {}
    incomplete: list[tuple[str, date]] = []
    rows = 0

    for series_root in sorted(swept):
        started = time.monotonic()
        before = rows
        groups = census(artifacts, scope, series_root)
        days = sorted(
            event_date for series, event_date in scope.event_days if series == series_root
        )
        for event_date in days:
            names = groups.get(event_date, ())
            tickers[(series_root, event_date)] = len(names)
            ladder = build_ladder(series_root, event_date, names)
            if ladder is None:
                incomplete.append((series_root, event_date))
                continue
            day = scope.event_days[(series_root, event_date)]
            # Warm-up rows before the window open each leg's state and rows after it close an
            # episode with a true end stamp; screen_windows applies the event-day rule instead.
            table = read_legs(artifacts, ladder, window_dates(day.window_start, day.window_end))
            rows += table.num_rows
            result = ladder_episodes(build_ladder_tape(ladder, table), t_persist_s=t_persist_s)
            split = split_of(scope, series_root, event_date)
            cluster = _key(series_root, event_date)
            for stream in STREAMS:
                episodes = [item for item in result.episodes if item.stream == stream]
                tradeable = [item for item in episodes if item.tradeable]
                offered = [
                    EvidenceWindow(
                        series=series_root,
                        event_date=event_date,
                        start=item.start,
                        end=item.end,
                    )
                    for item in tradeable
                ]
                passed = screen_windows(scope, offered)
                screened.add(passed)
                kept = [
                    item
                    for item, keep in zip(tradeable, keep_mask(offered, passed.kept), strict=True)
                    if keep
                ]
                tallies[stream].add(episodes, kept)
                tallies[stream].censored += result.censored[stream]
                tallies[stream].incomplete_states += result.incomplete_states[stream]
                admitted.extend(kept)
                if kept:
                    values[split].setdefault(cluster, []).extend(item.excess_cents for item in kept)
        logger.info(
            "ladder_consistency root=%s days=%d incomplete=%d rows=%d elapsed_s=%.1f",
            series_root,
            len(days),
            sum(1 for series, _ in incomplete if series == series_root),
            rows - before,
            time.monotonic() - started,
        )

    return Sweep(
        tallies=tallies,
        values=values,
        admitted=tuple(admitted),
        screened=screened,
        rows=rows,
        in_scope=sum(1 for series, _ in scope.event_days if series in swept),
        incomplete=tuple(incomplete),
        tickers=tickers,
    )


def bootstrap_of(clusters: Sequence[ValueCluster], seed: int) -> BootstrapResult:
    return cluster_median_bootstrap(
        clusters,
        null_value=NULL_VALUE,
        direction=DIRECTION,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=seed,
        ci_level=CI_LEVEL,
    )


def readout(
    values: Mapping[str, Sequence[Decimal]], *, split: str, population: int, seed: int
) -> SplitReadout:
    clusters = tuple(
        ValueCluster(cluster=name, values=tuple(values[name])) for name in sorted(values)
    )
    return SplitReadout(
        split=split,
        clusters=clusters,
        episodes=sum(len(item.values) for item in clusters),
        population=population,
        bootstrap=bootstrap_of(clusters, seed) if clusters else None,
    )


# No underpowered verdict is reachable here. This question reserves that verdict for a run with no
# measured latency floor, and the manifest aborts before any statistic runs unless a floor is
# present; too few qualifying city event-days is rarity in the tape, which closes the question.
def decide(discovery: SplitReadout, holdout: SplitReadout) -> Decision:
    gate = (
        None
        if discovery.bootstrap is None
        else evaluate_gate(
            estimate=discovery.bootstrap.estimate,
            p_value=discovery.bootstrap.p_value,
            result=discovery.bootstrap,
            threshold=EXCESS_BAR,
            direction=DIRECTION,
            alpha=ALPHA,
            n_min=CITY_DAY_MIN_DISCOVERY,
            n_unit=CITY_DAYS,
            undecidable=discovery.bootstrap.degenerate,
        )
    )
    replication = None
    if gate is None or holdout.bootstrap is None:
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
            discovery_n_min=CITY_DAY_MIN_DISCOVERY,
            alpha=HOLDOUT_ALPHA,
            n_unit=CITY_DAYS,
            undecidable=holdout.bootstrap.degenerate,
        )

    # An edge under the bar is closed on its economics whatever the resamples did, so degeneracy is
    # only read once the estimate has cleared the bar.
    if gate is None or not gate.economic:
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
    rtt_samples: Path,
    floor_source: FloorSource,
    maker_rate: Decimal,
    maker_rate_source: str,
    economic_bar_size: Decimal,
    economic_bar_price: Decimal,
    economic_bar_price_source: str,
    seed: int,
    run_root: Path,
    cohort: str | None = None,
) -> LadderRun:
    inputs = assemble_run_inputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        run_scope=run_scope,
        artifacts=artifacts,
        rtt_samples=rtt_samples,
        floor_source=floor_source,
        maker_rate=maker_rate,
        maker_rate_source=maker_rate_source,
        economic_bar_size=economic_bar_size,
        economic_bar_price=economic_bar_price,
        economic_bar_price_source=economic_bar_price_source,
        bootstrap_seed=seed,
        cohort=cohort,
    )
    digest = write_manifest(run_root, inputs)

    scope = load_run_scope(run_scope)
    t_persist_s = Decimal(str(inputs.floor.t_persist_s))
    swept = sweep_ladders(scope, artifacts, t_persist_s=t_persist_s, cohort=cohort)

    cities = set(in_cohort({series for series, _ in scope.event_days}, cohort))
    population = dict.fromkeys(SPLITS, 0)
    for series, event_date in scope.event_days:
        if series in cities:
            population[split_of(scope, series, event_date)] += 1

    discovery = readout(
        swept.values[DISCOVERY], split=DISCOVERY, population=population[DISCOVERY], seed=seed
    )
    holdout = readout(
        swept.values[HOLDOUT], split=HOLDOUT, population=population[HOLDOUT], seed=seed
    )
    run = LadderRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        seed=seed,
        floor=inputs.floor,
        t_persist_s=t_persist_s,
        universe=in_cohort(scope.universe.recorded, cohort),
        sweep=swept,
        discovery=discovery,
        holdout=holdout,
        decision=decide(discovery, holdout),
    )
    logger.info(
        "ladder_consistency verdict=%s discovery_cents=%s discovery_p=%s n=%d",
        run.decision.verdict,
        None if discovery.bootstrap is None else discovery.bootstrap.estimate,
        None if discovery.bootstrap is None else discovery.bootstrap.p_value,
        discovery.n_city_days,
    )
    return run


def result_payload(run: LadderRun) -> dict:
    screened = run.sweep.screened
    fraction = screened.excluded_fraction
    return {
        "run_id": run.run_id,
        "verdict": run.decision.verdict,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "bootstrap_seed": run.seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "excess_bar": str(EXCESS_BAR),
        "depth_min": str(DEPTH_MIN),
        "t_persist_s": str(run.t_persist_s),
        "latency_floor_source": run.floor.source.value,
        "universe": {"read": RECORDED, "series": list(run.universe)},
        "discovery": _split_payload(run.discovery),
        "holdout": _split_payload(run.holdout),
        "gate": _gate_payload(run.decision.gate),
        "replication": _replication_payload(run.decision.replication),
        "replication_skipped": run.decision.skipped,
        "episodes": _episode_payload(run.sweep.tallies),
        "distributions": {
            EVERY: _distributions(
                {stream: run.sweep.tallies[stream].all_episodes for stream in STREAMS}
            ),
            ADMITTED: _distributions(
                {stream: run.sweep.tallies[stream].kept_episodes for stream in STREAMS}
            ),
        },
        "exclusions": {
            "candidates": screened.candidates,
            "excluded": screened.excluded,
            "out_of_window": screened.out_of_window,
            "out_of_scope": screened.out_of_scope,
            "excluded_fraction": None if fraction is None else str(fraction),
            "by_class": dict(sorted(screened.by_class.items())),
        },
        "ladders": {
            "in_scope": run.sweep.in_scope,
            "complete": run.sweep.complete,
            "incomplete": len(run.sweep.incomplete),
            "incomplete_keys": [_key(series, day) for series, day in run.sweep.incomplete],
        },
        "rows": run.sweep.rows,
        "cities": sorted({series for series, _ in run.sweep.tickers}),
        "tickers_per_city_day": {
            _key(series, event_date): count
            for (series, event_date), count in sorted(run.sweep.tickers.items())
        },
    }


def _key(series: str, event_date: date) -> str:
    return f"{series} {event_date.isoformat()}"


def _split_payload(item: SplitReadout) -> dict:
    bootstrap = item.bootstrap
    return {
        "split": item.split,
        "median_excess_cents": None if bootstrap is None else str(bootstrap.estimate),
        "ci_low": None if bootstrap is None else bootstrap.ci_low,
        "ci_high": None if bootstrap is None else bootstrap.ci_high,
        "ci_level": None if bootstrap is None else bootstrap.ci_level,
        "p_value": None if bootstrap is None else bootstrap.p_value,
        "n_city_days": item.n_city_days,
        "population": item.population,
        "episodes": item.episodes,
        "clusters": len(item.clusters),
    }


def _episode_payload(tallies: Mapping[str, StreamTally]) -> dict:
    return {
        POOLED: {name: sum(getattr(tallies[stream], name) for stream in STREAMS) for name in COUNTS}
    } | {stream: {name: getattr(tallies[stream], name) for name in COUNTS} for stream in STREAMS}


def _distributions(chunks: Mapping[str, Mapping[str, list[np.ndarray]]]) -> dict:
    return {
        POOLED: {
            name: summarise([part for stream in STREAMS for part in chunks[stream][name]], scale)
            for name, (_, scale) in METRICS.items()
        }
    } | {
        stream: {
            name: summarise(chunks[stream][name], scale) for name, (_, scale) in METRICS.items()
        }
        for stream in STREAMS
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
