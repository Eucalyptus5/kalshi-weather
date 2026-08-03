import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME, write_manifest
from bot.lag.taker_flow import (
    CENT_BAR,
    HORIZONS_S,
    PRIMARY_HORIZON_S,
    PRINT_MIN_DISCOVERY,
    FlowCounts,
    HorizonResult,
    HorizonWindows,
    PrintOutcome,
    TickerBook,
    build_ticker_book,
    build_ticker_prints,
    cluster_aggregates,
    print_outcomes,
    resolve_anchors,
    resolve_horizon,
    screen_prints,
)
from bot.lag.tape_stats import (
    ALPHA,
    HOLDOUT_ALPHA,
    BootstrapResult,
    ClusterAggregate,
    GateVerdict,
    HoldoutVerdict,
    cluster_bootstrap,
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
    partition_files,
    read_window,
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
TOUCH_COLUMNS = ("id", "ticker", "received_at", "ts_ms", "yes_bid", "no_bid")
SPLITS = (DISCOVERY, HOLDOUT)
CI_LEVEL = 0.95
NULL_VALUE = Decimal("0")
ECONOMIC_BAR_SIZE: Decimal = Decimal("26")
ECONOMIC_BAR_PRICE: Decimal = Decimal("0.50")
ECONOMIC_BAR_PRICE_SOURCE = "preregistration"
DIRECTION = "greater"
TICKERS = "tickers"

PASS = "PASS"
CLOSED = "CLOSED"
UNDERPOWERED = "UNDERPOWERED"
UNDECIDABLE = "UNDECIDABLE"
PRINT_MIN_HOLDOUT = (PRINT_MIN_DISCOVERY + 1) // 2
# The gate resamples tickers, so its power is the cluster count, not the prints inside them. Set at
# the city event-day floor the other two questions carry rather than at anything read off this tape.
TICKER_MIN_DISCOVERY = 30
ZERO_ESTIMATE = "a discovery estimate of exactly zero fixes no direction to replicate"
NO_ESTIMATE = "a split with no usable prints carries no estimate to replicate"

_DAY = timedelta(days=1)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(slots=True)
class Tally:
    totals: dict[str, Decimal] = field(default_factory=dict)
    weights: dict[str, Decimal] = field(default_factory=dict)
    n_prints: int = 0
    contracts: Decimal = Decimal(0)
    counts: FlowCounts = FlowCounts()
    candidates: int = 0
    excluded: int = 0
    out_of_window: int = 0
    by_class: dict[str, int] = field(default_factory=dict)

    def add(self, outcomes: Sequence[PrintOutcome]) -> None:
        for item in cluster_aggregates(outcomes):
            self.totals[item.cluster] = self.totals.get(item.cluster, Decimal(0)) + item.total
            self.weights[item.cluster] = self.weights.get(item.cluster, Decimal(0)) + item.weight
        self.n_prints += len(outcomes)
        self.contracts += sum((item.contracts for item in outcomes), Decimal(0))

    def screen(self, screened: Screened) -> None:
        self.candidates += screened.candidates
        self.excluded += screened.excluded
        self.out_of_window += screened.out_of_window
        for name, count in screened.by_class.items():
            self.by_class[name] = self.by_class.get(name, 0) + count

    def clusters(self) -> tuple[ClusterAggregate, ...]:
        return tuple(
            ClusterAggregate(cluster=name, total=self.totals[name], weight=self.weights[name])
            for name in sorted(self.totals)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Sweep:
    tallies: Mapping[tuple[str, int], Tally]
    empty_side: int
    duplicates: int
    out_of_scope: int
    in_scope: Mapping[str, int]
    read_ts_violations: int
    fractional_size_prints: int
    outside_lock_window: int
    no_lock_prints: int
    off_universe_prints: int
    tickers: Mapping[tuple[str, date], frozenset[str]]


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizonReadout:
    result: HorizonResult
    bootstrap: BootstrapResult | None
    candidates: int
    excluded: int
    out_of_window: int
    by_class: Mapping[str, int]

    @property
    def excluded_fraction(self) -> Decimal | None:
        if self.candidates == 0:
            return None
        return Decimal(self.excluded) / Decimal(self.candidates)


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    gate: GateVerdict | None
    replication: HoldoutVerdict | None
    skipped: str
    verdict: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TakerFlowRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    seed: int
    sweep: Sweep
    discovery: tuple[HorizonReadout, ...]
    holdout: HorizonReadout
    decision: Decision

    @property
    def primary(self) -> HorizonReadout:
        return next(item for item in self.discovery if item.result.horizon_s == PRIMARY_HORIZON_S)


def read_touch(artifacts: Path, series_root: str, day: date) -> pa.Table:
    columns = list(TOUCH_COLUMNS)
    tables = []
    for path in partition_files(artifacts, TOUCH, series_root, [day]):
        stored = pq.ParquetFile(path)
        if not stored.schema_arrow.equals(TOUCH_SCHEMA):
            raise ValueError(f"{path} does not carry the frozen {TOUCH} schema")
        tables.append(stored.read(columns=columns))
    if not tables:
        return TOUCH_SCHEMA.empty_table().select(columns)
    return pa.concat_tables(tables)


def sweep_prints(
    scope: RunScope,
    artifacts: Path,
    *,
    lock_windows: Mapping[str, tuple[datetime, datetime]] | None = None,
    cohort: str | None = None,
) -> Sweep:
    tallies = {(split, horizon_s): Tally() for split in SPLITS for horizon_s in HORIZONS_S}
    hygiene = FlowCounts()
    out_of_scope = 0
    in_scope = dict.fromkeys(SPLITS, 0)
    violations = 0
    fractional = 0
    outside_lock = 0
    no_lock = 0
    off_universe = 0
    tickers: dict[tuple[str, date], set[str]] = {}
    days = window_dates(scope.scope_start, scope.scope_end)
    lock_dependent = set(scope.universe.lock_dependent)

    for series_root in in_cohort({series for series, _ in scope.event_days}, cohort):
        started = time.monotonic()
        outside_universe = lock_windows is not None and series_root not in lock_dependent
        hygienic = screen_prints(
            read_window(artifacts, TRADES, series_root, scope.scope_start, scope.scope_end)
        )
        hygiene += hygienic.counts
        event_dates = {}
        for ticker in pc.unique(hygienic.kept.column("ticker")).to_pylist():
            event_date = parse_ticker(ticker).event_date
            if (series_root, event_date) in scope.event_days:
                event_dates[ticker] = event_date
                tickers.setdefault((series_root, event_date), set()).add(ticker)
        scoped = hygienic.kept.filter(
            pc.is_in(
                hygienic.kept.column("ticker"),
                value_set=pa.array(sorted(event_dates), type=pa.string()),
            )
        )
        out_of_scope += hygienic.kept.num_rows - scoped.num_rows
        logger.info(
            "taker_flow root=%s prints=%d tickers=%d",
            series_root,
            scoped.num_rows,
            len(event_dates),
        )

        current = TOUCH_SCHEMA.empty_table().select(list(TOUCH_COLUMNS))
        following = current
        following_day = None
        for day in days:
            opens = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            today = scoped.filter(
                pc.and_(
                    pc.greater_equal(scoped.column("received_at"), opens),
                    pc.less(scoped.column("received_at"), opens + _DAY),
                )
            )
            if today.num_rows == 0:
                continue
            current = following if following_day == day else read_touch(artifacts, series_root, day)
            following_day = day + _DAY
            following = read_touch(artifacts, series_root, following_day)
            for ticker in sorted(pc.unique(today.column("ticker")).to_pylist()):
                event_date = event_dates[ticker]
                split = split_of(scope, series_root, event_date)
                prints_here = today.filter(pc.equal(today.column("ticker"), ticker))
                if outside_universe:
                    off_universe += prints_here.num_rows
                    continue
                if lock_windows is not None:
                    window = lock_windows.get(ticker)
                    if window is None:
                        no_lock += prints_here.num_rows
                        continue
                    inside = prints_here.filter(
                        pc.and_(
                            pc.greater_equal(prints_here.column("received_at"), window[0]),
                            pc.less_equal(prints_here.column("received_at"), window[1]),
                        )
                    )
                    outside_lock += prints_here.num_rows - inside.num_rows
                    if inside.num_rows == 0:
                        continue
                    prints_here = inside
                in_scope[split] += prints_here.num_rows
                head = current.filter(pc.equal(current.column("ticker"), ticker))
                rows = pa.concat_tables(
                    [head, following.filter(pc.equal(following.column("ticker"), ticker))]
                )
                # The pass emits a touch row per book event, so a ticker whose trades land on a
                # date its book did not carries no anchor at all.
                if rows.num_rows == 0:
                    for horizon_s in HORIZONS_S:
                        tallies[(split, horizon_s)].counts += FlowCounts(
                            unresolved=prints_here.num_rows
                        )
                    continue
                book = build_ticker_book(ticker, rows)
                prints = build_ticker_prints(ticker, prints_here)
                anchors = resolve_anchors(book, prints)
                violations += head_violations(book, head.num_rows)
                fractional += sum(1 for size in prints.contracts if size != int(size))
                for horizon_s in HORIZONS_S:
                    windows = resolve_horizon(book, prints, anchors, horizon_s=horizon_s)
                    tally = tallies[(split, horizon_s)]
                    tally.counts += windows.counts
                    positions, offered = _offer(windows, series_root, event_date)
                    screened = screen_windows(scope, offered)
                    tally.screen(screened)
                    usable = np.zeros(windows.usable.size, dtype=bool)
                    usable[positions[keep_mask(offered, screened.kept)]] = True
                    tally.add(
                        print_outcomes(book, prints, anchors, replace(windows, usable=usable))
                    )
        logger.info(
            "taker_flow root=%s done elapsed_s=%.1f", series_root, time.monotonic() - started
        )
        # The next root's trades are read at the top of the loop, and the recorder host cannot
        # hold two roots' buffers at once.
        del hygienic, scoped, current, following

    return Sweep(
        tallies=tallies,
        empty_side=hygiene.empty_side,
        duplicates=hygiene.duplicates,
        out_of_scope=out_of_scope,
        in_scope=in_scope,
        read_ts_violations=violations,
        fractional_size_prints=fractional,
        outside_lock_window=outside_lock,
        no_lock_prints=no_lock,
        off_universe_prints=off_universe,
        tickers={key: frozenset(names) for key, names in tickers.items()},
    )


# The rolling buffer carries the next arrival date as well, so the whole book's count would read
# that date's stamps again once it becomes the current one.
def head_violations(book: TickerBook, rows: int) -> int:
    seam = int(np.searchsorted(book.delta_rows, rows)) + 1
    return int(np.count_nonzero(np.diff(book.delta_ts[:seam]) < 0))


def bootstrap_of(clusters: Sequence[ClusterAggregate], seed: int) -> BootstrapResult:
    return cluster_bootstrap(
        clusters,
        null_value=NULL_VALUE,
        direction=DIRECTION,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=seed,
        ci_level=CI_LEVEL,
    )


def readout(tally: Tally, *, split: str, horizon_s: int, seed: int) -> HorizonReadout:
    result = HorizonResult(
        horizon_s=horizon_s,
        split=split,
        clusters=tally.clusters(),
        n_prints=tally.n_prints,
        contracts=tally.contracts,
        counts=tally.counts,
    )
    return HorizonReadout(
        result=result,
        bootstrap=bootstrap_of(result.clusters, seed) if result.clusters else None,
        candidates=tally.candidates,
        excluded=tally.excluded,
        out_of_window=tally.out_of_window,
        by_class=dict(tally.by_class),
    )


def decide(discovery: HorizonReadout, holdout: HorizonReadout) -> Decision:
    gate = (
        None
        if discovery.bootstrap is None
        else evaluate_gate(
            estimate=discovery.bootstrap.estimate,
            p_value=discovery.bootstrap.p_value,
            result=discovery.bootstrap,
            threshold=CENT_BAR,
            direction=DIRECTION,
            alpha=ALPHA,
            n_min=TICKER_MIN_DISCOVERY,
            n_unit=TICKERS,
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
            discovery_n_min=TICKER_MIN_DISCOVERY,
            alpha=HOLDOUT_ALPHA,
            n_unit=TICKERS,
            undecidable=holdout.bootstrap.degenerate,
        )

    # A shortfall in either unit keeps the question open for more tape, and an edge under the bar
    # is closed on its economics whatever the resamples did, so degeneracy is read last of the three.
    if (
        gate is None
        or not gate.powered
        or discovery.result.n_prints < PRINT_MIN_DISCOVERY
        or holdout.result.n_prints < PRINT_MIN_HOLDOUT
        or (replication is not None and not replication.powered)
    ):
        verdict = UNDERPOWERED
    elif not gate.economic:
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
    economic_bar_size: Decimal,
    economic_bar_price: Decimal,
    economic_bar_price_source: str,
    seed: int,
    run_root: Path,
) -> TakerFlowRun:
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
    swept = sweep_prints(scope, artifacts)
    discovery = tuple(
        readout(
            swept.tallies[(DISCOVERY, horizon_s)],
            split=DISCOVERY,
            horizon_s=horizon_s,
            seed=seed,
        )
        for horizon_s in HORIZONS_S
    )
    holdout = readout(
        swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)],
        split=HOLDOUT,
        horizon_s=PRIMARY_HORIZON_S,
        seed=seed,
    )
    run = TakerFlowRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        seed=seed,
        sweep=swept,
        discovery=discovery,
        holdout=holdout,
        decision=decide(
            next(item for item in discovery if item.result.horizon_s == PRIMARY_HORIZON_S),
            holdout,
        ),
    )
    bootstrap = run.primary.bootstrap
    logger.info(
        "taker_flow verdict=%s discovery_cents=%s discovery_p=%s n=%d",
        run.decision.verdict,
        None if bootstrap is None else bootstrap.estimate,
        None if bootstrap is None else bootstrap.p_value,
        run.primary.result.n_prints,
    )
    return run


def result_payload(run: TakerFlowRun) -> dict:
    primary = run.primary
    holdout = run.holdout
    candidates = primary.candidates + holdout.candidates
    excluded = primary.excluded + holdout.excluded
    counts = primary.result.counts + holdout.result.counts
    in_scope = dict(run.sweep.in_scope)
    return {
        "run_id": run.run_id,
        "verdict": run.decision.verdict,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "bootstrap_seed": run.seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "primary_horizon_s": PRIMARY_HORIZON_S,
        "cent_bar": str(CENT_BAR),
        "prints": {
            "in_scope_pooled": sum(in_scope.values()),
            "in_scope_discovery": in_scope[DISCOVERY],
            "in_scope_holdout": in_scope[HOLDOUT],
            "screened_pooled": primary.result.n_prints + holdout.result.n_prints,
            "screened_discovery": primary.result.n_prints,
            "screened_holdout": holdout.result.n_prints,
            "empty_side": run.sweep.empty_side,
            "duplicate_trade_id": run.sweep.duplicates,
            "out_of_scope": run.sweep.out_of_scope,
            "out_of_window": primary.out_of_window + holdout.out_of_window,
        },
        "discovery": _readout_payload(primary),
        "holdout": _readout_payload(holdout),
        "gate": _gate_payload(run.decision.gate),
        "replication": _replication_payload(run.decision.replication),
        "replication_skipped": run.decision.skipped,
        "horizon_curve": [_readout_payload(item) for item in run.discovery],
        "exclusions": {
            "candidates": candidates,
            "excluded": excluded,
            "excluded_fraction": (
                None if candidates == 0 else str(Decimal(excluded) / Decimal(candidates))
            ),
            "by_class": _pool_classes(primary.by_class, holdout.by_class),
        },
        "kernel_drops": {
            "unresolved": counts.unresolved,
            "uncovered": counts.uncovered,
            "one_sided": counts.one_sided,
            "host_clock": counts.host_clock,
            "read_ts_violations": run.sweep.read_ts_violations,
            "fractional_size_prints": run.sweep.fractional_size_prints,
        },
        "cities": sorted({series for series, _ in run.sweep.tickers}),
        "tickers_per_city_day": {
            f"{series} {event_date.isoformat()}": len(names)
            for (series, event_date), names in sorted(run.sweep.tickers.items())
        },
    }


def _readout_payload(item: HorizonReadout) -> dict:
    bootstrap = item.bootstrap
    fraction = item.excluded_fraction
    return {
        "split": item.result.split,
        "horizon_s": item.result.horizon_s,
        "mean_net_cents": None if bootstrap is None else str(bootstrap.estimate),
        "ci_low": None if bootstrap is None else bootstrap.ci_low,
        "ci_high": None if bootstrap is None else bootstrap.ci_high,
        "ci_level": None if bootstrap is None else bootstrap.ci_level,
        "n_prints": item.result.n_prints,
        "contracts": str(item.result.contracts),
        "clusters": len(item.result.clusters),
        "p_value": None if bootstrap is None else bootstrap.p_value,
        "candidates": item.candidates,
        "excluded": item.excluded,
        "excluded_fraction": None if fraction is None else str(fraction),
        "out_of_window": item.out_of_window,
        "by_class": dict(sorted(item.by_class.items())),
        "unresolved": item.result.counts.unresolved,
        "uncovered": item.result.counts.uncovered,
        "one_sided": item.result.counts.one_sided,
        "host_clock": item.result.counts.host_clock,
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


def _pool_classes(discovery: Mapping[str, int], holdout: Mapping[str, int]) -> dict[str, int]:
    return {
        name: discovery.get(name, 0) + holdout.get(name, 0)
        for name in sorted(set(discovery) | set(holdout))
    }


def _offer(
    windows: HorizonWindows, series: str, event_date: date
) -> tuple[np.ndarray, list[EvidenceWindow]]:
    positions = np.flatnonzero(windows.usable)
    return positions, [
        EvidenceWindow(
            series=series,
            event_date=event_date,
            start=_stamp(windows.start_us[position]),
            end=_stamp(windows.end_us[position]),
        )
        for position in positions
    ]


def _stamp(value: np.int64) -> datetime:
    return _EPOCH + timedelta(microseconds=int(value))
