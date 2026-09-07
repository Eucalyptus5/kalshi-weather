import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from bot.lag.fee_floor import MAKER_RATE_SOURCE, PUBLISHED_MAKER_RATE
from bot.lag.placement_grid import CloseSidecar, read_sidecar
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import (
    BOOTSTRAP_RESAMPLES,
    MANIFEST_NAME,
    Exemption,
    SettlementRecord,
    write_manifest,
)
from bot.lag.settlement_delta import DeltaPartition, delta_partition, supported_reading
from bot.lag.settlement_entry import CROSSING, EntryCounts, StraddleEntry, entry_counts, entry_of
from bot.lag.settlement_price import (
    SIZE,
    EntryScreen,
    PriceCounts,
    PricedStraddle,
    entry_window,
    ladder_census,
    price_counts,
    price_of,
    screen_entry_minutes,
)
from bot.lag.settlement_source import (
    BoundarySplit,
    SettlementProvenance,
    boundary_split,
    check_settlement_scope,
    distinct_notice_bodies,
    read_settlement_sources,
    source_root_counts,
)
from bot.lag.settlement_straddle import Straddle, straddle_of
from bot.lag.tape_stats import (
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
    LADDER,
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    RunScope,
    assemble_run_inputs,
    keep_mask,
    load_run_scope,
    read_window,
    split_of,
)
from bot.markets.observation_window import observation_window
from bot.observations.metar import StationObservation
from bot.replay.analysis_stations import HIGH, in_cohort
from bot.replay.run_scope import DISCOVERY, HOLDOUT


logger = logging.getLogger(__name__)

RESULTS_NAME = "results.json"
CI_LEVEL = 0.95
NULL_VALUE = Decimal("0")
DIRECTION = "greater"
CITY_EVENT_DAYS = "city-event-days"
DISCOVERY_N_MIN = 30
COHORT = HIGH
BOOTSTRAP_SEED = 20260820

# 0.05 Bonferroni-corrected at the two questions this pre-registration fixed. The divisor is that
# frozen family count and is never recounted. tape_stats.ALPHA is the closed campaign's 0.05/4 and
# is not this one's.
ALPHA_F2: float = 0.05 / 2
# The statistic charges every entry its own taker fee and its tick, so a figure above the bar has
# already paid its costs and only a strictly positive one is an edge.
STRICT = True

PASS = "PASS"
CLOSED = "CLOSED"
UNDERPOWERED = "UNDERPOWERED"
ZERO_ESTIMATE = "a discovery estimate of exactly zero fixes no direction to replicate"
NO_ESTIMATE = "a split with no priced straddle carries no estimate to replicate"
NO_GATE_ESTIMATE = "a split with no priced straddle carries no estimate to test against the bar"
PRE_BOUNDARY_REPORTED_ONLY = (
    "the holdout restricted to its days before the settlement source moved is reported only: it "
    "gates nothing, carries no multiplicity correction, and a figure here is not evidence of an "
    "edge"
)

EXEMPTIONS = (
    Exemption(
        field="r0_fraction_invalid_max",
        reason="the observation side is our own decoded readings, which the ladder validity "
        "screen does not cover",
    ),
    Exemption(
        field="latency_floor",
        reason="no order is placed, so no read-to-act round trip enters the statistic",
    ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Sweep:
    straddles: tuple[Straddle, ...]
    entries: tuple[StraddleEntry, ...]
    deltas: DeltaPartition
    city_event_days: int
    no_readings: int
    unsettled: int
    no_straddle: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Priced:
    rows: tuple[tuple[Straddle, StraddleEntry, PricedStraddle], ...]
    screen: EntryScreen
    counts: EntryCounts


@dataclass(frozen=True, slots=True, kw_only=True)
class Readout:
    split: str
    clusters: tuple[ClusterAggregate, ...]
    bootstrap: BootstrapResult | None
    priced_n: int
    counts: PriceCounts


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    gate: GateVerdict | None
    replication: HoldoutVerdict | None
    skipped: str
    verdict: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SettlementRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    seed: int
    cohort: str | None
    sweep: Sweep
    screen: EntryScreen
    counts: EntryCounts
    open_at_first_reading_n: int
    deltas: DeltaPartition
    provenance: SettlementProvenance
    boundary: BoundarySplit
    census: Mapping[tuple[str, date], int]
    row_counts: Mapping[str, int]
    discovery: Readout
    holdout: Readout
    holdout_pre_boundary: Readout
    decision: Decision


def event_dates_of(scope: RunScope, roots: Sequence[str]) -> tuple[date, ...]:
    named = set(roots)
    return tuple(sorted({event_date for series, event_date in scope.event_days if series in named}))


def sweep_settlements(
    scope: RunScope,
    closes: Path,
    observations: Mapping[str, Sequence[StationObservation]],
    settles: Mapping[tuple[str, date], Decimal],
    roots: Sequence[str],
) -> Sweep:
    straddles: list[Straddle] = []
    entries: list[StraddleEntry] = []
    rows: list[tuple[Straddle, Sequence[StationObservation]]] = []
    city_event_days = 0
    no_readings = 0
    unsettled = 0
    no_straddle = 0

    for series_root in roots:
        sidecar = read_sidecar(closes / f"{series_root}.json")
        found = len(straddles)
        days = sorted(
            event_date for series, event_date in scope.event_days if series == series_root
        )
        for event_date in days:
            day = scope.event_days[(series_root, event_date)]
            city_event_days += 1
            start, end = observation_window(day.timezone, event_date)
            window = [
                item for item in observations.get(day.station, ()) if start <= item.valid_time < end
            ]
            if not window:
                no_readings += 1
                continue
            acis_f = settles.get((day.station, event_date))
            if acis_f is None:
                unsettled += 1
                continue
            straddle = straddle_of(
                sidecar,
                root=series_root,
                station=day.station,
                event_date=event_date,
                timezone=day.timezone,
                observed_f=max(item.temp_f for item in window),
                acis_f=acis_f,
            )
            if straddle is None:
                no_straddle += 1
                continue
            straddles.append(straddle)
            entries.append(entry_of(straddle, window, sidecar))
            rows.append((straddle, window))
        logger.info(
            "settlement root=%s days=%d straddles=%d",
            series_root,
            len(days),
            len(straddles) - found,
        )

    return Sweep(
        straddles=tuple(straddles),
        entries=tuple(entries),
        deltas=delta_partition(rows),
        city_event_days=city_event_days,
        no_readings=no_readings,
        unsettled=unsettled,
        no_straddle=no_straddle,
    )


# One price read per straddle, and the ladder window it is read from is shared by every straddle on
# that city event-day, so the partitions are opened once per day rather than once per straddle.
def price_entries(scope: RunScope, artifacts: Path, closes: Path, sweep: Sweep) -> Priced:
    crossing = [
        (straddle, entry)
        for straddle, entry in zip(sweep.straddles, sweep.entries, strict=True)
        if entry.instant_class == CROSSING
    ]
    windows = [entry_window(entry) for _, entry in crossing]
    screen = screen_entry_minutes(scope, windows)
    kept = [
        pair for pair, keep in zip(crossing, keep_mask(windows, screen.kept), strict=True) if keep
    ]

    grouped: dict[tuple[str, date], list[tuple[Straddle, StraddleEntry]]] = {}
    for straddle, entry in kept:
        grouped.setdefault((straddle.root, straddle.event_date), []).append((straddle, entry))

    sidecars: dict[str, CloseSidecar] = {}
    rows: list[tuple[Straddle, StraddleEntry, PricedStraddle]] = []
    for (series_root, event_date), pairs in sorted(grouped.items()):
        if series_root not in sidecars:
            sidecars[series_root] = read_sidecar(closes / f"{series_root}.json")
        day = scope.event_days[(series_root, event_date)]
        table = read_window(artifacts, LADDER, series_root, day.window_start, day.window_end)
        priced = [price_of(entry, table, sidecars[series_root]) for _, entry in pairs]
        # PricedStraddle carries no ticker and no event day, so the seam back to the record it
        # priced is the position it came back in.
        rows.extend(
            (straddle, entry, record)
            for (straddle, entry), record in zip(pairs, priced, strict=True)
        )

    return Priced(
        rows=tuple(rows),
        screen=screen,
        counts=entry_counts([entry for _, entry, _ in rows]),
    )


# The cluster unit is the city event-day. An unpriced straddle leaves both sides of the ratio: at
# weight zero it would still count towards the cluster the interval is drawn from.
def cluster_aggregates(
    rows: Sequence[tuple[Straddle, StraddleEntry, PricedStraddle]],
) -> list[ClusterAggregate]:
    totals: dict[str, Decimal] = {}
    weights: dict[str, Decimal] = {}
    for straddle, _, record in rows:
        if not record.priced:
            continue
        cluster = f"{straddle.root} {straddle.event_date.isoformat()}"
        totals[cluster] = totals.get(cluster, Decimal(0)) + record.net_profit_cents * record.size
        weights[cluster] = weights.get(cluster, Decimal(0)) + record.size
    return [
        ClusterAggregate(cluster=name, total=totals[name], weight=weights[name])
        for name in sorted(totals)
    ]


def bootstrap_of(clusters: Sequence[ClusterAggregate], seed: int) -> BootstrapResult:
    return cluster_bootstrap(
        clusters,
        null_value=NULL_VALUE,
        direction=DIRECTION,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=seed,
        ci_level=CI_LEVEL,
    )


def readout(scope: RunScope, priced: Priced, *, split: str, seed: int) -> Readout:
    rows = tuple(
        (straddle, entry, record)
        for straddle, entry, record in priced.rows
        if split_of(scope, straddle.root, straddle.event_date) == split
    )
    clusters = tuple(cluster_aggregates(rows))
    return Readout(
        split=split,
        clusters=clusters,
        bootstrap=bootstrap_of(clusters, seed) if clusters else None,
        priced_n=sum(1 for _, _, record in rows if record.priced),
        counts=price_counts([record for _, _, record in rows]),
    )


def decide(discovery: Readout, holdout: Readout) -> Decision:
    gate = (
        None
        if discovery.bootstrap is None
        else evaluate_gate(
            estimate=discovery.bootstrap.estimate,
            p_value=discovery.bootstrap.p_value,
            result=discovery.bootstrap,
            threshold=SELF_CHARGED_BAR,
            direction=DIRECTION,
            alpha=ALPHA_F2,
            n_min=DISCOVERY_N_MIN,
            n_unit=CITY_EVENT_DAYS,
            undecidable=discovery.bootstrap.degenerate,
            strict=STRICT,
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
            discovery_n_min=DISCOVERY_N_MIN,
            alpha=HOLDOUT_ALPHA,
            n_unit=CITY_EVENT_DAYS,
            undecidable=holdout.bootstrap.degenerate,
        )

    # A shortfall in city event-days keeps the question open for more tape, so it is read before
    # the economics; a spread-free resample leaves the gate insignificant and closes on that.
    if gate is None or not gate.powered or (replication is not None and not replication.powered):
        verdict = UNDERPOWERED
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
    closes: Path,
    settlement_sources: Path,
    observations: Mapping[str, Sequence[StationObservation]],
    settles: Mapping[tuple[str, date], Decimal],
    rtt_samples: Path,
    floor_source: FloorSource,
    economic_bar_size: Decimal,
    economic_bar_price: Decimal,
    economic_bar_price_source: str,
    seed: int,
    run_root: Path,
    cohort: str | None = COHORT,
) -> SettlementRun:
    # Both wires are pre-flight: a run whose settlement sidecar or whose ladder tree is paired with
    # another freeze leaves no manifest behind, so both are read and checked before anything is
    # written.
    scope = load_run_scope(run_scope)
    roots = in_cohort({series for series, _ in scope.event_days}, cohort)
    provenance = read_settlement_sources(settlement_sources)
    check_settlement_scope(provenance, roots)
    census = ladder_census(artifacts, scope, roots)
    boundary = boundary_split(provenance, event_dates_of(scope, roots))

    assembled = assemble_run_inputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        run_scope=run_scope,
        artifacts=artifacts,
        kinds=(LADDER,),
        rtt_samples=rtt_samples,
        floor_source=floor_source,
        maker_rate=PUBLISHED_MAKER_RATE,
        maker_rate_source=MAKER_RATE_SOURCE,
        economic_bar_size=economic_bar_size,
        economic_bar_price=economic_bar_price,
        economic_bar_price_source=economic_bar_price_source,
        bootstrap_seed=seed,
        cohort=cohort,
    )
    # The assembler always fills the universe and always derives the floor, and a field declared
    # exempt that the run supplied anyway is refused, so the two are dropped here.
    inputs = replace(
        assembled,
        universe=None,
        floor=None,
        exemptions=EXEMPTIONS,
        settlement=SettlementRecord(provenance=provenance, boundary=boundary),
    )
    digest = write_manifest(run_root, inputs)

    sweep = sweep_settlements(scope, closes, observations, settles, roots)
    priced = price_entries(scope, artifacts, closes, sweep)
    discovery = readout(scope, priced, split=DISCOVERY, seed=seed)
    holdout = readout(scope, priced, split=HOLDOUT, seed=seed)
    # Reported beside the holdout and never gated on: decide reads the full holdout below.
    before_boundary = replace(
        priced,
        rows=tuple(
            (straddle, entry, record)
            for straddle, entry, record in priced.rows
            if straddle.event_date < boundary.boundary_date
        ),
    )
    holdout_pre_boundary = readout(scope, before_boundary, split=HOLDOUT, seed=seed)
    run = SettlementRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        seed=seed,
        cohort=cohort,
        sweep=sweep,
        screen=priced.screen,
        counts=priced.counts,
        open_at_first_reading_n=entry_counts(sweep.entries).open_at_first_reading_n,
        deltas=sweep.deltas,
        provenance=provenance,
        boundary=boundary,
        census=census,
        row_counts=inputs.row_counts,
        discovery=discovery,
        holdout=holdout,
        holdout_pre_boundary=holdout_pre_boundary,
        decision=decide(discovery, holdout),
    )
    bootstrap = discovery.bootstrap
    logger.info(
        "settlement verdict=%s discovery_cents=%s discovery_p=%s city_event_days=%d",
        run.decision.verdict,
        None if bootstrap is None else bootstrap.estimate,
        None if bootstrap is None else bootstrap.p_value,
        len(discovery.clusters),
    )
    return run


def result_payload(run: SettlementRun) -> dict:
    return {
        "run_id": run.run_id,
        "verdict": run.decision.verdict,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "bootstrap_seed": run.seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "alpha": ALPHA_F2,
        "bar": str(SELF_CHARGED_BAR),
        "bar_source": SELF_CHARGED_BAR_SOURCE,
        "bar_is_strict": STRICT,
        "size": str(SIZE),
        "cohort": run.cohort,
        "city_event_day_min_discovery": DISCOVERY_N_MIN,
        # The settling row is fixed by a daily extreme that does not exist until the window has
        # closed, so nobody standing at the entry instant can name it.
        "identifiable_ex_ante": any(entry.identifiable_ex_ante for entry in run.sweep.entries),
        "straddles": {
            "city_event_days": run.sweep.city_event_days,
            "straddles": len(run.sweep.straddles),
            "no_readings": run.sweep.no_readings,
            "unsettled": run.sweep.unsettled,
            "no_straddle": run.sweep.no_straddle,
        },
        "entries": {
            "n": run.counts.n,
            "entry_at_close_n": run.counts.entry_at_close_n,
            "open_at_first_reading_n": run.open_at_first_reading_n,
            "at_or_before_close_n": run.counts.at_or_before_close_n,
            "close_minus_entry_s": dict(sorted(run.counts.close_minus_entry_s.items())),
        },
        "deltas": _delta_payload(run.deltas),
        "settlement_source": _settlement_payload(run.provenance, run.boundary),
        "screen": {
            "candidates": run.screen.candidates,
            "excluded": run.screen.excluded,
            "dropped": run.screen.dropped,
            "city_event_days": run.screen.city_event_days,
            "city_event_days_kept": run.screen.city_event_days_kept,
            "city_event_days_lost": str(run.screen.city_event_days_lost),
        },
        "discovery": _readout_payload(run.discovery),
        "holdout": _readout_payload(run.holdout),
        "holdout_pre_boundary": _readout_payload(run.holdout_pre_boundary)
        | {
            "gates_nothing": True,
            "reported_only": PRE_BOUNDARY_REPORTED_ONLY,
            "boundary_date": run.boundary.boundary_date.isoformat(),
        },
        "gate": _gate_payload(run.decision.gate),
        "replication": _replication_payload(run.decision.replication),
        "replication_skipped": run.decision.skipped,
        "row_counts": dict(run.row_counts),
        "cities": sorted({series for series, _ in run.census}),
        "ladder_rows_per_city_day": {
            f"{series} {event_date.isoformat()}": rows
            for (series, event_date), rows in sorted(run.census.items())
        },
    }


def _delta_payload(deltas: DeltaPartition) -> dict:
    return {
        "abs_delta_gt_1": deltas.abs_delta_gt_1,
        "abs_delta_le_1": deltas.abs_delta_le_1,
        "delta_histogram": {
            str(delta): count for delta, count in sorted(deltas.delta_histogram.items())
        },
        "supported_reading": supported_reading(deltas),
        "abs_delta_gt_1_rows": [
            f"{station} {event_date.isoformat()}"
            for station, event_date in deltas.abs_delta_gt_1_rows
        ],
        "coverage_minutes": {
            f"{station} {event_date.isoformat()}": minutes
            for (station, event_date), minutes in sorted(deltas.coverage_minutes.items())
        },
    }


def _settlement_payload(provenance: SettlementProvenance, boundary: BoundarySplit) -> dict:
    return {
        "observed_at": provenance.observed_at.isoformat(),
        "observation_source": provenance.observation_source,
        "sha256": provenance.sha256,
        "roots_per_source": dict(sorted(source_root_counts(provenance).items())),
        "distinct_notice_bodies": distinct_notice_bodies(provenance),
        "boundary_date": boundary.boundary_date.isoformat(),
        "boundary_source": boundary.boundary_source,
        "days_before_boundary": boundary.days_before_boundary,
        "days_on_or_after_boundary": boundary.days_on_or_after_boundary,
    }


def _readout_payload(item: Readout) -> dict:
    bootstrap = item.bootstrap
    return {
        "split": item.split,
        "net_profit_cents_per_contract": None if bootstrap is None else str(bootstrap.estimate),
        "ci_low": None if bootstrap is None else bootstrap.ci_low,
        "ci_high": None if bootstrap is None else bootstrap.ci_high,
        "ci_level": None if bootstrap is None else bootstrap.ci_level,
        "p_value": None if bootstrap is None else bootstrap.p_value,
        "replicate_spread": None if bootstrap is None else bootstrap.replicate_spread,
        "degenerate": None if bootstrap is None else bootstrap.degenerate,
        "city_event_days": len(item.clusters),
        "straddles": item.counts.n,
        "priced": item.priced_n,
        "one_sided": item.counts.one_sided_n,
        "no_row": item.counts.no_row_n,
        "censored": item.counts.censored_n,
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
