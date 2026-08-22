import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow.compute as pc

from bot.lag.fee_floor import (
    MAKER_RATE_SOURCE as PUBLISHED_MAKER_RATE_SOURCE,
    PUBLISHED_MAKER_RATE,
)
from bot.lag.fill_convention import NO, YES, sweep_market_day
from bot.lag.maker_edge import (
    HORIZONS_S,
    PRIMARY_HORIZON_S,
    FillEdge,
    HorizonEdges,
    quote_book,
    resolve_curve,
    resolve_edges,
    window_cap_s,
)
from bot.lag.placement_grid import read_sidecar
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME, write_manifest
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
    TRADES,
    RunScope,
    assemble_run_inputs,
    load_run_scope,
    read_window,
    split_of,
)
from bot.markets.parser import parse_ticker
from bot.replay.analysis_stations import HIGH, in_cohort
from bot.replay.run_scope import DISCOVERY, HOLDOUT


logger = logging.getLogger(__name__)

RESULTS_NAME = "results.json"
SPLITS = (DISCOVERY, HOLDOUT)
CI_LEVEL = 0.95
NULL_VALUE = Decimal("0")
DIRECTION = "greater"
MARKET_DAYS = "market-days"
MARKET_DAY_MIN_DISCOVERY = 200
COHORT = HIGH
MAKER_RATE: Decimal = Decimal("0")
MAKER_RATE_SOURCE = "series_api_fee_type_quadratic_2026-08-19"

# 0.05 Bonferroni-corrected at the three questions this tape's pre-registration fixed. Written as
# the division because 0.05/3 has no exact decimal literal; the divisor is the frozen family count
# and is never recounted. tape_stats.ALPHA is the closed campaign's 0.05/4 and is not this one's.
ALPHA: float = 0.05 / 3
# The statistic charges every fill its own maker fee, so a figure above the bar has already paid
# its costs and only a strictly positive one is an edge.
STRICT = True

PASS = "PASS"
CLOSED = "CLOSED"
UNDERPOWERED = "UNDERPOWERED"
ZERO_ESTIMATE = "a discovery estimate of exactly zero fixes no direction to replicate"
NO_ESTIMATE = "a split with no scored fill carries no estimate to replicate"
NO_GATE_ESTIMATE = "a split with no scored fill carries no estimate to test against the bar"


# The cluster unit is the market-day: one Kalshi market on one event-day is exactly one ticker.
# Weighting by contracts is what makes the ratio read in cents per contract rather than per fill.
def cluster_aggregates(edges: Sequence[FillEdge]) -> list[ClusterAggregate]:
    totals: dict[str, Decimal] = {}
    weights: dict[str, Decimal] = {}
    for edge in edges:
        signed = edge.edge_cents_per_contract * edge.contracts
        totals[edge.ticker] = totals.get(edge.ticker, Decimal(0)) + signed
        weights[edge.ticker] = weights.get(edge.ticker, Decimal(0)) + edge.contracts
    return [
        ClusterAggregate(cluster=ticker, total=totals[ticker], weight=weights[ticker])
        for ticker in sorted(totals)
    ]


def weighted_estimate(clusters: Sequence[ClusterAggregate]) -> Decimal | None:
    if not clusters:
        return None
    return sum((item.total for item in clusters), Decimal(0)) / sum(
        (item.weight for item in clusters), Decimal(0)
    )


@dataclass(slots=True)
class Tally:
    totals: dict[str, Decimal] = field(default_factory=dict)
    weights: dict[str, Decimal] = field(default_factory=dict)
    n_fills: int = 0
    contracts: Decimal = Decimal(0)
    modelled: int = 0
    dropped: int = 0
    candidates: int = 0
    excluded: int = 0
    out_of_window: int = 0
    out_of_scope: int = 0
    by_class: dict[str, int] = field(default_factory=dict)

    def add(self, edges: Sequence[FillEdge]) -> None:
        for item in cluster_aggregates(edges):
            self.totals[item.cluster] = self.totals.get(item.cluster, Decimal(0)) + item.total
            self.weights[item.cluster] = self.weights.get(item.cluster, Decimal(0)) + item.weight
        self.n_fills += len(edges)
        self.contracts += sum((edge.contracts for edge in edges), Decimal(0))

    def group(self, group: HorizonEdges) -> None:
        self.modelled += group.modelled
        self.dropped += group.dropped
        self.candidates += group.screened.candidates
        self.excluded += group.screened.excluded
        self.out_of_window += group.screened.out_of_window
        self.out_of_scope += group.screened.out_of_scope
        for name, count in group.screened.by_class.items():
            self.by_class[name] = self.by_class.get(name, 0) + count
        self.add(group.edges)

    def clusters(self) -> tuple[ClusterAggregate, ...]:
        return tuple(
            ClusterAggregate(cluster=name, total=self.totals[name], weight=self.weights[name])
            for name in sorted(self.totals)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Sweep:
    tallies: Mapping[tuple[str, int], Tally]
    published: Mapping[str, Tally]
    offered: int
    yes_fills: int
    no_fills: int
    yes_empty: int
    no_empty: int
    unnamed_markets: int
    market_days: Mapping[tuple[str, date], int]


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizonReadout:
    split: str
    horizon_s: int
    clusters: tuple[ClusterAggregate, ...]
    bootstrap: BootstrapResult | None
    n_fills: int
    contracts: Decimal
    modelled: int
    dropped: int
    candidates: int
    excluded: int
    out_of_window: int
    out_of_scope: int
    by_class: Mapping[str, int]

    @property
    def excluded_fraction(self) -> Decimal | None:
        if self.candidates == 0:
            return None
        return Decimal(self.excluded) / Decimal(self.candidates)

    @property
    def no_mid_fraction(self) -> Decimal | None:
        if self.modelled == 0:
            return None
        return Decimal(self.dropped) / Decimal(self.modelled)


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    gate: GateVerdict | None
    replication: HoldoutVerdict | None
    skipped: str
    verdict: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MakerEdgeRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    seed: int
    maker_rate: Decimal
    sweep: Sweep
    discovery: tuple[HorizonReadout, ...]
    holdout: HorizonReadout
    decision: Decision

    @property
    def primary(self) -> HorizonReadout:
        return next(item for item in self.discovery if item.horizon_s == PRIMARY_HORIZON_S)


def sweep_fills(
    scope: RunScope,
    artifacts: Path,
    closes: Path,
    *,
    maker_rate: Decimal,
    cohort: str | None = None,
) -> Sweep:
    tallies = {(split, horizon_s): Tally() for split in SPLITS for horizon_s in HORIZONS_S}
    published = {split: Tally() for split in SPLITS}
    market_days: dict[tuple[str, date], int] = {}
    offered = 0
    filled = {YES: 0, NO: 0}
    empty = {YES: 0, NO: 0}
    unnamed = 0

    for series_root in in_cohort({series for series, _ in scope.event_days}, cohort):
        started = time.monotonic()
        sidecar = read_sidecar(closes / f"{series_root}.json")
        days = sorted(
            event_date for series, event_date in scope.event_days if series == series_root
        )
        for event_date in days:
            day = scope.event_days[(series_root, event_date)]
            ladder = read_window(artifacts, LADDER, series_root, day.window_start, day.window_end)
            prints = read_window(artifacts, TRADES, series_root, day.window_start, day.window_end)
            split = split_of(scope, series_root, event_date)
            names = [
                ticker
                for ticker in sorted(pc.unique(ladder.column("ticker")).to_pylist())
                if parse_ticker(ticker).event_date == event_date
            ]
            market_days[(series_root, event_date)] = len(names)
            for ticker in names:
                # A market the settle pull never named has no close, so it has no placement grid
                # and the convention has nothing to rest against.
                if ticker not in sidecar.markets:
                    unnamed += 1
                    continue
                swept = sweep_market_day(
                    ladder=ladder, prints=prints, sidecar=sidecar, ticker=ticker
                )
                offered += swept.offered
                filled[YES] += swept.yes_fills
                filled[NO] += swept.no_fills
                empty[YES] += swept.yes_empty
                empty[NO] += swept.no_empty
                book = quote_book(ladder, ticker)
                curve = resolve_curve(
                    book=book,
                    fills=swept.fills,
                    scope=scope,
                    series=series_root,
                    event_date=event_date,
                    maker_rate=maker_rate,
                )
                for horizon_s, group in curve.by_horizon.items():
                    tallies[(split, horizon_s)].group(group)
                published[split].group(
                    resolve_edges(
                        book=book,
                        fills=swept.fills,
                        scope=scope,
                        series=series_root,
                        event_date=event_date,
                        maker_rate=PUBLISHED_MAKER_RATE,
                        horizon_s=PRIMARY_HORIZON_S,
                    )
                )
        logger.info(
            "maker_edge root=%s days=%d markets=%d elapsed_s=%.1f",
            series_root,
            len(days),
            sum(count for (series, _), count in market_days.items() if series == series_root),
            time.monotonic() - started,
        )

    return Sweep(
        tallies=tallies,
        published=published,
        offered=offered,
        yes_fills=filled[YES],
        no_fills=filled[NO],
        yes_empty=empty[YES],
        no_empty=empty[NO],
        unnamed_markets=unnamed,
        market_days=market_days,
    )


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
    clusters = tally.clusters()
    return HorizonReadout(
        split=split,
        horizon_s=horizon_s,
        clusters=clusters,
        bootstrap=bootstrap_of(clusters, seed) if clusters else None,
        n_fills=tally.n_fills,
        contracts=tally.contracts,
        modelled=tally.modelled,
        dropped=tally.dropped,
        candidates=tally.candidates,
        excluded=tally.excluded,
        out_of_window=tally.out_of_window,
        out_of_scope=tally.out_of_scope,
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
            threshold=SELF_CHARGED_BAR,
            direction=DIRECTION,
            alpha=ALPHA,
            n_min=MARKET_DAY_MIN_DISCOVERY,
            n_unit=MARKET_DAYS,
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
            discovery_n_min=MARKET_DAY_MIN_DISCOVERY,
            alpha=HOLDOUT_ALPHA,
            n_unit=MARKET_DAYS,
            undecidable=holdout.bootstrap.degenerate,
        )

    # A shortfall in market-days keeps the question open for more tape, so it is read before the
    # economics; a degenerate resample leaves the gate insignificant and closes on that.
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
) -> MakerEdgeRun:
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
    swept = sweep_fills(scope, artifacts, closes, maker_rate=maker_rate, cohort=cohort)
    # One seed across the four horizons is the pre-registered draw: the horizons carrying equal
    # cluster counts then resample identical positions, so the curve's differences are the horizon
    # rather than the resampling, and only the 60-second group carries a verdict.
    discovery = tuple(
        readout(
            swept.tallies[(DISCOVERY, horizon_s)], split=DISCOVERY, horizon_s=horizon_s, seed=seed
        )
        for horizon_s in HORIZONS_S
    )
    holdout = readout(
        swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)],
        split=HOLDOUT,
        horizon_s=PRIMARY_HORIZON_S,
        seed=seed,
    )
    run = MakerEdgeRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        seed=seed,
        maker_rate=maker_rate,
        sweep=swept,
        discovery=discovery,
        holdout=holdout,
        decision=decide(
            next(item for item in discovery if item.horizon_s == PRIMARY_HORIZON_S), holdout
        ),
    )
    bootstrap = run.primary.bootstrap
    logger.info(
        "maker_edge verdict=%s discovery_cents=%s discovery_p=%s market_days=%d",
        run.decision.verdict,
        None if bootstrap is None else bootstrap.estimate,
        None if bootstrap is None else bootstrap.p_value,
        len(run.primary.clusters),
    )
    return run


def result_payload(run: MakerEdgeRun) -> dict:
    primary = run.primary
    holdout = run.holdout
    candidates = primary.candidates + holdout.candidates
    excluded = primary.excluded + holdout.excluded
    return {
        "run_id": run.run_id,
        "verdict": run.decision.verdict,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "bootstrap_seed": run.seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "primary_horizon_s": PRIMARY_HORIZON_S,
        "alpha": ALPHA,
        "bar": str(SELF_CHARGED_BAR),
        "bar_source": SELF_CHARGED_BAR_SOURCE,
        "bar_is_strict": STRICT,
        "maker_rate": str(run.maker_rate),
        "market_day_min_discovery": MARKET_DAY_MIN_DISCOVERY,
        "fills": {
            "offered": run.sweep.offered,
            "yes_fills": run.sweep.yes_fills,
            "no_fills": run.sweep.no_fills,
            "yes_empty": run.sweep.yes_empty,
            "no_empty": run.sweep.no_empty,
            "unnamed_markets": run.sweep.unnamed_markets,
            "scored_discovery": primary.n_fills,
            "scored_holdout": holdout.n_fills,
        },
        "discovery": _readout_payload(primary),
        "holdout": _readout_payload(holdout),
        "gate": _gate_payload(run.decision.gate),
        "replication": _replication_payload(run.decision.replication),
        "replication_skipped": run.decision.skipped,
        "horizon_curve": [_readout_payload(item) for item in run.discovery],
        "published_rate_sensitivity": _sensitivity_payload(run.sweep.published),
        "exclusions": {
            "candidates": candidates,
            "excluded": excluded,
            "excluded_fraction": (
                None if candidates == 0 else str(Decimal(excluded) / Decimal(candidates))
            ),
            "by_class": _pool_classes(primary.by_class, holdout.by_class),
        },
        "cities": sorted({series for series, _ in run.sweep.market_days}),
        "markets_per_city_day": {
            f"{series} {event_date.isoformat()}": count
            for (series, event_date), count in sorted(run.sweep.market_days.items())
        },
    }


def _readout_payload(item: HorizonReadout) -> dict:
    bootstrap = item.bootstrap
    excluded = item.excluded_fraction
    no_mid = item.no_mid_fraction
    return {
        "split": item.split,
        "horizon_s": item.horizon_s,
        "window_cap_s": window_cap_s(item.horizon_s),
        "edge_cents_per_contract": None if bootstrap is None else str(bootstrap.estimate),
        "ci_low": None if bootstrap is None else bootstrap.ci_low,
        "ci_high": None if bootstrap is None else bootstrap.ci_high,
        "ci_level": None if bootstrap is None else bootstrap.ci_level,
        "p_value": None if bootstrap is None else bootstrap.p_value,
        "replicate_spread": None if bootstrap is None else bootstrap.replicate_spread,
        "degenerate": None if bootstrap is None else bootstrap.degenerate,
        "market_days": len(item.clusters),
        "n_fills": item.n_fills,
        "contracts": str(item.contracts),
        "modelled": item.modelled,
        "no_mid_drops": item.dropped,
        "no_mid_fraction": None if no_mid is None else str(no_mid),
        "candidates": item.candidates,
        "excluded": item.excluded,
        "excluded_fraction": None if excluded is None else str(excluded),
        "out_of_window": item.out_of_window,
        "out_of_scope": item.out_of_scope,
        "by_class": dict(sorted(item.by_class.items())),
    }


# Reported beside the gating figure and never compared to the bar: the manifest stamps the gating
# rate alone, so one run still names one fee regime.
def _sensitivity_payload(published: Mapping[str, Tally]) -> dict:
    payload = {
        "rate": str(PUBLISHED_MAKER_RATE),
        "rate_source": PUBLISHED_MAKER_RATE_SOURCE,
        "horizon_s": PRIMARY_HORIZON_S,
        "gating": False,
    }
    for split in SPLITS:
        clusters = published[split].clusters()
        estimate = weighted_estimate(clusters)
        payload[f"{split}_edge_cents_per_contract"] = None if estimate is None else str(estimate)
        payload[f"{split}_market_days"] = len(clusters)
    return payload


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
