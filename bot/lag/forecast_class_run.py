import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bot.backtest.historical_open_meteo import SpreadCalibration
from bot.forecast.blend import BlendWeights, ClassScore, fit_weights
from bot.forecast.blend_score import blend_probability
from bot.lag.fee_floor import BAR_CONTEXT, MAKER_RATE_SOURCE, PUBLISHED_MAKER_RATE, fee_source
from bot.lag.forecast_classes import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    ClassRecord,
    class_freeze_path,
    read_class_freeze,
)
from bot.lag.forecast_entry import (
    SCREEN_RULE,
    SIZE,
    TICK_RULE,
    WALKED_TICK_REPORTED_ONLY,
    WALKED_TICK_RULE,
    DepthDistribution,
    DepthScreen,
    EntryCounts,
    TradedLeg,
    depth_distribution,
    depth_ok,
    entry_counts,
    entry_of,
    screen_depth,
)
from bot.lag.forecast_probability import (
    SIGMA_MULTIPLIERS,
    ClassProbability,
    LadderRung,
    SigmaSourceTally,
    baseline_brier,
    brier_skill,
    class_brier,
    class_cdf,
    event_ladder,
    ladder_sum,
    probabilities,
    sigma_for,
    sigma_source_tally,
)
from bot.lag.forecast_sample import F4_SERIES, SampleLeg, read_sample_freeze
from bot.lag.run_manifest import (
    BOOTSTRAP_RESAMPLES,
    MANIFEST_NAME,
    Exemption,
    RunInputs,
    write_manifest,
)
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
from bot.lag.tape_studies import SELF_CHARGED_BAR, SELF_CHARGED_BAR_SOURCE
from bot.markets.parser import parse_ticker
from bot.replay.analysis_stations import HIGH
from bot.replay.run_scope import DISCOVERY, HOLDOUT
from bot.validation.scoring import brier_score


logger = logging.getLogger(__name__)

RESULTS_NAME = "results.json"
CI_LEVEL = 0.95
NULL_VALUE = Decimal("0")
DIRECTION = "greater"
EVENT_DAYS = "event-days"
DISCOVERY_N_MIN = 200
COHORT = HIGH
BOOTSTRAP_SEED = 20260822

# 0.05 Bonferroni-corrected at the two questions the forecast-and-settlement family holds. The
# divisor is that frozen family count and is never recounted. tape_stats.ALPHA is the closed
# campaign's 0.05/4 and is not this one's.
ALPHA_F4: float = 0.05 / 2
# The statistic charges every entry its own taker fee and its tick, so a figure above the bar has
# already paid its costs and only a strictly positive one is an edge.
STRICT = True

GATING_LEAD = 24
REPORTED_LEAD = 36
BLEND_MEMBERS: tuple[str, ...] = ("ecmwf_ifs025", "hrrr", "icon_global", "nbm_nbs")
BLEND_LABEL = "blend"
CLASSES = (CLASS_A, CLASS_B, CLASS_C)
CLASS_MIN = 2

PASS = "PASS"
CLOSED = "CLOSED"
UNDERPOWERED = "UNDERPOWERED"
ZERO_ESTIMATE = "a discovery estimate of exactly zero fixes no direction to replicate"
NO_ESTIMATE = "a split with no traded leg carries no estimate to replicate"
NO_GATE_ESTIMATE = "a split with no traded leg carries no estimate to test against the bar"
LEAD_REPORTED_ONLY = (
    "the 36h lead is reported only: it gates nothing, carries no multiplicity correction, and is "
    "not evidence of an edge"
)
CLASS_REPORTED_ONLY = (
    "an individual class is reported only: it gates nothing, carries no multiplicity correction, "
    "and is not evidence of an edge"
)
SIGMA_BAND_REPORTED_ONLY = (
    "the sigma band is reported only: it gates nothing, carries no multiplicity correction, and "
    "is not evidence of an edge"
)

EXEMPTIONS = (
    Exemption(
        field="r0_fraction_invalid_max",
        reason="f4 reads no ladder and the r0 validity screen covers nothing it consumes",
    ),
    Exemption(
        field="latency_floor",
        reason="no order is placed and no read-to-act round trip enters the statistic",
    ),
)

_BLEND_SET = frozenset(BLEND_MEMBERS)


class ClassesUnavailable(RuntimeError):
    """Fewer than two forecast classes carry records, so the run stops at the availability check."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Readout:
    split: str
    clusters: tuple[ClusterAggregate, ...]
    bootstrap: BootstrapResult | None
    counts: EntryCounts


@dataclass(frozen=True, slots=True, kw_only=True)
class Brier:
    n: int
    model: Decimal
    baseline: Decimal
    skill: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class SigmaPoint:
    multiplier: Decimal
    estimate: Decimal | None
    traded_n: int
    event_days: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SigmaBand:
    label: str
    lead_hours: int
    split: str
    points: tuple[SigmaPoint, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class LadderSums:
    checked: int
    failed: int
    skipped: int
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BlendPanel:
    weights: BlendWeights
    counts: EntryCounts
    discovery: Readout
    holdout: Readout
    walked_discovery: Readout
    walked_holdout: Readout
    brier: Brier
    not_blendable_legs: int
    not_blendable_city_days: int
    not_blendable_event_days_lost: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassPanel:
    member: str
    forecast_class: str
    counts: EntryCounts
    discovery: Readout
    holdout: Readout
    brier: Brier


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadRun:
    lead_hours: int
    screen: DepthScreen
    candidates: DepthDistribution
    kept: DepthDistribution
    blend: BlendPanel
    classes: tuple[ClassPanel, ...]
    bands: tuple[SigmaBand, ...]
    sigma_sources: tuple[SigmaSourceTally, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    gate: GateVerdict | None
    replication: HoldoutVerdict | None
    skipped: str
    verdict: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastClassRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    seed: int
    cohort: str | None
    row_counts: Mapping[str, int]
    gating: LeadRun
    reported: LeadRun
    ladders: LadderSums
    decision: Decision


def read_class_records(classes: Path) -> list[ClassRecord]:
    found: dict[str, list[ClassRecord]] = {}
    for forecast_class in CLASSES:
        path = class_freeze_path(classes, forecast_class)
        found[forecast_class] = read_class_freeze(path) if path.is_file() else []
    retrieved = [name for name in CLASSES if found[name]]
    if len(retrieved) < CLASS_MIN:
        raise ClassesUnavailable(
            f"only {len(retrieved)} of {len(CLASSES)} forecast classes carry records under "
            f"{classes}, and the question closes at its availability check below {CLASS_MIN}"
        )
    return [row for name in CLASSES for row in found[name]]


def read_event_ladders(
    markets: Path, days: Collection[date]
) -> dict[tuple[str, date], tuple[str, ...]]:
    wanted = set(days)
    table = pq.read_table(markets, columns=["ticker", "series_ticker"])
    named = table.filter(pc.is_in(table.column("series_ticker"), value_set=pa.array(F4_SERIES)))
    ladders: dict[tuple[str, date], list[str]] = {}
    for ticker in named.column("ticker").to_pylist():
        parsed = parse_ticker(ticker)
        if parsed.event_date in wanted:
            ladders.setdefault((parsed.series, parsed.event_date), []).append(ticker)
    return {key: tuple(sorted(ladders[key])) for key in sorted(ladders)}


# The cluster unit is the event-day: one cluster pools the whole cohort's ladders for that date.
# A leg the entry rule never traded leaves both sides of the ratio rather than entering at SIZE
# weight against a zero total, which would dilute the estimate towards zero on a rule nobody
# registered.
def cluster_aggregates(rows: Sequence[TradedLeg], *, walked: bool) -> list[ClusterAggregate]:
    totals: dict[str, Decimal] = {}
    weights: dict[str, Decimal] = {}
    for row in rows:
        if not row.traded:
            continue
        cents = row.walked_net_profit_cents if walked else row.net_profit_cents
        cluster = row.event_date.isoformat()
        totals[cluster] = totals.get(cluster, Decimal(0)) + cents * row.size
        weights[cluster] = weights.get(cluster, Decimal(0)) + row.size
    return [
        ClusterAggregate(cluster=name, total=totals[name], weight=weights[name])
        for name in sorted(totals)
    ]


def point_estimate(clusters: Sequence[ClusterAggregate]) -> Decimal | None:
    if not clusters:
        return None
    total = Decimal(0)
    weight = Decimal(0)
    for item in clusters:
        total = BAR_CONTEXT.add(total, item.total)
        weight = BAR_CONTEXT.add(weight, item.weight)
    return BAR_CONTEXT.divide(total, weight)


def bootstrap_of(clusters: Sequence[ClusterAggregate], seed: int) -> BootstrapResult:
    return cluster_bootstrap(
        clusters,
        null_value=NULL_VALUE,
        direction=DIRECTION,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=seed,
        ci_level=CI_LEVEL,
    )


def readout(rows: Sequence[TradedLeg], *, split: str, seed: int, walked: bool = False) -> Readout:
    kept = tuple(row for row in rows if row.split == split)
    clusters = tuple(cluster_aggregates(kept, walked=walked))
    return Readout(
        split=split,
        clusters=clusters,
        bootstrap=bootstrap_of(clusters, seed) if clusters else None,
        counts=entry_counts(kept),
    )


def brier_of(model: Decimal, baseline: Decimal, n: int) -> Brier:
    return Brier(n=n, model=model, baseline=baseline, skill=brier_skill(model, baseline))


def sigma_band(
    label: str,
    lead_hours: int,
    entries: Sequence[tuple[SampleLeg, Mapping[Decimal, Decimal]]],
) -> SigmaBand:
    points: list[SigmaPoint] = []
    for multiplier in SIGMA_MULTIPLIERS:
        rows = [entry_of(leg, band[multiplier]) for leg, band in entries]
        clusters = cluster_aggregates(rows, walked=False)
        points.append(
            SigmaPoint(
                multiplier=multiplier,
                estimate=point_estimate(clusters),
                traded_n=sum(1 for row in rows if row.traded),
                event_days=len(clusters),
            )
        )
    return SigmaBand(label=label, lead_hours=lead_hours, split=DISCOVERY, points=tuple(points))


# as_of is fixed by (station, event_date, lead_hours), so every leg of the group answers the sigma
# lookup the same way and the lowest ticker is a deterministic pick among them.
def leg_per_group(legs: Sequence[SampleLeg]) -> dict[tuple[str, date, int], SampleLeg]:
    chosen: dict[tuple[str, date, int], SampleLeg] = {}
    for leg in legs:
        key = (leg.station, leg.event_date, leg.lead_hours)
        standing = chosen.get(key)
        if standing is None or leg.ticker < standing.ticker:
            chosen[key] = leg
    return chosen


# The full listed ladder is the object checked, not the legs that survived the staleness cap and
# the depth screen: a partial ladder cannot sum to one and would report a failure per gap.
def ladder_sums(
    records: Sequence[ClassRecord],
    legs: Sequence[SampleLeg],
    ladders: Mapping[tuple[str, date], tuple[str, ...]],
    calibration: SpreadCalibration,
) -> LadderSums:
    chosen = leg_per_group(legs)
    parsed: dict[tuple[str, date], tuple[LadderRung, ...]] = {}
    checked = 0
    skipped = 0
    failures: list[str] = []
    for record in records:
        leg = chosen.get((record.station, record.event_date, record.lead_hours))
        if leg is None or (leg.series, record.event_date) not in ladders:
            skipped += 1
            continue
        key = (leg.series, record.event_date)
        if key not in parsed:
            parsed[key] = event_ladder(ladders[key])
        sigma, _ = sigma_for(record, leg, calibration)
        event = (
            f"{leg.series} {record.event_date.isoformat()} lead={record.lead_hours} "
            f"{record.forecast_class}/{record.member}"
        )
        total = ladder_sum(parsed[key], class_cdf(record.daily_high_f, sigma), event=event)
        checked += 1
        if not total.ok:
            failures.append(f"{event} rungs={total.rungs} sum={total.total}")
    return LadderSums(
        checked=checked, failed=len(failures), skipped=skipped, failures=tuple(sorted(failures))
    )


def lead_run(
    *,
    lead_hours: int,
    legs: Sequence[SampleLeg],
    records: Sequence[ClassRecord],
    calibration: SpreadCalibration,
    seed: int,
) -> LeadRun:
    candidates = [leg for leg in legs if leg.lead_hours == lead_hours]
    screen = screen_depth(candidates)
    kept = [leg for leg in candidates if depth_ok(leg)]
    by_ticker = {leg.ticker: leg for leg in kept}
    priced = [
        row
        for row in probabilities(
            candidates, [row for row in records if row.lead_hours == lead_hours], calibration
        )
        if row.ticker in by_ticker
    ]

    per_leg: dict[str, dict[str, ClassProbability]] = {}
    for row in priced:
        per_leg.setdefault(row.ticker, {})[row.member] = row
    blendable = [leg.ticker for leg in kept if _BLEND_SET <= per_leg.get(leg.ticker, {}).keys()]
    blended_tickers = set(blendable)
    unblendable = [leg for leg in kept if leg.ticker not in blended_tickers]

    # The split travels with the record rather than being asserted here, so fit_weights' refusal
    # to fit anything but discovery is the barrier it was written to be.
    weights = fit_weights(
        [
            ClassScore(
                ticker=ticker,
                event_date=per_leg[ticker][member].event_date,
                split=per_leg[ticker][member].split,
                member=member,
                probability=per_leg[ticker][member].class_probability,
                outcome=per_leg[ticker][member].outcome,
            )
            for ticker in blendable
            if per_leg[ticker][BLEND_MEMBERS[0]].split == DISCOVERY
            for member in BLEND_MEMBERS
        ]
    )

    blend_rows: list[TradedLeg] = []
    blend_probabilities: list[Decimal] = []
    blend_prices: list[Decimal] = []
    blend_outcomes: list[int] = []
    for ticker in blendable:
        members = per_leg[ticker]
        blended = blend_probability(
            weights, {name: members[name].class_probability for name in BLEND_MEMBERS}
        )
        blend_rows.append(entry_of(by_ticker[ticker], blended))
        blend_probabilities.append(blended)
        blend_prices.append(by_ticker[ticker].entry_price)
        blend_outcomes.append(members[BLEND_MEMBERS[0]].outcome)

    blend = BlendPanel(
        weights=weights,
        counts=entry_counts(blend_rows),
        discovery=readout(blend_rows, split=DISCOVERY, seed=seed),
        holdout=readout(blend_rows, split=HOLDOUT, seed=seed),
        walked_discovery=readout(blend_rows, split=DISCOVERY, seed=seed, walked=True),
        walked_holdout=readout(blend_rows, split=HOLDOUT, seed=seed, walked=True),
        brier=brier_of(
            brier_score(blend_probabilities, blend_outcomes),
            brier_score(blend_prices, blend_outcomes),
            len(blend_rows),
        ),
        not_blendable_legs=len(unblendable),
        not_blendable_city_days=len({(leg.series, leg.event_date) for leg in unblendable}),
        not_blendable_event_days_lost=len(
            {leg.event_date for leg in unblendable}
            - {by_ticker[ticker].event_date for ticker in blendable}
        ),
    )

    named = sorted({row.member for row in priced})
    panels: list[ClassPanel] = []
    bands = [
        sigma_band(
            BLEND_LABEL,
            lead_hours,
            [
                (
                    by_ticker[ticker],
                    {
                        multiplier: blend_probability(
                            weights,
                            {
                                name: per_leg[ticker][name].sensitivity[multiplier]
                                for name in BLEND_MEMBERS
                            },
                        )
                        for multiplier in SIGMA_MULTIPLIERS
                    },
                )
                for ticker in blendable
                if per_leg[ticker][BLEND_MEMBERS[0]].split == DISCOVERY
            ],
        )
    ]
    for member in named:
        rows = [row for row in priced if row.member == member]
        traded = [entry_of(by_ticker[row.ticker], row.class_probability) for row in rows]
        panels.append(
            ClassPanel(
                member=member,
                forecast_class=rows[0].forecast_class,
                counts=entry_counts(traded),
                discovery=readout(traded, split=DISCOVERY, seed=seed),
                holdout=readout(traded, split=HOLDOUT, seed=seed),
                brier=brier_of(class_brier(rows), baseline_brier(rows), len(rows)),
            )
        )
        bands.append(
            sigma_band(
                member,
                lead_hours,
                [
                    (by_ticker[row.ticker], row.sensitivity)
                    for row in rows
                    if row.split == DISCOVERY
                ],
            )
        )

    logger.info(
        "forecast class lead=%d candidates=%d kept=%d blendable=%d classes=%d",
        lead_hours,
        len(candidates),
        len(kept),
        len(blendable),
        len(panels),
    )
    return LeadRun(
        lead_hours=lead_hours,
        screen=screen,
        candidates=depth_distribution(candidates),
        kept=depth_distribution(kept),
        blend=blend,
        classes=tuple(panels),
        bands=tuple(bands),
        sigma_sources=tuple(sigma_source_tally(priced)),
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
            alpha=ALPHA_F4,
            n_min=DISCOVERY_N_MIN,
            n_unit=EVENT_DAYS,
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
            n_unit=EVENT_DAYS,
            undecidable=holdout.bootstrap.degenerate,
        )

    # A shortfall in event-days keeps the question open for more tape, so it is read before the
    # economics; a spread-free resample leaves the gate insignificant and closes on that.
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
    sample: Path,
    classes: Path,
    markets: Path,
    calibration: Path,
    economic_bar_size: Decimal,
    economic_bar_price: Decimal,
    economic_bar_price_source: str,
    seed: int,
    run_root: Path,
    cohort: str | None = COHORT,
) -> ForecastClassRun:
    # Every freeze is opened and checked against its sidecar before the manifest is written, so a
    # run whose sample and classes are mis-paired leaves no manifest behind.
    legs = read_sample_freeze(sample)
    records = read_class_records(classes)
    spread = SpreadCalibration.load(calibration)
    ladders = read_event_ladders(markets, {leg.event_date for leg in legs})

    gating_legs = [leg for leg in legs if leg.lead_hours == GATING_LEAD]
    row_counts = {
        "sample_legs": len(legs),
        "class_records": len(records),
        "market_ladders": len(ladders),
        "event_days_24h": len({leg.event_date for leg in gating_legs}),
        "event_days_36h": len({leg.event_date for leg in legs if leg.lead_hours == REPORTED_LEAD}),
    }
    inputs = RunInputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        accrual_start=min(leg.as_of for leg in gating_legs),
        accrual_end=max(leg.close_time for leg in gating_legs),
        row_counts=row_counts,
        universe=None,
        fee=fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE),
        floor=None,
        economic_bar_size=economic_bar_size,
        economic_bar_price=economic_bar_price,
        economic_bar_price_source=economic_bar_price_source,
        bootstrap_seed=seed,
        cohort=cohort,
        exemptions=EXEMPTIONS,
    )
    digest = write_manifest(run_root, inputs)

    gating = lead_run(
        lead_hours=GATING_LEAD, legs=legs, records=records, calibration=spread, seed=seed
    )
    reported = lead_run(
        lead_hours=REPORTED_LEAD, legs=legs, records=records, calibration=spread, seed=seed
    )
    run = ForecastClassRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        seed=seed,
        cohort=cohort,
        row_counts=row_counts,
        gating=gating,
        reported=reported,
        ladders=ladder_sums(records, legs, ladders, spread),
        decision=decide(gating.blend.discovery, gating.blend.holdout),
    )
    bootstrap = gating.blend.discovery.bootstrap
    logger.info(
        "forecast class verdict=%s discovery_cents=%s discovery_p=%s event_days=%d",
        run.decision.verdict,
        None if bootstrap is None else bootstrap.estimate,
        None if bootstrap is None else bootstrap.p_value,
        len(gating.blend.discovery.clusters),
    )
    return run


def result_payload(run: ForecastClassRun) -> dict:
    gating = run.gating
    return {
        "run_id": run.run_id,
        "verdict": run.decision.verdict,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "bootstrap_seed": run.seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "alpha": ALPHA_F4,
        "bar": str(SELF_CHARGED_BAR),
        "bar_source": SELF_CHARGED_BAR_SOURCE,
        "bar_is_strict": STRICT,
        "size": str(SIZE),
        "cohort": run.cohort,
        "event_day_min_discovery": DISCOVERY_N_MIN,
        "gating_lead_hours": GATING_LEAD,
        # The pre-registration pins the screen and the tick reading, and the manifest has no slot
        # for either, so two runs under different pairings differ in the artifact as well.
        "screen_rule": SCREEN_RULE,
        "tick_rule": TICK_RULE,
        "blend": _blend_payload(gating.blend),
        "discovery": _readout_payload(gating.blend.discovery),
        "holdout": _readout_payload(gating.blend.holdout),
        "gate": _gate_payload(run.decision.gate),
        "replication": _replication_payload(run.decision.replication),
        "replication_skipped": run.decision.skipped,
        "classes": [_class_payload(panel) for panel in gating.classes],
        "lead_36h": _lead_payload(run.reported),
        "walked_tick": {
            "gating": False,
            "reported_only": WALKED_TICK_REPORTED_ONLY,
            "tick_rule": WALKED_TICK_RULE,
            "discovery": _readout_payload(gating.blend.walked_discovery),
            "holdout": _readout_payload(gating.blend.walked_holdout),
        },
        "sigma_band": {
            "gating": False,
            "reported_only": SIGMA_BAND_REPORTED_ONLY,
            "bands": [_band_payload(band) for band in gating.bands],
        },
        "depth": {
            "candidates": _depth_payload(gating.candidates),
            "kept": _depth_payload(gating.kept),
        },
        "screen": _screen_payload(gating.screen),
        "entries": {
            "n": gating.blend.counts.n,
            "traded": gating.blend.counts.traded_n,
            "untraded": gating.blend.counts.untraded_n,
            "not_blendable_legs": gating.blend.not_blendable_legs,
            "not_blendable_city_days": gating.blend.not_blendable_city_days,
            "not_blendable_event_days_lost": gating.blend.not_blendable_event_days_lost,
        },
        "ladder_sums": {
            "checked": run.ladders.checked,
            "failed": run.ladders.failed,
            "skipped": run.ladders.skipped,
            "failures": list(run.ladders.failures),
        },
        "sigma_source_tally": [_sigma_source_payload(item) for item in gating.sigma_sources],
        "briers": {
            BLEND_LABEL: _brier_payload(gating.blend.brier),
            **{panel.member: _brier_payload(panel.brier) for panel in gating.classes},
        },
        "row_counts": dict(run.row_counts),
    }


def _blend_payload(blend: BlendPanel) -> dict:
    weights = blend.weights
    return {
        "members": list(weights.members),
        "weights": {member: str(weights.weights[member]) for member in weights.members},
        "fitted_on_event_days": weights.fitted_on_event_days,
        "fitted_on_split": weights.fitted_on_split,
        "sha256": weights.sha256,
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
        "event_days": len(item.clusters),
        "legs": item.counts.n,
        "traded": item.counts.traded_n,
        "untraded": item.counts.untraded_n,
    }


def _brier_payload(brier: Brier) -> dict:
    return {
        "n": brier.n,
        "brier": str(brier.model),
        "baseline_brier": str(brier.baseline),
        "skill": str(brier.skill),
    }


def _class_payload(panel: ClassPanel) -> dict:
    return {
        "member": panel.member,
        "forecast_class": panel.forecast_class,
        "gating": False,
        "reported_only": CLASS_REPORTED_ONLY,
        "legs": panel.counts.n,
        "traded": panel.counts.traded_n,
        "untraded": panel.counts.untraded_n,
        "discovery": _readout_payload(panel.discovery),
        "holdout": _readout_payload(panel.holdout),
        "brier": _brier_payload(panel.brier),
    }


def _band_payload(band: SigmaBand) -> dict:
    return {
        "label": band.label,
        "lead_hours": band.lead_hours,
        "split": band.split,
        "points": [
            {
                "multiplier": str(point.multiplier),
                "net_profit_cents_per_contract": None
                if point.estimate is None
                else str(point.estimate),
                "traded": point.traded_n,
                "event_days": point.event_days,
            }
            for point in band.points
        ],
    }


def _depth_payload(item: DepthDistribution) -> dict:
    return {
        "prints_p10": item.prints_p10,
        "prints_p25": item.prints_p25,
        "prints_p50": item.prints_p50,
        "prints_p75": item.prints_p75,
        "prints_p90": item.prints_p90,
        "prints_max": item.prints_max,
        "prints_at_zero": item.prints_at_zero,
        "prints_below_size": item.prints_below_size,
        "contracts_p10": str(item.contracts_p10),
        "contracts_p25": str(item.contracts_p25),
        "contracts_p50": str(item.contracts_p50),
        "contracts_p75": str(item.contracts_p75),
        "contracts_p90": str(item.contracts_p90),
        "contracts_max": str(item.contracts_max),
        "contracts_at_zero": item.contracts_at_zero,
        "contracts_below_size": item.contracts_below_size,
    }


def _screen_payload(screen: DepthScreen) -> dict:
    return {
        "rule": SCREEN_RULE,
        "candidates": screen.candidates,
        "kept": screen.kept,
        "dropped": screen.dropped,
        "event_days": screen.event_days,
        "event_days_kept": screen.event_days_kept,
        "event_days_lost": str(screen.event_days_lost),
    }


def _sigma_source_payload(item: SigmaSourceTally) -> dict:
    return {
        "forecast_class": item.forecast_class,
        "native_xnd": item.native_xnd,
        "external_calibration": item.external_calibration,
    }


def _lead_payload(lead: LeadRun) -> dict:
    return {
        "lead_hours": lead.lead_hours,
        "gating": False,
        "reported_only": LEAD_REPORTED_ONLY,
        "blend": _blend_payload(lead.blend),
        "discovery": _readout_payload(lead.blend.discovery),
        "holdout": _readout_payload(lead.blend.holdout),
        "classes": [_class_payload(panel) for panel in lead.classes],
        "walked_tick": {
            "gating": False,
            "reported_only": WALKED_TICK_REPORTED_ONLY,
            "tick_rule": WALKED_TICK_RULE,
            "discovery": _readout_payload(lead.blend.walked_discovery),
            "holdout": _readout_payload(lead.blend.walked_holdout),
        },
        "sigma_band": {
            "gating": False,
            "reported_only": SIGMA_BAND_REPORTED_ONLY,
            "bands": [_band_payload(band) for band in lead.bands],
        },
        "depth": {
            "candidates": _depth_payload(lead.candidates),
            "kept": _depth_payload(lead.kept),
        },
        "screen": _screen_payload(lead.screen),
        "entries": {
            "n": lead.blend.counts.n,
            "traded": lead.blend.counts.traded_n,
            "untraded": lead.blend.counts.untraded_n,
            "not_blendable_legs": lead.blend.not_blendable_legs,
            "not_blendable_city_days": lead.blend.not_blendable_city_days,
            "not_blendable_event_days_lost": lead.blend.not_blendable_event_days_lost,
        },
        "sigma_source_tally": [_sigma_source_payload(item) for item in lead.sigma_sources],
        "briers": {
            BLEND_LABEL: _brier_payload(lead.blend.brier),
            **{panel.member: _brier_payload(panel.brier) for panel in lead.classes},
        },
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
