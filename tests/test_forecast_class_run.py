import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal, localcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.backtest.historical_open_meteo import SpreadCalibration, day_end_utc
from bot.forecast.blend import BlendWeights, ClassScore, fit_weights
from bot.lag import forecast_class_run
from bot.lag.fee_floor import MAKER_RATE_SOURCE, PUBLISHED_MAKER_RATE, fee_source
from bot.lag.forecast_class_run import (
    ALPHA_F4,
    BLEND_LABEL,
    BLEND_MEMBERS,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    CLASS_REPORTED_ONLY,
    CLASSES,
    CLOSED,
    COHORT,
    DISCOVERY_N_MIN,
    EVENT_DAYS,
    EXEMPTIONS,
    GATING_LEAD,
    LEAD_REPORTED_ONLY,
    PASS,
    REPORTED_LEAD,
    RESULTS_NAME,
    SIGMA_BAND_REPORTED_ONLY,
    STRICT,
    UNDERPOWERED,
    Brier,
    ClassesUnavailable,
    ForecastClassRun,
    Readout,
    bootstrap_of,
    cluster_aggregates,
    decide,
    execute,
    ladder_sums,
    point_estimate,
    read_class_records,
    read_event_ladders,
    result_payload,
)
from bot.lag.forecast_classes import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    LEAD_ANCHORED_COMPOSITE,
    MODEL_RUN,
    ClassRecord,
    class_freeze_path,
    leg_index,
    window_basis_for,
    write_class_freeze,
)
from bot.lag.forecast_entry import (
    SCREEN_RULE,
    SIZE,
    TICK_RULE,
    WALKED_TICK_REPORTED_ONLY,
    WALKED_TICK_RULE,
    depth_ok,
    entry_of,
)
from bot.lag.forecast_probability import (
    SIGMA_MULTIPLIERS,
    baseline_brier,
    brier_skill,
    class_brier,
    class_cdf,
    event_ladder,
    ladder_sum,
    probabilities,
    sigma_for,
)
from bot.lag.forecast_sample import F4_LEADS, SampleLeg, read_sample_freeze, write_sample_freeze
from bot.lag.r0_universe import Coverage, freeze_universe
from bot.lag.run_manifest import (
    BOOTSTRAP_RESAMPLES,
    MANIFEST_NAME,
    Exemption,
    ExemptionRefused,
    ManifestIncomplete,
    RunInputs,
    build_manifest,
)
from bot.lag.settlement_run import BOOTSTRAP_SEED as SETTLEMENT_SEED
from bot.lag.tape_stats import (
    ALPHA as CAMPAIGN_ALPHA,
    BootstrapResult,
    ClusterAggregate,
    GateVerdict,
    cluster_bootstrap,
    evaluate_gate,
)
from bot.lag.tape_studies import SELF_CHARGED_BAR, SELF_CHARGED_BAR_SOURCE
from bot.main import STATIONS
from bot.replay.analysis_stations import HIGH
from bot.replay.run_scope import DISCOVERY, HOLDOUT
from bot.validation.scoring import brier_score
from tests.test_tape_studies import seeded_repo, write_preregistration


RUN_ID = "f4-20260822"
SERIES = ("KXHIGHDEN", "KXHIGHNY")
BASE = {"KXHIGHDEN": 80, "KXHIGHNY": 60}
FIRST_DAY = date(2025, 8, 10)
DAYS = tuple(FIRST_DAY + timedelta(days=offset) for offset in range(10))
DISCOVERY_DAYS = DAYS[:5]
HOLDOUT_DAYS = DAYS[5:]
CLOSE_OFFSET = timedelta(seconds=60)
ISSUE_OFFSET = timedelta(hours=1)

ECMWF = "ecmwf_ifs025"
ICON = "icon_global"
UKMO = "ukmo_global_deterministic_10km"
NBM = "nbm_nbs"
HRRR = "hrrr"
CLASS_ROSTER = {CLASS_A: (ECMWF, ICON, UKMO), CLASS_B: (NBM,), CLASS_C: (HRRR,)}
MEMBER_SHIFT = {ECMWF: 0, ICON: 1, UKMO: 2, NBM: 3, HRRR: 4}
NBM_SIGMA = Decimal("2.5")
ALL_MEMBERS = (ECMWF, HRRR, ICON, NBM, UKMO)

THIN_DAY = ("KXHIGHDEN", DAYS[3])
THIN_RUNGS = 3
BROKEN_DAY = ("KXHIGHNY", DAYS[1])
MISSING_DAY = ("KNYC", DAYS[2])
SHALLOW_DAY = ("KXHIGHNY", DAYS[6])
SHALLOW_CONTRACTS = Decimal(10)
DEEP_CONTRACTS = Decimal(40)

AMBIENT_PRECISIONS = (20, 28, 50)
MARKETS_SCHEMA = pa.schema([("ticker", pa.string()), ("series_ticker", pa.string())])
CALIBRATION_BODY = {
    "model": "fixture",
    "buckets": [
        {"bucket": "8-24h", "sigma": 3.0},
        {"bucket": "24-72h", "sigma": 4.0},
        {"bucket": ">72h", "sigma": 8.0},
    ],
}

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SAMPLE = REPO_ROOT / "data" / "tape_studies" / "f4_inputs" / "sample.jsonl"
FROZEN_MARKETS = REPO_ROOT / "data" / "backtest" / "weather_markets.parquet"
FROZEN_LADDERS = 2302
FROZEN_RUNGS = {6: 2302}
# The one unread day the sample kept no leg on, whose seven listings are the corpus' only ladders
# that are not six rungs wide.
UNSAMPLED_DAY = date(2025, 11, 25)
WIDENED_LADDERS = 2309
WIDENED_RUNGS = {6: 2303, 3: 5, 5: 1}

needs_tape = pytest.mark.skipif(
    not (FROZEN_SAMPLE.exists() and FROZEN_MARKETS.exists()),
    reason="the frozen forecast corpus is not on this host",
)


def rungs(base: int) -> tuple[tuple[str, str, Decimal, Decimal | None], ...]:
    return (
        (f"T{base}", "below", Decimal(base), None),
        (f"B{base}.5", "bracket", Decimal(base), Decimal(base + 1)),
        (f"B{base + 2}.5", "bracket", Decimal(base + 2), Decimal(base + 3)),
        (f"B{base + 4}.5", "bracket", Decimal(base + 4), Decimal(base + 5)),
        (f"B{base + 6}.5", "bracket", Decimal(base + 6), Decimal(base + 7)),
        (f"T{base + 7}", "above", Decimal(base + 7), None),
    )


def ticker_of(series: str, event_date: date, suffix: str) -> str:
    return f"{series}-{event_date:%y%b%d}-{suffix}".upper()


def observed_high(series: str, day_index: int) -> int:
    return BASE[series] + 1 + 2 * (day_index % 4)


def settling_rung(day_index: int) -> int:
    return 1 + day_index % 4


def entry_price(rung_index: int, day_index: int) -> Decimal:
    return Decimal("0.08") + Decimal("0.03") * ((rung_index + day_index) % 6)


def sample_legs(
    *, thin: tuple[str, date] | None = THIN_DAY, shallow: tuple[str, date] | None = SHALLOW_DAY
) -> list[SampleLeg]:
    legs = []
    for lead_hours in F4_LEADS:
        for series in SERIES:
            station = STATIONS[series]
            for day_index, event_date in enumerate(DAYS):
                close_time = day_end_utc(event_date, station.timezone) - CLOSE_OFFSET
                ladder = rungs(BASE[series])
                if thin == (series, event_date):
                    ladder = ladder[:THIN_RUNGS]
                for rung_index, (suffix, kind, strike_lo, strike_hi) in enumerate(ladder):
                    legs.append(
                        SampleLeg(
                            ticker=ticker_of(series, event_date, suffix),
                            series=series,
                            station=station.station,
                            timezone=station.timezone,
                            event_date=event_date,
                            split=DISCOVERY if event_date in DISCOVERY_DAYS else HOLDOUT,
                            lead_hours=lead_hours,
                            close_time=close_time,
                            as_of=close_time - timedelta(hours=lead_hours),
                            entry_price=entry_price(rung_index, day_index),
                            staleness_minutes=Decimal("1"),
                            era="0010",
                            trailing_prints=40,
                            trailing_contracts=SHALLOW_CONTRACTS
                            if shallow == (series, event_date)
                            else DEEP_CONTRACTS,
                            strike_lo=strike_lo,
                            strike_hi=strike_hi,
                            kind=kind,
                            result="yes" if rung_index == settling_rung(day_index) else "no",
                        )
                    )
    return legs


def class_records(
    forecast_class: str,
    *,
    missing: Sequence[tuple[str, date]] = (),
    dropped_member: str | None = None,
) -> list[ClassRecord]:
    records = []
    for member in CLASS_ROSTER[forecast_class]:
        if member == dropped_member:
            continue
        for lead_hours in F4_LEADS:
            for series in SERIES:
                station = STATIONS[series]
                for day_index, event_date in enumerate(DAYS):
                    if member == ICON and (station.station, event_date) in missing:
                        continue
                    close_time = day_end_utc(event_date, station.timezone) - CLOSE_OFFSET
                    high = observed_high(series, day_index)
                    records.append(
                        ClassRecord(
                            station=station.station,
                            event_date=event_date,
                            lead_hours=lead_hours,
                            forecast_class=forecast_class,
                            member=member,
                            daily_high_f=Decimal(
                                high
                                + ((day_index + MEMBER_SHIFT[member] + lead_hours // 12) % 3)
                                - 1
                            ),
                            issue_time=close_time - timedelta(hours=lead_hours) - ISSUE_OFFSET,
                            issue_rule=LEAD_ANCHORED_COMPOSITE
                            if forecast_class == CLASS_A
                            else MODEL_RUN,
                            window_basis=window_basis_for(forecast_class, lead_hours),
                            native_sigma_f=NBM_SIGMA if member == NBM else None,
                            grid_latitude=station.latitude,
                            grid_longitude=station.longitude,
                            source_url=f"https://example.invalid/{member}",
                        )
                    )
    return records


def market_rows(*, broken: tuple[str, date] | None = None) -> list[dict]:
    rows = []
    for series in SERIES:
        for event_date in DAYS:
            ladder = rungs(BASE[series])
            if broken == (series, event_date):
                ladder = (*ladder[:3], *ladder[4:])
            rows.extend(
                {"ticker": ticker_of(series, event_date, suffix), "series_ticker": series}
                for suffix, _, _, _ in ladder
            )
    return rows


def write_corpus(
    root: Path,
    *,
    name: str = "inputs",
    missing: Sequence[tuple[str, date]] = (),
    dropped_member: str | None = None,
    thin: tuple[str, date] | None = THIN_DAY,
    broken: tuple[str, date] | None = None,
    written: Sequence[str] = CLASSES,
    corrupt: str | None = None,
) -> dict[str, Path]:
    directory = root / name
    directory.mkdir(parents=True)
    legs = sample_legs(thin=thin)
    sample = directory / "sample.jsonl"
    write_sample_freeze(legs, sample)

    indexed = leg_index(legs)
    for forecast_class in written:
        path = class_freeze_path(directory, forecast_class)
        write_class_freeze(
            class_records(forecast_class, missing=missing, dropped_member=dropped_member),
            path,
            indexed,
        )
        if forecast_class == corrupt:
            path.write_bytes(path.read_bytes() + b"\n")

    markets = directory / "weather_markets.parquet"
    pq.write_table(pa.Table.from_pylist(market_rows(broken=broken), schema=MARKETS_SCHEMA), markets)
    calibration = directory / "calibration.json"
    calibration.write_text(json.dumps(CALIBRATION_BODY))
    return {
        "preregistration": write_preregistration(directory / "preregistration.md"),
        "repo": seeded_repo(directory / "tree"),
        "sample": sample,
        "classes": directory,
        "markets": markets,
        "calibration": calibration,
    }


def run_at(
    paths: dict[str, Path], run_root: Path, *, run_id: str = RUN_ID, cohort: str | None = COHORT
) -> ForecastClassRun:
    return execute(
        run_id=run_id,
        preregistration=paths["preregistration"],
        repo=paths["repo"],
        sample=paths["sample"],
        classes=paths["classes"],
        markets=paths["markets"],
        calibration=paths["calibration"],
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        seed=BOOTSTRAP_SEED,
        run_root=run_root,
        cohort=cohort,
    )


@dataclass(frozen=True, slots=True)
class Corpus:
    root: Path
    paths: dict[str, Path]
    run_root: Path
    run: ForecastClassRun
    payload: dict


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    root = tmp_path_factory.mktemp("f4")
    paths = write_corpus(root)
    run_root = root / "tape_studies"
    run = run_at(paths, run_root)
    return Corpus(root=root, paths=paths, run_root=run_root, run=run, payload=result_payload(run))


def manifest_of(run_root: Path, run_id: str = RUN_ID) -> dict:
    return json.loads((run_root / run_id / MANIFEST_NAME).read_text())


def screened_rows(paths: dict[str, Path], lead_hours: int = GATING_LEAD) -> list:
    legs = read_sample_freeze(paths["sample"])
    records = read_class_records(paths["classes"])
    calibration = SpreadCalibration.load(paths["calibration"])
    candidates = [leg for leg in legs if leg.lead_hours == lead_hours]
    kept = {leg.ticker for leg in candidates if depth_ok(leg)}
    return [
        row
        for row in probabilities(
            candidates, [row for row in records if row.lead_hours == lead_hours], calibration
        )
        if row.ticker in kept
    ]


def refit_weights(paths: dict[str, Path]) -> BlendWeights:
    per_leg: dict[str, dict[str, object]] = {}
    for row in screened_rows(paths):
        per_leg.setdefault(row.ticker, {})[row.member] = row
    return fit_weights(
        [
            ClassScore(
                ticker=ticker,
                event_date=per_leg[ticker][member].event_date,
                split=DISCOVERY,
                member=member,
                probability=per_leg[ticker][member].class_probability,
                outcome=per_leg[ticker][member].outcome,
            )
            for ticker in sorted(per_leg)
            if set(BLEND_MEMBERS) <= per_leg[ticker].keys()
            and per_leg[ticker][BLEND_MEMBERS[0]].split == DISCOVERY
            for member in BLEND_MEMBERS
        ]
    )


def f4_inputs(paths: dict[str, Path], **overrides: object) -> RunInputs:
    legs = [leg for leg in read_sample_freeze(paths["sample"]) if leg.lead_hours == GATING_LEAD]
    inputs = RunInputs(
        run_id=RUN_ID,
        preregistration=paths["preregistration"],
        repo=paths["repo"],
        accrual_start=min(leg.as_of for leg in legs),
        accrual_end=max(leg.close_time for leg in legs),
        row_counts={"sample_legs": len(legs)},
        universe=None,
        fee=fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE),
        floor=None,
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        bootstrap_seed=BOOTSTRAP_SEED,
        cohort=COHORT,
        exemptions=EXEMPTIONS,
    )
    return replace(inputs, **overrides)


def bootstrap_at(
    p_value: float,
    *,
    estimate: str,
    n_clusters: int,
    degenerate: bool = False,
    replicate_spread: float = 0.5,
) -> BootstrapResult:
    return BootstrapResult(
        estimate=Decimal(estimate),
        null_value=forecast_class_run.NULL_VALUE,
        direction=forecast_class_run.DIRECTION,
        p_value=p_value,
        ci_level=CI_LEVEL,
        ci_low=0.1,
        ci_high=2.0,
        replicate_spread=replicate_spread,
        degenerate=degenerate,
        n_clusters=n_clusters,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )


def flat(total: str, count: int) -> tuple[ClusterAggregate, ...]:
    return tuple(
        ClusterAggregate(cluster=f"{index:04d}", total=Decimal(total), weight=SIZE)
        for index in range(count)
    )


def readout_at(bootstrap: BootstrapResult, *, split: str) -> Readout:
    return Readout(
        split=split,
        clusters=flat("1", bootstrap.n_clusters),
        bootstrap=bootstrap,
        counts=forecast_class_run.entry_counts([]),
    )


# Reruns the gate the run resolved with one flag moved, so that flag is the only thing that
# differs between the two verdicts.
def regate(gate: GateVerdict, result: BootstrapResult, **overrides: object) -> GateVerdict:
    arguments = {
        "estimate": gate.estimate,
        "p_value": gate.p_value,
        "result": result,
        "threshold": gate.threshold,
        "direction": gate.direction,
        "alpha": gate.alpha,
        "n_min": gate.n_min,
        "n_unit": gate.n_unit,
        "undecidable": gate.undecidable,
        "strict": STRICT,
    }
    return evaluate_gate(**(arguments | overrides))


def leg_at(
    *,
    entry_price: Decimal = Decimal("0.20"),
    event_date: date = DAYS[0],
    ticker: str = "KXHIGHDEN-25AUG10-B80.5",
) -> SampleLeg:
    close_time = day_end_utc(event_date, "America/Denver") - CLOSE_OFFSET
    return SampleLeg(
        ticker=ticker,
        series="KXHIGHDEN",
        station="KDEN",
        timezone="America/Denver",
        event_date=event_date,
        split=DISCOVERY,
        lead_hours=GATING_LEAD,
        close_time=close_time,
        as_of=close_time - timedelta(hours=GATING_LEAD),
        entry_price=entry_price,
        staleness_minutes=Decimal("1"),
        era="0010",
        trailing_prints=40,
        trailing_contracts=DEEP_CONTRACTS,
        strike_lo=Decimal(80),
        strike_hi=Decimal(81),
        kind="bracket",
        result="yes",
    )


def test_the_family_alpha_is_the_two_question_correction() -> None:
    assert ALPHA_F4 == 0.025
    assert ALPHA_F4 == 0.05 / 2
    assert CAMPAIGN_ALPHA == 0.0125
    assert ALPHA_F4 != CAMPAIGN_ALPHA
    assert ALPHA_F4 != 0.05 / 3


def test_a_run_that_reaches_the_closed_campaigns_alpha_lands_on_another_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = readout_at(bootstrap_at(0.02, estimate="1.5", n_clusters=220), split=DISCOVERY)
    holdout = readout_at(bootstrap_at(0.01, estimate="1.0", n_clusters=122), split=HOLDOUT)

    published = decide(discovery, holdout)
    monkeypatch.setattr(forecast_class_run, "ALPHA_F4", CAMPAIGN_ALPHA)
    narrowed = decide(discovery, holdout)

    assert CAMPAIGN_ALPHA <= 0.02 < ALPHA_F4
    assert published.gate is not None and narrowed.gate is not None
    assert published.gate.alpha == 0.025
    assert published.gate.significant is True
    assert published.verdict == PASS
    assert narrowed.gate.alpha == 0.0125
    assert narrowed.gate.significant is False
    assert narrowed.verdict == CLOSED
    assert narrowed.verdict != published.verdict


def test_the_seed_is_this_runs_and_not_the_one_already_spent(corpus: Corpus) -> None:
    assert BOOTSTRAP_SEED == 20260822
    assert BOOTSTRAP_SEED != SETTLEMENT_SEED
    assert SETTLEMENT_SEED == 20260820
    assert corpus.payload["bootstrap_seed"] == 20260822
    assert manifest_of(corpus.run_root)["bootstrap_seed"] == 20260822


def test_a_panel_whose_replicates_carry_no_spread_is_undecidable() -> None:
    bootstrap = bootstrap_at(
        0.0001, estimate="5", n_clusters=220, degenerate=True, replicate_spread=0.0
    )
    holdout = readout_at(bootstrap_at(0.01, estimate="1.0", n_clusters=122), split=HOLDOUT)

    gate = decide(readout_at(bootstrap, split=DISCOVERY), holdout).gate

    assert gate is not None
    assert gate.economic is True
    assert gate.undecidable is True
    assert gate.significant is False
    assert gate.passed is False


def test_the_discovery_floor_is_two_hundred_event_days() -> None:
    holdout = readout_at(bootstrap_at(0.01, estimate="1.0", n_clusters=122), split=HOLDOUT)
    short = decide(
        readout_at(bootstrap_at(0.02, estimate="1.5", n_clusters=199), split=DISCOVERY), holdout
    )
    full = decide(
        readout_at(bootstrap_at(0.02, estimate="1.5", n_clusters=200), split=DISCOVERY), holdout
    )

    assert DISCOVERY_N_MIN == 200
    assert short.gate is not None and full.gate is not None
    assert short.gate.n == 199
    assert short.gate.powered is False
    assert short.verdict == UNDERPOWERED
    assert full.gate.n == 200
    assert full.gate.powered is True
    assert full.gate.n_min == 200
    assert full.gate.n_unit == EVENT_DAYS
    assert full.verdict != UNDERPOWERED


def test_the_holdout_floor_is_half_the_discovery_one() -> None:
    discovery = readout_at(bootstrap_at(0.001, estimate="2.0", n_clusters=220), split=DISCOVERY)
    short = decide(
        discovery, readout_at(bootstrap_at(0.01, estimate="1.5", n_clusters=99), split=HOLDOUT)
    )
    full = decide(
        discovery, readout_at(bootstrap_at(0.01, estimate="1.5", n_clusters=100), split=HOLDOUT)
    )

    assert short.replication is not None and full.replication is not None
    assert short.replication.holdout_n_min == 100
    assert (DISCOVERY_N_MIN + 1) // 2 == 100
    assert short.replication.powered is False
    assert full.replication.powered is True
    assert full.replication.alpha == 0.05


def test_the_strict_gate_is_what_puts_a_zero_edge_below_the_zero_bar() -> None:
    bootstrap = bootstrap_at(0.001, estimate="0", n_clusters=220)
    holdout = readout_at(bootstrap_at(0.01, estimate="1.0", n_clusters=122), split=HOLDOUT)

    decision = decide(readout_at(bootstrap, split=DISCOVERY), holdout)
    gate = decision.gate

    assert STRICT is True
    assert SELF_CHARGED_BAR == Decimal("0")
    assert gate is not None
    assert gate.estimate == Decimal("0")
    assert gate.threshold == SELF_CHARGED_BAR
    assert gate.significant is True
    assert gate.powered is True
    assert gate.economic is False
    assert gate.passed is False
    assert regate(gate, bootstrap).economic is False
    assert regate(gate, bootstrap, strict=False).economic is True
    assert regate(gate, bootstrap, strict=False).passed is True
    assert decision.verdict == CLOSED
    assert decision.skipped == forecast_class_run.ZERO_ESTIMATE


def test_a_run_with_no_stated_bar_price_aborts_naming_it(corpus: Corpus) -> None:
    with pytest.raises(ManifestIncomplete) as refused:
        build_manifest(f4_inputs(corpus.paths, economic_bar_price=None))

    assert refused.value.fields == ("economic_bar_price",)
    assert "economic_bar_price" in str(refused.value)


def test_a_third_field_cannot_be_declared_exempt() -> None:
    with pytest.raises(ExemptionRefused, match="row_counts"):
        Exemption(field="row_counts", reason="f4 reads no rows")

    assert {item.field for item in EXEMPTIONS} == {"latency_floor", "r0_fraction_invalid_max"}
    assert all(item.reason for item in EXEMPTIONS)


def test_declaring_the_universe_exempt_while_supplying_it_is_refused(corpus: Corpus) -> None:
    universe = freeze_universe(
        fraction_invalid_max=Decimal("0.4"),
        passing=SERIES,
        coverage=Coverage(cities=SERIES, ladder_widths=(6,), in_scope_city_days=len(DAYS)),
    )

    with pytest.raises(ExemptionRefused, match="r0_fraction_invalid_max"):
        build_manifest(f4_inputs(corpus.paths, universe=universe))


def test_the_manifest_states_the_bar_the_statistic_charges_itself(corpus: Corpus) -> None:
    manifest = manifest_of(corpus.run_root)

    assert manifest["economic_bar_size"] == "0"
    assert manifest["economic_bar_price"] == "0"
    assert manifest["economic_bar_cents_per_contract"] == "0"
    assert manifest["economic_bar_price_source"] == SELF_CHARGED_BAR_SOURCE


def test_the_manifest_records_every_field_the_run_supplies(corpus: Corpus) -> None:
    manifest = manifest_of(corpus.run_root)
    legs = read_sample_freeze(corpus.paths["sample"])
    gating = [leg for leg in legs if leg.lead_hours == GATING_LEAD]

    assert manifest["sha256"] == corpus.run.manifest_sha256
    assert manifest["run_id"] == RUN_ID
    assert manifest["cohort"] == HIGH
    assert manifest["accrual_start"] == min(leg.as_of for leg in gating).isoformat()
    assert manifest["accrual_end"] == max(leg.close_time for leg in gating).isoformat()
    assert manifest["row_counts"] == corpus.payload["row_counts"]
    assert manifest["row_counts"]["sample_legs"] == len(legs)
    assert manifest["row_counts"]["event_days_24h"] == len(DAYS)
    assert manifest["row_counts"]["event_days_36h"] == len(DAYS)
    assert manifest["fee_maker_rate"] == str(PUBLISHED_MAKER_RATE)
    assert manifest["fee_maker_rate_source"] == MAKER_RATE_SOURCE
    assert manifest["r0_fraction_invalid_max"] is None
    assert manifest["latency_floor_source"] is None
    assert [item["field"] for item in manifest["exemptions"]] == [
        "latency_floor",
        "r0_fraction_invalid_max",
    ]
    assert "fee_type_check" not in manifest
    assert "settlement_source" not in manifest


def test_a_class_freeze_that_fails_its_sidecar_leaves_no_manifest_behind(tmp_path: Path) -> None:
    paths = write_corpus(tmp_path, corrupt=CLASS_B)
    run_root = tmp_path / "tape_studies"

    with pytest.raises(ValueError, match="does not match the sha256"):
        run_at(paths, run_root)

    assert not run_root.exists()


def test_the_manifest_lands_before_the_first_statistic_is_computed(tmp_path: Path) -> None:
    paths = write_corpus(tmp_path, dropped_member=ICON)
    run_root = tmp_path / "tape_studies"

    with pytest.raises(ValueError, match="no records to fit the blend on"):
        run_at(paths, run_root)

    assert manifest_of(run_root)["cohort"] == HIGH
    assert not (run_root / RUN_ID / RESULTS_NAME).exists()


def test_a_second_run_under_the_same_id_refuses_to_overwrite(corpus: Corpus) -> None:
    with pytest.raises(FileExistsError, match=MANIFEST_NAME):
        run_at(corpus.paths, corpus.run_root)

    assert manifest_of(corpus.run_root)["sha256"] == corpus.run.manifest_sha256


def test_a_corpus_carrying_one_class_stops_at_the_availability_check(tmp_path: Path) -> None:
    paths = write_corpus(tmp_path, written=(CLASS_A,))
    run_root = tmp_path / "tape_studies"

    with pytest.raises(ClassesUnavailable, match="1 of 3"):
        run_at(paths, run_root)

    assert not run_root.exists()
    assert (
        len(
            read_class_records(
                write_corpus(tmp_path, name="pair", written=(CLASS_A, CLASS_B))["classes"]
            )
        )
        > 0
    )


def test_the_run_resolves_one_gate_and_one_replication_over_the_whole_execute(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, int] = {"gate": 0, "holdout": 0}
    gate_seen = forecast_class_run.evaluate_gate
    holdout_seen = forecast_class_run.evaluate_holdout

    def counted_gate(**kwargs: object) -> object:
        calls["gate"] += 1
        return gate_seen(**kwargs)

    def counted_holdout(**kwargs: object) -> object:
        calls["holdout"] += 1
        return holdout_seen(**kwargs)

    monkeypatch.setattr(forecast_class_run, "evaluate_gate", counted_gate)
    monkeypatch.setattr(forecast_class_run, "evaluate_holdout", counted_holdout)
    run_at(corpus.paths, corpus.root / "counted", run_id=RUN_ID)

    assert calls["gate"] == 1
    assert calls["holdout"] == 1
    assert corpus.payload["gate"] is not None
    assert corpus.payload["replication"] is not None
    assert corpus.payload["replication_skipped"] == ""


def test_every_figure_beside_the_gate_publishes_that_it_gates_nothing(corpus: Corpus) -> None:
    payload = corpus.payload
    reported = payload["lead_36h"]

    assert reported["gating"] is False
    assert reported["reported_only"] == LEAD_REPORTED_ONLY
    assert reported["lead_hours"] == REPORTED_LEAD
    assert "gate" not in reported
    assert "replication" not in reported
    assert payload["walked_tick"]["gating"] is False
    assert payload["walked_tick"]["reported_only"] == WALKED_TICK_REPORTED_ONLY
    assert payload["walked_tick"]["tick_rule"] == WALKED_TICK_RULE
    assert payload["sigma_band"]["gating"] is False
    assert payload["sigma_band"]["reported_only"] == SIGMA_BAND_REPORTED_ONLY
    assert [item["member"] for item in payload["classes"]] == list(ALL_MEMBERS)
    for item in payload["classes"]:
        assert item["gating"] is False
        assert item["reported_only"] == CLASS_REPORTED_ONLY
        assert "gate" not in item
        assert "replication" not in item
    assert "gates nothing" in LEAD_REPORTED_ONLY
    assert "no multiplicity correction" in CLASS_REPORTED_ONLY
    assert "not evidence of an edge" in SIGMA_BAND_REPORTED_ONLY


def test_the_gate_reads_the_twenty_four_hour_blend(corpus: Corpus) -> None:
    payload = corpus.payload

    assert payload["gating_lead_hours"] == GATING_LEAD
    assert payload["gate"]["n_unit"] == EVENT_DAYS
    assert payload["gate"]["threshold"] == "0"
    assert payload["gate"]["alpha"] == ALPHA_F4
    assert payload["gate"]["n_min"] == DISCOVERY_N_MIN
    assert payload["gate"]["powered"] is False
    assert payload["verdict"] == UNDERPOWERED
    assert payload["gate"]["estimate"] == payload["discovery"]["net_profit_cents_per_contract"]
    assert payload["discovery"]["split"] == DISCOVERY
    assert payload["holdout"]["split"] == HOLDOUT
    assert payload["discovery"]["event_days"] == len(DISCOVERY_DAYS)
    assert payload["holdout"]["event_days"] == len(HOLDOUT_DAYS)


def test_the_blend_is_fitted_on_the_discovery_legs_and_never_refitted(corpus: Corpus) -> None:
    published = corpus.payload["blend"]
    refitted = refit_weights(corpus.paths)

    assert published["members"] == list(BLEND_MEMBERS)
    assert published["fitted_on_split"] == DISCOVERY
    assert published["fitted_on_event_days"] == len(DISCOVERY_DAYS)
    assert published["sha256"] == refitted.sha256
    assert published["weights"] == {
        member: str(refitted.weights[member]) for member in refitted.members
    }
    assert UKMO not in published["members"]
    assert list(corpus.payload).index("blend") < list(corpus.payload).index("holdout")


def test_fitting_on_a_holdout_record_is_refused() -> None:
    discovery = [
        ClassScore(
            ticker="KXHIGHDEN-25AUG10-B80.5",
            event_date=DAYS[0],
            split=DISCOVERY,
            member=member,
            probability=Decimal("0.4"),
            outcome=1,
        )
        for member in BLEND_MEMBERS
    ]

    with pytest.raises(ValueError, match=HOLDOUT):
        fit_weights([*discovery, replace(discovery[0], split=HOLDOUT)])


def test_a_leg_missing_a_blend_member_is_excluded_and_counted(tmp_path: Path) -> None:
    paths = write_corpus(tmp_path, missing=(MISSING_DAY,))
    payload = result_payload(run_at(paths, tmp_path / "tape_studies"))
    entries = payload["entries"]

    assert entries["not_blendable_legs"] == len(rungs(BASE["KXHIGHNY"]))
    assert entries["not_blendable_city_days"] == 1
    assert entries["not_blendable_event_days_lost"] == 0
    assert entries["n"] + entries["not_blendable_legs"] == payload["screen"]["kept"]
    assert payload["discovery"]["event_days"] == len(DISCOVERY_DAYS)


def test_an_untraded_leg_leaves_both_sides_of_the_ratio() -> None:
    tie = entry_of(leg_at(entry_price=Decimal("0.20")), Decimal("0.20"))
    traded = entry_of(leg_at(entry_price=Decimal("0.20"), event_date=DAYS[1]), Decimal("0.60"))

    clusters = cluster_aggregates([tie, traded], walked=False)

    assert tie.traded is False
    assert traded.traded is True
    assert [item.cluster for item in clusters] == [DAYS[1].isoformat()]
    assert cluster_aggregates([tie], walked=False) == []
    assert forecast_class_run.entry_counts([tie, traded]).untraded_n == 1


def test_a_cluster_carrying_no_weight_is_refused_outright() -> None:
    with pytest.raises(ValueError, match="non-positive weight"):
        cluster_bootstrap(
            [ClusterAggregate(cluster=DAYS[0].isoformat(), total=Decimal(0), weight=Decimal(0))],
            null_value=forecast_class_run.NULL_VALUE,
            direction=forecast_class_run.DIRECTION,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
            ci_level=CI_LEVEL,
        )


def test_the_sigma_band_at_one_is_the_gated_point_estimate(corpus: Corpus) -> None:
    payload = corpus.payload
    band = next(item for item in payload["sigma_band"]["bands"] if item["label"] == BLEND_LABEL)
    points = {item["multiplier"]: item for item in band["points"]}

    assert [str(multiplier) for multiplier in SIGMA_MULTIPLIERS] == ["0.5", "1", "2"]
    assert list(points) == ["0.5", "1", "2"]
    assert band["split"] == DISCOVERY
    assert points["1"]["net_profit_cents_per_contract"] == payload["gate"]["estimate"]
    assert points["1"]["event_days"] == len(DISCOVERY_DAYS)
    assert all(point["traded"] > 0 for point in band["points"])
    assert {item["label"] for item in payload["sigma_band"]["bands"]} == {
        BLEND_LABEL,
        *ALL_MEMBERS,
    }


def test_the_ladder_sum_check_reads_the_listed_ladder_and_not_the_survivors(
    corpus: Corpus,
) -> None:
    sums = corpus.payload["ladder_sums"]
    legs = read_sample_freeze(corpus.paths["sample"])
    records = read_class_records(corpus.paths["classes"])
    calibration = SpreadCalibration.load(corpus.paths["calibration"])
    survivors = tuple(
        sorted(
            leg.ticker
            for leg in legs
            if (leg.series, leg.event_date) == THIN_DAY and leg.lead_hours == GATING_LEAD
        )
    )
    record = next(
        row
        for row in records
        if (row.station, row.event_date, row.lead_hours) == ("KDEN", THIN_DAY[1], GATING_LEAD)
    )
    leg = next(leg for leg in legs if leg.ticker == survivors[0])
    sigma, _ = sigma_for(record, leg, calibration)
    partial = ladder_sum(
        event_ladder(survivors), class_cdf(record.daily_high_f, sigma), event="survivors"
    )

    assert len(survivors) == THIN_RUNGS
    assert partial.ok is False
    assert sums["checked"] == len(records)
    assert sums["failed"] == 0
    assert sums["skipped"] == 0
    assert sums["failures"] == []


def test_a_listed_ladder_that_does_not_sum_to_one_is_reported_with_its_event(
    tmp_path: Path,
) -> None:
    paths = write_corpus(tmp_path, broken=BROKEN_DAY)
    payload = result_payload(run_at(paths, tmp_path / "tape_studies"))
    sums = payload["ladder_sums"]
    named = f"{BROKEN_DAY[0]} {BROKEN_DAY[1].isoformat()}"

    assert sums["failed"] > 0
    assert all(named in failure for failure in sums["failures"])
    assert len(sums["failures"]) == sums["failed"]
    assert all("rungs=5" in failure for failure in sums["failures"])


def test_a_record_whose_event_has_no_listed_ladder_is_skipped(corpus: Corpus) -> None:
    records = read_class_records(corpus.paths["classes"])
    legs = read_sample_freeze(corpus.paths["sample"])
    calibration = SpreadCalibration.load(corpus.paths["calibration"])

    sums = ladder_sums(records, legs, {}, calibration)

    assert sums.checked == 0
    assert sums.skipped == len(records)
    assert sums.failed == 0


def test_the_depth_screen_drops_the_shallow_day_and_publishes_both_distributions(
    corpus: Corpus,
) -> None:
    payload = corpus.payload
    screen = payload["screen"]
    width = len(rungs(BASE["KXHIGHNY"]))

    assert screen["rule"] == SCREEN_RULE
    assert screen["dropped"] == width
    assert screen["kept"] == screen["candidates"] - width
    assert payload["depth"]["candidates"]["contracts_below_size"] == width
    assert payload["depth"]["kept"]["contracts_below_size"] == 0
    assert payload["depth"]["candidates"]["contracts_max"] == str(DEEP_CONTRACTS)
    assert payload["depth"]["kept"]["contracts_p10"] == str(DEEP_CONTRACTS)


def test_the_results_carry_the_pinned_screen_and_tick_rules(corpus: Corpus) -> None:
    payload = corpus.payload

    assert payload["screen_rule"] == "trailing_contracts_ge_26"
    assert payload["tick_rule"] == "one_tick_constant"
    assert payload["screen_rule"] == SCREEN_RULE
    assert payload["tick_rule"] == TICK_RULE
    assert payload["walked_tick"]["tick_rule"] == "one_tick_constant_entry_walked_one_tick"
    assert payload["size"] == str(SIZE)
    assert payload["bar"] == "0"
    assert payload["bar_source"] == SELF_CHARGED_BAR_SOURCE
    assert payload["bar_is_strict"] is True
    assert payload["cohort"] == HIGH
    assert payload["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES


def test_the_walked_tick_costs_a_tick_against_the_published_figure(corpus: Corpus) -> None:
    payload = corpus.payload
    published = Decimal(payload["discovery"]["net_profit_cents_per_contract"])
    walked = Decimal(payload["walked_tick"]["discovery"]["net_profit_cents_per_contract"])

    assert walked < published
    assert payload["walked_tick"]["discovery"]["event_days"] == len(DISCOVERY_DAYS)
    assert payload["walked_tick"]["holdout"]["split"] == HOLDOUT


def test_both_leads_are_measured_and_only_one_of_them_gates(corpus: Corpus) -> None:
    payload = corpus.payload
    reported = payload["lead_36h"]

    assert corpus.run.gating.lead_hours == GATING_LEAD
    assert corpus.run.reported.lead_hours == REPORTED_LEAD
    assert reported["discovery"]["net_profit_cents_per_contract"] is not None
    assert (
        reported["discovery"]["net_profit_cents_per_contract"]
        != payload["discovery"]["net_profit_cents_per_contract"]
    )
    assert [item["member"] for item in reported["classes"]] == list(ALL_MEMBERS)
    assert set(reported["briers"]) == {BLEND_LABEL, *ALL_MEMBERS}


def test_every_class_carries_a_brier_against_the_market_baseline(corpus: Corpus) -> None:
    payload = corpus.payload

    assert set(payload["briers"]) == {BLEND_LABEL, *ALL_MEMBERS}
    for name, item in payload["briers"].items():
        assert item["n"] > 0
        assert Decimal(item["brier"]) > 0
        assert Decimal(item["baseline_brier"]) > 0
        assert item["skill"] == str(
            brier_skill(Decimal(item["brier"]), Decimal(item["baseline_brier"]))
        )
        assert name in {BLEND_LABEL, *ALL_MEMBERS}


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_brier_figures_do_not_move_with_the_ambient_precision(
    corpus: Corpus, prec: int
) -> None:
    rows = [row for row in screened_rows(corpus.paths) if row.member == ECMWF]

    with localcontext(prec=prec):
        model = class_brier(rows)
        baseline = baseline_brier(rows)
        skill = brier_skill(model, baseline)

    published = corpus.payload["briers"][ECMWF]
    assert str(model) == published["brier"]
    assert str(baseline) == published["baseline_brier"]
    assert str(skill) == published["skill"]


def test_the_sigma_source_tally_names_both_roads_to_a_sigma(corpus: Corpus) -> None:
    tally = {item["forecast_class"]: item for item in corpus.payload["sigma_source_tally"]}

    assert set(tally) == {CLASS_A, CLASS_B, CLASS_C}
    assert tally[CLASS_B]["native_xnd"] > 0
    assert tally[CLASS_B]["external_calibration"] == 0
    assert tally[CLASS_A]["native_xnd"] == 0
    assert tally[CLASS_A]["external_calibration"] > 0
    assert tally[CLASS_C]["external_calibration"] > 0


def test_the_point_estimate_reads_the_same_ratio_the_bootstrap_publishes() -> None:
    clusters = tuple(
        ClusterAggregate(cluster=f"{index:04d}", total=Decimal(7 * index - 40), weight=SIZE)
        for index in range(12)
    )

    published = bootstrap_of(clusters, BOOTSTRAP_SEED)

    assert point_estimate(clusters) == published.estimate
    assert point_estimate(()) is None
    assert published.ci_level == CI_LEVEL
    assert published.seed == BOOTSTRAP_SEED


def test_the_same_seed_reads_the_same_p_value_twice(corpus: Corpus) -> None:
    second = run_at(corpus.paths, corpus.root / "second")

    first_bootstrap = corpus.run.gating.blend.discovery.bootstrap
    second_bootstrap = second.gating.blend.discovery.bootstrap
    assert first_bootstrap is not None and second_bootstrap is not None
    assert first_bootstrap.p_value == second_bootstrap.p_value
    assert first_bootstrap.seed == BOOTSTRAP_SEED
    assert second.manifest_sha256 == corpus.run.manifest_sha256


def test_the_results_publish_the_row_counts_the_manifest_states(corpus: Corpus) -> None:
    payload = corpus.payload
    legs = read_sample_freeze(corpus.paths["sample"])
    records = read_class_records(corpus.paths["classes"])

    assert payload["row_counts"] == {
        "sample_legs": len(legs),
        "class_records": len(records),
        "market_ladders": len(SERIES) * len(DAYS),
        "event_days_24h": len(DAYS),
        "event_days_36h": len(DAYS),
    }
    assert payload["row_counts"] == manifest_of(corpus.run_root)["row_counts"]
    assert payload["event_day_min_discovery"] == DISCOVERY_N_MIN


def test_the_brier_helper_carries_the_skill_it_was_built_from() -> None:
    model = brier_score([Decimal("0.4"), Decimal("0.6")], [0, 1])
    baseline = brier_score([Decimal("0.5"), Decimal("0.5")], [0, 1])

    brier = forecast_class_run.brier_of(model, baseline, 2)

    assert brier == Brier(n=2, model=model, baseline=baseline, skill=brier_skill(model, baseline))
    assert brier.skill > 0


@needs_tape
def test_the_listed_ladders_over_the_sampled_days_are_six_rungs_wide() -> None:
    days = {leg.event_date for leg in read_sample_freeze(FROZEN_SAMPLE)}

    ladders = read_event_ladders(FROZEN_MARKETS, days)
    widened = read_event_ladders(FROZEN_MARKETS, days | {UNSAMPLED_DAY})

    assert len(ladders) == FROZEN_LADDERS
    assert _rung_counts(ladders) == FROZEN_RUNGS
    assert UNSAMPLED_DAY not in days
    assert len(widened) == WIDENED_LADDERS
    assert _rung_counts(widened) == WIDENED_RUNGS


def _rung_counts(ladders: dict) -> dict[int, int]:
    counts: dict[int, int] = {}
    for tickers in ladders.values():
        counts[len(tickers)] = counts.get(len(tickers), 0) + 1
    return dict(sorted(counts.items()))
