import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.placement_grid import MarketClose, sidecar_payload
from bot.lag.r0_universe import Coverage, freeze_digest, freeze_universe, write_universe
from bot.lag.fee_floor import MAKER_RATE_SOURCE, PUBLISHED_MAKER_RATE
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import (
    BOOTSTRAP_RESAMPLES,
    MANIFEST_NAME,
    Exemption,
    ExemptionRefused,
    ManifestIncomplete,
    RunInputs,
    build_manifest,
)
from bot.lag.settlement_price import SIZE, PriceCounts, price_counts
from bot.lag.settlement_run import (
    ALPHA_F2,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    CITY_EVENT_DAYS,
    CLOSED,
    COHORT,
    DIRECTION,
    DISCOVERY_N_MIN,
    EXEMPTIONS,
    NULL_VALUE,
    PRE_BOUNDARY_REPORTED_ONLY,
    RESULTS_NAME,
    STRICT,
    UNDERPOWERED,
    Readout,
    SettlementRun,
    bootstrap_of,
    event_dates_of,
    decide,
    execute,
    result_payload,
)
from bot.lag.settlement_source import (
    BOUNDARY_SOURCE,
    SettlementScopeShort,
    boundary_split,
    pull_settlement_sources,
    read_settlement_sources,
)
from bot.lag.settlement_straddle import event_ticker_of
from bot.lag.tape_stats import (
    ALPHA as CAMPAIGN_ALPHA,
    BootstrapResult,
    ClusterAggregate,
    GateVerdict,
    cluster_bootstrap,
    evaluate_gate,
)
from bot.lag.tape_studies import (
    LADDER,
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    TOUCH,
    TRADES,
    assemble_run_inputs,
    load_run_scope,
)
from bot.markets.observation_window import observation_window
from bot.observations.metar import StationObservation
from bot.replay.analysis_stations import HIGH, LOW, in_cohort
from bot.replay.artifacts import LADDER_SCHEMA
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    QUIET_BAND,
    Split,
    write_split,
)
from scripts import f2_report
from scripts.q4_report import write_settles_cache
from tests.test_settlement_source import series_body, transport_for
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    exclusion_row,
    seeded_repo,
    write_preregistration,
    write_rtt_samples,
)


UTC = timezone.utc
SERIES = "KXHIGHDEN"
LOW_SERIES = "KXLOWTDEN"
OTHER_SERIES = "KXHIGHNY"
STATION = "KDEN"
ZONE = "America/Denver"
RUN_ID = "2026-08-20-settlement"
FIRST_DAY = date(2026, 8, 2)
WINDOW = tuple(FIRST_DAY + timedelta(days=offset) for offset in range(14))
DISCOVERY_DAYS = WINDOW[:10]
HOLDOUT_DAYS = WINDOW[10:]
UNNAMED_DAY = FIRST_DAY + timedelta(days=14)
SCOPE_START = datetime(2026, 8, 2, 7, tzinfo=UTC)
SCOPE_END = datetime(2026, 8, 16, 7, tzinfo=UTC)
BOUNDARY_DATE = date(2026, 8, 14)
ON_OR_AFTER = 2
PRE_BOUNDARY_DAYS = tuple(day for day in HOLDOUT_DAYS if day < BOUNDARY_DATE)
LONG_WINDOW = tuple(FIRST_DAY + timedelta(days=offset) for offset in range(22))
LATE_HOLDOUT_WINDOW = (*WINDOW[:10], *WINDOW[12:])
DISCOVERY_ACROSS_BOUNDARY = (*DISCOVERY_DAYS, *WINDOW[12:])

ACIS_F = Decimal("93")
SETTLING_STRIKE = "B92.5"
RISING = ("80", "88", "94")
WIDE = ("80", "88", "96")
CAPPED = ("94", "88", "90")
WIDE_DAY = WINDOW[4]
OPEN_DAY = WINDOW[6]
CROSSING_DAYS = 13
NO_BID = "0.5800"
QUIET_START = datetime(2026, 8, 2, 23, tzinfo=UTC)
QUIET_END = datetime(2026, 8, 2, 23, 30, tzinfo=UTC)

LADDER_ROW_OFFSET = timedelta(hours=7)
FIRST_READING_OFFSET = timedelta(hours=6)

SKEWED_HISTOGRAM = {3: 1, -5: 2, 0: 4}
SKEWED_COVERAGE = {
    ("KDEN", date(2026, 8, 1)): 5,
    ("KAUS", date(2026, 8, 3)): 7,
    ("KNYC", date(2026, 8, 2)): 9,
}
SKEWED_CLOSE_GAPS = {300: 2, 60: 1, 900: 3}
SKEWED_CENSUS = {
    ("KXHIGHLAX", date(2026, 8, 3)): 4,
    ("KXHIGHDEN", date(2026, 8, 1)): 5,
    ("KXHIGHNY", date(2026, 8, 2)): 6,
    ("KXHIGHCHI", date(2026, 8, 5)): 7,
    ("KXHIGHMIA", date(2026, 8, 4)): 8,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCOPE = REPO_ROOT / "data" / "tape_studies" / "run_scope_v2"
CLOSES = REPO_ROOT / "data" / "tape_studies" / "closes_v2"
FROZEN_ROOTS = 20
FROZEN_DAYS = 14

needs_tape = pytest.mark.skipif(
    not (FROZEN_SCOPE.exists() and CLOSES.exists()),
    reason="the recorded tape is not on this host",
)


def reading(stamp: datetime, temp_f: str) -> StationObservation:
    return StationObservation(
        station=STATION,
        valid_time=stamp,
        publication_time=stamp,
        temp_f=Decimal(temp_f),
        is_special=False,
        raw=f"{STATION} {temp_f}F",
        source="tape",
    )


def temps_for(event_date: date) -> tuple[str, ...]:
    if event_date == WIDE_DAY:
        return WIDE
    if event_date == OPEN_DAY:
        return CAPPED
    return RISING


def observations(days: Sequence[date] = WINDOW) -> dict[str, list[StationObservation]]:
    rows = []
    for event_date in days:
        start, _ = observation_window(ZONE, event_date)
        rows.extend(
            reading(start + FIRST_READING_OFFSET + timedelta(hours=index), temp)
            for index, temp in enumerate(temps_for(event_date))
        )
    return {STATION: rows}


def settles(days: Sequence[date] = WINDOW) -> dict[tuple[str, date], Decimal]:
    return {(STATION, event_date): ACIS_F for event_date in days}


def market(event_date: date, suffix: str, **fields: object) -> MarketClose:
    event_ticker = event_ticker_of(SERIES, event_date)
    close = datetime.combine(event_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    return MarketClose(
        ticker=f"{event_ticker}-{suffix}",
        event_ticker=event_ticker,
        close_time=close + timedelta(hours=6, minutes=59),
        status="finalized",
        **fields,
    )


def day_markets(event_date: date) -> list[MarketClose]:
    rows = [
        market(
            event_date,
            "T88",
            floor_strike=None,
            cap_strike=88,
            strike_type="less",
            result="no",
        ),
        market(
            event_date,
            "T95",
            floor_strike=95,
            cap_strike=None,
            strike_type="greater",
            result="no",
        ),
    ]
    for floor in (88, 90, 92, 94):
        rows.append(
            market(
                event_date,
                f"B{floor}.5",
                floor_strike=floor,
                cap_strike=floor + 1,
                strike_type="between",
                result="yes" if floor == 92 else "no",
            )
        )
    return rows


def closes_dir(tmp_path: Path, *, days: Sequence[date] = WINDOW, name: str = "closes") -> Path:
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    markets = [row for event_date in days for row in day_markets(event_date)]
    payload = sidecar_payload(SERIES, markets, [])
    (directory / f"{SERIES}.json").write_text(
        json.dumps({**payload, "sha256": freeze_digest(payload)}, indent=1)
    )
    return directory


def price(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.0001")))


def ladder_row(row_id: int, ticker: str, when: datetime, yes_bid: Decimal) -> dict:
    no_bid = Decimal(NO_BID)
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": when,
        "ts_ms": row_id * 1000,
        "yes_bid": price(yes_bid),
        "yes_bid_depth": "30.00",
        "yes_ask": price(Decimal(1) - no_bid),
        "yes_ask_depth": "30.00",
        "no_bid": price(no_bid),
        "no_bid_depth": "30.00",
        "no_ask": price(Decimal(1) - yes_bid),
        "no_ask_depth": "30.00",
        "yes_prices": [price(yes_bid)],
        "yes_sizes": ["30.00"],
        "yes_levels": 1,
        "no_prices": [price(no_bid)],
        "no_sizes": ["30.00"],
        "no_levels": 1,
    }


def artifacts_dir(
    tmp_path: Path,
    *,
    days: Sequence[date] = WINDOW,
    one_sided: frozenset[date] = frozenset(),
    name: str = "artifacts",
) -> Path:
    root = tmp_path / name
    (root / LADDER).mkdir(parents=True, exist_ok=True)
    for index, event_date in enumerate(days):
        start, _ = observation_window(ZONE, event_date)
        ticker = f"{event_ticker_of(SERIES, event_date)}-{SETTLING_STRIKE}"
        row = ladder_row(
            index + 1,
            ticker,
            start + LADDER_ROW_OFFSET,
            Decimal("0.40") + Decimal("0.01") * index,
        )
        if event_date in one_sided:
            row = row | {
                "no_bid": price(Decimal(0)),
                "no_bid_depth": "0.00",
                "no_prices": [],
                "no_sizes": [],
                "no_levels": 0,
            }
        pq.write_table(
            pa.Table.from_pylist([row], schema=LADDER_SCHEMA),
            root / LADDER / f"{SERIES}-{event_date.isoformat()}-b000001.parquet",
        )
    return root


def event_day_row(series: str, event_date: date, day_index: int, split: str) -> dict:
    start, end = observation_window(ZONE, event_date)
    return {
        "series": series,
        "station": STATION,
        "timezone": ZONE,
        "event_date": event_date,
        "window_start": start,
        "window_end": end,
        "tickers": 6,
        "ladder_rows": 900,
        "first_event_at": start,
        "last_event_at": end,
        "covered": True,
        "evaluable": True,
        "in_scope": True,
        "day_index": day_index,
        "split": split,
        "excluded_us": 0,
        "span_us": 0,
    }


def event_day_table(
    series: Sequence[str], days: Sequence[date], discovery: Sequence[date]
) -> pa.Table:
    rows = [
        event_day_row(
            name,
            event_date,
            index,
            DISCOVERY if event_date in discovery else HOLDOUT,
        )
        for name in series
        for index, event_date in enumerate(days, start=1)
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def scope_dir(
    tmp_path: Path,
    *,
    series: Sequence[str] = (SERIES,),
    days: Sequence[date] = WINDOW,
    discovery: Sequence[date] = DISCOVERY_DAYS,
    bands: Sequence[tuple[str, datetime, datetime]] = ((QUIET_BAND, QUIET_START, QUIET_END),),
    name: str = "scope",
) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                exclusion_row(index, exclusion_class, start, end)
                for index, (exclusion_class, start, end) in enumerate(bands)
            ],
            schema=EXCLUSIONS_SCHEMA,
        ),
        directory / "exclusions.parquet",
    )
    pq.write_table(event_day_table(series, days, discovery), directory / "event_days.parquet")
    write_split(
        directory / "split.json",
        Split(
            cities=tuple(series),
            discovery_days=tuple(day for day in days if day in discovery),
            holdout_days=tuple(day for day in days if day not in discovery),
            boundary_event_day=HOLDOUT_DAYS[0],
            scope_start=SCOPE_START,
            scope_end=SCOPE_END,
        ),
    )
    write_universe(
        directory / "r0_universe.json",
        freeze_universe(
            fraction_invalid_max=Decimal("0.4"),
            passing=tuple(series),
            coverage=Coverage(
                cities=tuple(series),
                ladder_widths=(6,),
                in_scope_city_days=len(series) * len(days),
            ),
        ),
    )
    return directory


def settlement_sidecar(
    tmp_path: Path, *, roots: Sequence[str] = (SERIES,), name: str = "settlement_source.json"
) -> Path:
    path = tmp_path / name
    bodies = {root: series_body(root) for root in roots}
    pull_settlement_sources(
        sorted(bodies), datetime(2026, 8, 19, 17, 30, tzinfo=UTC), path, transport_for(bodies)
    )
    return path


def run_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "artifacts": artifacts_dir(tmp_path),
        "closes": closes_dir(tmp_path),
        "settlement_sources": settlement_sidecar(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
    }


def run_at(
    tmp_path: Path,
    paths: dict[str, Path],
    *,
    cohort: str | None = COHORT,
    days: Sequence[date] = WINDOW,
    run_root: Path | None = None,
) -> SettlementRun:
    return execute(
        run_id=RUN_ID,
        observations=observations(days),
        settles=settles(days),
        floor_source=FloorSource.SIGNED_READ,
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        seed=BOOTSTRAP_SEED,
        run_root=tmp_path / "tape_studies" if run_root is None else run_root,
        cohort=cohort,
        **paths,
    )


def write_arrivals(path: Path) -> Path:
    rows = [
        {
            "station": item.station,
            "source": item.source,
            "obs_time": item.valid_time.isoformat(),
            "tmpf": str(item.temp_f),
            "received_at": item.publication_time.isoformat(),
        }
        for readings in observations().values()
        for item in readings
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def report_args(paths: dict[str, Path], tmp_path: Path, run_root: Path) -> argparse.Namespace:
    settles_path = tmp_path / "acis_settles.json"
    write_settles_cache(settles_path, settles())
    return argparse.Namespace(
        run_id=RUN_ID,
        preregistration=paths["preregistration"],
        repo=paths["repo"],
        run_scope=paths["run_scope"],
        artifacts=paths["artifacts"],
        closes=paths["closes"],
        settlement_sources=paths["settlement_sources"],
        observations=write_arrivals(tmp_path / "arrivals.jsonl"),
        settles=settles_path,
        rtt_samples=paths["rtt_samples"],
        floor_source=FloorSource.SIGNED_READ.value,
        cohort=COHORT,
        run_root=run_root,
    )


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return run_paths(tmp_path)


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "tape_studies"


def manifest_of(run_root: Path) -> dict:
    return json.loads((run_root / RUN_ID / MANIFEST_NAME).read_text())


def flat(total: str, count: int) -> tuple[ClusterAggregate, ...]:
    return tuple(
        ClusterAggregate(cluster=f"{SERIES} {index:04d}", total=Decimal(total), weight=SIZE)
        for index in range(count)
    )


def spread(total: str, count: int) -> tuple[ClusterAggregate, ...]:
    return tuple(
        ClusterAggregate(
            cluster=f"{SERIES} {index:04d}",
            total=Decimal(total) + Decimal(2 * index - (count - 1)) / 2,
            weight=SIZE,
        )
        for index in range(count)
    )


def readout_of(clusters: tuple[ClusterAggregate, ...], *, split: str) -> Readout:
    return Readout(
        split=split,
        clusters=clusters,
        bootstrap=bootstrap_of(clusters, BOOTSTRAP_SEED) if clusters else None,
        priced_n=len(clusters),
        counts=price_counts([]),
    )


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
        null_value=NULL_VALUE,
        direction=DIRECTION,
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


def readout_at(bootstrap: BootstrapResult, *, split: str) -> Readout:
    return Readout(
        split=split,
        clusters=flat("1", bootstrap.n_clusters),
        bootstrap=bootstrap,
        priced_n=bootstrap.n_clusters,
        counts=PriceCounts(n=bootstrap.n_clusters, one_sided_n=0, no_row_n=0, censored_n=0),
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


def test_the_family_alpha_is_the_two_question_correction() -> None:
    assert ALPHA_F2 == 0.025
    assert ALPHA_F2 == 0.05 / 2
    assert CAMPAIGN_ALPHA == 0.0125
    assert ALPHA_F2 != CAMPAIGN_ALPHA
    assert ALPHA_F2 != 0.05 / 3


def test_a_p_value_between_the_two_alphas_is_significant_under_this_ones() -> None:
    bootstrap = bootstrap_at(0.02, estimate="1.5", n_clusters=40)
    holdout = readout_at(bootstrap_at(0.03, estimate="1.0", n_clusters=20), split=HOLDOUT)

    gate = decide(readout_at(bootstrap, split=DISCOVERY), holdout).gate

    assert CAMPAIGN_ALPHA <= bootstrap.p_value < ALPHA_F2
    assert gate is not None
    assert gate.alpha == 0.025
    assert gate.significant is True
    assert regate(gate, bootstrap, alpha=CAMPAIGN_ALPHA).significant is False


def test_a_panel_whose_replicates_carry_no_spread_is_undecidable() -> None:
    bootstrap = bootstrap_at(
        0.0001, estimate="5", n_clusters=40, degenerate=True, replicate_spread=0.0
    )
    holdout = readout_at(bootstrap_at(0.03, estimate="1.0", n_clusters=20), split=HOLDOUT)

    gate = decide(readout_at(bootstrap, split=DISCOVERY), holdout).gate

    assert gate is not None
    assert gate.economic is True
    assert gate.undecidable is True
    assert gate.significant is False
    assert gate.passed is False


def test_the_strict_gate_is_what_puts_a_zero_edge_below_the_zero_bar() -> None:
    bootstrap = bootstrap_at(0.001, estimate="0", n_clusters=40)
    holdout = readout_at(bootstrap_at(0.03, estimate="1.0", n_clusters=20), split=HOLDOUT)

    decision = decide(readout_at(bootstrap, split=DISCOVERY), holdout)
    gate = decision.gate

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


def test_the_interval_the_readout_publishes_covers_ninety_five_percent() -> None:
    clusters = tuple(
        ClusterAggregate(
            cluster=f"{SERIES} {index:04d}", total=Decimal(7 * index - 40), weight=SIZE
        )
        for index in range(12)
    )

    published = bootstrap_of(clusters, BOOTSTRAP_SEED)
    narrower = cluster_bootstrap(
        clusters,
        null_value=NULL_VALUE,
        direction=DIRECTION,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        ci_level=0.90,
    )

    assert CI_LEVEL == 0.95
    assert published.ci_level == 0.95
    assert published.ci_low == pytest.approx(-0.573718, abs=1e-6)
    assert published.ci_high == pytest.approx(0.480769, abs=1e-6)
    assert narrower.estimate == published.estimate
    assert narrower.ci_low == pytest.approx(-0.483974, abs=1e-6)
    assert narrower.ci_high == pytest.approx(0.391026, abs=1e-6)


def test_a_flat_panel_resamples_without_spread() -> None:
    bootstrap = bootstrap_of(flat("26", 30), BOOTSTRAP_SEED)

    assert bootstrap.degenerate is True
    assert bootstrap.replicate_spread == 0.0
    assert bootstrap_of(spread("26", 30), BOOTSTRAP_SEED).degenerate is False


def test_the_discovery_floor_is_thirty_city_event_days() -> None:
    holdout = readout_at(bootstrap_at(0.03, estimate="1.0", n_clusters=20), split=HOLDOUT)
    short = decide(readout_of(spread("26", 29), split=DISCOVERY), holdout)
    full = decide(readout_of(spread("26", 30), split=DISCOVERY), holdout)

    assert DISCOVERY_N_MIN == 30
    assert short.gate is not None and full.gate is not None
    assert short.gate.n == 29
    assert short.gate.powered is False
    assert full.gate.n == 30
    assert full.gate.powered is True
    assert full.gate.n_min == 30
    assert full.gate.n_unit == CITY_EVENT_DAYS


def test_the_holdout_floor_is_half_the_discovery_one() -> None:
    discovery = readout_at(bootstrap_at(0.001, estimate="2.0", n_clusters=40), split=DISCOVERY)
    short = decide(
        discovery, readout_at(bootstrap_at(0.01, estimate="1.5", n_clusters=14), split=HOLDOUT)
    )
    full = decide(
        discovery, readout_at(bootstrap_at(0.01, estimate="1.5", n_clusters=15), split=HOLDOUT)
    )

    assert short.replication is not None and full.replication is not None
    assert short.replication.holdout_n_min == 15
    assert short.replication.powered is False
    assert full.replication.powered is True
    assert full.replication.alpha == 0.05


def test_a_complete_run_writes_its_manifest_and_reads_the_ladder(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)

    manifest = manifest_of(run_root)
    assert manifest["sha256"] == run.manifest_sha256
    assert manifest["cohort"] == HIGH
    assert manifest["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert manifest["bootstrap_seed"] == BOOTSTRAP_SEED
    assert run.decision.verdict == UNDERPOWERED
    assert run.cohort == HIGH


def test_the_manifest_states_the_bar_the_statistic_charges_itself(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    run_at(tmp_path, paths)

    manifest = manifest_of(run_root)
    assert manifest["economic_bar_size"] == "0"
    assert manifest["economic_bar_price"] == "0"
    assert manifest["economic_bar_cents_per_contract"] == "0"
    assert manifest["economic_bar_price_source"] == SELF_CHARGED_BAR_SOURCE


def test_the_manifest_declares_both_exemptions_and_supplies_neither_field(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    run_at(tmp_path, paths)

    manifest = manifest_of(run_root)
    assert [item["field"] for item in manifest["exemptions"]] == [
        "latency_floor",
        "r0_fraction_invalid_max",
    ]
    assert all(item["reason"] for item in manifest["exemptions"])
    assert manifest["r0_fraction_invalid_max"] is None
    assert manifest["r0_universe_sha256"] is None
    assert manifest["latency_floor_source"] is None
    assert {item.field for item in EXEMPTIONS} == {"latency_floor", "r0_fraction_invalid_max"}


def assembled_inputs(paths: dict[str, Path]) -> RunInputs:
    return assemble_run_inputs(
        run_id=RUN_ID,
        preregistration=paths["preregistration"],
        repo=paths["repo"],
        run_scope=paths["run_scope"],
        artifacts=paths["artifacts"],
        kinds=(LADDER,),
        rtt_samples=paths["rtt_samples"],
        floor_source=FloorSource.SIGNED_READ,
        maker_rate=PUBLISHED_MAKER_RATE,
        maker_rate_source=MAKER_RATE_SOURCE,
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        bootstrap_seed=BOOTSTRAP_SEED,
        cohort=COHORT,
    )


def test_declaring_the_exemptions_without_dropping_the_fields_is_refused(
    paths: dict[str, Path],
) -> None:
    inputs = assembled_inputs(paths)

    with pytest.raises(ExemptionRefused, match="r0_fraction_invalid_max"):
        build_manifest(replace(inputs, exemptions=EXEMPTIONS))

    assert inputs.universe is not None
    assert inputs.floor is not None


def test_a_third_field_cannot_be_declared_exempt() -> None:
    with pytest.raises(ExemptionRefused, match="row_counts"):
        Exemption(field="row_counts", reason="the run reads no rows")


def test_a_run_with_no_stated_bar_price_aborts_naming_it(paths: dict[str, Path]) -> None:
    inputs = assembled_inputs(paths)

    with pytest.raises(ManifestIncomplete) as refused:
        build_manifest(
            replace(
                inputs,
                universe=None,
                floor=None,
                exemptions=EXEMPTIONS,
                economic_bar_price=None,
            )
        )

    assert refused.value.fields == ("economic_bar_price",)
    assert "economic_bar_price" in str(refused.value)


def test_a_sidecar_short_of_the_scope_leaves_no_manifest_behind(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["settlement_sources"] = settlement_sidecar(
        tmp_path, roots=(OTHER_SERIES,), name="short.json"
    )

    with pytest.raises(SettlementScopeShort, match=SERIES):
        run_at(tmp_path, paths)

    assert not run_root.exists()


def test_a_ladder_the_census_cannot_find_leaves_no_manifest_behind(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["artifacts"] = artifacts_dir(tmp_path, days=WINDOW[:-1], name="short_artifacts")

    with pytest.raises(ValueError, match="no ladder partition"):
        run_at(tmp_path, paths)

    assert not run_root.exists()


def test_the_manifest_lands_before_the_first_statistic_is_computed(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    days = (*WINDOW, UNNAMED_DAY)
    paths["run_scope"] = scope_dir(tmp_path, days=days, name="wider")
    paths["artifacts"] = artifacts_dir(tmp_path, days=days, name="wider_artifacts")

    with pytest.raises(ValueError, match="close sidecar"):
        run_at(tmp_path, paths, days=days)

    assert manifest_of(run_root)["cohort"] == HIGH


def test_a_scope_spanning_both_ladders_is_refused_without_a_cohort(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    paths["run_scope"] = scope_dir(tmp_path, series=(SERIES, LOW_SERIES), name="both")

    with pytest.raises(ValueError, match="spans both ladders"):
        run_at(tmp_path, paths, cohort=None)

    assert not run_root.exists()
    assert COHORT == HIGH
    assert COHORT != LOW


def test_the_sweep_keeps_one_straddle_for_every_city_event_day(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)

    assert run.sweep.city_event_days == 14
    assert len(run.sweep.straddles) == 14
    assert len(run.sweep.entries) == 14
    assert run.sweep.no_readings == 0
    assert run.sweep.unsettled == 0
    assert run.sweep.no_straddle == 0


def test_the_open_window_never_reaches_the_crossing_class(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)

    assert run.open_at_first_reading_n == 1
    assert run.counts.n == CROSSING_DAYS
    assert run.counts.open_at_first_reading_n == 0
    assert run.counts.at_or_before_close_n == CROSSING_DAYS
    assert run.counts.at_or_before_close_n <= run.counts.n


def test_the_entry_minute_screen_keeps_every_crossing_day(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)

    assert run.screen.candidates == CROSSING_DAYS
    assert run.screen.excluded == 0
    assert run.screen.dropped == 0
    assert run.screen.city_event_days_kept == CROSSING_DAYS


def test_the_wide_delta_day_falsifies_a_rounding_difference(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)

    assert run.deltas.abs_delta_gt_1 == 1
    assert run.deltas.abs_delta_le_1 == 13
    assert run.deltas.abs_delta_gt_1_rows == ((STATION, WIDE_DAY),)
    assert run.deltas.delta_histogram == {1: 13, 3: 1}


def test_the_boundary_split_counts_the_days_read_under_the_moved_source(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)

    assert run.boundary.boundary_date == BOUNDARY_DATE
    assert run.boundary.days_on_or_after_boundary == ON_OR_AFTER
    assert run.boundary.days_before_boundary == 12
    assert run.boundary.boundary_source == BOUNDARY_SOURCE


def test_the_pre_boundary_holdout_stops_at_the_day_the_source_moved(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)
    payload = result_payload(run)
    restricted = payload["holdout_pre_boundary"]
    covered = [item.cluster for item in run.holdout_pre_boundary.clusters]
    settled_on_the_boundary = f"{SERIES} {BOUNDARY_DATE.isoformat()}"

    assert payload["holdout"]["city_event_days"] == 4
    assert payload["holdout"]["straddles"] == 4
    assert restricted["city_event_days"] == 2
    assert restricted["straddles"] == 2
    assert restricted["split"] == HOLDOUT
    assert covered == [f"{SERIES} {day.isoformat()}" for day in PRE_BOUNDARY_DAYS]
    assert settled_on_the_boundary in [item.cluster for item in run.holdout.clusters]
    assert settled_on_the_boundary not in covered
    assert run.holdout_pre_boundary.bootstrap is not None
    assert run.holdout_pre_boundary.bootstrap.seed == BOOTSTRAP_SEED


def test_the_replication_reads_the_whole_holdout_and_not_the_restricted_one(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    paths["run_scope"] = scope_dir(tmp_path, days=LONG_WINDOW, name="long")
    paths["artifacts"] = artifacts_dir(tmp_path, days=LONG_WINDOW, name="long_artifacts")
    paths["closes"] = closes_dir(tmp_path, days=LONG_WINDOW, name="long_closes")

    payload = result_payload(run_at(tmp_path, paths, days=LONG_WINDOW))
    replication = payload["replication"]
    restricted = payload["holdout_pre_boundary"]

    assert replication["holdout_n"] == 12
    assert replication["holdout_estimate"] == "48.48076923076923076923076923"
    assert replication["holdout_estimate"] == payload["holdout"]["net_profit_cents_per_contract"]
    assert restricted["city_event_days"] == 2
    assert restricted["net_profit_cents_per_contract"] == "50.98076923076923076923076923"
    assert restricted["net_profit_cents_per_contract"] != replication["holdout_estimate"]


def test_the_restricted_holdout_publishes_that_it_gates_nothing(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    payload = result_payload(run_at(tmp_path, paths))
    restricted = payload["holdout_pre_boundary"]

    assert restricted["gating"] is False
    assert restricted["reported_only"] == PRE_BOUNDARY_REPORTED_ONLY
    assert "gates nothing" in restricted["reported_only"]
    assert "no multiplicity correction" in restricted["reported_only"]
    assert restricted["boundary_date"] == BOUNDARY_DATE.isoformat()
    assert restricted["boundary_date"] == payload["settlement_source"]["boundary_date"]


def test_a_holdout_that_begins_on_the_boundary_leaves_the_restricted_figure_empty(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    paths["run_scope"] = scope_dir(tmp_path, days=LATE_HOLDOUT_WINDOW, name="late")
    paths["artifacts"] = artifacts_dir(tmp_path, days=LATE_HOLDOUT_WINDOW, name="late_artifacts")

    payload = result_payload(run_at(tmp_path, paths, days=LATE_HOLDOUT_WINDOW))
    restricted = payload["holdout_pre_boundary"]

    assert payload["holdout"]["city_event_days"] == 2
    assert restricted["city_event_days"] == 0
    assert restricted["straddles"] == 0
    assert restricted["priced"] == 0
    assert restricted["net_profit_cents_per_contract"] is None
    assert restricted["ci_low"] is None
    assert restricted["p_value"] is None
    assert restricted["degenerate"] is None
    assert restricted["gating"] is False


def test_the_gate_reads_the_discovery_days_that_sit_past_the_boundary(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    paths["run_scope"] = scope_dir(tmp_path, discovery=DISCOVERY_ACROSS_BOUNDARY, name="spanning")

    run = run_at(tmp_path, paths)
    payload = result_payload(run)
    discovery = payload["discovery"]
    settled_on_the_boundary = f"{SERIES} {BOUNDARY_DATE.isoformat()}"

    assert settled_on_the_boundary in [item.cluster for item in run.discovery.clusters]
    assert payload["holdout"]["city_event_days"] == 2
    assert payload["gate"]["n"] == 11
    assert payload["gate"]["estimate"] == "53.34615384615384615384615385"
    assert payload["gate"]["estimate"] == discovery["net_profit_cents_per_contract"]
    assert discovery["replicate_spread"] == pytest.approx(0.6281, abs=1e-4)


def test_the_manifest_records_the_settlement_source_it_read_under(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    run_at(tmp_path, paths)

    manifest = manifest_of(run_root)
    assert manifest["settlement_source"] == {SERIES: "The Weather Company"}
    assert manifest["boundary_date"] == BOUNDARY_DATE.isoformat()
    assert manifest["boundary_source"] == BOUNDARY_SOURCE
    assert manifest["days_on_or_after_boundary"] == ON_OR_AFTER
    assert manifest["last_updated_ts"][SERIES].startswith(BOUNDARY_DATE.isoformat())


def test_an_unpriced_straddle_leaves_both_sides_of_the_ratio(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    complete = run_at(tmp_path, paths)
    paths["artifacts"] = artifacts_dir(tmp_path, one_sided=frozenset({WINDOW[0]}), name="one_sided")
    thinned = run_at(tmp_path, paths, run_root=tmp_path / "second")
    published = result_payload(thinned)["discovery"]

    assert len(complete.discovery.clusters) == 9
    assert len(thinned.discovery.clusters) == 8
    assert thinned.discovery.counts.one_sided_n == 1
    assert thinned.discovery.counts.censored_n == 1
    assert thinned.discovery.priced_n == 8
    assert thinned.discovery.counts.n == 9
    assert thinned.discovery.bootstrap is not None
    assert thinned.discovery.bootstrap.n_clusters == 8
    assert published["one_sided"] == 1
    assert published["censored"] == 1
    assert published["no_row"] == 0


def test_every_row_is_unidentifiable_when_its_instant_happens(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)

    assert all(entry.identifiable_ex_ante is False for entry in run.sweep.entries)
    assert result_payload(run)["identifiable_ex_ante"] is False


def test_the_results_carry_every_figure_the_report_reads(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    payload = result_payload(run_at(tmp_path, paths))

    assert payload["deltas"]["abs_delta_gt_1"] == 1
    assert payload["settlement_source"]["days_on_or_after_boundary"] == ON_OR_AFTER
    assert payload["settlement_source"]["boundary_source"] == BOUNDARY_SOURCE
    assert payload["settlement_source"]["distinct_notice_bodies"] == 1
    assert payload["entries"]["open_at_first_reading_n"] == 1
    assert payload["entries"]["entry_at_close_n"] == 0
    assert payload["alpha"] == ALPHA_F2
    assert payload["bar"] == "0"
    assert payload["bar_source"] == SELF_CHARGED_BAR_SOURCE
    assert payload["bar_is_strict"] is True
    assert payload["size"] == str(SIZE)
    assert payload["city_event_day_min_discovery"] == DISCOVERY_N_MIN
    assert payload["cohort"] == HIGH
    assert payload["gate"]["threshold"] == "0"
    assert payload["gate"]["alpha"] == ALPHA_F2
    assert payload["discovery"]["split"] == DISCOVERY
    assert payload["holdout"]["split"] == HOLDOUT
    assert payload["discovery"]["ci_level"] == 0.95
    assert payload["holdout"]["ci_level"] == 0.95
    assert CI_LEVEL == 0.95


def test_the_results_publish_the_row_counts_the_manifest_states(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    payload = result_payload(run_at(tmp_path, paths))

    assert payload["row_counts"] == {
        TOUCH: 0,
        LADDER: 14,
        TRADES: 0,
        "exclusions": 1,
        "event_days": 14,
    }
    assert payload["row_counts"] == manifest_of(run_root)["row_counts"]
    assert payload["ladder_rows_per_city_day"] == {
        f"{SERIES} {day.isoformat()}": 1 if day == WINDOW[-1] else 2 for day in WINDOW
    }


def test_the_results_publish_every_map_in_one_order_whatever_order_it_was_built_in(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)
    shuffled = replace(
        run,
        deltas=replace(
            run.deltas, delta_histogram=SKEWED_HISTOGRAM, coverage_minutes=SKEWED_COVERAGE
        ),
        counts=replace(run.counts, close_minus_entry_s=SKEWED_CLOSE_GAPS),
        census=SKEWED_CENSUS,
    )

    payload = result_payload(shuffled)

    assert list(SKEWED_HISTOGRAM) != sorted(SKEWED_HISTOGRAM)
    assert list(SKEWED_COVERAGE) != sorted(SKEWED_COVERAGE)
    assert list(SKEWED_CLOSE_GAPS) != sorted(SKEWED_CLOSE_GAPS)
    assert list(SKEWED_CENSUS) != sorted(SKEWED_CENSUS)
    assert list(payload["deltas"]["delta_histogram"]) == ["-5", "0", "3"]
    assert list(payload["deltas"]["coverage_minutes"]) == [
        "KAUS 2026-08-03",
        "KDEN 2026-08-01",
        "KNYC 2026-08-02",
    ]
    assert list(payload["entries"]["close_minus_entry_s"]) == [60, 300, 900]
    assert payload["cities"] == [
        "KXHIGHCHI",
        "KXHIGHDEN",
        "KXHIGHLAX",
        "KXHIGHMIA",
        "KXHIGHNY",
    ]
    assert list(payload["ladder_rows_per_city_day"]) == [
        "KXHIGHCHI 2026-08-05",
        "KXHIGHDEN 2026-08-01",
        "KXHIGHLAX 2026-08-03",
        "KXHIGHMIA 2026-08-04",
        "KXHIGHNY 2026-08-02",
    ]


def test_the_report_seeds_every_bootstrap_with_the_one_frozen_seed(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    assert f2_report.run(report_args(paths, tmp_path, run_root)) == 0

    results = json.loads((run_root / RUN_ID / RESULTS_NAME).read_text())
    assert BOOTSTRAP_SEED == 20260820
    assert results["bootstrap_seed"] == 20260820
    assert manifest_of(run_root)["bootstrap_seed"] == 20260820


def test_the_same_seed_reads_the_same_p_value_twice(paths: dict[str, Path], tmp_path: Path) -> None:
    first = run_at(tmp_path, paths, run_root=tmp_path / "first")
    second = run_at(tmp_path, paths, run_root=tmp_path / "second")

    assert first.discovery.bootstrap is not None
    assert second.discovery.bootstrap is not None
    assert first.discovery.bootstrap.p_value == second.discovery.bootstrap.p_value
    assert first.discovery.bootstrap.seed == BOOTSTRAP_SEED


def test_a_second_run_under_the_same_id_refuses_to_overwrite(
    paths: dict[str, Path], run_root: Path, tmp_path: Path
) -> None:
    digest = run_at(tmp_path, paths).manifest_sha256

    with pytest.raises(FileExistsError, match=MANIFEST_NAME):
        run_at(tmp_path, paths)

    assert manifest_of(run_root)["sha256"] == digest


def test_the_cluster_unit_weighs_every_straddle_by_the_size_it_was_priced_at(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    run = run_at(tmp_path, paths)

    assert {item.weight for item in run.discovery.clusters} == {SIZE}
    assert [item.cluster for item in run.discovery.clusters] == sorted(
        f"{SERIES} {day.isoformat()}" for day in DISCOVERY_DAYS if day != OPEN_DAY
    )


@needs_tape
def test_the_frozen_scope_spans_both_ladders_and_the_run_names_the_high_one() -> None:
    scope = load_run_scope(FROZEN_SCOPE)
    series = {name for name, _ in scope.event_days}

    roots = in_cohort(series, COHORT)

    assert len(series) == 2 * FROZEN_ROOTS
    assert len(roots) == FROZEN_ROOTS
    assert all(name.startswith("KXHIGH") for name in roots)
    with pytest.raises(ValueError, match="spans both ladders"):
        in_cohort(series, None)


@needs_tape
def test_the_frozen_window_reads_two_days_under_the_moved_source(tmp_path: Path) -> None:
    scope = load_run_scope(FROZEN_SCOPE)
    roots = in_cohort({name for name, _ in scope.event_days}, COHORT)
    dates = event_dates_of(scope, roots)
    provenance = read_settlement_sources(settlement_sidecar(tmp_path, roots=roots))

    split = boundary_split(provenance, dates)

    assert len(dates) == FROZEN_DAYS
    assert dates[0] == FIRST_DAY
    assert dates[-1] == date(2026, 8, 15)
    assert split.boundary_date == BOUNDARY_DATE
    assert split.days_on_or_after_boundary == ON_OR_AFTER
    assert split.days_before_boundary == FROZEN_DAYS - ON_OR_AFTER
