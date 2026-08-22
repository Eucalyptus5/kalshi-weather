import json
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.fee_floor import (
    MAKER_RATE_SOURCE as PUBLISHED_MAKER_RATE_SOURCE,
    PUBLISHED_MAKER_RATE,
)
from bot.lag.fill_convention import NO, NO_CONTRACTS, YES, YES_CONTRACTS
from bot.lag.maker_edge import HORIZONS_S, PRIMARY_HORIZON_S, FillEdge
from bot.lag.maker_edge_run import (
    ALPHA,
    CI_LEVEL,
    CLOSED,
    COHORT,
    DIRECTION,
    MAKER_RATE,
    MAKER_RATE_SOURCE,
    MARKET_DAY_MIN_DISCOVERY,
    MARKET_DAYS,
    NO_ESTIMATE,
    PASS,
    STRICT,
    UNDERPOWERED,
    ZERO_ESTIMATE,
    HorizonReadout,
    Sweep,
    bootstrap_of,
    cluster_aggregates,
    decide,
    execute,
    readout,
    result_payload,
    sweep_fills,
    weighted_estimate,
)
from bot.lag.placement_grid import MarketClose, sidecar_payload
from bot.lag.r0_universe import Coverage, freeze_digest, freeze_universe, write_universe
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME, ManifestIncomplete
from bot.lag.tape_stats import (
    ALPHA as CAMPAIGN_ALPHA,
    BootstrapResult,
    ClusterAggregate,
    GateVerdict,
    evaluate_gate,
    evaluate_holdout,
)
from bot.lag.tape_studies import (
    LADDER,
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    TRADES,
    RunScope,
    load_run_scope,
)
from bot.replay.analysis_stations import HIGH, LOW
from bot.replay.artifacts import LADDER_SCHEMA, TRADES_SCHEMA
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    QUIET_BAND,
    RECORDED_GAP,
    RESUBSCRIBE_BLIND,
    SUBSCRIPTION_WIDE,
    Split,
    write_split,
)
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    SHORT_SAMPLES,
    event_day_row,
    exclusion_row,
    seeded_repo,
    write_partition,
    write_preregistration,
    write_rtt_samples,
)


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parent.parent

SERIES = "KXHIGHDEN"
LOW_SERIES = "KXLOWTDEN"
DISCOVERY_DAY = date(2026, 8, 10)
HOLDOUT_DAY = date(2026, 8, 11)
OPENS = timedelta(hours=6)
# The window runs a full day from its open, so shutting it three minutes after the fills means
# opening it the day before the event day.
TAIL_OPENS = timedelta(hours=7, minutes=3) - timedelta(days=1)
SCOPE_START = datetime(2026, 8, 10, 6, tzinfo=UTC)
SCOPE_END = datetime(2026, 8, 12, 6, tzinfo=UTC)

# parse_ticker refuses a B-form strike that is not a half-integer, so the swept fixture names its
# markets in the T form the parser accepts and only the cluster fixtures carry the B names.
B62 = "KXHIGHDEN-26AUG10-B62"
B64 = "KXHIGHDEN-26AUG10-B64"
T62 = "KXHIGHDEN-26AUG10-T62"
T64 = "KXHIGHDEN-26AUG10-T64"
T66 = "KXHIGHDEN-26AUG10-T66"
NEXT_TICKER = "KXHIGHDEN-26AUG11-T62"

DISCOVERY_CLOSE = datetime(2026, 8, 11, 7, tzinfo=UTC)
HOLDOUT_CLOSE = datetime(2026, 8, 12, 7, tzinfo=UTC)
FIRST_PLACEMENT = datetime(2026, 8, 10, 7, tzinfo=UTC)
NEXT_PLACEMENT = datetime(2026, 8, 11, 7, tzinfo=UTC)

SEED = 20260819
RUN_ID = "2026-08-19-f1"
FREE = Decimal("0")
YES_BID = "0.4000"
STEP_BID = "0.4100"
NO_BID = "0.5800"
YES_TOUCH = "0.4200"
QUIET_START = datetime(2026, 8, 10, 23, tzinfo=UTC)
QUIET_END = datetime(2026, 8, 10, 23, 30, tzinfo=UTC)
MARK_BAND = (
    datetime(2026, 8, 10, 7, 0, 30, tzinfo=UTC),
    datetime(2026, 8, 10, 7, 0, 40, tzinfo=UTC),
)
HOLDOUT_BAND = (
    datetime(2026, 8, 11, 7, 0, 30, tzinfo=UTC),
    datetime(2026, 8, 11, 7, 0, 40, tzinfo=UTC),
)
LISTS_EARLY = datetime(2026, 8, 10, 14, tzinfo=UTC)

QUANTUM = Decimal("0.0001")
FLAT_EDGE = Decimal("0.5000")
GATING_EDGE = Decimal("0.3222")
PLACEMENTS_PER_MARKET = 48
MARKETS_SWEPT = 3

HEADLINE_SPREAD = 0.0251
POOLED_SPREAD = 2.8e-17


def edge(ticker: str, side: str, cents: str, horizon_s: int = PRIMARY_HORIZON_S) -> FillEdge:
    return FillEdge(
        ticker=ticker,
        side=side,
        contracts=YES_CONTRACTS if side == YES else NO_CONTRACTS,
        placement_price=Decimal("0.50"),
        horizon_s=horizon_s,
        edge_cents_per_contract=Decimal(cents),
    )


GATING_EDGES = (
    edge(B62, YES, "-0.40"),
    edge(B62, NO, "0.30"),
    edge(B64, YES, "-0.60"),
    edge(B64, NO, "0.50"),
)
OFF_GATE_EDGES = tuple(
    item
    for horizon_s in (1, 10, 300)
    for item in (
        edge(B62, YES, "-0.40", horizon_s),
        edge(B62, NO, "0.10", horizon_s),
        edge(B64, YES, "-0.60", horizon_s),
        edge(B64, NO, "0.10", horizon_s),
    )
)


def city_event_day(item: FillEdge) -> str:
    return item.ticker.rsplit("-", 1)[0]


def keyed(edges: Sequence[FillEdge], key) -> tuple[ClusterAggregate, ...]:
    totals: dict[str, Decimal] = {}
    weights: dict[str, Decimal] = {}
    for item in edges:
        name = key(item)
        totals[name] = totals.get(name, Decimal(0)) + item.edge_cents_per_contract * item.contracts
        weights[name] = weights.get(name, Decimal(0)) + item.contracts
    return tuple(
        ClusterAggregate(cluster=name, total=totals[name], weight=weights[name])
        for name in sorted(totals)
    )


def flat(total: str, count: int, *, split: str) -> tuple[ClusterAggregate, ...]:
    return tuple(
        ClusterAggregate(
            cluster=f"{SERIES}-{split}-{index:04d}", total=Decimal(total), weight=Decimal("10")
        )
        for index in range(count)
    )


def spread(total: str, count: int, *, split: str) -> tuple[ClusterAggregate, ...]:
    return tuple(
        ClusterAggregate(
            cluster=f"{SERIES}-{split}-{index:04d}",
            total=Decimal(total) + Decimal(2 * index - (count - 1)) / 2,
            weight=Decimal("10"),
        )
        for index in range(count)
    )


def readout_of(
    clusters: tuple[ClusterAggregate, ...], *, split: str, seed: int = SEED
) -> HorizonReadout:
    return HorizonReadout(
        split=split,
        horizon_s=PRIMARY_HORIZON_S,
        clusters=clusters,
        bootstrap=bootstrap_of(clusters, seed) if clusters else None,
        n_fills=2 * len(clusters),
        contracts=sum((item.weight for item in clusters), Decimal(0)),
        modelled=2 * len(clusters),
        dropped=0,
        candidates=2 * len(clusters),
        excluded=0,
        out_of_window=0,
        out_of_scope=0,
        by_class={},
    )


# Reruns the gate off the verdict the run module built, so the flag under test is the only thing
# that moves between the two calls.
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


def ladder_row(
    row_id: int, at: datetime, ticker: str, *, yes_bid: str = YES_BID, no_bid: str = NO_BID
) -> dict:
    empty = Decimal(no_bid) == 0
    no_depth = "0.00" if empty else "5.00"
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at,
        "ts_ms": None,
        "yes_bid": yes_bid,
        "yes_bid_depth": "3.00",
        "yes_ask": str(Decimal("1") - Decimal(no_bid)),
        "yes_ask_depth": no_depth,
        "no_bid": no_bid,
        "no_bid_depth": no_depth,
        "no_ask": str(Decimal("1") - Decimal(yes_bid)),
        "no_ask_depth": "3.00",
        "yes_prices": [yes_bid],
        "yes_sizes": ["3.00"],
        "yes_levels": 1,
        "no_prices": [] if empty else [no_bid],
        "no_sizes": [] if empty else ["5.00"],
        "no_levels": 0 if empty else 1,
    }


def trade_row(
    row_id: int, at: datetime, ticker: str, yes_price: str, count: str, side: str
) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at,
        "ts_ms": None,
        "yes_price": yes_price,
        "no_price": str(Decimal("1") - Decimal(yes_price)),
        "count": count,
        "taker_side": side,
        "trade_id": f"t{row_id}",
    }


# The yes bid ticks up for the minute after the fills and settles back, so the mark-out is zero at
# 1, 10 and 300 seconds and one half cent at 60: a readout taken off any other horizon of this book
# reads a different edge than the gating one.
def book_rows(row_id: int, ticker: str, base: datetime, *, one_sided: bool = False) -> list[dict]:
    minutes = (-1, 0, 1, 3, 6, 10)
    return [
        ladder_row(
            row_id + index,
            base + timedelta(minutes=offset),
            ticker,
            yes_bid=STEP_BID if offset == 1 else YES_BID,
            no_bid="0.0000" if one_sided and offset in (3, 10) else NO_BID,
        )
        for index, offset in enumerate(minutes)
    ]


def fill_trades(row_id: int, ticker: str, base: datetime, *, no_count: str = "6.00") -> list[dict]:
    return [
        trade_row(row_id, base + timedelta(seconds=10), ticker, YES_BID, "4.00", NO),
        trade_row(row_id + 1, base + timedelta(seconds=20), ticker, YES_TOUCH, no_count, YES),
    ]


def artifacts_dir(
    tmp_path: Path,
    *,
    one_sided: bool = False,
    no_side_short: bool = False,
    strays: bool = False,
    name: str = "artifacts",
) -> Path:
    root = tmp_path / name
    strayed = (
        book_rows(31, T66, FIRST_PLACEMENT) + book_rows(41, NEXT_TICKER, LISTS_EARLY)
        if strays
        else []
    )
    write_partition(
        root,
        DISCOVERY_DAY,
        1,
        book_rows(1, T62, FIRST_PLACEMENT, one_sided=one_sided)
        + book_rows(11, T64, FIRST_PLACEMENT, one_sided=one_sided)
        + strayed,
        kind=LADDER,
        schema=LADDER_SCHEMA,
        series=SERIES,
    )
    write_partition(
        root,
        HOLDOUT_DAY,
        1,
        book_rows(21, NEXT_TICKER, NEXT_PLACEMENT, one_sided=one_sided),
        kind=LADDER,
        schema=LADDER_SCHEMA,
        series=SERIES,
    )
    write_partition(
        root,
        DISCOVERY_DAY,
        1,
        fill_trades(101, T62, FIRST_PLACEMENT)
        + fill_trades(111, T64, FIRST_PLACEMENT, no_count="5.00" if no_side_short else "6.00"),
        kind=TRADES,
        schema=TRADES_SCHEMA,
        series=SERIES,
    )
    write_partition(
        root,
        HOLDOUT_DAY,
        1,
        fill_trades(121, NEXT_TICKER, NEXT_PLACEMENT),
        kind=TRADES,
        schema=TRADES_SCHEMA,
        series=SERIES,
    )
    return root


def market(ticker: str, close: datetime) -> MarketClose:
    return MarketClose(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0],
        close_time=close,
        floor_strike=62,
        cap_strike=None,
        strike_type="greater",
        status="finalized",
        result="yes",
    )


def closes_dir(tmp_path: Path, *, roots: Sequence[str] = (SERIES,)) -> Path:
    directory = tmp_path / "closes"
    directory.mkdir(exist_ok=True)
    for root in roots:
        markets = [
            market(T62.replace(SERIES, root), DISCOVERY_CLOSE),
            market(T64.replace(SERIES, root), DISCOVERY_CLOSE),
            market(NEXT_TICKER.replace(SERIES, root), HOLDOUT_CLOSE),
        ]
        payload = sidecar_payload(root, markets, [])
        (directory / f"{root}.json").write_text(
            json.dumps({**payload, "sha256": freeze_digest(payload)}, indent=1)
        )
    return directory


def event_day_table(series: Sequence[str] = (SERIES,), *, opens: timedelta = OPENS) -> pa.Table:
    rows = [
        event_day_row(
            event_date, in_scope=True, split=split, day_index=index, opens=opens, series=name
        )
        for name in series
        for index, (event_date, split) in enumerate(
            ((DISCOVERY_DAY, DISCOVERY), (HOLDOUT_DAY, HOLDOUT)), start=1
        )
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def scope_dir(
    tmp_path: Path,
    *,
    series: Sequence[str] = (SERIES,),
    bands: Sequence[tuple[str, datetime, datetime]] = ((QUIET_BAND, QUIET_START, QUIET_END),),
    opens: timedelta = OPENS,
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
    pq.write_table(event_day_table(series, opens=opens), directory / "event_days.parquet")
    write_split(
        directory / "split.json",
        Split(
            cities=tuple(series),
            discovery_days=(DISCOVERY_DAY,),
            holdout_days=(HOLDOUT_DAY,),
            boundary_event_day=HOLDOUT_DAY,
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
                cities=tuple(series), ladder_widths=(6,), in_scope_city_days=2 * len(series)
            ),
        ),
    )
    return directory


def run_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "artifacts": artifacts_dir(tmp_path),
        "closes": closes_dir(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
    }


def run_at(
    tmp_path: Path, paths: dict[str, Path], *, rate: Decimal = FREE, cohort: str | None = COHORT
):
    return execute(
        run_id=RUN_ID,
        floor_source=FloorSource.SIGNED_READ,
        maker_rate=rate,
        maker_rate_source=MAKER_RATE_SOURCE,
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        cohort=cohort,
        **paths,
    )


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return run_paths(tmp_path)


@pytest.fixture
def scope(tmp_path: Path, paths: dict[str, Path]) -> RunScope:
    return load_run_scope(paths["run_scope"])


@pytest.fixture
def swept(tmp_path: Path, paths: dict[str, Path], scope: RunScope) -> Sweep:
    return sweep_fills(scope, paths["artifacts"], paths["closes"], maker_rate=FREE, cohort=COHORT)


def test_the_cluster_unit_is_the_market_day_and_the_total_is_contract_weighted() -> None:
    clusters = cluster_aggregates(GATING_EDGES)
    bootstrap = bootstrap_of(tuple(clusters), SEED)

    assert [item.cluster for item in clusters] == [B62, B64]
    assert clusters[0].total == Decimal("2.8560")
    assert clusters[1].total == Decimal("5.5840")
    assert [item.weight for item in clusters] == [Decimal("38.36"), Decimal("38.36")]
    assert bootstrap.n_clusters == 2
    assert bootstrap.estimate.quantize(Decimal("0.0001")) == Decimal("0.1100")
    assert bootstrap.estimate > SELF_CHARGED_BAR
    assert bootstrap.degenerate is False
    assert bootstrap.replicate_spread == pytest.approx(HEADLINE_SPREAD, abs=1e-4)


def test_the_bootstrap_measures_against_the_same_zero_the_gate_reads() -> None:
    result = bootstrap_of(cluster_aggregates(GATING_EDGES), SEED)

    assert result.null_value == SELF_CHARGED_BAR
    assert result.direction == DIRECTION
    assert result.resamples == BOOTSTRAP_RESAMPLES
    assert result.ci_level == CI_LEVEL
    assert CI_LEVEL == 0.95


def test_a_size_blind_total_reads_the_other_side_of_the_zero_bar() -> None:
    size_blind = tuple(
        ClusterAggregate(
            cluster=item.cluster,
            total=sum(
                (
                    fill.edge_cents_per_contract
                    for fill in GATING_EDGES
                    if fill.ticker == item.cluster
                ),
                Decimal(0),
            ),
            weight=item.weight,
        )
        for item in cluster_aggregates(GATING_EDGES)
    )

    weighted = weighted_estimate(cluster_aggregates(GATING_EDGES))
    blind = weighted_estimate(size_blind)

    assert weighted.quantize(Decimal("0.0001")) == Decimal("0.1100")
    assert blind.quantize(Decimal("0.0001")) == Decimal("-0.0026")
    assert weighted > SELF_CHARGED_BAR
    assert blind < SELF_CHARGED_BAR


def test_the_city_event_day_keying_moves_the_bootstrap_and_not_the_estimate() -> None:
    market_day = cluster_aggregates(GATING_EDGES)
    city_day = keyed(GATING_EDGES, city_event_day)

    by_market_day = bootstrap_of(tuple(market_day), SEED)
    by_city_day = bootstrap_of(city_day, SEED)

    assert [item.cluster for item in city_day] == ["KXHIGHDEN-26AUG10"]
    assert by_market_day.n_clusters == 2
    assert by_city_day.n_clusters == 1
    assert by_market_day.degenerate is False
    assert by_city_day.degenerate is True
    assert by_market_day.replicate_spread == pytest.approx(HEADLINE_SPREAD, abs=1e-4)
    assert by_city_day.replicate_spread == pytest.approx(POOLED_SPREAD, abs=1e-18)
    # A ratio of sums is invariant to every partition of the fills, so no estimate literal can ever
    # tell the two keyings apart.
    assert by_market_day.estimate == by_city_day.estimate


def test_pooling_the_horizons_reads_a_different_edge_from_the_gating_group() -> None:
    every = GATING_EDGES + OFF_GATE_EDGES
    pooled = cluster_aggregates(every)
    by_horizon = keyed(every, lambda item: f"{item.ticker}-{item.horizon_s}")

    assert [item.weight for item in pooled] == [Decimal("153.44"), Decimal("153.44")]
    assert [item.total for item in pooled] == [Decimal("-4.1760"), Decimal("-8.8640")]
    assert weighted_estimate(pooled).quantize(Decimal("0.0001")) == Decimal("-0.0425")
    assert len(by_horizon) == 8
    assert weighted_estimate(by_horizon).quantize(Decimal("0.0001")) == Decimal("-0.0425")
    assert weighted_estimate(cluster_aggregates(GATING_EDGES)).quantize(
        Decimal("0.0001")
    ) == Decimal("0.1100")


def test_the_family_alpha_is_the_three_question_correction_not_the_closed_campaigns() -> None:
    discovery = readout_of(spread("20", MARKET_DAY_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY)
    holdout = readout_of(spread("20", 100, split=HOLDOUT), split=HOLDOUT)

    decision = decide(discovery, holdout)

    assert ALPHA == 0.05 / 3
    assert ALPHA != CAMPAIGN_ALPHA
    assert CAMPAIGN_ALPHA == 0.0125
    assert decision.gate.alpha == ALPHA
    assert decision.replication.alpha == 0.05


def test_a_p_value_between_the_two_alphas_clears_this_familys_gate() -> None:
    between = 0.014
    discovery = readout_of(spread("20", MARKET_DAY_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY)
    gate = decide(discovery, readout_of(spread("20", 100, split=HOLDOUT), split=HOLDOUT)).gate

    at_family = regate(gate, discovery.bootstrap, p_value=between)
    at_campaign = regate(gate, discovery.bootstrap, p_value=between, alpha=CAMPAIGN_ALPHA)

    assert CAMPAIGN_ALPHA < between < ALPHA
    assert at_family.significant
    assert not at_campaign.significant


def test_the_strict_gate_is_what_puts_a_zero_edge_below_the_zero_bar() -> None:
    balanced = flat("5", 100, split=DISCOVERY) + tuple(
        ClusterAggregate(cluster=f"mirror-{index:04d}", total=Decimal("-5"), weight=Decimal("10"))
        for index in range(100)
    )
    discovery = readout_of(balanced, split=DISCOVERY)
    holdout = readout_of(spread("20", 100, split=HOLDOUT), split=HOLDOUT)

    decision = decide(discovery, holdout)
    gate = decision.gate

    assert gate.estimate == Decimal("0")
    assert gate.threshold == SELF_CHARGED_BAR
    assert gate.n == MARKET_DAY_MIN_DISCOVERY
    assert not discovery.bootstrap.degenerate
    assert not gate.economic
    assert not gate.passed
    assert regate(gate, discovery.bootstrap).economic is False
    assert regate(gate, discovery.bootstrap, strict=False).economic is True
    assert decision.replication is None
    assert decision.skipped == ZERO_ESTIMATE
    assert decision.verdict == CLOSED


def test_a_degenerate_bootstrap_refuses_the_pass_the_same_call_otherwise_reaches() -> None:
    discovery = readout_of(flat("20", MARKET_DAY_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY)
    holdout = readout_of(spread("20", 100, split=HOLDOUT), split=HOLDOUT)

    decision = decide(discovery, holdout)
    gate = decision.gate

    assert discovery.bootstrap.degenerate
    assert gate.undecidable is True
    assert gate.significant is False
    assert gate.passed is False
    assert gate.economic is True
    assert regate(gate, discovery.bootstrap, undecidable=False).passed is True
    assert decision.verdict == CLOSED


@pytest.mark.parametrize(("market_days", "verdict"), [(199, UNDERPOWERED), (200, PASS)])
def test_the_discovery_floor_binds_at_two_hundred_market_days(
    market_days: int, verdict: str
) -> None:
    discovery = readout_of(spread("20", market_days, split=DISCOVERY), split=DISCOVERY)
    holdout = readout_of(spread("20", 100, split=HOLDOUT), split=HOLDOUT)

    decision = decide(discovery, holdout)

    assert MARKET_DAY_MIN_DISCOVERY == 200
    assert MARKET_DAYS == "market-days"
    assert decision.gate.n == market_days
    assert decision.gate.n_min == 200
    assert decision.gate.n_unit == MARKET_DAYS
    assert decision.gate.powered is (market_days == 200)
    assert decision.verdict == verdict


def test_a_holdout_one_market_day_short_is_unpowered_though_it_holds_the_direction() -> None:
    discovery = readout_of(spread("20", MARKET_DAY_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY)
    holdout = readout_of(spread("20", 99, split=HOLDOUT), split=HOLDOUT)

    decision = decide(discovery, holdout)
    replication = decision.replication

    assert replication.holdout_n == 99
    assert replication.holdout_n_min == 100
    assert replication.powered is False
    assert replication.same_sign is True
    assert replication.magnitude is True
    assert replication.replicated is False
    assert decision.verdict == UNDERPOWERED


def test_a_split_with_no_scored_fill_carries_no_gate() -> None:
    decision = decide(readout_of((), split=DISCOVERY), readout_of((), split=HOLDOUT))

    assert decision.gate is None
    assert decision.replication is None
    assert decision.skipped == NO_ESTIMATE
    assert decision.verdict == UNDERPOWERED


def test_a_gate_that_passed_against_an_empty_holdout_replicates_nothing() -> None:
    discovery = readout_of(spread("20", MARKET_DAY_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY)

    decision = decide(discovery, readout_of((), split=HOLDOUT))

    assert decision.gate.passed is True
    assert decision.replication is None
    assert decision.skipped == NO_ESTIMATE
    assert decision.verdict == CLOSED


def test_the_holdout_floor_is_the_shared_half_of_the_discovery_floor() -> None:
    verdict = evaluate_holdout(
        discovery_estimate=Decimal("1"),
        holdout_estimate=Decimal("1"),
        holdout_p_value=0.001,
        holdout_result=bootstrap_of(spread("20", 100, split=HOLDOUT), SEED),
        discovery_n_min=MARKET_DAY_MIN_DISCOVERY,
        alpha=0.05,
        n_unit=MARKET_DAYS,
        undecidable=False,
    )

    assert verdict.holdout_n_min == 100
    assert verdict.powered


def test_the_sweep_scores_one_market_day_per_ticker(swept: Sweep) -> None:
    discovery = swept.tallies[(DISCOVERY, PRIMARY_HORIZON_S)]
    holdout = swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)]

    assert sorted(discovery.totals) == [T62, T64]
    assert sorted(holdout.totals) == [NEXT_TICKER]
    assert swept.yes_fills == 3
    assert swept.no_fills == 3
    assert swept.yes_empty == 0
    assert swept.no_empty == 0
    assert swept.offered == MARKETS_SWEPT * PLACEMENTS_PER_MARKET * 2
    assert swept.unnamed_markets == 0
    assert discovery.n_fills == 4
    assert discovery.contracts == 2 * (YES_CONTRACTS + NO_CONTRACTS)
    assert discovery.weights[T62] == YES_CONTRACTS + NO_CONTRACTS
    # The half-tick capture is 0.50, and the bid step adds half a cent to the filled yes quote at 60
    # seconds while taking the same half cent off the filled no one.
    assert discovery.totals[T62] == Decimal("1.00") * YES_CONTRACTS + Decimal("0") * NO_CONTRACTS
    assert swept.tallies[(DISCOVERY, 1)].totals[T62] == Decimal("0.50") * (
        YES_CONTRACTS + NO_CONTRACTS
    )
    assert swept.market_days == {(SERIES, DISCOVERY_DAY): 2, (SERIES, HOLDOUT_DAY): 1}


def test_a_print_that_only_matches_the_queue_ahead_credits_that_side_nothing(
    tmp_path: Path, paths: dict[str, Path], scope: RunScope
) -> None:
    swept = sweep_fills(
        scope,
        artifacts_dir(tmp_path, no_side_short=True, name="no_side_short"),
        paths["closes"],
        maker_rate=FREE,
        cohort=COHORT,
    )
    discovery = swept.tallies[(DISCOVERY, PRIMARY_HORIZON_S)]

    assert swept.yes_fills == 3
    assert swept.no_fills == 2
    assert swept.offered == MARKETS_SWEPT * PLACEMENTS_PER_MARKET * 2
    assert discovery.weights[T62] == YES_CONTRACTS + NO_CONTRACTS
    assert discovery.weights[T64] == YES_CONTRACTS


def test_a_side_the_book_never_quotes_is_never_offered_and_counted_on_its_own_side(
    tmp_path: Path, paths: dict[str, Path], scope: RunScope
) -> None:
    swept = sweep_fills(
        scope,
        artifacts_dir(tmp_path, one_sided=True, name="one_sided"),
        paths["closes"],
        maker_rate=FREE,
        cohort=COHORT,
    )
    resting = MARKETS_SWEPT * (PLACEMENTS_PER_MARKET - 1)

    assert swept.yes_empty == 0
    assert swept.no_empty == resting
    assert swept.offered == MARKETS_SWEPT * PLACEMENTS_PER_MARKET * 2 - resting


def test_the_sweep_reads_only_the_markets_this_event_day_settles_and_the_pull_named(
    tmp_path: Path, paths: dict[str, Path], scope: RunScope
) -> None:
    swept = sweep_fills(
        scope,
        artifacts_dir(tmp_path, strays=True, name="strays"),
        paths["closes"],
        maker_rate=FREE,
        cohort=COHORT,
    )

    assert swept.market_days == {(SERIES, DISCOVERY_DAY): 3, (SERIES, HOLDOUT_DAY): 1}
    assert swept.unnamed_markets == 1
    assert sorted(swept.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].totals) == [T62, T64]
    assert sorted(swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)].totals) == [NEXT_TICKER]


def test_every_horizon_reads_the_same_seed(swept: Sweep) -> None:
    seeds = {
        horizon_s: readout(
            swept.tallies[(DISCOVERY, horizon_s)],
            split=DISCOVERY,
            horizon_s=horizon_s,
            seed=SEED,
        ).bootstrap.seed
        for horizon_s in HORIZONS_S
    }

    assert seeds == dict.fromkeys(HORIZONS_S, SEED)


def test_the_two_rates_price_the_same_fills_and_only_one_reaches_the_manifest(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    run = run_at(tmp_path, paths, rate=FREE)
    payload = result_payload(run)
    sensitivity = payload["published_rate_sensitivity"]
    manifest = json.loads((tmp_path / "tape_studies" / RUN_ID / MANIFEST_NAME).read_text())

    assert Decimal(payload["discovery"]["edge_cents_per_contract"]).quantize(QUANTUM) == GATING_EDGE
    assert sensitivity["rate"] == str(MAKER_RATE)
    assert sensitivity["gating"] is False
    assert Decimal(sensitivity[f"{DISCOVERY}_edge_cents_per_contract"]) < FLAT_EDGE
    assert sensitivity[f"{DISCOVERY}_market_days"] == 2
    assert payload["gate"]["estimate"] == payload["discovery"]["edge_cents_per_contract"]
    assert Decimal(payload["gate"]["estimate"]) != Decimal(
        sensitivity[f"{DISCOVERY}_edge_cents_per_contract"]
    )
    assert set(sensitivity).isdisjoint({"economic", "significant", "powered", "passed", "bar"})
    assert payload["maker_rate"] == str(FREE)
    assert payload["maker_rate"] != str(MAKER_RATE)
    assert manifest["fee_maker_rate"] == "0"
    assert manifest["fee_maker_rate"] != str(MAKER_RATE)
    assert manifest["economic_bar_size"] == "0"
    assert manifest["economic_bar_cents_per_contract"] == "0"
    assert manifest["cohort"] == HIGH


def test_the_no_mid_drops_are_empirical_and_never_a_missing_quote(
    tmp_path: Path, paths: dict[str, Path], scope: RunScope
) -> None:
    initial = (REPO_ROOT / "alembic" / "versions" / "0001_initial.py").read_text()
    later = (
        REPO_ROOT / "alembic" / "versions" / "0002_orderbook_depth_and_trade_features.py"
    ).read_text()

    two_sided = sweep_fills(
        scope, paths["artifacts"], paths["closes"], maker_rate=FREE, cohort=COHORT
    )
    one_sided = sweep_fills(
        scope,
        artifacts_dir(tmp_path, one_sided=True, name="one_sided"),
        paths["closes"],
        maker_rate=FREE,
        cohort=COHORT,
    )

    assert 'sa.Column("no_bid", sa.Numeric(precision=10, scale=6), nullable=False)' in initial
    assert '"no_bid"' not in later
    assert two_sided.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].dropped == 0
    assert two_sided.tallies[(DISCOVERY, 300)].dropped == 0
    assert one_sided.tallies[(DISCOVERY, 300)].dropped == 4
    assert one_sided.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].dropped == 0


def test_a_horizon_that_modelled_no_fill_reports_no_fraction() -> None:
    empty = readout_of((), split=DISCOVERY)

    assert empty.modelled == 0
    assert empty.no_mid_fraction is None
    assert empty.excluded_fraction is None


def test_the_manifest_lands_before_the_first_statistic(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    run_root = tmp_path / "tape_studies"
    paths["rtt_samples"] = write_rtt_samples(tmp_path / "short.jsonl", SHORT_SAMPLES)

    with pytest.raises(ManifestIncomplete, match="latency_floor"):
        run_at(tmp_path, paths)

    assert not (run_root / RUN_ID / MANIFEST_NAME).exists()
    assert list(run_root.glob("**/*.json")) == []


def test_the_manifest_is_on_disk_before_the_sweep_reads_a_single_market(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    empty = tmp_path / "no_closes"
    empty.mkdir()
    paths["closes"] = empty

    with pytest.raises(FileNotFoundError, match=SERIES):
        run_at(tmp_path, paths)

    assert (tmp_path / "tape_studies" / RUN_ID / MANIFEST_NAME).exists()


def test_one_seed_carries_every_horizon_and_the_holdout(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    run = run_at(tmp_path, paths)

    assert {item.bootstrap.seed for item in run.discovery} == {SEED}
    assert [item.horizon_s for item in run.discovery] == list(HORIZONS_S)
    assert run.holdout.bootstrap.seed == SEED
    assert run.seed == SEED


def test_a_run_naming_the_other_ladder_refuses_rather_than_sweeping_nothing(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="holds no low series"):
        run_at(tmp_path, paths, cohort=LOW)

    assert not (tmp_path / "tape_studies" / RUN_ID / MANIFEST_NAME).exists()


def test_a_two_ladder_scope_needs_the_run_to_name_its_cohort(tmp_path: Path) -> None:
    paths = run_paths(tmp_path)
    paths["run_scope"] = scope_dir(tmp_path, series=(SERIES, LOW_SERIES), name="both")

    with pytest.raises(ValueError, match="spans both ladders"):
        run_at(tmp_path, paths, cohort=None)

    assert COHORT == HIGH


def test_the_run_states_the_zero_bar_and_the_regime_it_prices_under(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    payload = result_payload(run_at(tmp_path, paths, rate=MAKER_RATE))

    assert payload["bar"] == "0"
    assert payload["bar_source"] == SELF_CHARGED_BAR_SOURCE
    assert payload["bar_is_strict"] is True
    assert payload["alpha"] == ALPHA
    assert payload["maker_rate"] == str(MAKER_RATE)
    assert payload["market_day_min_discovery"] == 200
    assert payload["gate"]["threshold"] == "0"
    assert payload["verdict"] == UNDERPOWERED
    assert [item["horizon_s"] for item in payload["horizon_curve"]] == list(HORIZONS_S)
    assert json.loads(json.dumps(payload)) == payload


def test_the_gate_reads_the_sixty_second_horizon_and_not_its_neighbours(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    payload = result_payload(run_at(tmp_path, paths, rate=FREE))
    curve = {item["horizon_s"]: item for item in payload["horizon_curve"]}

    assert payload["primary_horizon_s"] == PRIMARY_HORIZON_S
    assert payload["discovery"] == curve[PRIMARY_HORIZON_S]
    assert payload["gate"]["estimate"] == curve[PRIMARY_HORIZON_S]["edge_cents_per_contract"]
    assert payload["gate"]["n"] == payload["discovery"]["market_days"]
    assert (
        Decimal(curve[PRIMARY_HORIZON_S]["edge_cents_per_contract"]).quantize(QUANTUM)
        == GATING_EDGE
    )
    assert {Decimal(curve[horizon_s]["edge_cents_per_contract"]) for horizon_s in (1, 10, 300)} == {
        FLAT_EDGE
    }
    assert payload["discovery"]["n_fills"] == 4
    assert payload["discovery"]["contracts"] == str(2 * (YES_CONTRACTS + NO_CONTRACTS))
    assert payload["discovery"]["ci_level"] == CI_LEVEL
    assert payload["fills"]["scored_discovery"] == 4
    assert payload["fills"]["scored_holdout"] == 2


def test_the_published_rate_edge_comes_off_the_horizon_the_gate_reads(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    payload = result_payload(run_at(tmp_path, paths, rate=MAKER_RATE))
    sensitivity = payload["published_rate_sensitivity"]
    holdout = payload["holdout"]["edge_cents_per_contract"]
    curve = {
        item["horizon_s"]: item["edge_cents_per_contract"] for item in payload["horizon_curve"]
    }

    assert sensitivity["horizon_s"] == PRIMARY_HORIZON_S
    assert sensitivity["rate"] == "0.0175"
    assert sensitivity["rate_source"] == "published_formula"
    assert sensitivity[f"{DISCOVERY}_edge_cents_per_contract"] == curve[PRIMARY_HORIZON_S]
    assert sensitivity[f"{DISCOVERY}_edge_cents_per_contract"] not in {
        curve[horizon_s] for horizon_s in (1, 10, 300)
    }
    assert sensitivity[f"{HOLDOUT}_edge_cents_per_contract"] == holdout
    assert sensitivity[f"{HOLDOUT}_market_days"] == 1


# Which regime gates and which is only reported is a pre-registration decision, so both stand
# pinned where the run left them and a change to either has to show up as one.
def test_the_gating_rate_and_the_reported_rate_stand_where_the_run_left_them() -> None:
    assert MAKER_RATE == Decimal("0.0175")
    assert MAKER_RATE_SOURCE == "published_formula"
    assert PUBLISHED_MAKER_RATE == Decimal("0.0175")
    assert PUBLISHED_MAKER_RATE_SOURCE == "published_formula"


def test_a_resample_spread_that_only_vanishes_against_its_scale_is_still_degenerate(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    payload = result_payload(run_at(tmp_path, paths, rate=FREE))

    assert payload["discovery"]["market_days"] == 2
    assert payload["discovery"]["replicate_spread"] > 0
    assert payload["discovery"]["degenerate"] is True
    assert payload["gate"]["undecidable"] is True
    assert payload["gate"]["significant"] is False


def test_a_degenerate_holdout_refuses_the_replication_the_run_otherwise_reaches(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    payload = result_payload(run_at(tmp_path, paths, rate=FREE))
    replication = payload["replication"]

    assert payload["holdout"]["market_days"] == 1
    assert payload["holdout"]["degenerate"] is True
    assert replication["holdout_n"] == 1
    assert replication["same_sign"] is True
    assert replication["magnitude"] is True
    assert replication["undecidable"] is True
    assert replication["significant"] is False
    assert replication["replicated"] is False
    assert payload["replication_skipped"] == ""


def test_the_manifest_names_the_seed_the_bootstrap_drew_and_the_sources_it_priced_under(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    run = run_at(tmp_path, paths, rate=MAKER_RATE)
    payload = result_payload(run)
    manifest = json.loads((tmp_path / "tape_studies" / RUN_ID / MANIFEST_NAME).read_text())

    assert manifest["bootstrap_seed"] == SEED
    assert manifest["bootstrap_seed"] == run.primary.bootstrap.seed
    assert manifest["bootstrap_seed"] == run.holdout.bootstrap.seed
    assert manifest["fee_maker_rate"] == str(MAKER_RATE)
    assert manifest["fee_maker_rate_source"] == MAKER_RATE_SOURCE
    assert manifest["economic_bar_price_source"] == SELF_CHARGED_BAR_SOURCE
    assert payload["manifest"] == str(tmp_path / "tape_studies" / RUN_ID / MANIFEST_NAME)
    assert Path(payload["manifest"]).exists()


def test_a_band_over_the_mark_out_drops_the_fill_and_names_the_class_that_dropped_it(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    paths["run_scope"] = scope_dir(
        tmp_path,
        bands=((QUIET_BAND, *MARK_BAND), (RECORDED_GAP, *HOLDOUT_BAND)),
        name="banded",
    )
    payload = result_payload(run_at(tmp_path, paths, rate=FREE))
    curve = {item["horizon_s"]: item for item in payload["horizon_curve"]}
    sensitivity = payload["published_rate_sensitivity"]

    assert curve[1]["candidates"] == 4
    assert curve[1]["excluded"] == 0
    assert Decimal(curve[1]["excluded_fraction"]) == 0
    assert curve[10]["excluded"] == 2
    assert curve[PRIMARY_HORIZON_S]["candidates"] == 4
    assert curve[PRIMARY_HORIZON_S]["excluded"] == 4
    assert Decimal(curve[PRIMARY_HORIZON_S]["excluded_fraction"]) == 1
    assert curve[PRIMARY_HORIZON_S]["out_of_window"] == 0
    assert curve[PRIMARY_HORIZON_S]["out_of_scope"] == 0
    assert curve[PRIMARY_HORIZON_S]["by_class"][QUIET_BAND] == 4
    assert curve[PRIMARY_HORIZON_S]["by_class"][RECORDED_GAP] == 0
    assert payload["holdout"]["excluded"] == 2
    assert payload["holdout"]["by_class"][RECORDED_GAP] == 2
    assert payload["exclusions"]["candidates"] == 6
    assert payload["exclusions"]["excluded"] == 6
    assert Decimal(payload["exclusions"]["excluded_fraction"]) == 1
    assert payload["exclusions"]["by_class"] == {
        QUIET_BAND: 4,
        RECORDED_GAP: 2,
        RESUBSCRIBE_BLIND: 0,
        SUBSCRIPTION_WIDE: 0,
    }
    assert payload["discovery"]["edge_cents_per_contract"] is None
    assert sensitivity[f"{DISCOVERY}_edge_cents_per_contract"] is None
    assert sensitivity[f"{DISCOVERY}_market_days"] == 0
    assert payload["gate"] is None
    assert payload["replication_skipped"] == NO_ESTIMATE
    assert payload["verdict"] == UNDERPOWERED


def test_a_mark_out_running_past_the_day_window_is_out_of_window_and_not_excluded(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    paths["run_scope"] = scope_dir(tmp_path, opens=TAIL_OPENS, name="tail")

    payload = result_payload(run_at(tmp_path, paths, rate=FREE))
    curve = {item["horizon_s"]: item for item in payload["horizon_curve"]}

    assert curve[300]["candidates"] == 4
    assert curve[300]["out_of_window"] == 4
    assert curve[300]["excluded"] == 0
    assert curve[300]["out_of_scope"] == 0
    assert curve[300]["edge_cents_per_contract"] is None
    assert curve[PRIMARY_HORIZON_S]["out_of_window"] == 0
    assert payload["holdout"]["out_of_window"] == 0
    assert Decimal(payload["discovery"]["edge_cents_per_contract"]).quantize(QUANTUM) == GATING_EDGE
