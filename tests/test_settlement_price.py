from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.fee_floor import (
    MAKER_RATE_SOURCE,
    PUBLISHED_MAKER_RATE,
    TICK_CENTS,
    economic_bar_cents_per_contract,
    fee_source,
    published_taker_fee,
)
from bot.lag.placement_grid import CloseSidecar, MarketClose
from bot.lag.r0_universe import Coverage, freeze_universe
from bot.lag.read_rtt import FloorSource, LatencyFloor
from bot.lag.run_manifest import ManifestIncomplete, RunInputs, build_manifest
from bot.lag.settlement_entry import StraddleEntry
from bot.lag.settlement_price import (
    ENTRY_WINDOW,
    SIZE,
    PriceCounts,
    PricedStraddle,
    entry_window,
    fee_cents_per_contract,
    ladder_census,
    price_counts,
    price_of,
    screen_entry_minutes,
)
from bot.lag.tape_stats import (
    BootstrapResult,
    ClusterAggregate,
    cluster_bootstrap,
    evaluate_gate,
)
from bot.lag.tape_studies import (
    LADDER,
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    EvidenceWindow,
    RunScope,
    load_run_scope,
    window_dates,
)
from bot.main import STATIONS
from bot.markets.observation_window import observation_window
from bot.replay.artifacts import LADDER_SCHEMA


BAR_PRICE = Decimal("0.50")
FEE_PER_CONTRACT = Decimal("1.769230769230769230769230769")
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "2026-08-19-settlement"
ACCRUAL_START = datetime(2026, 8, 2, 5, tzinfo=timezone.utc)
ACCRUAL_END = datetime(2026, 8, 16, 8, tzinfo=timezone.utc)
FROZEN_SCOPE = REPO_ROOT / "data" / "tape_studies" / "run_scope_v2"
CLOSES = REPO_ROOT / "data" / "tape_studies" / "closes_v2"
ROOT = "KXHIGHDEN"
TICKER = "KXHIGHDEN-26AUG01-B94.5"
EVENT_DATE = date(2026, 8, 1)
INSTANT = datetime(2026, 8, 1, 20, tzinfo=timezone.utc)
SEED = 20260819
HALF_BOOK = (("0.49", "30"),)
LEVELS_SHORT = tuple((str(Decimal("0.49") - Decimal("0.01") * step), "4") for step in range(6))
LEVELS_DEEP = tuple((str(Decimal("0.49") - Decimal("0.01") * step), "5") for step in range(6))

needs_tape = pytest.mark.skipif(
    not FROZEN_SCOPE.exists(), reason="the recorded tape is not on this host"
)


def bootstrapped(n_clusters: int) -> BootstrapResult:
    return cluster_bootstrap(
        [
            ClusterAggregate(cluster=f"city-{index}", total=Decimal("1"), weight=Decimal("1"))
            for index in range(n_clusters)
        ],
        null_value=Decimal("0"),
        direction="greater",
        resamples=99,
        seed=1,
        ci_level=0.95,
    )


def run_inputs(preregistration: Path, repo: Path, *, price: Decimal | None) -> RunInputs:
    return RunInputs(
        run_id=RUN_ID,
        preregistration=preregistration,
        repo=repo,
        accrual_start=ACCRUAL_START,
        accrual_end=ACCRUAL_END,
        row_counts={"ladder": 1},
        universe=freeze_universe(
            fraction_invalid_max=Decimal("0.5"),
            passing=("KXHIGHDEN",),
            coverage=Coverage(cities=("KXHIGHDEN",), ladder_widths=(6,), in_scope_city_days=14),
        ),
        fee=fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE),
        floor=LatencyFloor(
            source=FloorSource.SIGNED_READ, floor_s=1.0, t_persist_s=2.0, n_usable=30
        ),
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=price,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        bootstrap_seed=SEED,
    )


def test_the_size_is_stated_outside_the_evidence() -> None:
    assert SIZE == Decimal(26)
    assert isinstance(SIZE, Decimal)


def test_the_entry_leg_costs_what_the_published_formula_charges() -> None:
    assert published_taker_fee(Decimal(26), BAR_PRICE) == Decimal("0.46")
    assert fee_cents_per_contract(Decimal(26), BAR_PRICE) == FEE_PER_CONTRACT


def test_the_fee_and_the_tick_are_the_one_leg_bar_itself() -> None:
    assert fee_cents_per_contract(Decimal(26), BAR_PRICE) + TICK_CENTS == (
        economic_bar_cents_per_contract(Decimal(26), BAR_PRICE)
    )
    assert economic_bar_cents_per_contract(Decimal(26), BAR_PRICE) == Decimal(
        "2.769230769230769230769230769"
    )


def test_a_statistic_that_charges_its_own_fee_states_a_bar_of_zero(tmp_path: Path) -> None:
    preregistration = tmp_path / "note.md"
    preregistration.write_text("alpha = 0.0125\n")
    manifest = build_manifest(run_inputs(preregistration, REPO_ROOT, price=SELF_CHARGED_BAR))
    assert manifest.economic_bar_size == Decimal("0")
    assert manifest.economic_bar_price == Decimal("0")
    assert str(manifest.economic_bar_cents_per_contract) == "0"


def test_a_stated_size_with_no_stated_price_aborts(tmp_path: Path) -> None:
    preregistration = tmp_path / "note.md"
    preregistration.write_text("alpha = 0.0125\n")
    with pytest.raises(ManifestIncomplete) as refused:
        build_manifest(run_inputs(preregistration, REPO_ROOT, price=None))
    assert refused.value.fields == ("economic_bar_price",)
    assert "economic_bar_price" in str(refused.value)


def test_a_zero_bar_scores_a_zero_estimate_by_its_strictness() -> None:
    shared = {
        "estimate": Decimal("0"),
        "p_value": 0.001,
        "result": bootstrapped(30),
        "threshold": Decimal("0"),
        "direction": "greater",
        "alpha": 0.0125,
        "n_min": 20,
        "n_unit": "city-days",
        "undecidable": False,
    }
    assert evaluate_gate(**shared, strict=False).economic is True
    assert evaluate_gate(**shared, strict=True).economic is False


@pytest.fixture(scope="module")
def scope() -> RunScope:
    return load_run_scope(FROZEN_SCOPE)


def high_roots() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in CLOSES.glob("KXHIGH*.json")))


def high_days(scope: RunScope) -> tuple[tuple[str, date], ...]:
    roots = set(high_roots())
    return tuple(sorted(key for key in scope.event_days if key[0] in roots))


def straddle_entry(instant: datetime = INSTANT) -> StraddleEntry:
    return StraddleEntry(
        entry_instant=instant,
        strike=94,
        settlement_side="yes",
        ticker=TICKER,
        close_time=instant + timedelta(hours=11),
        entry_at_or_before_close=True,
        instant_class="crossing",
        identifiable_ex_ante=False,
    )


def test_the_evidence_window_is_the_minute_the_price_is_read_in() -> None:
    window = entry_window(straddle_entry())
    assert window == EvidenceWindow(
        series=ROOT,
        event_date=EVENT_DATE,
        start=INSTANT,
        end=INSTANT + timedelta(seconds=60),
    )
    assert ENTRY_WINDOW == timedelta(seconds=60)


@needs_tape
def test_the_observation_window_keeps_no_city_event_day(scope: RunScope) -> None:
    days = high_days(scope)
    assert len(high_roots()) == 20
    assert len(days) == 280
    windows = []
    for series, event_date in days:
        start, end = observation_window(STATIONS[series].timezone, event_date)
        windows.append(EvidenceWindow(series=series, event_date=event_date, start=start, end=end))

    screened = screen_entry_minutes(scope, windows)

    assert screened.candidates == 280
    assert len(screened.kept) == 0
    assert screened.excluded == 280
    assert screened.city_event_days == 280
    assert screened.city_event_days_kept == 0
    assert screened.city_event_days_lost == Decimal(1)


@needs_tape
def test_the_entry_minute_leaves_every_city_event_day_a_surviving_minute(scope: RunScope) -> None:
    windows = []
    for series, event_date in high_days(scope):
        start, end = observation_window(STATIONS[series].timezone, event_date)
        minute = start
        while minute + ENTRY_WINDOW <= end:
            windows.append(
                EvidenceWindow(
                    series=series,
                    event_date=event_date,
                    start=minute,
                    end=minute + ENTRY_WINDOW,
                )
            )
            minute += ENTRY_WINDOW

    screened = screen_entry_minutes(scope, windows)

    assert screened.candidates == 403200
    assert len(screened.kept) == 366334
    assert screened.excluded == 36866
    assert screened.dropped == 36866
    assert screened.city_event_days == 280
    assert screened.city_event_days_kept == 280
    assert screened.city_event_days_lost == Decimal(0)


def price(value: str) -> str:
    return str(Decimal(value).quantize(Decimal("0.0001")))


def depth(value: str) -> str:
    return str(Decimal(value).quantize(Decimal("0.01")))


def book_row(
    row_id: int,
    when: datetime,
    *,
    yes: Sequence[tuple[str, str]] = (("0.50", "30"),),
    no: Sequence[tuple[str, str]] = (("0.50", "30"),),
    ticker: str = TICKER,
) -> dict:
    top_yes, yes_depth = (price(yes[0][0]), depth(yes[0][1])) if yes else (price("0"), depth("0"))
    top_no, no_depth = (price(no[0][0]), depth(no[0][1])) if no else (price("0"), depth("0"))
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": when,
        "ts_ms": row_id * 1000,
        "yes_bid": top_yes,
        "yes_bid_depth": yes_depth,
        "yes_ask": price(str(Decimal("1") - Decimal(top_no))),
        "yes_ask_depth": no_depth,
        "no_bid": top_no,
        "no_bid_depth": no_depth,
        "no_ask": price(str(Decimal("1") - Decimal(top_yes))),
        "no_ask_depth": yes_depth,
        "yes_prices": [price(level) for level, _ in yes],
        "yes_sizes": [depth(size) for _, size in yes],
        "yes_levels": len(yes),
        "no_prices": [price(level) for level, _ in no],
        "no_sizes": [depth(size) for _, size in no],
        "no_levels": len(no),
    }


def ladder_tree(
    tmp_path: Path,
    scope: RunScope,
    roots: Sequence[str],
    *,
    absent: frozenset[str] = frozenset(),
    stubs: frozenset[str] = frozenset(),
) -> Path:
    artifacts = tmp_path / "artifacts"
    (artifacts / LADDER).mkdir(parents=True, exist_ok=True)
    for series in roots:
        if series in absent:
            continue
        rows = [] if series in stubs else [book_row(1, INSTANT, ticker=f"{series}-26AUG01-B94.5")]
        table = pa.Table.from_pylist(rows, schema=LADDER_SCHEMA)
        for day in sorted(
            {
                stamp
                for key, event_day in scope.event_days.items()
                if key[0] == series
                for stamp in window_dates(event_day.window_start, event_day.window_end)
            }
        ):
            pq.write_table(
                table, artifacts / LADDER / f"{series}-{day.isoformat()}-b000001.parquet"
            )
    return artifacts


@needs_tape
def test_the_census_counts_a_ladder_partition_for_every_city_event_day(
    tmp_path: Path, scope: RunScope
) -> None:
    roots = high_roots()
    census = ladder_census(ladder_tree(tmp_path, scope, roots), scope, roots)
    assert len(census) == 280
    assert set(census) == set(high_days(scope))
    assert min(census.values()) > 0


@needs_tape
def test_the_census_aborts_naming_the_first_partition_it_cannot_find(
    tmp_path: Path, scope: RunScope
) -> None:
    roots = high_roots()
    artifacts = ladder_tree(tmp_path, scope, roots, absent=frozenset({"KXHIGHCHI", "KXHIGHDEN"}))
    with pytest.raises(ValueError) as refused:
        ladder_census(artifacts, scope, roots)
    assert "KXHIGHCHI 2026-08-02 has no ladder partition" in str(refused.value)
    assert "KXHIGHDEN" not in str(refused.value)


@needs_tape
def test_the_census_aborts_naming_the_first_partition_that_is_a_stub(
    tmp_path: Path, scope: RunScope
) -> None:
    roots = high_roots()
    artifacts = ladder_tree(tmp_path, scope, roots, stubs=frozenset({"KXHIGHLAX"}))
    with pytest.raises(ValueError) as refused:
        ladder_census(artifacts, scope, roots)
    assert "KXHIGHLAX 2026-08-02 has an empty ladder partition" in str(refused.value)


def book(rows: Sequence[dict]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=LADDER_SCHEMA)


def closes(result: str) -> CloseSidecar:
    market = MarketClose(
        ticker=TICKER,
        event_ticker="KXHIGHDEN-26AUG01",
        close_time=INSTANT + timedelta(hours=11),
        floor_strike=94,
        cap_strike=None,
        strike_type="greater_or_equal",
        status="finalized",
        result=result,
    )
    return CloseSidecar(root=ROOT, markets={TICKER: market}, voided=(), sha256="0" * 64)


def test_the_entry_price_is_the_two_sided_mid_not_the_ask() -> None:
    table = book([book_row(1, INSTANT, yes=HALF_BOOK, no=HALF_BOOK)])

    record = price_of(straddle_entry(), table, closes("yes"))

    assert isinstance(record, PricedStraddle)
    assert record.priced is True
    assert record.entry_price == Decimal("0.50")
    assert Decimal(table.column("yes_ask")[0].as_py()) == Decimal("0.51")
    assert Decimal(0) <= record.entry_price <= Decimal(1)
    assert record.size == SIZE


def test_a_mid_off_the_half_reads_where_the_two_bids_leave_it() -> None:
    table = book([book_row(1, INSTANT, yes=(("0.48", "30"),), no=HALF_BOOK)])

    record = price_of(straddle_entry(), table, closes("yes"))

    assert record.entry_price == Decimal("0.495")


def test_the_settlement_sides_profit_is_signed_by_whether_it_paid() -> None:
    table = book([book_row(1, INSTANT, yes=HALF_BOOK, no=HALF_BOOK)])

    paid = price_of(straddle_entry(), table, closes("yes"))
    unpaid = price_of(straddle_entry(), table, closes("no"))

    assert paid.net_profit_cents == Decimal("47.23076923076923076923076923")
    assert unpaid.net_profit_cents == Decimal("-52.76923076923076923076923077")
    assert paid.entry_fee_cents == FEE_PER_CONTRACT
    assert paid.entry_tick_cents == TICK_CENTS
    assert paid.entry_fee_cents + paid.entry_tick_cents == (
        economic_bar_cents_per_contract(Decimal(26), BAR_PRICE)
    )


def test_a_book_one_sided_at_the_instant_yields_no_price() -> None:
    table = book([book_row(1, INSTANT, yes=HALF_BOOK, no=())])

    record = price_of(straddle_entry(), table, closes("yes"))

    assert record.priced is False
    assert record.entry_price is None
    assert record.entry_fee_cents is None
    assert record.net_profit_cents is None
    assert price_counts([record]).one_sided_n == 1


def test_an_empty_no_book_does_not_read_back_as_a_live_quote() -> None:
    table = book([book_row(1, INSTANT, yes=HALF_BOOK, no=())])

    assert Decimal(table.column("yes_ask")[0].as_py()) == Decimal(1)
    assert Decimal(table.column("yes_ask_depth")[0].as_py()) == Decimal(0)
    assert price_of(straddle_entry(), table, closes("yes")).priced is False


def test_six_levels_that_cannot_fill_the_stated_size_are_censored() -> None:
    table = book([book_row(1, INSTANT, yes=HALF_BOOK, no=LEVELS_SHORT)])

    record = price_of(straddle_entry(), table, closes("yes"))

    assert record.censored is True
    assert record.priced is True
    assert price_counts([record]).censored_n == 1


def test_six_levels_that_fill_the_stated_size_are_not_censored() -> None:
    table = book([book_row(1, INSTANT, yes=HALF_BOOK, no=LEVELS_DEEP)])

    record = price_of(straddle_entry(), table, closes("yes"))

    assert record.censored is False
    assert price_counts([record]).censored_n == 0


def test_the_row_read_is_the_last_one_at_or_before_the_instant() -> None:
    table = book(
        [
            book_row(1, INSTANT - timedelta(seconds=30), yes=(("0.10", "30"),), no=HALF_BOOK),
            book_row(2, INSTANT, yes=(("0.48", "30"),), no=HALF_BOOK),
            book_row(3, INSTANT + timedelta(seconds=1), yes=(("0.90", "30"),), no=HALF_BOOK),
        ]
    )

    record = price_of(straddle_entry(), table, closes("yes"))

    assert record.entry_price == Decimal("0.495")


def test_a_row_for_another_ticker_does_not_price_this_one() -> None:
    table = book(
        [
            book_row(1, INSTANT - timedelta(seconds=30), yes=(("0.48", "30"),), no=HALF_BOOK),
            book_row(2, INSTANT, ticker="KXHIGHDEN-26AUG01-B92.5", yes=HALF_BOOK, no=HALF_BOOK),
        ]
    )

    record = price_of(straddle_entry(), table, closes("yes"))

    assert record.entry_price == Decimal("0.495")


def test_a_ticker_with_no_row_at_or_before_the_instant_is_refused() -> None:
    table = book([book_row(1, INSTANT + timedelta(seconds=1), yes=HALF_BOOK, no=HALF_BOOK)])

    with pytest.raises(ValueError) as refused:
        price_of(straddle_entry(), table, closes("yes"))
    assert TICKER in str(refused.value)


def test_the_counts_split_the_one_sided_reads_from_the_censored_ones() -> None:
    entry = straddle_entry()
    records = [
        price_of(entry, book([book_row(1, INSTANT, yes=HALF_BOOK, no=LEVELS_DEEP)]), closes("yes")),
        price_of(
            entry, book([book_row(1, INSTANT, yes=HALF_BOOK, no=LEVELS_SHORT)]), closes("yes")
        ),
        price_of(entry, book([book_row(1, INSTANT, yes=HALF_BOOK, no=())]), closes("yes")),
    ]

    counts = price_counts(records)

    assert isinstance(counts, PriceCounts)
    assert counts.n == 3
    assert counts.one_sided_n == 1
    assert counts.censored_n == 2
