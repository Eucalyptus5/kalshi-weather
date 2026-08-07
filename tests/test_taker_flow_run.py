from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.r0_universe import Coverage, freeze_universe, write_universe
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import MANIFEST_NAME
from bot.lag.taker_flow import (
    CENT_BAR,
    HORIZONS_S,
    PRIMARY_HORIZON_S,
    PRINT_MIN_DISCOVERY,
    FlowCounts,
    HorizonResult,
)
from bot.lag.taker_flow_run import (
    CLOSED,
    ECONOMIC_BAR_PRICE,
    ECONOMIC_BAR_PRICE_SOURCE,
    ECONOMIC_BAR_SIZE,
    NO_ESTIMATE,
    PASS,
    PRINT_MIN_HOLDOUT,
    TICKER_MIN_DISCOVERY,
    TICKERS,
    TOUCH_COLUMNS,
    UNDECIDABLE,
    UNDERPOWERED,
    ZERO_ESTIMATE,
    HorizonReadout,
    Sweep,
    Tally,
    bootstrap_of,
    decide,
    execute,
    read_touch,
    readout,
    result_payload,
    sweep_prints,
)
from bot.lag.tape_stats import ClusterAggregate, evaluate_holdout
from bot.lag.tape_studies import RunScope, load_run_scope
from bot.replay.analysis_stations import HIGH
from bot.replay.artifacts import TOUCH_SCHEMA, TRADES_SCHEMA
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    QUIET_BAND,
    RESUBSCRIBE_BLIND,
    Split,
    write_split,
)
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    event_day_row,
    exclusion_row,
    seeded_repo,
    write_partition,
    write_preregistration,
    write_rtt_samples,
)


UTC = timezone.utc
MICROSECOND = timedelta(microseconds=1)
OPENS = timedelta(hours=6)

SERIES = "KXHIGHDEN"
LOW_SERIES = "KXLOWTDEN"
DISCOVERY_DAY = date(2026, 7, 18)
HOLDOUT_DAY = date(2026, 7, 19)
LATE_DAY = date(2026, 7, 20)
STRANGER_DAY = date(2026, 7, 25)
SCOPE_START = datetime(2026, 7, 18, 6, tzinfo=UTC)
SCOPE_END = datetime(2026, 7, 20, 6, tzinfo=UTC)

DAY_TICKER = "KXHIGHDEN-26JUL18-T70"
NEXT_TICKER = "KXHIGHDEN-26JUL19-T70"
STRANGER_TICKER = "KXHIGHDEN-26JUL25-T70"

QUIET_START = datetime(2026, 7, 18, 7, tzinfo=UTC)
QUIET_END = datetime(2026, 7, 18, 9, tzinfo=UTC)
BLINK = datetime(2026, 7, 18, 12, tzinfo=UTC)

SEED = 20260812
RUN_ID = "2026-08-17-q3"
POWERED = PRINT_MIN_DISCOVERY + 1_000
HOLDOUT_POWERED = 3_000
TICKER_MIN_HOLDOUT = (TICKER_MIN_DISCOVERY + 1) // 2
THIN_TICKERS = 3


def when(day: date, hour: int, minute: int, second: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=UTC)


def touch(
    row_id: int, ticker: str, received_at: datetime, ts_ms: int, yes_bid: str, no_bid: str
) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": received_at,
        "ts_ms": ts_ms,
        "yes_bid": yes_bid,
        "yes_bid_depth": "10",
        "yes_ask": str(Decimal("1") - Decimal(no_bid)),
        "yes_ask_depth": "10",
        "no_bid": no_bid,
        "no_bid_depth": "10",
        "no_ask": str(Decimal("1") - Decimal(yes_bid)),
        "no_ask_depth": "10",
    }


def trade(
    row_id: int,
    ticker: str,
    received_at: datetime,
    ts_ms: int,
    *,
    side: str = "yes",
    trade_id: str | None = None,
    count: str = "10",
) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": received_at,
        "ts_ms": ts_ms,
        "yes_price": "0.50",
        "no_price": "0.50",
        "count": count,
        "taker_side": side,
        "trade_id": f"t{row_id}" if trade_id is None else trade_id,
    }


TOUCH_DISCOVERY_DAY = [
    touch(1, DAY_TICKER, when(DISCOVERY_DAY, 5, 59, 59), 8500, "0.50", "0.48"),
    touch(2, DAY_TICKER, when(DISCOVERY_DAY, 6, 0, 30), 8600, "0.50", "0.48"),
    touch(3, DAY_TICKER, when(DISCOVERY_DAY, 11, 59, 58), 9000, "0.40", "0.58"),
    touch(4, DAY_TICKER, when(DISCOVERY_DAY, 12, 0, 0), 10000, "0.50", "0.48"),
    touch(5, DAY_TICKER, when(DISCOVERY_DAY, 12, 5, 10), 11000, "0.50", "0.48"),
    touch(6, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 0), 12000, "0.40", "0.58"),
    touch(7, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 2), 13000, "0.50", "0.48"),
    touch(8, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 3), 14000, "0.51", "0.47"),
    touch(9, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 12), 15000, "0.52", "0.46"),
    touch(10, DAY_TICKER, when(DISCOVERY_DAY, 18, 1, 2), 16000, "0.55", "0.43"),
    touch(11, DAY_TICKER, when(DISCOVERY_DAY, 18, 5, 2), 17000, "0.60", "0.38"),
    touch(12, DAY_TICKER, when(DISCOVERY_DAY, 18, 10, 0), 18000, "0.60", "0.38"),
]

TOUCH_HOLDOUT_DAY = [
    touch(30, DAY_TICKER, when(HOLDOUT_DAY, 0, 0, 2), 21000, "0.50", "0.48"),
    touch(31, DAY_TICKER, when(HOLDOUT_DAY, 0, 0, 3), 21500, "0.51", "0.47"),
    touch(32, DAY_TICKER, when(HOLDOUT_DAY, 0, 0, 12), 22000, "0.52", "0.46"),
    touch(33, DAY_TICKER, when(HOLDOUT_DAY, 0, 1, 2), 23000, "0.55", "0.43"),
    touch(34, DAY_TICKER, when(HOLDOUT_DAY, 0, 5, 2), 24000, "0.60", "0.38"),
    touch(35, DAY_TICKER, when(HOLDOUT_DAY, 0, 10, 0), 25000, "0.60", "0.38"),
    touch(40, NEXT_TICKER, when(HOLDOUT_DAY, 18, 0, 0), 30000, "0.40", "0.58"),
    touch(41, NEXT_TICKER, when(HOLDOUT_DAY, 18, 0, 2), 31000, "0.50", "0.48"),
    touch(42, NEXT_TICKER, when(HOLDOUT_DAY, 18, 0, 3), 32000, "0.51", "0.47"),
    touch(43, NEXT_TICKER, when(HOLDOUT_DAY, 18, 0, 12), 33000, "0.52", "0.46"),
    touch(44, NEXT_TICKER, when(HOLDOUT_DAY, 18, 1, 2), 34000, "0.55", "0.43"),
    touch(45, NEXT_TICKER, when(HOLDOUT_DAY, 18, 5, 2), 35000, "0.60", "0.38"),
    touch(46, NEXT_TICKER, when(HOLDOUT_DAY, 18, 10, 0), 36000, "0.60", "0.38"),
]

TOUCH_LATE_DAY = [
    touch(60, NEXT_TICKER, when(LATE_DAY, 5, 59, 30), 40000, "0.50", "0.48"),
    touch(61, NEXT_TICKER, when(LATE_DAY, 6, 1, 0), 41000, "0.50", "0.48"),
]

TRADES_DISCOVERY_DAY = [
    trade(100, DAY_TICKER, when(DISCOVERY_DAY, 6, 0, 1), 8000),
    trade(101, DAY_TICKER, when(DISCOVERY_DAY, 11, 59, 59), 9500),
    trade(102, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 1), 12500),
    trade(103, STRANGER_TICKER, when(DISCOVERY_DAY, 18, 0, 1), 12500),
    trade(104, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 1), 12500, side=""),
    trade(105, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 1), 12500, trade_id="t102"),
    trade(106, DAY_TICKER, when(DISCOVERY_DAY, 23, 59, 0), 20000),
]

TRADES_HOLDOUT_DAY = [trade(110, NEXT_TICKER, when(HOLDOUT_DAY, 18, 0, 1), 30500)]
TRADES_LATE_DAY = [trade(120, NEXT_TICKER, when(LATE_DAY, 5, 59, 0), 39000)]

FRACTIONAL_TRADES_DAY = [
    trade(140, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 1), 12500, count="1.24"),
    trade(141, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 2), 12600, count="4"),
]

EARLY_TOUCH_DAY = [
    touch(90, NEXT_TICKER, when(DISCOVERY_DAY, 18, 0, 0), 12000, "0.50", "0.48"),
    touch(91, NEXT_TICKER, when(DISCOVERY_DAY, 18, 0, 2), 13000, "0.50", "0.48"),
    touch(92, NEXT_TICKER, when(DISCOVERY_DAY, 18, 1, 2), 16000, "0.55", "0.43"),
    touch(93, NEXT_TICKER, when(DISCOVERY_DAY, 18, 10, 0), 18000, "0.60", "0.38"),
]
EARLY_TRADES_DAY = [trade(160, NEXT_TICKER, when(DISCOVERY_DAY, 18, 0, 1), 12500)]

BOOKLESS_TOUCH_DAY = [touch(70, NEXT_TICKER, when(DISCOVERY_DAY, 18, 0, 0), 12000, "0.50", "0.48")]
BOOKLESS_TRADES_DAY = [
    trade(130, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 1), 12500),
    trade(131, DAY_TICKER, when(DISCOVERY_DAY, 18, 0, 2), 12600),
]

VIOLATION_DISCOVERY_DAY = [
    touch(1, DAY_TICKER, when(DISCOVERY_DAY, 12, 0, 0), 1000, "0.50", "0.48"),
    touch(2, DAY_TICKER, when(DISCOVERY_DAY, 12, 0, 10), 3000, "0.50", "0.48"),
    touch(3, DAY_TICKER, when(DISCOVERY_DAY, 12, 0, 20), 2000, "0.50", "0.48"),
    touch(4, DAY_TICKER, when(DISCOVERY_DAY, 12, 0, 30), 4000, "0.50", "0.48"),
]
VIOLATION_HOLDOUT_DAY = [
    touch(5, DAY_TICKER, when(HOLDOUT_DAY, 12, 0, 0), 3500, "0.50", "0.48"),
    touch(6, DAY_TICKER, when(HOLDOUT_DAY, 12, 0, 10), 6000, "0.50", "0.48"),
    touch(7, DAY_TICKER, when(HOLDOUT_DAY, 12, 0, 20), 5000, "0.50", "0.48"),
    touch(8, DAY_TICKER, when(HOLDOUT_DAY, 12, 0, 30), 7000, "0.50", "0.48"),
]


def exclusion_table() -> pa.Table:
    rows = [
        exclusion_row(0, QUIET_BAND, QUIET_START, QUIET_END),
        exclusion_row(1, RESUBSCRIBE_BLIND, BLINK, BLINK + MICROSECOND),
    ]
    return pa.Table.from_pylist(rows, schema=EXCLUSIONS_SCHEMA)


def event_day_table() -> pa.Table:
    rows = [
        event_day_row(DISCOVERY_DAY, in_scope=True, split=DISCOVERY, day_index=1, opens=OPENS),
        event_day_row(HOLDOUT_DAY, in_scope=True, split=HOLDOUT, day_index=2, opens=OPENS),
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def scope_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "scope"
    directory.mkdir()
    pq.write_table(exclusion_table(), directory / "exclusions.parquet")
    pq.write_table(event_day_table(), directory / "event_days.parquet")
    write_split(
        directory / "split.json",
        Split(
            cities=(SERIES,),
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
            passing=(SERIES,),
            coverage=Coverage(cities=(SERIES,), ladder_widths=(6,), in_scope_city_days=2),
        ),
    )
    return directory


def both_ladder_scope_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "both_scope"
    directory.mkdir()
    pq.write_table(exclusion_table(), directory / "exclusions.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                event_day_row(
                    event_date,
                    in_scope=True,
                    split=split,
                    day_index=index,
                    opens=OPENS,
                    series=series,
                )
                for series in (SERIES, LOW_SERIES)
                for index, (event_date, split) in enumerate(
                    ((DISCOVERY_DAY, DISCOVERY), (HOLDOUT_DAY, HOLDOUT)), start=1
                )
            ],
            schema=EVENT_DAYS_SCHEMA,
        ),
        directory / "event_days.parquet",
    )
    write_split(
        directory / "split.json",
        Split(
            cities=(SERIES, LOW_SERIES),
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
            passing=(SERIES, LOW_SERIES),
            coverage=Coverage(
                cities=(SERIES, LOW_SERIES), ladder_widths=(6,), in_scope_city_days=4
            ),
        ),
    )
    return directory


def run_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
    }


def artifacts_dir(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    write_partition(root, DISCOVERY_DAY, 1, TOUCH_DISCOVERY_DAY)
    write_partition(root, HOLDOUT_DAY, 1, TOUCH_HOLDOUT_DAY)
    write_partition(root, LATE_DAY, 1, TOUCH_LATE_DAY)
    for day, rows in (
        (DISCOVERY_DAY, TRADES_DISCOVERY_DAY),
        (HOLDOUT_DAY, TRADES_HOLDOUT_DAY),
        (LATE_DAY, TRADES_LATE_DAY),
    ):
        write_partition(root, day, 1, rows, kind="trades", schema=TRADES_SCHEMA)
    return root


def bookless_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "bookless"
    write_partition(root, DISCOVERY_DAY, 1, BOOKLESS_TOUCH_DAY)
    write_partition(
        root, DISCOVERY_DAY, 1, BOOKLESS_TRADES_DAY, kind="trades", schema=TRADES_SCHEMA
    )
    return root


def early_listing_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "early_listing"
    write_partition(root, DISCOVERY_DAY, 1, EARLY_TOUCH_DAY)
    write_partition(root, DISCOVERY_DAY, 1, EARLY_TRADES_DAY, kind="trades", schema=TRADES_SCHEMA)
    return root


def discovery_only_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "discovery_only"
    write_partition(root, DISCOVERY_DAY, 1, TOUCH_DISCOVERY_DAY)
    write_partition(root, HOLDOUT_DAY, 1, TOUCH_HOLDOUT_DAY)
    write_partition(
        root, DISCOVERY_DAY, 1, TRADES_DISCOVERY_DAY, kind="trades", schema=TRADES_SCHEMA
    )
    return root


def fractional_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "fractional"
    write_partition(root, DISCOVERY_DAY, 1, TOUCH_DISCOVERY_DAY)
    write_partition(root, HOLDOUT_DAY, 1, TOUCH_HOLDOUT_DAY)
    write_partition(
        root, DISCOVERY_DAY, 1, FRACTIONAL_TRADES_DAY, kind="trades", schema=TRADES_SCHEMA
    )
    return root


def violation_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "violations"
    write_partition(root, DISCOVERY_DAY, 1, VIOLATION_DISCOVERY_DAY)
    write_partition(root, HOLDOUT_DAY, 1, VIOLATION_HOLDOUT_DAY)
    write_partition(
        root,
        DISCOVERY_DAY,
        1,
        [trade(200, DAY_TICKER, when(DISCOVERY_DAY, 12, 0, 5), 500)],
        kind="trades",
        schema=TRADES_SCHEMA,
    )
    write_partition(
        root,
        HOLDOUT_DAY,
        1,
        [trade(210, DAY_TICKER, when(HOLDOUT_DAY, 12, 0, 5), 3200)],
        kind="trades",
        schema=TRADES_SCHEMA,
    )
    return root


def clusters_of(*totals: tuple[str, str, str]) -> tuple[ClusterAggregate, ...]:
    return tuple(
        ClusterAggregate(cluster=name, total=Decimal(total), weight=Decimal(weight))
        for name, total, weight in totals
    )


def spread(total: str, count: int, *, split: str) -> tuple[ClusterAggregate, ...]:
    return tuple(
        ClusterAggregate(
            cluster=f"{SERIES}-{split}-{index:03d}",
            total=Decimal(total) + Decimal(2 * index - (count - 1)) / 2,
            weight=Decimal("10"),
        )
        for index in range(count)
    )


def flat(total: str, count: int, *, split: str) -> tuple[ClusterAggregate, ...]:
    return tuple(
        ClusterAggregate(
            cluster=f"{SERIES}-{split}-{index:03d}", total=Decimal(total), weight=Decimal("10")
        )
        for index in range(count)
    )


def readout_of(
    clusters: tuple[ClusterAggregate, ...], *, split: str, n_prints: int
) -> HorizonReadout:
    result = HorizonResult(
        horizon_s=PRIMARY_HORIZON_S,
        split=split,
        clusters=clusters,
        n_prints=n_prints,
        contracts=sum((item.weight for item in clusters), Decimal(0)),
        counts=FlowCounts(),
    )
    return HorizonReadout(
        result=result,
        bootstrap=bootstrap_of(clusters, SEED),
        candidates=n_prints,
        excluded=0,
        out_of_window=0,
        by_class={},
    )


@pytest.fixture
def scope(tmp_path: Path) -> RunScope:
    return load_run_scope(scope_dir(tmp_path))


@pytest.fixture
def swept(tmp_path: Path, scope: RunScope) -> Sweep:
    return sweep_prints(scope, artifacts_dir(tmp_path))


def test_the_hygiene_counts_come_off_the_whole_root(swept: Sweep) -> None:
    assert swept.empty_side == 1
    assert swept.duplicates == 1
    assert swept.out_of_scope == 1
    assert swept.in_scope == {DISCOVERY: 4, HOLDOUT: 2}


def test_a_print_outside_the_frozen_scope_never_reaches_a_cluster(swept: Sweep) -> None:
    clusters = swept.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].clusters()

    assert [item.cluster for item in clusters] == [DAY_TICKER]
    assert STRANGER_TICKER not in {item.cluster for item in clusters}
    assert (SERIES, STRANGER_DAY) not in swept.tickers


def test_the_two_splits_carry_their_own_event_days(swept: Sweep) -> None:
    discovery = swept.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].clusters()
    holdout = swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)].clusters()

    assert [item.cluster for item in discovery] == [DAY_TICKER]
    assert [item.cluster for item in holdout] == [NEXT_TICKER]
    assert swept.tickers == {
        (SERIES, DISCOVERY_DAY): frozenset({DAY_TICKER}),
        (SERIES, HOLDOUT_DAY): frozenset({NEXT_TICKER}),
    }


def test_a_window_over_an_exclusion_is_dropped_whole_and_counted_by_class(swept: Sweep) -> None:
    tally = swept.tallies[(DISCOVERY, PRIMARY_HORIZON_S)]

    assert tally.candidates == 4
    assert tally.excluded == 1
    assert tally.by_class[RESUBSCRIBE_BLIND] == 1
    assert tally.by_class[QUIET_BAND] == 0
    assert tally.n_prints == 2
    assert tally.contracts == Decimal("20")


def test_a_window_opening_before_its_event_day_is_dropped_as_out_of_window(
    swept: Sweep, scope: RunScope
) -> None:
    assert scope.event_days[(SERIES, DISCOVERY_DAY)].window_start == when(DISCOVERY_DAY, 6, 0, 0)
    for horizon_s in HORIZONS_S:
        assert swept.tallies[(DISCOVERY, horizon_s)].out_of_window == 1


def test_a_window_running_past_its_event_day_is_dropped_as_out_of_window(
    swept: Sweep, scope: RunScope
) -> None:
    assert scope.event_days[(SERIES, HOLDOUT_DAY)].window_end == when(LATE_DAY, 6, 0, 0)
    assert swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)].out_of_window == 1
    assert swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)].n_prints == 1
    assert swept.tallies[(HOLDOUT, 1)].out_of_window == 0
    assert swept.tallies[(HOLDOUT, 1)].n_prints == 2


def test_a_market_printing_before_its_own_event_day_opens_never_reaches_a_cluster(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_prints(scope, early_listing_artifacts(tmp_path))

    assert when(DISCOVERY_DAY, 18, 10, 0) < scope.event_days[(SERIES, HOLDOUT_DAY)].window_start
    assert sweep.in_scope == {DISCOVERY: 0, HOLDOUT: 1}
    for horizon_s in HORIZONS_S:
        tally = sweep.tallies[(HOLDOUT, horizon_s)]
        assert tally.candidates == 1
        assert tally.out_of_window == 1
        assert tally.excluded == 0
        assert tally.n_prints == 0
        assert tally.clusters() == ()


def test_the_rolling_buffer_reaches_an_anchor_on_the_next_arrival_date(swept: Sweep) -> None:
    longest = swept.tallies[(DISCOVERY, HORIZONS_S[-1])]

    assert longest.n_prints == 2
    assert longest.totals[DAY_TICKER] == Decimal("128")


def test_the_discovery_curve_is_the_contract_weighted_mean(swept: Sweep) -> None:
    means = {
        horizon_s: readout(
            swept.tallies[(DISCOVERY, horizon_s)],
            split=DISCOVERY,
            horizon_s=horizon_s,
            seed=SEED,
        ).bootstrap.estimate
        for horizon_s in HORIZONS_S
    }

    assert means == {
        1: Decimal("-2.6"),
        10: Decimal("-1.6"),
        60: Decimal("1.4"),
        300: Decimal("6.4"),
    }
    assert means[PRIMARY_HORIZON_S] > CENT_BAR


def test_the_holdout_carries_its_own_mean(swept: Sweep) -> None:
    result = readout(
        swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)],
        split=HOLDOUT,
        horizon_s=PRIMARY_HORIZON_S,
        seed=SEED,
    )

    assert result.bootstrap.estimate == Decimal("1.4")
    assert result.result.n_prints == 1
    assert result.result.contracts == 10


def test_the_kernel_drops_are_summed_per_horizon(swept: Sweep) -> None:
    assert swept.tallies[(HOLDOUT, HORIZONS_S[-1])].counts.uncovered == 1
    assert swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)].counts.uncovered == 0


def test_a_fractional_size_is_counted_and_weighted_at_the_size_the_wire_sent(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_prints(scope, fractional_artifacts(tmp_path))
    tally = sweep.tallies[(DISCOVERY, PRIMARY_HORIZON_S)]

    assert sweep.fractional_size_prints == 1
    assert tally.n_prints == 2
    assert tally.contracts == Decimal("5.24")
    assert tally.weights[DAY_TICKER] == Decimal("5.24")


def test_a_run_of_whole_sizes_counts_no_fractional_prints(swept: Sweep) -> None:
    assert swept.fractional_size_prints == 0


def test_stamps_that_went_backwards_are_counted_once_across_the_rows_the_run_read(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_prints(scope, violation_artifacts(tmp_path))

    assert sweep.read_ts_violations == 3


def test_a_ticker_whose_book_never_arrived_counts_its_prints_unresolved(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_prints(scope, bookless_artifacts(tmp_path))

    assert sweep.in_scope[DISCOVERY] == 2
    for horizon_s in HORIZONS_S:
        tally = sweep.tallies[(DISCOVERY, horizon_s)]
        assert tally.counts.unresolved == 2
        assert tally.candidates == 0
        assert tally.n_prints == 0
        assert tally.clusters() == ()


def test_a_cluster_carries_the_batches_a_ticker_printed_across(swept: Sweep) -> None:
    clusters = swept.tallies[(HOLDOUT, 1)].clusters()

    assert [item.cluster for item in clusters] == [NEXT_TICKER]
    assert clusters[0].total == Decimal("-62")
    assert clusters[0].weight == Decimal("20")


def test_the_same_seed_gives_the_same_p_value(swept: Sweep) -> None:
    tally = swept.tallies[(DISCOVERY, PRIMARY_HORIZON_S)]

    first = readout(tally, split=DISCOVERY, horizon_s=PRIMARY_HORIZON_S, seed=SEED)
    second = readout(tally, split=DISCOVERY, horizon_s=PRIMARY_HORIZON_S, seed=SEED)

    assert first.bootstrap.p_value == second.bootstrap.p_value
    assert first.bootstrap.seed == SEED


def test_the_column_pruned_read_carries_only_what_the_kernel_needs(tmp_path: Path) -> None:
    root = artifacts_dir(tmp_path)

    table = read_touch(root, SERIES, DISCOVERY_DAY)

    assert table.schema.names == list(TOUCH_COLUMNS)
    assert table.column("id").to_pylist() == [row["id"] for row in TOUCH_DISCOVERY_DAY]


def test_a_touch_partition_carrying_another_schema_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    path = write_partition(root, DISCOVERY_DAY, 1, TRADES_HOLDOUT_DAY, schema=TRADES_SCHEMA)

    with pytest.raises(ValueError, match=str(path.name)):
        read_touch(root, SERIES, DISCOVERY_DAY)


def test_a_missing_partition_still_carries_the_pruned_columns(tmp_path: Path) -> None:
    table = read_touch(tmp_path, SERIES, DISCOVERY_DAY)

    assert table.num_rows == 0
    assert table.schema.names == list(TOUCH_COLUMNS)
    assert TOUCH_SCHEMA.names != list(TOUCH_COLUMNS)


def test_a_discovery_too_thin_in_prints_to_power_the_gate_is_underpowered() -> None:
    discovery = readout_of(
        spread("20", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=100
    )
    holdout = readout_of(
        spread("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.powered
    assert decision.verdict == UNDERPOWERED


def test_a_discovery_over_too_few_tickers_is_underpowered_at_any_print_count() -> None:
    discovery = readout_of(
        spread("20", THIN_TICKERS, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        spread("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert discovery.result.n_prints > PRINT_MIN_DISCOVERY
    assert decision.gate.n == THIN_TICKERS
    assert decision.gate.n_min == TICKER_MIN_DISCOVERY
    assert decision.gate.n_unit == TICKERS
    assert not decision.gate.powered
    assert decision.verdict == UNDERPOWERED


def test_a_holdout_too_thin_in_prints_to_replicate_is_underpowered() -> None:
    discovery = readout_of(
        spread("20", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        spread("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=100
    )

    decision = decide(discovery, holdout)

    assert holdout.result.n_prints < PRINT_MIN_HOLDOUT
    assert decision.gate.powered
    assert decision.replication.powered
    assert decision.verdict == UNDERPOWERED


def test_a_holdout_over_too_few_tickers_is_underpowered() -> None:
    discovery = readout_of(
        spread("20", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        spread("20", THIN_TICKERS, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.replication.holdout_n == THIN_TICKERS
    assert decision.replication.holdout_n_min == TICKER_MIN_HOLDOUT
    assert not decision.replication.powered
    assert decision.verdict == UNDERPOWERED


def test_an_estimate_under_the_cent_bar_closes_the_question() -> None:
    discovery = readout_of(
        spread("5", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        spread("5", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.estimate == Decimal("0.5")
    assert not decision.gate.economic
    assert decision.gate.significant
    assert decision.replication.replicated
    assert decision.verdict == CLOSED


def test_an_estimate_under_the_cent_bar_closes_it_however_the_resamples_landed() -> None:
    discovery = readout_of(
        flat("5", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        flat("5", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.estimate == Decimal("0.5")
    assert decision.gate.undecidable
    assert not decision.gate.economic
    assert decision.verdict == CLOSED


def test_an_estimate_over_the_bar_that_does_not_replicate_closes_the_question() -> None:
    discovery = readout_of(
        spread("20", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        spread("-20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.passed
    assert not decision.replication.same_sign
    assert not decision.replication.replicated
    assert decision.verdict == CLOSED


def test_a_gate_that_clears_and_replicates_passes() -> None:
    discovery = readout_of(
        spread("20", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        spread("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.passed
    assert decision.gate.n == TICKER_MIN_DISCOVERY
    assert decision.replication.replicated
    assert decision.verdict == PASS


def test_an_estimate_no_resample_moved_refuses_a_verdict() -> None:
    discovery = readout_of(
        flat("20", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        flat("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.economic
    assert decision.gate.powered
    assert decision.gate.undecidable
    assert not decision.gate.significant
    assert not decision.gate.passed
    assert decision.verdict == UNDECIDABLE


def test_a_holdout_no_resample_moved_refuses_a_verdict_the_discovery_alone_would_pass() -> None:
    discovery = readout_of(
        spread("20", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        flat("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.passed
    assert not decision.gate.undecidable
    assert decision.replication.undecidable
    assert decision.replication.powered
    assert not decision.replication.replicated
    assert decision.verdict == UNDECIDABLE


def test_a_zero_estimate_on_a_discovery_too_thin_to_power_the_gate_is_underpowered() -> None:
    discovery = readout_of(
        spread("0", THIN_TICKERS, split=DISCOVERY), split=DISCOVERY, n_prints=100
    )
    holdout = readout_of(
        spread("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.estimate == Decimal("0")
    assert not decision.gate.powered
    assert decision.replication is None
    assert decision.skipped == ZERO_ESTIMATE
    assert decision.verdict == UNDERPOWERED


def test_a_split_with_no_usable_prints_carries_no_bootstrap() -> None:
    empty = readout(Tally(), split=HOLDOUT, horizon_s=PRIMARY_HORIZON_S, seed=SEED)

    assert empty.bootstrap is None
    assert empty.result.n_prints == 0
    assert empty.result.clusters == ()
    assert empty.excluded_fraction is None


def test_a_holdout_with_no_usable_prints_is_underpowered() -> None:
    discovery = readout_of(
        spread("20", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout(Tally(), split=HOLDOUT, horizon_s=PRIMARY_HORIZON_S, seed=SEED)

    decision = decide(discovery, holdout)

    assert decision.gate.powered
    assert decision.replication is None
    assert decision.skipped == NO_ESTIMATE
    assert decision.verdict == UNDERPOWERED


def test_a_discovery_with_no_usable_prints_carries_no_gate() -> None:
    discovery = readout(Tally(), split=DISCOVERY, horizon_s=PRIMARY_HORIZON_S, seed=SEED)
    holdout = readout_of(
        spread("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate is None
    assert decision.replication is None
    assert decision.skipped == NO_ESTIMATE
    assert decision.verdict == UNDERPOWERED


def test_a_discovery_estimate_of_exactly_zero_closes_without_a_replication() -> None:
    discovery = readout_of(
        spread("0", TICKER_MIN_DISCOVERY, split=DISCOVERY), split=DISCOVERY, n_prints=POWERED
    )
    holdout = readout_of(
        spread("20", TICKER_MIN_HOLDOUT, split=HOLDOUT), split=HOLDOUT, n_prints=HOLDOUT_POWERED
    )

    decision = decide(discovery, holdout)

    assert decision.gate.estimate == Decimal("0")
    assert decision.gate.powered
    assert decision.replication is None
    assert decision.skipped == ZERO_ESTIMATE
    assert decision.verdict == CLOSED
    with pytest.raises(ValueError):
        evaluate_holdout(
            discovery_estimate=Decimal("0"),
            holdout_estimate=Decimal("2"),
            holdout_p_value=0.0,
            holdout_result=holdout.bootstrap,
            discovery_n_min=TICKER_MIN_DISCOVERY,
            alpha=0.05,
            n_unit=TICKERS,
            undecidable=False,
        )


def test_a_scope_spanning_both_ladders_is_not_swept_without_a_cohort(
    tmp_path: Path, scope: RunScope
) -> None:
    paired = next(iter(scope.event_days.values()))
    both = replace(scope, event_days={**scope.event_days, (LOW_SERIES, paired.event_date): paired})

    with pytest.raises(ValueError, match="names no cohort"):
        sweep_prints(both, artifacts_dir(tmp_path))

    swept = sweep_prints(both, artifacts_dir(tmp_path), cohort=HIGH)

    assert swept.in_scope == {DISCOVERY: 4, HOLDOUT: 2}
    assert LOW_SERIES not in {series for series, _ in swept.tickers}


def test_a_two_ladder_run_naming_no_cohort_writes_no_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "tape_studies"

    with pytest.raises(ValueError, match="names no cohort"):
        execute(
            run_id=RUN_ID,
            run_scope=both_ladder_scope_dir(tmp_path),
            artifacts=artifacts_dir(tmp_path),
            floor_source=FloorSource.SIGNED_READ,
            economic_bar_size=ECONOMIC_BAR_SIZE,
            economic_bar_price=ECONOMIC_BAR_PRICE,
            economic_bar_price_source=ECONOMIC_BAR_PRICE_SOURCE,
            seed=SEED,
            run_root=run_root,
            **run_paths(tmp_path),
        )

    assert not (run_root / RUN_ID / MANIFEST_NAME).exists()
    assert not run_root.exists()


def test_a_two_ladder_run_reads_only_the_cohort_it_names(tmp_path: Path) -> None:
    scope_root = both_ladder_scope_dir(tmp_path)

    run = execute(
        run_id=RUN_ID,
        run_scope=scope_root,
        artifacts=artifacts_dir(tmp_path),
        floor_source=FloorSource.SIGNED_READ,
        economic_bar_size=ECONOMIC_BAR_SIZE,
        economic_bar_price=ECONOMIC_BAR_PRICE,
        economic_bar_price_source=ECONOMIC_BAR_PRICE_SOURCE,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        cohort=HIGH,
        **run_paths(tmp_path),
    )

    payload = result_payload(run)
    assert len(load_run_scope(scope_root).event_days) == 4
    assert payload["cities"] == [SERIES]
    assert LOW_SERIES not in {series for series, _ in run.sweep.tickers}
    assert run.sweep.in_scope == {DISCOVERY: 4, HOLDOUT: 2}
