from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

from bot.lag.depth_map import (
    BUCKET_HOURS,
    CUMULATIVE_TICKS,
    FINAL_HOURS,
    NO_MATCH,
    NO_SIDE,
    REPLENISH_FRACTION,
    REPLENISH_WINDOW_S,
    SLIPPAGE_BAR_CENTS,
    TOUCH_DEPTH_BAR,
    YES_SIDE,
    WeightedTally,
    _FOLD_PAIRS,
    atm_leg,
    cell_edges,
    classify_print,
    cumulative_depth,
    hour_of_day,
    hours_to_close_bucket,
    level_units,
    scaled_units,
    screen_cells,
    walk_capacity,
    weighted_quantile,
)
from bot.lag.ladder_consistency import PRICE_TICKS, SIZE_UNITS
from bot.lag.lead_lag import atm_series
from bot.lag.r0_universe import R0Universe
from bot.lag.tape_studies import (
    EXCLUSION_CLASSES,
    EvidenceWindow,
    RunScope,
    keep_mask,
    merge_intervals,
    screen_windows,
)
from bot.replay.artifacts import TOUCH_SCHEMA
from bot.replay.run_scope import (
    DISCOVERY,
    QUIET_BAND,
    RECORDED_GAP,
    RESUBSCRIBE_BLIND,
    SUBSCRIPTION_WIDE,
    EventDay,
    Exclusion,
)


UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MICROSECOND = timedelta(microseconds=1)
MICROS_PER_HOUR = 3_600_000_000

SERIES = "KXHIGHDEN"
EVENT_DATE = date(2026, 7, 18)
LEG_A = "KXHIGHDEN-26JUL18-B93.5"
LEG_B = "KXHIGHDEN-26JUL18-B95.5"
WINDOW_START = datetime(2026, 7, 18, 6, tzinfo=UTC)
WINDOW_END = WINDOW_START + timedelta(days=1)

QUIET_START = datetime(2026, 7, 18, 7, tzinfo=UTC)
QUIET_END = datetime(2026, 7, 18, 9, tzinfo=UTC)
GAP_START = datetime(2026, 7, 18, 8, tzinfo=UTC)
GAP_END = datetime(2026, 7, 18, 8, 10, tzinfo=UTC)
BLINK = datetime(2026, 7, 18, 12, tzinfo=UTC)
WIDE_START = datetime(2026, 7, 18, 18, tzinfo=UTC)
WIDE_END = datetime(2026, 7, 18, 18, 30, tzinfo=UTC)

WIDTH = 6
BAR_UNITS = int(SLIPPAGE_BAR_CENTS * PRICE_TICKS // 100)

PRICE_TABLE: tuple[tuple[str, int, int], ...] = (
    ("0.0000", 4, 0),
    ("1.0000", 4, 10_000),
    ("0.0100", 4, 100),
    ("0.9900", 4, 9_900),
    ("0.0700", 4, 700),
    ("0.39", 2, 39),
    ("0.00", 2, 0),
    ("250.00", 2, 25_000),
)

SPANS: tuple[tuple[str, datetime, datetime], ...] = (
    ("kept", datetime(2026, 7, 18, 10, tzinfo=UTC), datetime(2026, 7, 18, 10, 30, tzinfo=UTC)),
    ("opening", WINDOW_START, WINDOW_START + timedelta(minutes=1)),
    ("closing", WINDOW_END - timedelta(minutes=1), WINDOW_END),
    ("early", WINDOW_START - timedelta(hours=1), WINDOW_START - timedelta(minutes=30)),
    ("late", WINDOW_END - timedelta(minutes=1), WINDOW_END + timedelta(minutes=1)),
    ("quiet", datetime(2026, 7, 18, 7, 10, tzinfo=UTC), datetime(2026, 7, 18, 7, 20, tzinfo=UTC)),
    (
        "two_classes",
        datetime(2026, 7, 18, 8, 1, tzinfo=UTC),
        datetime(2026, 7, 18, 8, 5, tzinfo=UTC),
    ),
    ("blind", BLINK - timedelta(seconds=1), BLINK + timedelta(seconds=1)),
    ("wide", datetime(2026, 7, 18, 18, 10, tzinfo=UTC), datetime(2026, 7, 18, 18, 20, tzinfo=UTC)),
)


def micros(stamp: datetime) -> int:
    return (stamp - EPOCH) // MICROSECOND


def units(values: Sequence[str], scale: int, width: int = WIDTH) -> list[int]:
    row = [int(Decimal(value) * scale) for value in values]
    return row + [0] * (width - len(row))


def book(prices: Sequence[str], sizes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([units(prices, PRICE_TICKS)], dtype=np.int64),
        np.array([units(sizes, SIZE_UNITS)], dtype=np.int64),
    )


def average_slippage(prices: Sequence[str], sizes: Sequence[str], walk: Decimal) -> Decimal:
    best = Decimal(prices[0])
    remaining = walk
    cost = Decimal(0)
    for price, size in zip(prices, sizes, strict=True):
        taken = min(Decimal(size), remaining)
        cost += (best - Decimal(price)) * taken
        remaining -= taken
    return cost / walk


def largest_walk(prices: Sequence[str], sizes: Sequence[str], bar_cents: Decimal) -> int:
    bar = bar_cents / 100
    total = int(sum(Decimal(size) for size in sizes) * SIZE_UNITS)
    return max(
        step
        for step in range(1, total + 1)
        if average_slippage(prices, sizes, Decimal(step) / SIZE_UNITS) <= bar
    )


def touch(seconds: int, ticker: str, yes_bid: str, no_bid: str) -> dict:
    return {
        "ticker": ticker,
        "received_at": WINDOW_START + timedelta(seconds=seconds),
        "ts_ms": None,
        "yes_bid": yes_bid,
        "yes_bid_depth": "50",
        "yes_ask": str(Decimal("1") - Decimal(no_bid)),
        "yes_ask_depth": "50",
        "no_bid": no_bid,
        "no_bid_depth": "50",
        "no_ask": str(Decimal("1") - Decimal(yes_bid)),
        "no_ask_depth": "50",
    }


def quoted(seconds: int, ticker: str, dollars: str) -> dict:
    return touch(
        seconds,
        ticker,
        str(Decimal(dollars) - Decimal("0.01")),
        str(Decimal("0.99") - Decimal(dollars)),
    )


def touch_table(rows: Sequence[dict]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"id": index, **row} for index, row in enumerate(rows, start=1)], schema=TOUCH_SCHEMA
    )


# The sweep hands atm_leg the rows it already filtered, so the fixture filters the same way.
def leg_arrays(rows: Sequence[dict]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    legs = tuple(sorted({row["ticker"] for row in rows}))
    seats = np.array([legs.index(row["ticker"]) for row in rows], dtype=np.int64)
    yes_bid = np.array([int(Decimal(row["yes_bid"]) * PRICE_TICKS) for row in rows], dtype=np.int64)
    no_bid = np.array([int(Decimal(row["no_bid"]) * PRICE_TICKS) for row in rows], dtype=np.int64)
    received = np.array([micros(row["received_at"]) for row in rows], dtype=np.int64)
    quoted = (
        (yes_bid > 0)
        & (no_bid > 0)
        & (received >= micros(WINDOW_START))
        & (received <= micros(WINDOW_END))
    )
    return legs, seats[quoted], (yes_bid + PRICE_TICKS - no_bid)[quoted]


def pooled_span(tally: WeightedTally, group: int) -> int:
    return sum(tally.pooled([group]).values())


def pooled_quantile(
    tally: WeightedTally, group: int, numerator: int, denominator: int
) -> int | None:
    return weighted_quantile(tally.pooled([group]), numerator, denominator)


def event_day() -> EventDay:
    return EventDay(
        series=SERIES,
        station="KDEN",
        timezone="America/Denver",
        event_date=EVENT_DATE,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        tickers=6,
        ladder_rows=900,
        first_event_at=WINDOW_START,
        last_event_at=WINDOW_END,
        covered=True,
        evaluable=True,
        in_scope=True,
        day_index=1,
        split=DISCOVERY,
        excluded_us=0,
        span_us=0,
    )


def exclusion(name: str, start: datetime, end: datetime) -> Exclusion:
    return Exclusion(
        exclusion_class=name,
        start=start,
        end=end,
        boundary_id=None,
        gap_id=None,
        gap_reason=None,
        padded=False,
    )


@pytest.fixture
def scope() -> RunScope:
    exclusions = (
        exclusion(QUIET_BAND, QUIET_START, QUIET_END),
        exclusion(RECORDED_GAP, GAP_START, GAP_END),
        exclusion(RESUBSCRIBE_BLIND, BLINK, BLINK + MICROSECOND),
        exclusion(SUBSCRIPTION_WIDE, WIDE_START, WIDE_END),
    )
    return RunScope(
        exclusions=exclusions,
        merged=merge_intervals((item.start, item.end) for item in exclusions),
        by_class={
            name: merge_intervals(
                (item.start, item.end) for item in exclusions if item.exclusion_class == name
            )
            for name in EXCLUSION_CLASSES
        },
        event_days={(SERIES, EVENT_DATE): event_day()},
        discovery_days=frozenset({EVENT_DATE}),
        holdout_days=frozenset(),
        scope_start=WINDOW_START,
        scope_end=WINDOW_END,
        universe=R0Universe(
            fraction_invalid_max=Decimal("0.4"),
            passing=(SERIES,),
            lock_dependent=(),
            recorded=(SERIES,),
            ladder_widths=(6,),
            in_scope_city_days=14,
            reconciliation="agree",
            recorded_not_passing=(),
            passing_not_recorded=(),
        ),
    )


def test_the_bars_are_the_preregistered_numbers() -> None:
    assert SLIPPAGE_BAR_CENTS == Decimal("1")
    assert CUMULATIVE_TICKS == 5
    assert REPLENISH_FRACTION == Decimal("0.5")
    assert REPLENISH_WINDOW_S == 60
    assert FINAL_HOURS == 6
    assert TOUCH_DEPTH_BAR == Decimal("200")
    assert BUCKET_HOURS == 6


@pytest.mark.parametrize(
    ("value", "decimals", "expected"), PRICE_TABLE, ids=[row[0] for row in PRICE_TABLE]
)
def test_scaled_units_is_a_golden_table(value: str, decimals: int, expected: int) -> None:
    scaled = scaled_units(pa.array([value]), decimals)
    assert scaled.dtype == np.int64
    assert scaled.tolist() == [expected]


def test_scaled_units_converts_a_whole_column_at_once() -> None:
    column = pa.array(["0.0000", "0.5000", "0.0700"])
    assert scaled_units(column, 4).tolist() == [0, 5_000, 700]


def test_scaled_units_rejects_a_wrong_decimal_count() -> None:
    with pytest.raises(ValueError, match="0.500"):
        scaled_units(pa.array(["0.5000", "0.500"]), 4)


def test_scaled_units_rejects_a_string_with_no_point() -> None:
    with pytest.raises(ValueError, match="50"):
        scaled_units(pa.array(["50"]), 2)


def test_level_units_pads_ragged_rows_to_the_stored_width() -> None:
    column = pa.array(
        [
            ["0.4000", "0.3900"],
            [],
            ["0.4000", "0.3900", "0.3800", "0.3700", "0.3600", "0.3500"],
        ],
        type=pa.list_(pa.string()),
    )
    dense = level_units(column, 4, WIDTH)
    assert dense.dtype == np.int64
    assert dense.shape == (3, WIDTH)
    assert dense.tolist() == [
        [4_000, 3_900, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [4_000, 3_900, 3_800, 3_700, 3_600, 3_500],
    ]


def test_level_units_reads_sizes_at_two_decimals() -> None:
    column = pa.array([["10.00", "0.39"]], type=pa.list_(pa.string()))
    assert level_units(column, 2, WIDTH).tolist() == [[1_000, 39, 0, 0, 0, 0]]


def test_level_units_reads_a_sliced_list_column_off_its_own_offsets() -> None:
    column = pa.array(
        [["0.4000", "0.3900"], ["0.3000"], [], ["0.2000"]], type=pa.list_(pa.string())
    )
    assert level_units(column.slice(1, 2), 4, WIDTH).tolist() == [
        [3_000, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]


def test_an_empty_ladder_column_converts_to_an_empty_matrix() -> None:
    dense = level_units(pa.array([], type=pa.list_(pa.string())), 4, WIDTH)
    assert dense.shape == (0, WIDTH)
    assert scaled_units(pa.array([], type=pa.string()), 4).tolist() == []


def test_level_units_rejects_a_row_past_the_width() -> None:
    column = pa.array([["0.4000"], ["0.4000", "0.3900", "0.3800"]], type=pa.list_(pa.string()))
    with pytest.raises(ValueError, match="3"):
        level_units(column, 4, 2)


def test_a_contiguous_six_level_book_sits_wholly_inside_five_cents() -> None:
    prices, sizes = book(
        ["0.4000", "0.3900", "0.3800", "0.3700", "0.3600", "0.3500"], ["1", "1", "1", "1", "1", "1"]
    )
    assert cumulative_depth(prices, sizes, CUMULATIVE_TICKS).tolist() == [6 * SIZE_UNITS]


def test_a_level_past_five_cents_does_not_count() -> None:
    prices, sizes = book(["0.4000", "0.3900", "0.3400"], ["1", "2", "3"])
    assert cumulative_depth(prices, sizes, CUMULATIVE_TICKS).tolist() == [3 * SIZE_UNITS]


def test_the_padding_below_a_short_book_carries_no_depth() -> None:
    prices, sizes = book(["0.0300", "0.0200"], ["1", "1"])
    assert cumulative_depth(prices, sizes, CUMULATIVE_TICKS).tolist() == [2 * SIZE_UNITS]


def test_an_empty_side_has_no_cumulative_depth() -> None:
    prices, sizes = book([], [])
    assert cumulative_depth(prices, sizes, CUMULATIVE_TICKS).tolist() == [0]


def test_a_crossing_walk_matches_the_average_slippage_definition() -> None:
    prices = ["0.4000", "0.3900", "0.3500"]
    sizes = ["1", "1", "1"]
    walk = walk_capacity(*book(prices, sizes), np.array([3]), BAR_UNITS)
    assert walk.capacity.tolist() == [largest_walk(prices, sizes, SLIPPAGE_BAR_CENTS)]
    assert walk.censored.tolist() == [False]
    assert walk.exhausted.tolist() == [False]


def test_a_walk_that_runs_out_of_book_is_exhausted_not_censored() -> None:
    prices, sizes = book(
        ["0.4000", "0.3900", "0.3800", "0.3700", "0.3600", "0.3500"],
        ["100", "1", "1", "1", "1", "1"],
    )
    walk = walk_capacity(prices, sizes, np.array([WIDTH]), BAR_UNITS)
    assert walk.capacity.tolist() == [105 * SIZE_UNITS]
    assert walk.exhausted.tolist() == [True]
    assert walk.censored.tolist() == [False]


def test_a_walk_that_runs_out_of_ladder_is_censored() -> None:
    prices, sizes = book(
        ["0.4000", "0.3900", "0.3800", "0.3700", "0.3600", "0.3500"],
        ["100", "1", "1", "1", "1", "1"],
    )
    walk = walk_capacity(prices, sizes, np.array([WIDTH + 2]), BAR_UNITS)
    assert walk.capacity.tolist() == [105 * SIZE_UNITS]
    assert walk.censored.tolist() == [True]
    assert walk.exhausted.tolist() == [False]


def test_an_empty_side_carries_no_capacity_and_is_a_complete_observation() -> None:
    prices, sizes = book([], [])
    walk = walk_capacity(prices, sizes, np.array([0]), BAR_UNITS)
    assert walk.capacity.tolist() == [0]
    assert walk.exhausted.tolist() == [True]
    assert walk.censored.tolist() == [False]


def test_no_row_is_both_censored_and_exhausted() -> None:
    prices = np.array(
        [
            units(["0.4000", "0.3900", "0.3500"], PRICE_TICKS),
            units(["0.4000", "0.3900", "0.3800", "0.3700", "0.3600", "0.3500"], PRICE_TICKS),
            units(["0.4000", "0.3900", "0.3800", "0.3700", "0.3600", "0.3500"], PRICE_TICKS),
            units([], PRICE_TICKS),
        ],
        dtype=np.int64,
    )
    sizes = np.array(
        [
            units(["1", "1", "1"], SIZE_UNITS),
            units(["100", "1", "1", "1", "1", "1"], SIZE_UNITS),
            units(["100", "1", "1", "1", "1", "1"], SIZE_UNITS),
            units([], SIZE_UNITS),
        ],
        dtype=np.int64,
    )
    walk = walk_capacity(prices, sizes, np.array([3, WIDTH, WIDTH + 2, 0]), BAR_UNITS)
    assert not np.any(walk.censored & walk.exhausted)
    assert walk.capacity.tolist() == [225, 10_500, 10_500, 0]


def test_capacity_is_zero_only_where_the_side_is_empty() -> None:
    prices, sizes = book(["0.4000"], ["0.01"])
    walk = walk_capacity(prices, sizes, np.array([1]), BAR_UNITS)
    assert walk.capacity.tolist() == [1]


def test_an_hour_at_depth_one_outweighs_a_second_at_depth_two_hundred() -> None:
    tally = WeightedTally()
    tally.add(
        np.array([0, 0], dtype=np.int64),
        np.array([1, 200], dtype=np.int64),
        np.array([3_600_000_000, 1_000_000], dtype=np.int64),
    )
    assert pooled_quantile(tally, 0, 1, 2) == 1
    assert pooled_span(tally, 0) == 3_601_000_000


def test_the_tally_keeps_its_groups_apart() -> None:
    tally = WeightedTally()
    tally.add(
        np.array([0, 1, 1], dtype=np.int64),
        np.array([5, 9, 11], dtype=np.int64),
        np.array([10, 30, 10], dtype=np.int64),
    )
    assert tally.groups() == [0, 1]
    assert pooled_span(tally, 0) == 10
    assert pooled_span(tally, 1) == 40
    assert pooled_quantile(tally, 0, 1, 2) == 5
    assert pooled_quantile(tally, 1, 1, 2) == 9
    assert pooled_quantile(tally, 1, 9, 10) == 11


def test_repeated_batches_fold_into_the_same_bin() -> None:
    tally = WeightedTally()
    for _ in range(3):
        tally.add(
            np.array([2, 2], dtype=np.int64),
            np.array([4, 4], dtype=np.int64),
            np.array([7, 5], dtype=np.int64),
        )
    assert pooled_span(tally, 2) == 36
    assert pooled_quantile(tally, 2, 1, 2) == 4


def test_microsecond_weights_stay_exact_across_the_accrual_window() -> None:
    tally = WeightedTally()
    tally.add(
        np.array([0, 0], dtype=np.int64),
        np.array([3, 4], dtype=np.int64),
        np.array([1_200_000_000_000, 1], dtype=np.int64),
    )
    assert pooled_span(tally, 0) == 1_200_000_000_001


def test_pooling_groups_merges_their_bins() -> None:
    tally = WeightedTally()
    tally.add(
        np.array([0, 1, 1], dtype=np.int64),
        np.array([5, 5, 11], dtype=np.int64),
        np.array([10, 30, 10], dtype=np.int64),
    )

    assert tally.pooled([0, 1]) == {5: 40, 11: 10}
    assert tally.pooled([1]) == {5: 30, 11: 10}
    assert tally.pooled([7]) == {}
    assert weighted_quantile(tally.pooled([0, 1]), 1, 2) == 5


def test_an_unseen_group_has_no_span_and_no_quantile() -> None:
    tally = WeightedTally()
    assert tally.groups() == []
    assert pooled_span(tally, 7) == 0
    assert pooled_quantile(tally, 7, 1, 2) is None


def test_a_pair_seen_on_both_sides_of_a_fold_lands_in_one_bin() -> None:
    tally = WeightedTally()
    batch = _FOLD_PAIRS // 4
    for _ in range(6):
        tally.add(
            np.full(batch, 3, dtype=np.int64),
            np.full(batch, 11, dtype=np.int64),
            np.ones(batch, dtype=np.int64),
        )
    assert tally.pooled([3]) == {11: 6 * batch}
    assert tally.entries() == 1


def test_interleaved_groups_and_values_read_back_sorted() -> None:
    tally = WeightedTally()
    tally.add(
        np.array([1, 0, 1], dtype=np.int64),
        np.array([9, 5, 2], dtype=np.int64),
        np.array([10, 4, 30], dtype=np.int64),
    )
    tally.add(
        np.array([0, 1], dtype=np.int64),
        np.array([3, 9], dtype=np.int64),
        np.array([6, 10], dtype=np.int64),
    )
    assert tally.groups() == [0, 1]
    assert list(tally.pooled([1])) == [2, 9]
    assert tally.pooled([1]) == {2: 30, 9: 20}
    assert tally.pooled([0]) == {3: 6, 5: 4}
    assert weighted_quantile(tally.pooled([0, 1]), 1, 2) == 2


def test_weights_past_the_float_mantissa_sum_without_losing_the_low_bit() -> None:
    tally = WeightedTally()
    heavy = 2**53 + 1
    tally.add(
        np.zeros(3, dtype=np.int64),
        np.full(3, 7, dtype=np.int64),
        np.full(3, heavy, dtype=np.int64),
    )
    tally.add(
        np.zeros(2, dtype=np.int64),
        np.full(2, 7, dtype=np.int64),
        np.full(2, heavy, dtype=np.int64),
    )
    assert tally.pooled([0]) == {7: 5 * heavy}


def test_entries_counts_each_packed_bin_once() -> None:
    tally = WeightedTally()
    assert tally.entries() == 0
    tally.add(
        np.array([0, 0, 1], dtype=np.int64),
        np.array([5, 6, 5], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
    )
    assert tally.entries() == 3
    tally.add(
        np.array([0, 1], dtype=np.int64),
        np.array([5, 7], dtype=np.int64),
        np.array([4, 5], dtype=np.int64),
    )
    assert tally.entries() == 4
    assert tally.pooled([0]) == {5: 5, 6: 2}


def test_a_group_never_added_pools_to_nothing_beside_a_packed_one() -> None:
    tally = WeightedTally()
    tally.add(
        np.array([2], dtype=np.int64), np.array([4], dtype=np.int64), np.array([9], dtype=np.int64)
    )
    assert tally.pooled([5]) == {}
    assert weighted_quantile(tally.pooled([5]), 1, 2) is None
    assert tally.entries() == 1


def test_weighted_quantile_reads_a_bare_mapping() -> None:
    bins = {1: 3_600_000_000, 200: 1_000_000}
    assert weighted_quantile(bins, 1, 2) == 1
    assert weighted_quantile(bins, 1, 1) == 200
    assert weighted_quantile({}, 1, 2) is None


def test_a_quantile_landing_exactly_on_a_bin_edge_takes_that_bin() -> None:
    assert weighted_quantile({1: 50, 2: 50}, 1, 2) == 1
    assert weighted_quantile({1: 50, 2: 50}, 51, 100) == 2


@pytest.mark.parametrize(("name", "start", "end"), SPANS, ids=[row[0] for row in SPANS])
def test_screen_cells_matches_screen_windows(
    scope: RunScope, name: str, start: datetime, end: datetime
) -> None:
    window = EvidenceWindow(series=SERIES, event_date=EVENT_DATE, start=start, end=end)
    screened = screen_windows(scope, [window])
    cells = screen_cells(
        scope,
        SERIES,
        EVENT_DATE,
        np.array([micros(start)], dtype=np.int64),
        np.array([micros(end)], dtype=np.int64),
    )
    assert cells.kept.tolist() == keep_mask([window], screened.kept).tolist()
    assert cells.out_of_window == screened.out_of_window
    assert cells.excluded == screened.excluded
    assert cells.by_class == screened.by_class


def test_screen_cells_matches_screen_windows_over_the_whole_batch(scope: RunScope) -> None:
    windows = [
        EvidenceWindow(series=SERIES, event_date=EVENT_DATE, start=start, end=end)
        for _, start, end in SPANS
    ]
    screened = screen_windows(scope, windows)
    cells = screen_cells(
        scope,
        SERIES,
        EVENT_DATE,
        np.array([micros(window.start) for window in windows], dtype=np.int64),
        np.array([micros(window.end) for window in windows], dtype=np.int64),
    )
    assert cells.kept.tolist() == keep_mask(windows, screened.kept).tolist()
    assert cells.out_of_window == screened.out_of_window == 2
    assert cells.excluded == screened.excluded == 4
    assert cells.by_class == screened.by_class


def test_one_cell_can_be_counted_under_two_exclusion_classes(scope: RunScope) -> None:
    cells = screen_cells(
        scope,
        SERIES,
        EVENT_DATE,
        np.array([micros(datetime(2026, 7, 18, 8, 1, tzinfo=UTC))], dtype=np.int64),
        np.array([micros(datetime(2026, 7, 18, 8, 5, tzinfo=UTC))], dtype=np.int64),
    )
    assert cells.excluded == 1
    assert cells.by_class[QUIET_BAND] == 1
    assert cells.by_class[RECORDED_GAP] == 1
    assert sum(cells.by_class.values()) == 2


def test_a_class_carrying_no_exclusion_counts_nothing(scope: RunScope) -> None:
    quiet = exclusion(QUIET_BAND, QUIET_START, QUIET_END)
    bare = RunScope(
        exclusions=(quiet,),
        merged=merge_intervals([(quiet.start, quiet.end)]),
        by_class={
            name: merge_intervals([(quiet.start, quiet.end)] if name == QUIET_BAND else [])
            for name in EXCLUSION_CLASSES
        },
        event_days=scope.event_days,
        discovery_days=scope.discovery_days,
        holdout_days=scope.holdout_days,
        scope_start=scope.scope_start,
        scope_end=scope.scope_end,
        universe=scope.universe,
    )
    cells = screen_cells(
        bare,
        SERIES,
        EVENT_DATE,
        np.array([micros(QUIET_START), micros(BLINK)], dtype=np.int64),
        np.array([micros(QUIET_END), micros(BLINK)], dtype=np.int64),
    )
    assert cells.kept.tolist() == [False, True]
    assert cells.by_class == {
        QUIET_BAND: 1,
        RECORDED_GAP: 0,
        RESUBSCRIBE_BLIND: 0,
        SUBSCRIPTION_WIDE: 0,
    }


def test_an_event_day_outside_the_frozen_scope_is_rejected(scope: RunScope) -> None:
    with pytest.raises(ValueError, match="frozen scope"):
        screen_cells(
            scope,
            SERIES,
            date(2026, 7, 25),
            np.array([micros(WINDOW_START)], dtype=np.int64),
            np.array([micros(WINDOW_END)], dtype=np.int64),
        )


def test_a_break_outside_the_span_is_ignored() -> None:
    times = np.array([10, 20, 30], dtype=np.int64)
    assert cell_edges(times, np.array([5, 35], dtype=np.int64)).tolist() == [10, 20, 30]


def test_a_break_on_a_state_time_does_not_duplicate_it() -> None:
    times = np.array([10, 20, 30], dtype=np.int64)
    assert cell_edges(times, np.array([10, 20, 30], dtype=np.int64)).tolist() == [10, 20, 30]


def test_an_interior_break_splits_the_state_that_outlived_it() -> None:
    times = np.array([10, 20, 30], dtype=np.int64)
    edges = cell_edges(times, np.array([5, 20, 25, 35], dtype=np.int64))
    assert edges.tolist() == [10, 20, 25, 30]


def test_hour_of_day_reads_utc() -> None:
    stamps = np.array(
        [
            micros(datetime(2026, 7, 18, tzinfo=UTC)),
            micros(datetime(2026, 7, 18, 7, 30, tzinfo=UTC)),
            micros(datetime(2026, 7, 18, 23, 59, 59, tzinfo=UTC)),
        ],
        dtype=np.int64,
    )
    assert hour_of_day(stamps).tolist() == [0, 7, 23]


def test_hours_to_close_falls_into_six_hour_buckets() -> None:
    close = micros(WINDOW_END)
    received = np.array(
        [close, close - MICROS_PER_HOUR, close - 6 * MICROS_PER_HOUR, close - 19 * MICROS_PER_HOUR],
        dtype=np.int64,
    )
    buckets = hours_to_close_bucket(np.full(received.shape, close), received, BUCKET_HOURS)
    assert buckets.tolist() == [0, 0, 1, 3]


def test_the_atm_leg_matches_the_series_picker_on_two_legs() -> None:
    rows = [
        quoted(0, LEG_A, "0.50"),
        quoted(0, LEG_B, "0.70"),
        quoted(10, LEG_A, "0.90"),
        quoted(10, LEG_B, "0.55"),
        quoted(20, LEG_A, "0.90"),
        quoted(20, LEG_B, "0.55"),
        quoted(30, LEG_A, "0.90"),
        quoted(30, LEG_B, "0.55"),
    ]
    picked = atm_series(
        SERIES, EVENT_DATE, touch_table(rows), window_start=WINDOW_START, window_end=WINDOW_END
    )
    legs, seats, mid2 = leg_arrays(rows)
    seat = atm_leg(legs, seats, mid2)
    assert legs[seat] == picked.ticker == LEG_B


def test_the_atm_leg_breaks_a_tie_the_same_way_the_series_picker_does() -> None:
    rows = [
        quoted(0, LEG_B, "0.55"),
        quoted(0, LEG_A, "0.45"),
        quoted(10, LEG_B, "0.55"),
        quoted(10, LEG_A, "0.45"),
    ]
    picked = atm_series(
        SERIES, EVENT_DATE, touch_table(rows), window_start=WINDOW_START, window_end=WINDOW_END
    )
    legs, seats, mid2 = leg_arrays(rows)
    assert legs[atm_leg(legs, seats, mid2)] == picked.ticker == LEG_A


def test_the_atm_leg_picks_nothing_when_no_state_is_two_sided() -> None:
    rows = [touch(0, LEG_A, "0.6000", "0.0000"), touch(10, LEG_B, "0.0000", "0.4000")]
    assert (
        atm_series(
            SERIES, EVENT_DATE, touch_table(rows), window_start=WINDOW_START, window_end=WINDOW_END
        )
        is None
    )
    legs, seats, mid2 = leg_arrays(rows)
    assert atm_leg(legs, seats, mid2) is None


def test_the_atm_leg_ignores_states_outside_the_window() -> None:
    rows = [
        quoted(-60, LEG_A, "0.50"),
        quoted(0, LEG_A, "0.90"),
        quoted(0, LEG_B, "0.55"),
        quoted(10, LEG_A, "0.90"),
        quoted(10, LEG_B, "0.55"),
    ]
    picked = atm_series(
        SERIES, EVENT_DATE, touch_table(rows), window_start=WINDOW_START, window_end=WINDOW_END
    )
    legs, seats, mid2 = leg_arrays(rows)
    assert legs[atm_leg(legs, seats, mid2)] == picked.ticker == LEG_B


def test_a_print_at_the_yes_bid_consumed_the_yes_side() -> None:
    assert classify_print(4_000, 4_000, 5_900) == YES_SIDE


def test_a_print_at_the_no_bid_consumed_the_no_side() -> None:
    assert classify_print(4_100, 4_000, PRICE_TICKS - 4_100) == NO_SIDE


def test_a_print_matching_neither_side_is_unclassified() -> None:
    assert classify_print(4_050, 4_000, 5_900) == NO_MATCH
