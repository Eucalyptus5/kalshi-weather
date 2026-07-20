from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from bot.execution.fees import taker_fee
from bot.lag.fee_floor import published_taker_fee
from bot.lag.tape_stats import ALPHA, HOLDOUT_ALPHA, cluster_bootstrap
from bot.lag.taker_flow import (
    CENT_BAR,
    HORIZONS_S,
    PRICE_TICKS,
    PRIMARY_HORIZON_S,
    PRINT_MIN_DISCOVERY,
    Anchors,
    HorizonResult,
    HorizonWindows,
    PrintOutcome,
    TickerBook,
    TickerPrints,
    build_ticker_book,
    build_ticker_prints,
    cluster_aggregates,
    horizon_result,
    print_net_cents,
    print_outcomes,
    resolve_anchors,
    resolve_horizon,
    screen_prints,
    yes_pressure,
)
from bot.replay.artifacts import TOUCH_SCHEMA, TRADES_SCHEMA


TICKER = "KXHIGHDEN-26JUL01-B85.5"
OTHER = "KXHIGHDEN-26JUL01-B87.5"
SPLIT = "discovery"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
START = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)


def at(seconds: int) -> datetime:
    return START + timedelta(seconds=seconds)


def us(seconds: int) -> int:
    return (at(seconds) - EPOCH) // timedelta(microseconds=1)


def touch(
    row_id: int,
    seconds: int,
    ts_ms: int | None,
    yes_bid: str = "0.40",
    no_bid: str = "0.58",
    ticker: str = TICKER,
) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at(seconds),
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
    seconds: int,
    ts_ms: int,
    side: str = "yes",
    yes_price: str = "0.45",
    count: str = "4",
    trade_id: str | None = None,
    ticker: str = TICKER,
) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at(seconds),
        "ts_ms": ts_ms,
        "yes_price": yes_price,
        "no_price": str(Decimal("1") - Decimal(yes_price)),
        "count": count,
        "taker_side": side,
        "trade_id": f"t{row_id}" if trade_id is None else trade_id,
    }


def touch_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=TOUCH_SCHEMA)


def trades_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=TRADES_SCHEMA)


def for_ticker(table: pa.Table, ticker: str) -> pa.Table:
    return table.filter(pc.equal(table.column("ticker"), ticker))


def pieces(
    touch_rows: list[dict], trade_rows: list[dict], horizon_s: int = PRIMARY_HORIZON_S
) -> tuple[TickerBook, TickerPrints, Anchors, HorizonWindows]:
    book = build_ticker_book(TICKER, touch_table(touch_rows))
    prints = build_ticker_prints(TICKER, screen_prints(trades_table(trade_rows)).kept)
    anchors = resolve_anchors(book, prints)
    return book, prints, anchors, resolve_horizon(book, prints, anchors, horizon_s=horizon_s)


def study(
    touch_rows: list[dict], trade_rows: list[dict], horizon_s: int = PRIMARY_HORIZON_S
) -> HorizonResult:
    touch_rows_table = touch_table(touch_rows)
    hygiene = screen_prints(trades_table(trade_rows))
    counts = hygiene.counts
    outcomes: list[PrintOutcome] = []
    for ticker in sorted(set(touch_rows_table.column("ticker").to_pylist())):
        book = build_ticker_book(ticker, for_ticker(touch_rows_table, ticker))
        prints = build_ticker_prints(ticker, for_ticker(hygiene.kept, ticker))
        anchors = resolve_anchors(book, prints)
        windows = resolve_horizon(book, prints, anchors, horizon_s=horizon_s)
        counts += windows.counts
        outcomes.extend(print_outcomes(book, prints, anchors, windows))
    return horizon_result(horizon_s=horizon_s, split=SPLIT, outcomes=outcomes, counts=counts)


RISING = [
    touch(1, 0, 1000, "0.40", "0.58"),
    touch(2, 2, 2000, "0.41", "0.57"),
    touch(3, 30, 3000, "0.50", "0.48"),
    touch(4, 100, 4000, "0.50", "0.48"),
]

END_TO_END_TOUCH = [
    touch(1, 0, 1000, "0.40", "0.58"),
    touch(2, 2, 2000, "0.45", "0.53"),
    touch(3, 50, 3000, "0.50", "0.48"),
    touch(4, 120, 4000, "0.50", "0.48"),
    touch(5, 0, 1000, "0.55", "0.43", ticker=OTHER),
    touch(6, 3, 2500, "0.60", "0.38", ticker=OTHER),
    touch(7, 40, 3500, "0.50", "0.48", ticker=OTHER),
    touch(8, 200, 4500, "0.50", "0.48", ticker=OTHER),
]

END_TO_END_TRADES = [
    trade(10, 1, 1500, "yes", "0.45"),
    trade(11, 2, 2000, "no", "0.55", ticker=OTHER),
]


def test_frozen_constants_hold() -> None:
    assert HORIZONS_S == (1, 10, 60, 300)
    assert PRIMARY_HORIZON_S == 60
    assert PRIMARY_HORIZON_S in HORIZONS_S
    assert PRINT_MIN_DISCOVERY == 5_000
    assert CENT_BAR == Decimal("1.0")
    assert PRICE_TICKS == 10_000
    assert ALPHA == 0.0125
    assert ALPHA * 4 == HOLDOUT_ALPHA
    assert HOLDOUT_ALPHA == 0.05


def test_yes_pressure_signs_the_outcome_the_aggressor_bought() -> None:
    assert yes_pressure("yes") == 1
    assert yes_pressure("no") == -1


@pytest.mark.parametrize("value", ["", "YES", "No", "both", "buy"])
def test_unsigned_taker_side_raises(value: str) -> None:
    with pytest.raises(ValueError):
        yes_pressure(value)


def test_mid_is_the_doubled_tick_mid() -> None:
    book = build_ticker_book(TICKER, touch_table([touch(1, 0, 1000, "0.40", "0.58")]))
    assert int(book.mid2[0]) == 8200
    assert bool(book.two_sided[0])


def test_an_empty_book_carries_no_mid() -> None:
    book = build_ticker_book(TICKER, touch_table([touch(1, 0, 1000, "0.00", "0.00")]))
    naive = Decimal(int(book.mid2[0])) / Decimal(2 * PRICE_TICKS)
    assert naive == Decimal("0.5")
    assert not bool(book.two_sided[0])


def test_anchor_is_the_first_delta_stamped_after_the_print() -> None:
    rows = [touch(1, 0, 1000), touch(2, 10, 2000), touch(3, 20, 3000), touch(4, 200, 4000)]
    _, _, anchors, _ = pieces(rows, [trade(10, 1, 1500), trade(11, 2, 2000)])
    assert anchors.resolved.tolist() == [True, True]
    assert anchors.index.tolist() == [1, 2]
    assert not anchors.host_clock.any()


def test_anchor_is_the_state_after_the_event() -> None:
    book, _, anchors, _ = pieces(
        [touch(1, 0, 1000, "0.40", "0.58"), touch(2, 2, 2000, "0.50", "0.48"), touch(3, 200, 3000)],
        [trade(10, 1, 1500)],
    )
    assert int(anchors.index[0]) == 1
    assert int(book.mid2[anchors.index[0]]) == 10200


def test_a_snapshot_between_the_print_and_the_delta_forces_the_host_clock() -> None:
    rows = [touch(1, 0, 1000), touch(2, 5, None), touch(3, 6, 2000), touch(4, 200, 3000)]
    _, _, anchors, _ = pieces(rows, [trade(10, 1, 1500)])
    assert anchors.index.tolist() == [1]
    assert anchors.host_clock.tolist() == [True]


def test_a_snapshot_after_the_anchor_leaves_the_exchange_clock_alone() -> None:
    rows = [touch(1, 0, 1000), touch(2, 5, 2000), touch(3, 6, None), touch(4, 200, 3000)]
    _, _, anchors, _ = pieces(rows, [trade(10, 1, 1500)])
    assert anchors.index.tolist() == [1]
    assert anchors.host_clock.tolist() == [False]


def test_no_later_delta_leaves_the_print_unresolved() -> None:
    rows = [touch(1, 0, 1000), touch(2, 10, 2000)]
    _, _, anchors, windows = pieces(rows, [trade(10, 1, 5000)])
    assert anchors.resolved.tolist() == [False]
    assert not windows.usable.any()
    assert windows.counts.unresolved == 1


def test_horizon_end_is_the_last_row_at_or_before_the_deadline() -> None:
    rows = [touch(1, 0, 1000), touch(2, 2, 2000), touch(3, 50, 3000), touch(4, 61, 4000)]
    rows.append(touch(5, 70, 5000))
    book, _, anchors, windows = pieces(rows, [trade(10, 1, 1500)])
    assert int(anchors.index[0]) == 1
    assert int(windows.end_index[0]) == 3
    assert int(windows.end_us[0]) == us(62)
    assert int(book.received_us[windows.end_index[0]]) == us(61)
    assert windows.usable.tolist() == [True]


def test_a_tape_that_stops_short_of_the_horizon_drops_the_print() -> None:
    rows = [touch(1, 0, 1000), touch(2, 2, 2000), touch(3, 50, 3000), touch(4, 61, 4000)]
    _, _, _, windows = pieces(rows, [trade(10, 1, 1500)])
    assert windows.usable.tolist() == [False]
    assert windows.counts.uncovered == 1
    assert windows.counts.one_sided == 0


def test_a_one_sided_anchor_drops_the_print() -> None:
    rows = [touch(1, 0, 1000), touch(2, 2, 2000, "0.41", "0.00"), touch(3, 200, 3000)]
    _, _, _, windows = pieces(rows, [trade(10, 1, 1500)])
    assert windows.usable.tolist() == [False]
    assert windows.counts.one_sided == 1


def test_a_one_sided_horizon_end_drops_the_print() -> None:
    rows = [touch(1, 0, 1000), touch(2, 2, 2000), touch(3, 50, 3000, "0.00", "0.00")]
    rows.append(touch(4, 200, 4000))
    _, _, _, windows = pieces(rows, [trade(10, 1, 1500)])
    assert windows.usable.tolist() == [False]
    assert windows.counts.one_sided == 1


def test_the_evidence_window_opens_at_the_earlier_of_print_and_anchor() -> None:
    rows = [touch(1, 0, 1000), touch(2, 1, 2000), touch(3, 100, 3000)]
    _, _, anchors, windows = pieces(rows, [trade(10, 5, 1500)])
    assert int(anchors.index[0]) == 1
    assert int(windows.start_us[0]) == us(1)
    assert int(windows.end_us[0]) == us(61)


def test_a_rising_mid_pays_the_yes_taker_and_costs_the_no_taker() -> None:
    trades = [trade(10, 1, 1500, "yes", "0.50", "10"), trade(11, 1, 1500, "no", "0.50", "10")]
    book, prints, anchors, windows = pieces(RISING, trades)
    outcomes = print_outcomes(book, prints, anchors, windows)
    assert [outcome.net_cents for outcome in outcomes] == [Decimal("54"), Decimal("-126")]
    assert outcomes[0].net_cents + outcomes[1].net_cents == Decimal("-72")


FEE_CASES = [
    (1, "0.05", 400, "0"),
    (1, "0.05", 0, "-2"),
    (4, "0.45", 1000, "6"),
    (4, "0.45", -1000, "-34"),
    (10, "0.50", 2000, "64"),
    (100, "0.50", 200, "-250"),
]


@pytest.mark.parametrize(("contracts", "price", "signed_move2", "expected"), FEE_CASES)
def test_net_cents_golden_table(
    contracts: int, price: str, signed_move2: int, expected: str
) -> None:
    assert print_net_cents(
        contracts=contracts, price=Decimal(price), signed_move2=signed_move2
    ) == Decimal(expected)


@pytest.mark.parametrize(("contracts", "price"), [(1, "0.05"), (4, "0.45"), (100, "0.50")])
def test_the_fee_module_agrees_with_the_published_formula(contracts: int, price: str) -> None:
    assert taker_fee(contracts, Decimal(price)) == published_taker_fee(contracts, Decimal(price))


def test_a_move_worth_the_round_trip_fee_nets_zero() -> None:
    fee_cents = Decimal(2) * Decimal(100) * taker_fee(1, Decimal("0.05"))
    assert fee_cents == Decimal("2")
    assert print_net_cents(contracts=1, price=Decimal("0.05"), signed_move2=400) == Decimal("0")


def test_a_fractional_count_raises() -> None:
    table = trades_table([trade(10, 1, 1500, count="3.50")])
    with pytest.raises(ValueError):
        build_ticker_prints(TICKER, table)


def test_screen_prints_drops_empty_sides_and_duplicate_trade_ids() -> None:
    rows = [
        trade(9, 1, 1500, trade_id="a"),
        trade(5, 1, 1500, trade_id="a"),
        trade(6, 2, 1600, side="", trade_id="b"),
        trade(7, 3, 1700, trade_id="c"),
    ]
    hygiene = screen_prints(trades_table(rows))
    assert hygiene.kept.column("id").to_pylist() == [5, 7]
    assert hygiene.counts.empty_side == 1
    assert hygiene.counts.duplicates == 1


def test_clusters_group_by_ticker_and_feed_the_bootstrap() -> None:
    outcomes = [
        PrintOutcome(ticker=TICKER, net_cents=Decimal("6"), contracts=4),
        PrintOutcome(ticker=OTHER, net_cents=Decimal("26"), contracts=4),
        PrintOutcome(ticker=TICKER, net_cents=Decimal("-2"), contracts=1),
    ]
    clusters = cluster_aggregates(outcomes)
    assert [item.cluster for item in clusters] == [TICKER, OTHER]
    assert [item.total for item in clusters] == [Decimal("4"), Decimal("26")]
    assert [item.weight for item in clusters] == [Decimal("5"), Decimal("4")]
    bootstrap = cluster_bootstrap(
        clusters,
        null_value=Decimal("0"),
        direction="greater",
        resamples=99,
        seed=7,
        ci_level=0.9,
    )
    assert bootstrap.n_clusters == 2
    assert bootstrap.estimate == Decimal("30") / Decimal("9")


def test_a_zero_contract_cluster_is_dropped() -> None:
    outcomes = [
        PrintOutcome(ticker=TICKER, net_cents=Decimal("0"), contracts=0),
        PrintOutcome(ticker=OTHER, net_cents=Decimal("26"), contracts=4),
    ]
    clusters = cluster_aggregates(outcomes)
    assert [item.cluster for item in clusters] == [OTHER]


def test_non_monotone_delta_stamps_are_counted() -> None:
    rows = [touch(1, 0, 3000), touch(2, 1, 2000), touch(3, 2, 4000)]
    book = build_ticker_book(TICKER, touch_table(rows))
    assert book.ts_violations == 1


def test_monotone_delta_stamps_count_no_violation() -> None:
    book = build_ticker_book(TICKER, touch_table([touch(1, 0, 1000), touch(2, 1, 1000)]))
    assert book.ts_violations == 0


def test_out_of_order_stamps_anchor_on_the_first_row_stamped_after_the_print() -> None:
    rows = [touch(1, 0, 1000), touch(2, 10, 3000), touch(3, 20, 2000), touch(4, 200, 4000)]
    book, _, anchors, _ = pieces(rows, [trade(10, 1, 1500), trade(11, 2, 2500)])
    assert book.ts_violations == 1
    assert anchors.resolved.tolist() == [True, True]
    assert anchors.index.tolist() == [1, 1]
    assert not anchors.host_clock.any()


def test_a_horizon_with_no_clusters_has_no_mean() -> None:
    result = study(END_TO_END_TOUCH, END_TO_END_TRADES, horizon_s=HORIZONS_S[-1])
    assert result.clusters == ()
    with pytest.raises(ValueError):
        result.mean_net_cents


def test_end_to_end_contract_weighted_mean() -> None:
    result = study(END_TO_END_TOUCH, END_TO_END_TRADES)
    assert result.horizon_s == PRIMARY_HORIZON_S
    assert result.split == SPLIT
    assert result.n_prints == 2
    assert result.contracts == 8
    assert [item.cluster for item in result.clusters] == [TICKER, OTHER]
    assert [item.total for item in result.clusters] == [Decimal("6"), Decimal("26")]
    assert result.mean_net_cents == Decimal("4")
    assert result.counts.unresolved == 0
    assert result.counts.uncovered == 0
    assert result.counts.one_sided == 0
    assert result.counts.host_clock == 0


def test_a_one_second_horizon_pays_only_the_fee() -> None:
    result = study(END_TO_END_TOUCH, END_TO_END_TRADES, horizon_s=1)
    assert result.mean_net_cents == Decimal("-3.5")


def test_the_longest_horizon_runs_off_the_tape() -> None:
    result = study(END_TO_END_TOUCH, END_TO_END_TRADES, horizon_s=HORIZONS_S[-1])
    assert result.n_prints == 0
    assert result.counts.uncovered == 2


def test_counts_add_across_tickers() -> None:
    result = study(END_TO_END_TOUCH, END_TO_END_TRADES + [trade(12, 1, 1500, side="")])
    assert result.counts.empty_side == 1
    assert result.n_prints == 2


def test_prints_arrays_carry_the_taker_price() -> None:
    prints = build_ticker_prints(
        TICKER,
        trades_table([trade(10, 1, 1500, "yes", "0.45"), trade(11, 2, 1600, "no", "0.45")]),
    )
    assert prints.prices == (Decimal("0.45"), Decimal("0.55"))
    assert prints.pressure.tolist() == [1, -1]
    assert prints.contracts.tolist() == [4, 4]
    assert np.array_equal(prints.received_us, np.array([us(1), us(2)]))
