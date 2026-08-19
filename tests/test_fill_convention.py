from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pyarrow as pa
import pytest

from bot.lag.depth_map import depth_at_price
from bot.lag.fill_convention import (
    NO_CONTRACTS,
    REST_S,
    YES_CONTRACTS,
    MarketDayFills,
    credited,
    sweep_market_day,
)
from bot.lag.placement_grid import PLACEMENTS, CloseSidecar, MarketClose
from bot.replay.artifacts import LADDER_SCHEMA, TRADES_SCHEMA


UTC = timezone.utc
SERIES = "KXHIGHDEN"
AUG13 = "KXHIGHDEN-26AUG13-T58"
AUG14 = "KXHIGHDEN-26AUG14-T58"
BRACKET = "KXHIGHDEN-26AUG13-B8889"
UNNAMED = "KXHIGHDEN-26AUG15-T58"
AUG13_CLOSE = datetime(2026, 8, 14, 6, 59, tzinfo=UTC)
AUG14_CLOSE = datetime(2026, 8, 15, 7, tzinfo=UTC)
FIRST = datetime(2026, 8, 13, 7, tzinfo=UTC)
OPENED = FIRST - timedelta(minutes=1)
SIBLING_AT = OPENED + timedelta(seconds=1)
STEP = timedelta(seconds=900)
RESTED = timedelta(seconds=REST_S)

PRICE = Decimal("0.0001")
SIZE = Decimal("0.01")
YES_BID = "0.40"
NO_BID = "0.58"
# A print stores no_price as the complement of yes_price on the same row, so a print at 0.42 is
# the one that can take the no quote resting at 0.58.
NO_TOUCH_PRINT = "0.42"
SIBLING_BID = "0.20"
DEEP = "500"
HALF = "0.50"

PLAN_LEVELS: tuple[tuple[str, str], ...] = (
    ("0.40", "9"),
    ("0.39", "1"),
    ("0.38", "1"),
    ("0.37", "2.50"),
)
SEAM_LEVELS: tuple[tuple[str, str], ...] = (
    ("0.40", "2.50"),
    ("0.39", "1"),
    ("0.38", "1"),
    ("0.37", "9"),
)


def price(value: str) -> str:
    return str(Decimal(value).quantize(PRICE))


def size(value: str) -> str:
    return str(Decimal(value).quantize(SIZE))


def ladder_row(
    row_id: int,
    at: datetime,
    yes: Sequence[tuple[str, str]],
    no: Sequence[tuple[str, str]],
    *,
    ticker: str = AUG13,
) -> dict:
    yes_bid, yes_depth = yes[0] if yes else ("0", "0")
    no_bid, no_depth = no[0] if no else ("0", "0")
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at,
        "ts_ms": None,
        "yes_bid": price(yes_bid),
        "yes_bid_depth": size(yes_depth),
        "yes_ask": price(str(Decimal("1") - Decimal(no_bid))),
        "yes_ask_depth": size(no_depth),
        "no_bid": price(no_bid),
        "no_bid_depth": size(no_depth),
        "no_ask": price(str(Decimal("1") - Decimal(yes_bid))),
        "no_ask_depth": size(yes_depth),
        "yes_prices": [price(level) for level, _ in yes],
        "yes_sizes": [size(depth) for _, depth in yes],
        "yes_levels": len(yes),
        "no_prices": [price(level) for level, _ in no],
        "no_sizes": [size(depth) for _, depth in no],
        "no_levels": len(no),
    }


def flat(
    row_id: int,
    at: datetime,
    *,
    yes_depth: str = "3",
    no_depth: str = "5",
    ticker: str = AUG13,
) -> dict:
    return ladder_row(row_id, at, [(YES_BID, yes_depth)], [(NO_BID, no_depth)], ticker=ticker)


def trade_row(
    row_id: int,
    at: datetime,
    yes_price: str,
    count: str,
    taker_side: str,
    *,
    ticker: str = AUG13,
) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at,
        "ts_ms": None,
        "yes_price": price(yes_price),
        "no_price": price(str(Decimal("1") - Decimal(yes_price))),
        "count": size(count),
        "taker_side": taker_side,
        "trade_id": f"t{row_id}",
    }


def ladder_table(rows: Sequence[dict]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=LADDER_SCHEMA)


def trades_table(rows: Sequence[dict]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=TRADES_SCHEMA)


def market(ticker: str, close: datetime) -> MarketClose:
    return MarketClose(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0],
        close_time=close,
        floor_strike=58,
        cap_strike=None,
        strike_type="greater",
        status="finalized",
        result="yes",
    )


SIDECAR = CloseSidecar(
    root=SERIES,
    markets={
        item.ticker: item
        for item in (
            market(AUG13, AUG13_CLOSE),
            market(BRACKET, AUG13_CLOSE),
            market(AUG14, AUG14_CLOSE),
        )
    },
    voided=(),
    sha256="0" * 64,
)


def sweep(
    rows: Sequence[dict],
    prints: Sequence[dict] = (),
    *,
    ticker: str = AUG13,
    sidecar: CloseSidecar = SIDECAR,
) -> MarketDayFills:
    return sweep_market_day(
        ladder=ladder_table(rows),
        prints=trades_table(prints),
        sidecar=sidecar,
        ticker=ticker,
    )


def test_the_convention_is_fixed_at_the_preregistered_sizes_and_rest() -> None:
    assert YES_CONTRACTS == Decimal("12.36")
    assert NO_CONTRACTS == Decimal("26")
    assert REST_S == 300


def test_the_credit_test_is_strict() -> None:
    assert credited(Decimal("2.50"), Decimal("2.50")) is False
    assert credited(Decimal("2.51"), Decimal("2.50")) is True
    assert credited(Decimal("3.00"), Decimal("3")) is False


@pytest.mark.parametrize(("traded", "fills"), [("3", 0), ("4", 1)])
def test_volume_fills_only_once_it_passes_the_depth_ahead(traded: str, fills: int) -> None:
    swept = sweep(
        [flat(1, OPENED, yes_depth="3")],
        [trade_row(1, FIRST + timedelta(seconds=10), YES_BID, traded, "no")],
    )

    assert swept.yes_fills == fills
    assert swept.no_fills == 0


@pytest.mark.parametrize(
    ("heavy", "light", "yes_fills", "no_fills"),
    [("no", "yes", 1, 0), ("yes", "no", 0, 1)],
)
def test_the_quotes_are_separated_by_taker_side_not_by_price(
    heavy: str, light: str, yes_fills: int, no_fills: int
) -> None:
    at = FIRST + timedelta(seconds=10)
    swept = sweep(
        [ladder_row(1, OPENED, [("0.40", "1")], [("0.60", "3")])],
        [trade_row(1, at, "0.40", "4", heavy), trade_row(2, at, "0.40", "1", light)],
    )

    assert swept.yes_fills == yes_fills
    assert swept.no_fills == no_fills
    assert [fill.side for fill in swept.fills] == ["yes"] * yes_fills + ["no"] * no_fills


@pytest.mark.parametrize(
    ("moved", "arrives_s", "fills"),
    [(True, 200, 0), (True, 100, 1), (False, 200, 1)],
)
def test_a_touch_that_leaves_the_placement_price_cancels_the_order(
    moved: bool, arrives_s: int, fills: int
) -> None:
    rows = [flat(1, OPENED)]
    if moved:
        rows.append(ladder_row(2, FIRST + timedelta(seconds=120), [("0.39", "3")], [(NO_BID, "5")]))

    swept = sweep(rows, [trade_row(1, FIRST + timedelta(seconds=arrives_s), YES_BID, "10", "no")])

    assert swept.yes_fills == fills


def test_a_touch_that_re_publishes_at_the_same_price_keeps_the_order_resting() -> None:
    rows = [
        flat(1, OPENED),
        ladder_row(2, FIRST + timedelta(seconds=120), [("0.39", "3")], [(NO_BID, "5")]),
    ]
    prints = [trade_row(1, FIRST + timedelta(seconds=200), NO_TOUCH_PRINT, "10", "yes")]

    swept = sweep(rows, prints)

    assert swept.no_fills == 1
    assert swept.yes_fills == 0


@pytest.mark.parametrize(("arrives_s", "fills"), [(250, 1), (350, 0)])
def test_a_touch_that_moves_only_after_the_deadline_still_ends_the_rest_at_it(
    arrives_s: int, fills: int
) -> None:
    rows = [
        flat(1, OPENED),
        ladder_row(2, FIRST + RESTED + timedelta(seconds=100), [("0.39", "3")], [(NO_BID, "5")]),
    ]

    swept = sweep(rows, [trade_row(1, FIRST + timedelta(seconds=arrives_s), YES_BID, "10", "no")])

    assert swept.yes_fills == fills


def test_a_fill_is_credited_whole_at_the_side_size() -> None:
    at = FIRST + timedelta(seconds=10)
    swept = sweep(
        [flat(1, OPENED)],
        [
            trade_row(1, at, YES_BID, DEEP, "no"),
            trade_row(2, at, NO_TOUCH_PRINT, DEEP, "yes"),
        ],
    )

    assert {fill.side: fill.contracts for fill in swept.fills} == {
        "yes": Decimal("12.36"),
        "no": Decimal("26"),
    }
    assert sum((fill.contracts for fill in swept.fills), Decimal(0)) == Decimal("38.36")


def test_both_sides_are_placed_at_every_instant() -> None:
    swept = sweep([flat(1, OPENED)])

    assert PLACEMENTS == 48
    assert len(swept.instants) == 48
    assert swept.offered == 96
    assert swept.yes_empty == 0
    assert swept.no_empty == 0
    assert swept.fills == ()


def test_the_two_sides_fill_counts_come_back_apart() -> None:
    prints = [
        trade_row(index + 1, FIRST + index * STEP + timedelta(seconds=10), YES_BID, "10", "no")
        for index in range(3)
    ]
    prints.append(trade_row(4, FIRST + timedelta(seconds=10), NO_TOUCH_PRINT, "10", "yes"))

    swept = sweep([flat(1, OPENED)], prints)

    yes_contracts = sum((fill.contracts for fill in swept.fills if fill.side == "yes"), Decimal(0))
    no_contracts = sum((fill.contracts for fill in swept.fills if fill.side == "no"), Decimal(0))
    pooled = swept.yes_fills + swept.no_fills

    assert (swept.yes_fills, swept.no_fills) == (3, 1)
    assert yes_contracts == Decimal("37.08")
    assert no_contracts == Decimal("26")
    assert yes_contracts + no_contracts != pooled * YES_CONTRACTS
    assert yes_contracts + no_contracts != pooled * NO_CONTRACTS


def test_an_empty_side_places_no_order_and_is_counted() -> None:
    rows = [ladder_row(1, OPENED, [(YES_BID, "3")], [])]

    swept = sweep(rows)

    assert rows[0]["yes_ask"] == "1.0000"
    assert swept.no_empty == 48
    assert swept.yes_empty == 0
    assert swept.offered == 48


def test_a_touch_resting_under_one_contract_is_placed_not_counted_empty() -> None:
    rows = [flat(1, OPENED, yes_depth=HALF)]

    swept = sweep(rows)

    assert Decimal(rows[0]["yes_bid_depth"]) < Decimal("1")
    assert swept.yes_empty == 0
    assert swept.no_empty == 0
    assert swept.offered == 96


def test_the_modelled_fill_rate_comes_back_beside_the_fills() -> None:
    prints = [
        trade_row(index + 1, FIRST + index * STEP + timedelta(seconds=10), YES_BID, "10", "no")
        for index in range(2)
    ]

    swept = sweep([flat(1, OPENED)], prints)

    assert len(swept.fills) == 2
    assert swept.fill_rate == Decimal("2")


def test_the_depth_read_is_named_by_price_not_by_level() -> None:
    row = ladder_row(1, OPENED, PLAN_LEVELS, [(NO_BID, "5")])

    assert depth_at_price(row, "yes", Decimal("0.37")) == (Decimal("2.50"), False)
    assert depth_at_price(row, "yes", Decimal("0.40")) == (Decimal("9"), False)


@pytest.mark.parametrize(("traded", "fills"), [("2.50", 0), ("2.51", 1)])
def test_a_print_at_the_last_resting_second_clears_the_depth_ahead_or_does_not(
    traded: str, fills: int
) -> None:
    swept = sweep(
        [ladder_row(1, OPENED, SEAM_LEVELS, [(NO_BID, "5")])],
        [trade_row(1, FIRST + RESTED, YES_BID, traded, "no")],
    )

    assert swept.yes_fills == fills


def test_the_sweep_reads_the_close_stored_for_the_ticker_it_sweeps() -> None:
    aug13 = sweep([flat(1, OPENED)], ticker=AUG13)
    aug14 = sweep([], ticker=AUG14)

    assert aug13.close == datetime(2026, 8, 14, 6, 59, tzinfo=UTC)
    assert aug14.close == datetime(2026, 8, 15, 7, tzinfo=UTC)
    assert aug13.close != aug14.close


def test_the_two_closes_generate_the_same_forty_eight_clock_instants() -> None:
    aug13 = sweep([], ticker=AUG13)
    aug14 = sweep([], ticker=AUG14)

    assert [instant.timetz() for instant in aug13.instants] == [
        instant.timetz() for instant in aug14.instants
    ]
    assert aug13.instants[0] == datetime(2026, 8, 13, 7, tzinfo=UTC)
    assert aug13.instants[47] == datetime(2026, 8, 13, 18, 45, tzinfo=UTC)
    assert aug14.instants[0] == datetime(2026, 8, 14, 7, tzinfo=UTC)
    assert aug14.instants[47] == datetime(2026, 8, 14, 18, 45, tzinfo=UTC)


def test_a_ticker_the_sidecar_does_not_name_is_refused_even_with_its_root_carried() -> None:
    assert UNNAMED.split("-")[0] == SIDECAR.root

    with pytest.raises(ValueError, match=UNNAMED):
        sweep([], ticker=UNNAMED)


def test_a_fill_carries_two_instants_neither_read_off_the_other() -> None:
    prints = [
        trade_row(1, FIRST + timedelta(seconds=10), YES_BID, "10", "no"),
        trade_row(2, FIRST + STEP + timedelta(seconds=250), YES_BID, "10", "no"),
    ]

    swept = sweep([flat(1, OPENED)], prints)

    assert [fill.placed_at for fill in swept.fills] == [FIRST, FIRST + STEP]
    assert [fill.filled_at for fill in swept.fills] == [
        FIRST + timedelta(seconds=10),
        FIRST + STEP + timedelta(seconds=250),
    ]
    assert {fill.filled_at - fill.placed_at for fill in swept.fills} == {
        timedelta(seconds=10),
        timedelta(seconds=250),
    }
    assert {fill.placed_at.utcoffset() for fill in swept.fills} == {timedelta(0)}
    assert {fill.filled_at.utcoffset() for fill in swept.fills} == {timedelta(0)}


def test_the_print_that_carries_the_volume_past_the_queue_is_the_fill() -> None:
    prints = [
        trade_row(1, FIRST + timedelta(seconds=10), YES_BID, "3", "no"),
        trade_row(2, FIRST + timedelta(seconds=20), YES_BID, "1", "no"),
        trade_row(3, FIRST + timedelta(seconds=30), YES_BID, "1", "no"),
    ]

    swept = sweep([flat(1, OPENED, yes_depth="3")], prints)

    assert [fill.filled_at for fill in swept.fills] == [FIRST + timedelta(seconds=20)]
    assert swept.fills[0].placement_price == Decimal("0.40")


def test_a_sibling_tickers_print_does_not_credit_our_quote() -> None:
    rows = [flat(1, OPENED), flat(2, OPENED, ticker=BRACKET)]
    prints = [trade_row(1, FIRST + timedelta(seconds=10), YES_BID, "10", "no", ticker=BRACKET)]

    assert sweep(rows, prints, ticker=AUG13).yes_fills == 0
    assert sweep(rows, prints, ticker=BRACKET).yes_fills == 1


def test_a_sibling_tickers_book_is_not_read_as_our_own() -> None:
    rows = [
        flat(1, OPENED, yes_depth="3"),
        ladder_row(2, SIBLING_AT, [(SIBLING_BID, "50")], [("0.79", "50")], ticker=BRACKET),
    ]
    prints = [
        trade_row(1, FIRST + timedelta(seconds=10), YES_BID, "10", "no"),
        trade_row(2, FIRST + timedelta(seconds=10), SIBLING_BID, "60", "no", ticker=BRACKET),
    ]

    assert rows[1]["received_at"] > rows[0]["received_at"]
    assert [fill.placement_price for fill in sweep(rows, prints, ticker=AUG13).fills] == [
        Decimal("0.40")
    ]
    assert [fill.placement_price for fill in sweep(rows, prints, ticker=BRACKET).fills] == [
        Decimal("0.20")
    ]
