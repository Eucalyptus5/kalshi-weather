from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Sequence

import pyarrow as pa
import pytest

from bot.lag.fee_floor import published_taker_fee
from bot.lag.ladder_consistency import (
    CITY_DAY_MIN_DISCOVERY,
    CITY_DAY_MIN_HOLDOUT,
    DEPTH_MIN,
    EXCESS_BAR,
    LADDER_LEGS,
    MONOTONICITY,
    SPREAD_TICKS_PER_LEG,
    STREAM_FAMILY,
    STREAM_LEGS,
    SUM_BUY,
    SUM_SELL,
    build_ladder,
    build_ladder_tape,
    fee_floor_cents,
    ladder_episodes,
)
from bot.replay.artifacts import TOUCH_SCHEMA


SERIES = "KXHIGHNY"
EVENT_DATE = date(2026, 7, 1)
LEGS = (
    "KXHIGHNY-26JUL01-T93",
    "KXHIGHNY-26JUL01-B93.5",
    "KXHIGHNY-26JUL01-B95.5",
    "KXHIGHNY-26JUL01-B97.5",
    "KXHIGHNY-26JUL01-B99.5",
    "KXHIGHNY-26JUL01-T100",
)
OUTSIDE = "KXHIGHNY-26JUL01-B101.5"
START = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)
PERSIST = Decimal("1")

Quote = tuple[str, str, str, str]

DEAD: Quote = ("0.0000", "0.00", "1.0000", "0.00")
CLEAN: tuple[Quote, ...] = (
    ("0.10", "50", "0.12", "50"),
    ("0.20", "50", "0.22", "50"),
    ("0.30", "50", "0.32", "50"),
    ("0.20", "50", "0.22", "50"),
    ("0.10", "50", "0.12", "50"),
    ("0.05", "50", "0.07", "50"),
)
BUY: tuple[Quote, ...] = (
    ("0.08", "50", "0.10", "50"),
    ("0.18", "50", "0.20", "50"),
    ("0.28", "50", "0.30", "50"),
    ("0.18", "50", "0.20", "50"),
    ("0.08", "50", "0.10", "50"),
    ("0.06", "50", "0.08", "50"),
)
SELL: tuple[Quote, ...] = (
    ("0.20", "50", "0.22", "50"),
    ("0.20", "50", "0.22", "50"),
    ("0.25", "50", "0.27", "50"),
    ("0.20", "50", "0.22", "50"),
    ("0.10", "50", "0.12", "50"),
    ("0.08", "50", "0.10", "50"),
)
WIDE_ABOVE: Quote = ("0.12", "100", "0.14", "100")
DEEP_BELOW: Quote = ("0.98", "100", "0.99", "100")


def at(seconds: int) -> datetime:
    return START + timedelta(seconds=seconds)


def quote_row(seconds: int, position: int, quote: Quote) -> dict:
    yes_bid, yes_bid_depth, yes_ask, yes_ask_depth = quote
    return {
        "ticker": LEGS[position],
        "received_at": at(seconds),
        "ts_ms": None,
        "yes_bid": yes_bid,
        "yes_bid_depth": yes_bid_depth,
        "yes_ask": yes_ask,
        "yes_ask_depth": yes_ask_depth,
        "no_bid": str(Decimal("1") - Decimal(yes_ask)),
        "no_bid_depth": yes_ask_depth,
        "no_ask": str(Decimal("1") - Decimal(yes_bid)),
        "no_ask_depth": yes_bid_depth,
    }


def opening(book: Sequence[Quote]) -> list[dict]:
    return [quote_row(position, position, book[position]) for position in range(LADDER_LEGS)]


def tails(below: Quote, above: Quote) -> tuple[Quote, ...]:
    return (below, DEAD, DEAD, DEAD, DEAD, above)


def table(rows: Sequence[dict]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"id": index + 1, **row} for index, row in enumerate(rows)], schema=TOUCH_SCHEMA
    )


def ladder():
    return build_ladder(SERIES, EVENT_DATE, list(LEGS))


def run(rows: Sequence[dict], *, t_persist_s: Decimal = PERSIST):
    return ladder_episodes(build_ladder_tape(ladder(), table(rows)), t_persist_s=t_persist_s)


def only(result, stream: str):
    found = [episode for episode in result.episodes if episode.stream == stream]
    assert len(found) == 1
    return found[0]


def test_gate_constants_are_pinned():
    assert LADDER_LEGS == 6
    assert DEPTH_MIN == Decimal("10")
    assert EXCESS_BAR == Decimal("2")
    assert SPREAD_TICKS_PER_LEG == Decimal("1")
    assert CITY_DAY_MIN_DISCOVERY == 30
    assert CITY_DAY_MIN_HOLDOUT == 15
    assert STREAM_FAMILY == {SUM_BUY: "sum", SUM_SELL: "sum", MONOTONICITY: "monotonicity"}
    assert STREAM_LEGS == {SUM_BUY: 6, SUM_SELL: 6, MONOTONICITY: 2}


def test_six_leg_ladder_resolves_in_strike_order():
    built = ladder()
    assert built.legs == LEGS
    assert built.below == 0
    assert built.above == 5
    assert built.series == SERIES
    assert built.event_date == EVENT_DATE


def test_ladder_legs_sort_regardless_of_input_order():
    shuffled = [LEGS[3], LEGS[5], LEGS[0], LEGS[2], LEGS[4], LEGS[1]]
    assert build_ladder(SERIES, EVENT_DATE, shuffled).legs == LEGS


@pytest.mark.parametrize(
    "tickers",
    [
        pytest.param(list(LEGS[:5]), id="five_legs"),
        pytest.param([*LEGS, OUTSIDE], id="seven_legs"),
        pytest.param([*LEGS[:5], OUTSIDE], id="one_tail_only"),
        pytest.param([*LEGS[:5], "KXHIGHNY-26JUL02-T100"], id="other_event_date"),
        pytest.param([*LEGS[:5], "KXHIGHNY-26JUL01-X93"], id="unparseable"),
    ],
)
def test_incomplete_groups_are_not_ladders(tickers):
    assert build_ladder(SERIES, EVENT_DATE, tickers) is None


def test_published_fee_is_read_at_the_episode_size():
    assert published_taker_fee(Decimal("1"), Decimal("0.07")) == Decimal("0.01")
    assert published_taker_fee(Decimal("10"), Decimal("0.07")) == Decimal("0.05")
    one = fee_floor_cents(contracts=Decimal("1"), prices=[Decimal("0.07")])
    ten = fee_floor_cents(contracts=Decimal("10"), prices=[Decimal("0.07")])
    assert one == Decimal("2")
    assert ten == Decimal("1.5")
    assert (ten - SPREAD_TICKS_PER_LEG) * 2 == one - SPREAD_TICKS_PER_LEG


def test_spread_term_is_one_tick_per_leg():
    contracts = Decimal("10")
    price = Decimal("0.07")
    per_leg = Decimal(100) * published_taker_fee(contracts, price) / contracts
    six = fee_floor_cents(contracts=contracts, prices=[price] * 6)
    two = fee_floor_cents(contracts=contracts, prices=[price] * 2)
    assert six - 6 * per_leg == Decimal("6")
    assert two - 2 * per_leg == Decimal("2")


@pytest.mark.parametrize(
    ("contracts", "low", "high"),
    [
        pytest.param(Decimal("10.50"), Decimal("1.47"), Decimal("1.48"), id="ten_and_a_half"),
        pytest.param(Decimal("0.39"), Decimal("3.56"), Decimal("3.57"), id="under_one"),
    ],
)
def test_fractional_sizes_price_without_an_integer_assumption(contracts, low, high):
    floor = fee_floor_cents(contracts=contracts, prices=[Decimal("0.07")])
    assert low < floor < high


@pytest.mark.parametrize(
    ("contracts", "prices"),
    [
        pytest.param(Decimal("0"), [Decimal("0.07")], id="zero_contracts"),
        pytest.param(Decimal("-1"), [Decimal("0.07")], id="negative_contracts"),
        pytest.param(Decimal("10"), [], id="no_legs"),
    ],
)
def test_fee_floor_rejects_unpriceable_input(contracts, prices):
    with pytest.raises(ValueError):
        fee_floor_cents(contracts=contracts, prices=prices)


def test_correctly_ordered_ladder_fires_nothing():
    assert Decimal(CLEAN[0][2]) + Decimal(CLEAN[5][2]) < Decimal("1")
    result = run(opening(CLEAN) + [quote_row(6, 5, CLEAN[5])])
    assert result.episodes == ()
    assert result.censored == {SUM_BUY: 0, SUM_SELL: 0, MONOTONICITY: 0}


def test_six_asks_under_a_dollar_are_one_sum_buy_episode():
    result = run(opening(BUY) + [quote_row(6, 5, ("0.06", "50", "0.20", "50"))])
    episode = only(result, SUM_BUY)
    assert result.episodes == (episode,)
    assert episode.magnitude_cents == Decimal("2")
    assert episode.signed_magnitude_cents == Decimal("2")
    assert episode.legs == LADDER_LEGS
    assert episode.family == "sum"


def test_six_bids_over_a_dollar_are_one_sum_sell_episode():
    result = run(opening(SELL) + [quote_row(6, 5, ("0.02", "50", "0.10", "50"))])
    episode = only(result, SUM_SELL)
    assert result.episodes == (episode,)
    assert episode.magnitude_cents == Decimal("3")
    assert episode.signed_magnitude_cents == Decimal("-3")
    assert episode.legs == LADDER_LEGS


def test_crossed_tail_bids_are_one_monotonicity_episode():
    below = ("0.52", "50", "0.54", "50")
    above = ("0.52", "50", "0.54", "50")
    result = run(opening(tails(below, above)) + [quote_row(6, 5, ("0.40", "50", "0.54", "50"))])
    episode = only(result, MONOTONICITY)
    assert result.episodes == (episode,)
    assert episode.magnitude_cents == Decimal("4")
    assert episode.signed_magnitude_cents == Decimal("-4")
    assert episode.legs == 2
    assert episode.family == "monotonicity"


def test_empty_no_side_cannot_make_a_sum_buy():
    book = list(BUY)
    book[2] = ("0.28", "50", "1.0000", "0.00")
    result = run(opening(book) + [quote_row(6, 5, BUY[5])])
    assert [episode for episode in result.episodes if episode.stream == SUM_BUY] == []


def test_unquoted_leg_cannot_form_a_sell_package():
    quoted = ("0.30", "50", "0.32", "50")
    book = (quoted, quoted, quoted, quoted, DEAD, quoted)
    result = run(opening(book) + [quote_row(6, 5, quoted)])
    assert result.episodes == ()


def test_unprinted_leg_is_not_a_state():
    result = run(opening(BUY) + [quote_row(6, 5, ("0.06", "50", "0.20", "50"))])
    assert result.rows == 7
    for stream in (SUM_BUY, SUM_SELL, MONOTONICITY):
        assert result.incomplete_states[stream] == LADDER_LEGS - 1
    assert all(episode.start == at(5) for episode in result.episodes)


def test_contiguous_run_is_one_episode():
    rows = opening(BUY) + [
        quote_row(6, 4, BUY[4]),
        quote_row(7, 4, BUY[4]),
        quote_row(8, 5, ("0.06", "50", "0.20", "50")),
    ]
    episode = only(run(rows), SUM_BUY)
    assert episode.start == at(5)
    assert episode.end == at(8)
    assert episode.states == 3
    assert episode.duration_s == Decimal("3")
    assert isinstance(episode.duration_s, Decimal)


def test_violation_that_restarts_is_two_episodes():
    rows = opening(BUY) + [
        quote_row(6, 5, ("0.06", "50", "0.20", "50")),
        quote_row(7, 5, BUY[5]),
        quote_row(8, 5, ("0.06", "50", "0.20", "50")),
    ]
    found = [episode for episode in run(rows).episodes if episode.stream == SUM_BUY]
    assert [(episode.start, episode.end) for episode in found] == [
        (at(5), at(6)),
        (at(7), at(8)),
    ]


def test_run_open_at_the_last_row_is_censored():
    result = run(opening(BUY))
    assert [episode for episode in result.episodes if episode.stream == SUM_BUY] == []
    assert result.censored[SUM_BUY] == 1
    assert result.censored[SUM_SELL] == 0


def sell_book(depth: str) -> tuple[Quote, ...]:
    quoted = ("0.15", depth, "0.17", depth)
    return (quoted, quoted, quoted, quoted, quoted, ("0.45", depth, "0.47", depth))


def lock_rows(above: Quote, closing: Quote, *, seconds: int = 6) -> list[dict]:
    return opening(tails(DEEP_BELOW, above)) + [quote_row(seconds, 5, closing)]


def test_episode_clearing_every_gate_is_tradeable():
    episode = only(run(lock_rows(WIDE_ABOVE, ("0.01", "100", "0.03", "100"))), MONOTONICITY)
    assert episode.magnitude_cents == Decimal("10")
    assert episode.depth == Decimal("100")
    assert episode.fee_floor_cents == Decimal("2.88")
    assert episode.excess_cents == Decimal("7.12")
    assert episode.tradeable is True


def test_one_microsecond_under_persistence_is_not_tradeable():
    rows = lock_rows(WIDE_ABOVE, ("0.01", "100", "0.03", "100"))
    episode = only(run(rows, t_persist_s=Decimal("1.000001")), MONOTONICITY)
    assert episode.duration_s == Decimal("1")
    assert episode.excess_cents > 0
    assert episode.tradeable is False


def test_one_thin_leg_blocks_tradeability():
    thin = list(sell_book("50"))
    thin[3] = ("0.15", "9.99", "0.17", "50")
    result = run(opening(thin) + [quote_row(6, 5, ("0.05", "50", "0.47", "50"))])
    episode = only(result, SUM_SELL)
    assert episode.magnitude_cents == Decimal("20")
    assert episode.depth == Decimal("9.99")
    assert episode.excess_cents > 0
    assert episode.tradeable is False


def test_worst_state_governs_the_score():
    rows = opening(tails(DEEP_BELOW, WIDE_ABOVE)) + [
        quote_row(6, 5, WIDE_ABOVE),
        quote_row(7, 5, WIDE_ABOVE),
        quote_row(8, 5, ("0.045", "10", "0.06", "10")),
        quote_row(9, 5, ("0.01", "100", "0.03", "100")),
    ]
    episode = only(run(rows), MONOTONICITY)
    assert episode.states == 4
    assert episode.magnitude_cents == Decimal("2.5")
    assert episode.depth == Decimal("10")
    assert episode.fee_floor_cents == Decimal("2.6")
    assert episode.excess_cents == Decimal("-0.1")
    assert episode.tradeable is False

    first = fee_floor_cents(contracts=Decimal("100"), prices=[Decimal("0.02"), Decimal("0.12")])
    assert Decimal("10") - first > 0
    assert Decimal("100") >= DEPTH_MIN


@pytest.mark.parametrize(
    ("depth", "floor"),
    [
        pytest.param("20", Decimal("12.25"), id="shallow"),
        pytest.param("200", Decimal("12.21"), id="deep"),
    ],
)
def test_floor_is_read_at_the_episode_size(depth, floor):
    result = run(opening(sell_book(depth)) + [quote_row(6, 5, ("0.05", depth, "0.47", depth))])
    episode = only(result, SUM_SELL)
    assert episode.depth == Decimal(depth)
    assert episode.fee_floor_cents == floor


def test_episode_over_the_floor_but_under_the_bar_is_tradeable():
    rows = lock_rows(("0.055", "100", "0.07", "100"), ("0.01", "100", "0.03", "100"))
    episode = only(run(rows), MONOTONICITY)
    assert episode.magnitude_cents == Decimal("3.5")
    assert episode.fee_floor_cents == Decimal("2.51")
    assert Decimal(0) < episode.excess_cents < EXCESS_BAR
    assert episode.tradeable is True


def test_fractional_depths_flow_through_the_floor():
    below = ("0.98", "104.80", "0.99", "104.80")
    above = ("0.12", "10.37", "0.14", "10.37")
    rows = opening(tails(below, above)) + [quote_row(6, 5, ("0.01", "10.37", "0.03", "10.37"))]
    episode = only(run(rows), MONOTONICITY)
    assert episode.depth == Decimal("10.37")
    assert Decimal("2.96") < episode.fee_floor_cents < Decimal("2.97")
    assert episode.excess_cents == episode.magnitude_cents - episode.fee_floor_cents
    assert episode.tradeable is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("yes_bid", "0.12345", id="price"),
        pytest.param("yes_bid_depth", "50.001", id="size"),
    ],
)
def test_values_off_the_stored_grid_name_their_ticker_and_field(field, value):
    rows = opening(CLEAN)
    rows[2][field] = value
    with pytest.raises(ValueError) as raised:
        build_ladder_tape(ladder(), table(rows))
    assert LEGS[2] in str(raised.value)
    assert field in str(raised.value)


def test_rows_out_of_id_order_raise():
    payload = [{"id": LADDER_LEGS - index, **row} for index, row in enumerate(opening(CLEAN))]
    with pytest.raises(ValueError, match="id order"):
        build_ladder_tape(ladder(), pa.Table.from_pylist(payload, schema=TOUCH_SCHEMA))


def test_row_outside_the_ladder_raises():
    rows = opening(CLEAN)
    rows[3]["ticker"] = OUTSIDE
    with pytest.raises(ValueError, match="not a leg"):
        build_ladder_tape(ladder(), table(rows))
