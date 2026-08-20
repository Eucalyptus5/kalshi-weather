from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pytest

from bot.lag.fee_floor import published_maker_fee
from bot.lag.fill_convention import NO, NO_CONTRACTS, REST_S, YES, YES_CONTRACTS, Fill
from bot.lag.maker_edge import (
    HORIZONS_S,
    PRIMARY_HORIZON_S,
    EdgeCurve,
    HorizonEdges,
    QuoteBook,
    edge_cents,
    evidence_window,
    mark_out_cents,
    mid2_at,
    quote_book,
    resolve_curve,
    resolve_edges,
    window_cap_s,
)
from bot.lag.maker_headroom import maker_fee_cents_per_contract
from bot.lag.tape_studies import RunScope, Screened, load_run_scope
from bot.replay.artifacts import LADDER_SCHEMA
from tests.test_tape_studies import BLINK, DISCOVERY_DAY, HOLDOUT_DAY, SERIES, scope_dir


UTC = timezone.utc
TICKER = "KXHIGHDEN-26JUL18-T70"
BRACKET = "KXHIGHDEN-26JUL18-T72"
PRICE = Decimal("0.0001")
SIZE = Decimal("0.01")

HALF = "0.5000"
FELL = "0.4960"
CENT_LOWER = "0.4800"
SIBLING_YES = "0.1000"
SIBLING_NO = "0.8000"
EMPTY = "0.0000"
GONE = Decimal("0")

T0 = datetime(2026, 7, 18, 12, tzinfo=UTC)
FILL_S = 200
MAKER_RATE = Decimal("0.0175")
FREE = Decimal("0")
STRIKE = Decimal("0.50")
CHEAP = Decimal("0.05")
TRUNCATED = Decimal(12)


def price(value: str) -> str:
    return str(Decimal(value).quantize(PRICE))


def size(value: str) -> str:
    return str(Decimal(value).quantize(SIZE))


def at(seconds: int, *, base: datetime = T0) -> datetime:
    return base + timedelta(seconds=seconds)


def book_row(
    row_id: int, when: datetime, yes_bid: str, no_bid: str, *, ticker: str = TICKER
) -> dict:
    yes = [] if Decimal(yes_bid) == GONE else [(price(yes_bid), size("3"))]
    no = [] if Decimal(no_bid) == GONE else [(price(no_bid), size("5"))]
    top_yes, yes_depth = yes[0] if yes else (price("0"), size("0"))
    top_no, no_depth = no[0] if no else (price("0"), size("0"))
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": when,
        "ts_ms": None,
        "yes_bid": top_yes,
        "yes_bid_depth": yes_depth,
        "yes_ask": price(str(Decimal("1") - Decimal(top_no))),
        "yes_ask_depth": no_depth,
        "no_bid": top_no,
        "no_bid_depth": no_depth,
        "no_ask": price(str(Decimal("1") - Decimal(top_yes))),
        "no_ask_depth": yes_depth,
        "yes_prices": [level for level, _ in yes],
        "yes_sizes": [depth for _, depth in yes],
        "yes_levels": len(yes),
        "no_prices": [level for level, _ in no],
        "no_sizes": [depth for _, depth in no],
        "no_levels": len(no),
    }


def book(rows: Sequence[dict]) -> QuoteBook:
    return quote_book(pa.Table.from_pylist(list(rows), schema=LADDER_SCHEMA), TICKER)


def flat_book(base: datetime = T0) -> QuoteBook:
    return book(
        [
            book_row(1, at(0, base=base), HALF, HALF),
            book_row(2, at(FILL_S, base=base), HALF, HALF),
            book_row(3, at(700, base=base), HALF, HALF),
        ]
    )


# The mid steps by 0.20c, which the whole-cent quote grid cannot express; the artifact stores
# prices to four places, so the fixture reaches the pre-registered literal the grid does not.
def falling_book(base: datetime = T0) -> QuoteBook:
    return book(
        [
            book_row(1, at(0, base=base), HALF, HALF),
            book_row(2, at(FILL_S, base=base), HALF, HALF),
            book_row(3, at(260, base=base), FELL, HALF),
            book_row(4, at(700, base=base), FELL, HALF),
        ]
    )


def fill(
    side: str,
    *,
    placed: int = 0,
    filled: int = FILL_S,
    contracts: Decimal | None = None,
    price: Decimal = STRIKE,
    base: datetime = T0,
) -> Fill:
    return Fill(
        ticker=TICKER,
        side=side,
        placement_price=price,
        contracts=(YES_CONTRACTS if side == YES else NO_CONTRACTS)
        if contracts is None
        else contracts,
        placed_at=at(placed, base=base),
        filled_at=at(filled, base=base),
    )


def scope_at(tmp_path: Path) -> RunScope:
    return load_run_scope(scope_dir(tmp_path))


def resolve(
    tmp_path: Path,
    quotes: QuoteBook,
    fills: Sequence[Fill],
    *,
    rate: Decimal = FREE,
    event_date: date = DISCOVERY_DAY,
) -> EdgeCurve:
    return resolve_curve(
        book=quotes,
        fills=fills,
        scope=scope_at(tmp_path),
        series=SERIES,
        event_date=event_date,
        maker_rate=rate,
    )


def test_a_sibling_brackets_book_is_not_read_as_our_own() -> None:
    ladder = pa.Table.from_pylist(
        [
            book_row(1, at(0), HALF, HALF),
            book_row(2, at(FILL_S), HALF, HALF),
            book_row(3, at(260), SIBLING_YES, SIBLING_NO, ticker=BRACKET),
            book_row(4, at(700), HALF, HALF),
        ],
        schema=LADDER_SCHEMA,
    )

    ours = quote_book(ladder, TICKER)
    theirs = quote_book(ladder, BRACKET)

    assert ours.stamps == (at(0), at(FILL_S), at(700))
    assert theirs.stamps == (at(260),)
    assert mid2_at(ours, at(260)) == 10_000
    assert mid2_at(theirs, at(260)) == 3_000
    assert mark_out_cents(
        ours, anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=YES
    ) == Decimal("0")
    assert mark_out_cents(theirs, anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=YES) is None


def test_two_hundred_doubled_ticks_is_one_cent_on_one_contract() -> None:
    moved = book(
        [
            book_row(1, at(0), HALF, HALF),
            book_row(2, at(60), CENT_LOWER, HALF),
            book_row(3, at(700), CENT_LOWER, HALF),
        ]
    )

    assert mid2_at(moved, at(0)) == 10_000
    assert mid2_at(moved, at(60)) == 9_800
    assert mark_out_cents(moved, anchor=at(0), horizon_s=60, side=YES) == Decimal("-1.00")


def test_a_mid_is_undefined_off_the_book_and_on_a_one_sided_row() -> None:
    moved = book(
        [
            book_row(1, at(0), HALF, HALF),
            book_row(2, at(230), HALF, EMPTY),
        ]
    )

    assert mid2_at(moved, at(0)) == 10_000
    assert mid2_at(moved, at(230)) is None
    assert mid2_at(moved, at(400)) is None
    assert mid2_at(moved, at(-1)) is None


def test_a_mid_past_the_last_row_is_undefined_even_when_that_row_is_two_sided(
    tmp_path: Path,
) -> None:
    ended = book(
        [
            book_row(1, at(0), HALF, HALF),
            book_row(2, at(230), HALF, HALF),
        ]
    )

    curve = resolve(tmp_path, ended, [fill(YES)])

    assert mid2_at(ended, at(230)) == 10_000
    assert mid2_at(ended, at(400)) is None
    assert mid2_at(ended, at(-1)) is None
    assert mark_out_cents(ended, anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=YES) is None
    assert curve.gate.dropped == 1
    assert curve.gate.edges == ()
    assert curve.by_horizon[10].edges[0].edge_cents_per_contract == Decimal("0.50")


def test_the_mark_out_signs_off_the_side_the_quote_filled() -> None:
    moved = falling_book()

    long_yes = mark_out_cents(moved, anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=YES)
    long_no = mark_out_cents(moved, anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=NO)

    assert long_yes == Decimal("-0.20")
    assert long_no == Decimal("0.20")


def test_a_free_fill_pays_the_half_tick_less_the_adverse_move() -> None:
    moved = falling_book()
    mark_out = mark_out_cents(moved, anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=YES)

    edge = edge_cents(mark_out=mark_out, contracts=YES_CONTRACTS, price=STRIKE, rate=FREE)

    assert maker_fee_cents_per_contract(YES_CONTRACTS, STRIKE, FREE) == Decimal("0")
    assert mark_out == Decimal("-0.20")
    assert edge == Decimal("0.30")
    assert edge != Decimal("0.70")


def test_a_free_flat_fill_keeps_the_whole_half_tick() -> None:
    flat = mark_out_cents(flat_book(), anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=YES)

    assert flat == Decimal("0")
    assert maker_fee_cents_per_contract(YES_CONTRACTS, STRIKE, FREE) == Decimal("0.0000")
    assert edge_cents(mark_out=flat, contracts=YES_CONTRACTS, price=STRIKE, rate=FREE) == Decimal(
        "0.50"
    )


def test_the_fee_spreads_the_ceilinged_aggregate_over_the_fill_own_size() -> None:
    fee = maker_fee_cents_per_contract(YES_CONTRACTS, STRIKE, MAKER_RATE)

    assert published_maker_fee(YES_CONTRACTS, STRIKE, MAKER_RATE) == Decimal("0.06")
    assert fee == Decimal("0.4854")
    assert fee != Decimal("0.49")
    assert edge_cents(
        mark_out=Decimal("0"), contracts=YES_CONTRACTS, price=STRIKE, rate=MAKER_RATE
    ) == Decimal("0.0146")
    assert edge_cents(
        mark_out=Decimal("-0.20"), contracts=YES_CONTRACTS, price=STRIKE, rate=MAKER_RATE
    ) == Decimal("-0.1854")


def test_each_side_is_charged_the_fee_its_own_size_earns(tmp_path: Path) -> None:
    yes_fee = maker_fee_cents_per_contract(YES_CONTRACTS, STRIKE, MAKER_RATE)
    no_fee = maker_fee_cents_per_contract(NO_CONTRACTS, STRIKE, MAKER_RATE)

    curve = resolve(tmp_path, flat_book(), [fill(YES), fill(NO)], rate=MAKER_RATE)
    priced = {edge.side: edge.edge_cents_per_contract for edge in curve.gate.edges}
    sized = {edge.side: edge.contracts for edge in curve.gate.edges}

    assert no_fee == Decimal("0.4615")
    assert no_fee != yes_fee
    assert priced[YES] == Decimal("0.0146")
    assert priced[NO] == Decimal("0.0385")
    assert sized[YES] == YES_CONTRACTS
    assert sized[NO] == NO_CONTRACTS


def test_the_fee_is_charged_at_the_fill_own_placement_price(tmp_path: Path) -> None:
    cheap = maker_fee_cents_per_contract(YES_CONTRACTS, CHEAP, MAKER_RATE)

    curve = resolve(tmp_path, flat_book(), [fill(YES, price=CHEAP)], rate=MAKER_RATE)
    edge = curve.gate.edges[0]

    assert published_maker_fee(YES_CONTRACTS, CHEAP, MAKER_RATE) == Decimal("0.02")
    assert cheap == Decimal("0.1618")
    assert cheap != maker_fee_cents_per_contract(YES_CONTRACTS, STRIKE, MAKER_RATE)
    assert edge.placement_price == CHEAP
    assert edge.edge_cents_per_contract == Decimal("0.3382")
    assert edge.edge_cents_per_contract != Decimal("0.0146")


def test_a_truncated_size_reads_the_fee_and_the_verdict_wrong() -> None:
    assert maker_fee_cents_per_contract(YES_CONTRACTS, STRIKE, MAKER_RATE) == Decimal("0.4854")
    assert maker_fee_cents_per_contract(TRUNCATED, STRIKE, MAKER_RATE) == Decimal("0.5000")
    assert edge_cents(
        mark_out=Decimal("-0.20"), contracts=YES_CONTRACTS, price=STRIKE, rate=MAKER_RATE
    ) == Decimal("-0.1854")
    assert edge_cents(
        mark_out=Decimal("-0.20"), contracts=TRUNCATED, price=STRIKE, rate=MAKER_RATE
    ) == Decimal("-0.2000")


def test_the_truncated_size_is_the_one_place_the_verdict_turns() -> None:
    modelled = edge_cents(
        mark_out=Decimal("0"), contracts=YES_CONTRACTS, price=STRIKE, rate=MAKER_RATE
    )
    truncated = edge_cents(
        mark_out=Decimal("0"), contracts=TRUNCATED, price=STRIKE, rate=MAKER_RATE
    )

    assert modelled == Decimal("0.0146")
    assert truncated == Decimal("0.0000")
    assert modelled > 0
    assert not truncated > 0


def test_the_side_is_carried_rather_than_inferred_from_the_size(tmp_path: Path) -> None:
    both = [fill(YES), fill(NO, contracts=YES_CONTRACTS)]

    curve = resolve(tmp_path, flat_book(), both)
    edges = curve.gate.edges

    assert {edge.contracts for edge in edges} == {YES_CONTRACTS}
    assert [edge.ticker for edge in edges] == [TICKER, TICKER]
    assert [edge.side for edge in edges] == [YES, NO]
    assert [edge.placement_price for edge in edges] == [STRIKE, STRIKE]
    assert [edge.horizon_s for edge in edges] == [PRIMARY_HORIZON_S, PRIMARY_HORIZON_S]


@pytest.mark.parametrize(
    ("horizon_s", "cap_s"),
    [(1, 301), (10, 310), (60, 360), (300, 600)],
)
def test_the_window_cap_is_the_longest_rest_plus_the_horizon(horizon_s: int, cap_s: int) -> None:
    assert window_cap_s(horizon_s) == cap_s
    assert window_cap_s(horizon_s) == REST_S + horizon_s


def test_the_gate_reads_the_sixty_second_group_by_name() -> None:
    empty = Screened(
        kept=(), candidates=0, excluded=0, out_of_scope=0, out_of_window=0, by_class={}
    )
    curve = EdgeCurve(
        by_horizon={
            horizon_s: HorizonEdges(
                horizon_s=horizon_s,
                window_cap_s=window_cap_s(horizon_s),
                edges=(),
                modelled=0,
                dropped=0,
                screened=empty,
            )
            for horizon_s in reversed(HORIZONS_S)
        }
    )

    assert tuple(curve.by_horizon) == tuple(reversed(HORIZONS_S))
    assert curve.gate.horizon_s == PRIMARY_HORIZON_S
    assert curve.gate.window_cap_s == 360


def test_the_four_horizons_come_back_together(tmp_path: Path) -> None:
    curve = resolve(tmp_path, falling_book(), [fill(YES)])

    assert tuple(curve.by_horizon) == HORIZONS_S
    assert curve.gate is curve.by_horizon[PRIMARY_HORIZON_S]
    assert [group.horizon_s for group in curve.by_horizon.values()] == list(HORIZONS_S)
    assert curve.by_horizon[1].edges[0].edge_cents_per_contract == Decimal("0.50")
    assert curve.by_horizon[10].edges[0].edge_cents_per_contract == Decimal("0.50")
    assert curve.gate.edges[0].edge_cents_per_contract == Decimal("0.30")
    assert curve.by_horizon[300].edges[0].edge_cents_per_contract == Decimal("0.30")


def test_every_edge_is_stamped_with_the_horizon_that_scored_it(tmp_path: Path) -> None:
    curve = resolve(tmp_path, falling_book(), [fill(YES)])

    assert [group.edges[0].horizon_s for group in curve.by_horizon.values()] == list(HORIZONS_S)
    assert curve.by_horizon[1].edges[0].horizon_s == 1
    assert curve.by_horizon[300].edges[0].horizon_s == 300


def test_a_one_sided_horizon_end_drops_the_fill_at_that_horizon_alone(tmp_path: Path) -> None:
    rows = [
        book_row(1, at(0), HALF, HALF),
        book_row(2, at(FILL_S), HALF, HALF),
        book_row(3, at(230), HALF, EMPTY),
        book_row(4, at(400), HALF, HALF),
        book_row(5, at(700), HALF, HALF),
    ]

    curve = resolve(tmp_path, book(rows), [fill(YES), fill(YES, placed=300, filled=400)])

    assert {group.horizon_s: group.dropped for group in curve.by_horizon.values()} == {
        1: 0,
        10: 0,
        60: 1,
        300: 0,
    }
    assert curve.gate.modelled == 2
    assert len(curve.gate.edges) == 1
    assert curve.gate.dropped_fraction == Decimal("0.5")
    assert curve.by_horizon[10].dropped_fraction == Decimal("0")


def test_a_one_sided_book_at_the_fill_drops_it_as_surely_as_at_the_horizon(
    tmp_path: Path,
) -> None:
    quotes = book(
        [
            book_row(1, at(0), HALF, HALF),
            book_row(2, at(FILL_S), HALF, EMPTY),
            book_row(3, at(240), HALF, HALF),
            book_row(4, at(700), HALF, HALF),
        ]
    )

    curve = resolve(tmp_path, quotes, [fill(YES)])

    assert mid2_at(quotes, at(FILL_S)) is None
    assert mid2_at(quotes, at(260)) == 10_000
    assert mark_out_cents(quotes, anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=YES) is None
    assert curve.gate.dropped == 1
    assert curve.gate.edges == ()


def test_an_exclusion_clear_of_the_gate_still_drops_the_three_hundred_second_point(
    tmp_path: Path,
) -> None:
    base = BLINK - timedelta(seconds=420)

    curve = resolve(
        tmp_path,
        flat_book(base),
        [fill(YES, base=base)],
        event_date=HOLDOUT_DAY,
    )

    assert curve.gate.screened.candidates == 1
    assert curve.gate.screened.excluded == 0
    assert len(curve.gate.edges) == 1
    assert curve.by_horizon[300].screened.excluded == 1
    assert curve.by_horizon[300].screened.excluded_fraction == Decimal("1")
    assert curve.by_horizon[300].edges == ()


def test_a_horizon_counts_its_exclusions_apart_from_its_drops(tmp_path: Path) -> None:
    early = BLINK - timedelta(seconds=420)
    late = BLINK + timedelta(seconds=600)
    quotes = book(
        [
            book_row(1, early, HALF, HALF),
            book_row(2, at(FILL_S, base=early), HALF, HALF),
            book_row(3, BLINK, HALF, EMPTY),
            book_row(4, late, HALF, HALF),
            book_row(5, at(FILL_S, base=late), HALF, HALF),
            book_row(6, at(400, base=late), HALF, EMPTY),
            book_row(7, at(2000, base=late), HALF, HALF),
        ]
    )

    curve = resolve(
        tmp_path,
        quotes,
        [fill(YES, base=early), fill(YES, base=late)],
        event_date=HOLDOUT_DAY,
    )
    far = curve.by_horizon[300]

    assert curve.gate.modelled == 2
    assert curve.gate.dropped == 0
    assert curve.gate.screened.excluded == 0
    assert len(curve.gate.edges) == 2
    assert far.modelled == 2
    assert far.dropped == 1
    assert far.dropped_fraction == Decimal("0.5")
    assert far.screened.excluded == 1
    assert far.screened.candidates == 2
    assert far.edges == ()


def test_a_horizon_that_modelled_no_fills_has_no_dropped_fraction(tmp_path: Path) -> None:
    empty = resolve_edges(
        book=flat_book(),
        fills=[],
        scope=scope_at(tmp_path),
        series=SERIES,
        event_date=DISCOVERY_DAY,
        maker_rate=FREE,
        horizon_s=PRIMARY_HORIZON_S,
    )

    assert empty.modelled == 0
    assert empty.dropped == 0
    assert empty.edges == ()
    with pytest.raises(InvalidOperation):
        _ = empty.dropped_fraction


def test_the_mark_out_anchors_on_the_fill_and_the_window_opens_at_the_placement(
    tmp_path: Path,
) -> None:
    moved = falling_book()
    placed = fill(YES)

    mark_out = mark_out_cents(moved, anchor=placed.filled_at, horizon_s=PRIMARY_HORIZON_S, side=YES)
    window = evidence_window(
        placed, series=SERIES, event_date=DISCOVERY_DAY, horizon_s=PRIMARY_HORIZON_S
    )
    curve = resolve(tmp_path, moved, [placed])

    assert mark_out == Decimal("-0.20")
    assert edge_cents(
        mark_out=mark_out, contracts=YES_CONTRACTS, price=STRIKE, rate=FREE
    ) == Decimal("0.30")
    assert curve.gate.edges[0].edge_cents_per_contract == Decimal("0.30")
    assert mark_out_cents(
        moved, anchor=placed.placed_at, horizon_s=PRIMARY_HORIZON_S, side=YES
    ) == Decimal("0")
    assert window.start == T0
    assert window.end == T0 + timedelta(seconds=260)
    assert (window.end - window.start).total_seconds() == 260
    assert (window.end - window.start) != timedelta(seconds=PRIMARY_HORIZON_S)
    assert (window.end - window.start) != timedelta(seconds=window_cap_s(PRIMARY_HORIZON_S))


def test_the_seam_fill_at_the_published_rate_reads_the_fractional_size(tmp_path: Path) -> None:
    moved = falling_book()

    curve = resolve(tmp_path, moved, [fill(YES)], rate=MAKER_RATE)
    mark_out = mark_out_cents(moved, anchor=at(FILL_S), horizon_s=PRIMARY_HORIZON_S, side=YES)

    assert curve.gate.edges[0].edge_cents_per_contract == Decimal("-0.1854")
    assert edge_cents(
        mark_out=mark_out, contracts=TRUNCATED, price=STRIKE, rate=MAKER_RATE
    ) == Decimal("-0.2000")


def test_every_horizon_screens_its_own_windows(tmp_path: Path) -> None:
    scope = scope_at(tmp_path)
    placed = fill(YES)

    groups = [
        resolve_edges(
            book=falling_book(),
            fills=[placed],
            scope=scope,
            series=SERIES,
            event_date=DISCOVERY_DAY,
            maker_rate=FREE,
            horizon_s=horizon_s,
        )
        for horizon_s in HORIZONS_S
    ]

    assert [group.window_cap_s for group in groups] == [301, 310, 360, 600]
    assert all(group.screened.candidates == 1 for group in groups)
    assert all(group.modelled == 1 for group in groups)
