import inspect
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

from bot.lag.lead_lag import (
    CITY_SERIES,
    CORRIDORS,
    MOVE_BAR_CENTS,
    PAIRS,
    WINDOW_S,
    AtmSeries,
    Pair,
    atm_series,
    follower_response,
    follower_view,
    leader_crossings,
    pair_episodes,
    round_trip_fee_cents,
)
from bot.main import STATIONS
from bot.replay.artifacts import TOUCH_SCHEMA


SERIES = "KXHIGHDEN"
OTHER_SERIES = "KXHIGHTOKC"
EVENT_DATE = date(2026, 7, 1)
START = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)
WINDOW_END = START + timedelta(hours=6)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LEG_A = "KXHIGHDEN-26JUL01-B93.5"
LEG_B = "KXHIGHDEN-26JUL01-B95.5"

CITY_TABLE: tuple[tuple[str, str], ...] = (
    ("DEN", "KXHIGHDEN"),
    ("OKC", "KXHIGHTOKC"),
    ("DFW", "KXHIGHTDAL"),
    ("AUS", "KXHIGHAUS"),
    ("SAT", "KXHIGHTSATX"),
    ("IAH", "KXHIGHTHOU"),
    ("MSY", "KXHIGHTNOLA"),
    ("ATL", "KXHIGHTATL"),
    ("MSP", "KXHIGHTMIN"),
    ("CHI", "KXHIGHCHI"),
    ("DCA", "KXHIGHTDC"),
    ("PHL", "KXHIGHPHIL"),
    ("NYC", "KXHIGHNY"),
    ("BOS", "KXHIGHTBOS"),
    ("PHX", "KXHIGHTPHX"),
    ("LAS", "KXHIGHTLV"),
    ("SFO", "KXHIGHTSFO"),
    ("LAX", "KXHIGHLAX"),
)

PAIR_TABLE: tuple[tuple[str, str, str], ...] = (
    ("gulf", "DEN", "OKC"),
    ("gulf", "OKC", "DFW"),
    ("gulf", "DFW", "AUS"),
    ("gulf", "AUS", "SAT"),
    ("gulf", "SAT", "IAH"),
    ("gulf", "IAH", "MSY"),
    ("gulf", "MSY", "ATL"),
    ("northeast", "MSP", "CHI"),
    ("northeast", "CHI", "DCA"),
    ("northeast", "DCA", "PHL"),
    ("northeast", "PHL", "NYC"),
    ("northeast", "NYC", "BOS"),
    ("southwest", "PHX", "LAS"),
    ("california", "SFO", "LAX"),
)

Point = tuple[int, str, bool]


def at(seconds: int) -> datetime:
    return START + timedelta(seconds=seconds)


def us(seconds: int) -> int:
    return (at(seconds) - EPOCH) // timedelta(microseconds=1)


def mid2(cents: str) -> int:
    doubled = Decimal(cents) * 200
    assert doubled == int(doubled)
    return int(doubled)


def quote(seconds: int, ticker: str, yes_bid: str, no_bid: str) -> dict:
    return {
        "ticker": ticker,
        "received_at": at(seconds),
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


def mid(seconds: int, ticker: str, dollars: str) -> dict:
    bid = Decimal(dollars) - Decimal("0.01")
    return quote(seconds, ticker, str(bid), str(Decimal("0.99") - Decimal(dollars)))


def table(rows: Sequence[dict], *, ids: Sequence[int] | None = None) -> pa.Table:
    numbered = list(ids) if ids is not None else list(range(1, len(rows) + 1))
    return pa.Table.from_pylist(
        [{"id": row_id, **row} for row_id, row in zip(numbered, rows, strict=True)],
        schema=TOUCH_SCHEMA,
    )


def read(rows: Sequence[dict]) -> AtmSeries | None:
    return atm_series(SERIES, EVENT_DATE, table(rows), window_start=START, window_end=WINDOW_END)


def series_of(points: Sequence[Point], *, series: str = SERIES, ticker: str = LEG_A) -> AtmSeries:
    return AtmSeries(
        series=series,
        event_date=EVENT_DATE,
        ticker=ticker,
        received_us=np.array([us(second) for second, _, _ in points], dtype=np.int64),
        mid2=np.array([mid2(cents) for _, cents, _ in points], dtype=np.int64),
        two_sided=np.array([live for _, _, live in points], dtype=bool),
    )


def ramp(seconds: Sequence[int], cents: Sequence[str]) -> list[Point]:
    return [(second, value, True) for second, value in zip(seconds, cents, strict=True)]


def follower_of(points: Sequence[Point]) -> AtmSeries:
    return series_of(points, series=OTHER_SERIES, ticker="KXHIGHTOKC-26JUL01-B93.5")


def test_city_series_covers_eighteen_recorded_roots():
    assert len(CITY_SERIES) == 18
    assert set(CITY_SERIES.values()) <= set(STATIONS)


@pytest.mark.parametrize(("city", "root"), CITY_TABLE, ids=[city for city, _ in CITY_TABLE])
def test_city_series_is_a_golden_table(city: str, root: str):
    assert CITY_SERIES[city] == root


def test_chicago_dallas_and_houston_break_the_station_code_rule():
    assert CITY_SERIES["CHI"] == "KXHIGHCHI"
    assert STATIONS["KXHIGHCHI"].station == "KMDW"
    assert CITY_SERIES["DFW"] == "KXHIGHTDAL"
    assert STATIONS["KXHIGHTDAL"].station == "KDFW"
    assert CITY_SERIES["IAH"] == "KXHIGHTHOU"
    assert STATIONS["KXHIGHTHOU"].station == "KHOU"
    stripped = {
        city: STATIONS[root].station
        for city, root in CITY_SERIES.items()
        if city not in {"CHI", "IAH"}
    }
    assert stripped == {city: f"K{city}" for city in stripped}


def test_mapped_roots_and_the_two_unpaired_ones_are_the_whole_universe():
    assert set(CITY_SERIES.values()) | {"KXHIGHTSEA", "KXHIGHMIA"} == set(STATIONS)
    assert len(CITY_SERIES) + 2 == len(STATIONS) == 20


def test_pairs_are_the_adjacent_members_of_each_corridor():
    assert len(PAIRS) == 14
    assert PAIRS[0] == Pair(corridor="gulf", upstream="DEN", downstream="OKC")
    assert [(pair.corridor, pair.upstream, pair.downstream) for pair in PAIRS] == list(PAIR_TABLE)


def test_corridor_pair_counts_are_seven_five_one_one():
    counts = {name: sum(1 for pair in PAIRS if pair.corridor == name) for name in CORRIDORS}
    assert counts == {"gulf": 7, "northeast": 5, "southwest": 1, "california": 1}


def test_every_pair_joins_two_cities_of_its_own_corridor():
    for pair in PAIRS:
        chain = CORRIDORS[pair.corridor]
        assert chain.index(pair.downstream) == chain.index(pair.upstream) + 1


def test_five_cent_bar_clears_a_round_trip_taker_fee_by_one_cent():
    fee = round_trip_fee_cents(Decimal("1"), Decimal("0.50"))
    assert fee == Decimal("4")
    assert MOVE_BAR_CENTS - fee == Decimal("1")


def test_atm_leg_is_the_one_nearest_half_over_the_day_not_at_the_open():
    rows = [
        mid(0, LEG_A, "0.50"),
        mid(0, LEG_B, "0.70"),
        mid(10, LEG_A, "0.90"),
        mid(10, LEG_B, "0.55"),
        mid(20, LEG_A, "0.90"),
        mid(20, LEG_B, "0.55"),
        mid(30, LEG_A, "0.90"),
        mid(30, LEG_B, "0.55"),
    ]
    picked = read(rows)
    assert picked.ticker == LEG_B
    assert picked.series == SERIES
    assert picked.event_date == EVENT_DATE
    assert picked.received_us.tolist() == [us(0), us(10), us(20), us(30)]
    assert picked.mid2.tolist() == [mid2("70"), mid2("55"), mid2("55"), mid2("55")]


def test_a_spike_off_half_does_not_cost_the_leg_the_pick():
    rows = [
        mid(0, LEG_A, "0.50"),
        mid(0, LEG_B, "0.56"),
        mid(10, LEG_A, "0.50"),
        mid(10, LEG_B, "0.56"),
        mid(20, LEG_A, "0.50"),
        mid(20, LEG_B, "0.56"),
        mid(30, LEG_A, "0.95"),
        mid(30, LEG_B, "0.56"),
        mid(40, LEG_A, "0.95"),
        mid(40, LEG_B, "0.56"),
    ]
    picked = read(rows)
    assert picked.ticker == LEG_A
    assert picked.two_sided.all()
    assert picked.mid2.tolist() == [mid2(value) for value in ("50", "50", "50", "95", "95")]


def test_ties_on_the_median_distance_go_to_the_lower_ticker():
    rows = [
        mid(0, LEG_B, "0.55"),
        mid(0, LEG_A, "0.45"),
        mid(10, LEG_B, "0.55"),
        mid(10, LEG_A, "0.45"),
    ]
    assert read(rows).ticker == LEG_A


def test_a_one_sided_leg_reading_exactly_half_is_never_selected():
    rows = [
        quote(0, LEG_A, "0.0000", "0.0000"),
        mid(0, LEG_B, "0.60"),
        quote(10, LEG_A, "0.0000", "0.0000"),
        mid(10, LEG_B, "0.60"),
    ]
    picked = read(rows)
    assert picked.ticker == LEG_B
    assert picked.two_sided.all()


def test_a_zero_no_bid_is_not_two_sided_though_the_stored_ask_reads_one():
    rows = [mid(0, LEG_A, "0.60"), quote(10, LEG_A, "0.60", "0.0000"), mid(20, LEG_A, "0.60")]
    assert table(rows).column("yes_ask").to_pylist()[1] == "1.0000"
    picked = read(rows)
    assert picked.two_sided.tolist() == [True, False, True]


def test_rows_outside_the_window_are_dropped_with_both_endpoints_kept():
    rows = [
        mid(-1, LEG_A, "0.50"),
        mid(0, LEG_A, "0.51"),
        mid(21600, LEG_A, "0.52"),
        mid(21601, LEG_A, "0.53"),
    ]
    picked = read(rows)
    assert picked.received_us.tolist() == [us(0), us(21600)]
    assert picked.mid2.tolist() == [mid2("51"), mid2("52")]


def test_no_two_sided_row_in_the_window_yields_nothing():
    rows = [mid(-5, LEG_B, "0.50"), quote(0, LEG_A, "0.60", "0.0000")]
    assert read(rows) is None


def test_out_of_order_ids_are_rejected():
    rows = [mid(0, LEG_A, "0.50"), mid(10, LEG_A, "0.51")]
    with pytest.raises(ValueError, match="id order"):
        atm_series(
            SERIES,
            EVENT_DATE,
            table(rows, ids=[7, 2]),
            window_start=START,
            window_end=WINDOW_END,
        )


def test_a_price_off_the_tick_grid_is_rejected():
    with pytest.raises(ValueError, match="grid"):
        read([quote(0, LEG_A, "0.50005", "0.49")])


def test_a_move_reaching_exactly_the_bar_crosses():
    crossings = leader_crossings(series_of(ramp([0, 30], ["50", "55"])))
    assert len(crossings) == 1
    assert crossings[0].anchor_us == us(0)
    assert crossings[0].cross_us == us(30)
    assert crossings[0].deadline_us == us(WINDOW_S)
    assert crossings[0].direction == 1
    assert crossings[0].move2 == mid2("5")


def test_a_move_one_tick_short_of_the_bar_does_not_cross():
    assert leader_crossings(series_of(ramp([0, 30], ["50", "54.995"]))) == []


def test_a_downward_move_crosses_with_a_negative_direction():
    crossings = leader_crossings(series_of(ramp([0, 30], ["50", "44"])))
    assert len(crossings) == 1
    assert crossings[0].direction == -1
    assert crossings[0].move2 == -mid2("6")


def test_a_move_landing_after_the_window_does_not_cross():
    assert leader_crossings(series_of(ramp([0, WINDOW_S + 1], ["50", "60"]))) == []


def test_a_monotone_ramp_yields_non_overlapping_crossings():
    seconds = list(range(21))
    crossings = leader_crossings(series_of(ramp(seconds, [str(50 + step) for step in seconds])))
    assert [(item.anchor_us, item.cross_us) for item in crossings] == [
        (us(0), us(5)),
        (us(6), us(11)),
        (us(12), us(17)),
    ]


def test_one_sided_states_neither_anchor_nor_close_a_window():
    crossings = leader_crossings(
        series_of([(0, "50", True), (1, "60", False), (2, "50", True), (3, "55", True)])
    )
    assert len(crossings) == 1
    assert crossings[0].anchor_us == us(0)
    assert crossings[0].cross_us == us(3)


def test_leader_crossings_reads_the_leader_alone():
    assert list(inspect.signature(leader_crossings).parameters) == ["series"]


def test_the_follower_view_starts_strictly_after_the_crossing():
    leader = series_of(ramp([0, 100], ["50", "55"]))
    crossing = leader_crossings(leader)[0]
    follower = follower_of(ramp([0, 50, 100, 150, 200], ["50", "50", "50", "52", "55"]))
    view = follower_view(follower, crossing)
    assert int(view.received_us.min()) > crossing.cross_us
    assert view.received_us.size == int(np.count_nonzero(follower.received_us > crossing.cross_us))
    assert view.received_us.tolist() == [us(150), us(200)]
    assert view.base_us == us(100)
    assert view.base_mid2 == mid2("50")


def test_a_follower_that_moved_before_the_crossing_and_then_sat_still_is_no_episode():
    leader = series_of(ramp([0, 100], ["50", "55"]))
    follower = follower_of(ramp([0, 50, 100, 200, 400], ["50", "55", "55", "55", "55"]))
    assert pair_episodes(PAIRS[0], leader, follower, reverse=False) == []


def test_the_lead_runs_from_the_leader_crossing_not_from_the_anchor():
    leader = series_of(ramp([0, 100], ["50", "55"]))
    follower = follower_of(ramp([0, 100, 160], ["50", "50", "55"]))
    episodes = pair_episodes(PAIRS[0], leader, follower, reverse=False)
    assert len(episodes) == 1
    assert episodes[0].lead_s == Decimal("60")
    assert episodes[0].anchor == at(0)
    assert episodes[0].leader_cross == at(100)
    assert episodes[0].follower_cross == at(160)
    assert episodes[0].evidence_span() == (at(0), at(160))


def test_the_leader_crossings_do_not_move_with_the_follower():
    steps = list(range(21))
    leader = series_of(ramp(steps, [str(50 + step) for step in steps]))
    crossings = leader_crossings(leader)
    calm = follower_of(ramp([0, 30], ["50", "50"]))
    quick = follower_of(ramp(list(range(41)), [str(50 + step // 2) for step in range(41)]))
    assert leader_crossings(leader) == crossings
    assert pair_episodes(PAIRS[0], leader, calm, reverse=False) == []
    episodes = pair_episodes(PAIRS[0], leader, quick, reverse=False)
    assert [(item.anchor, item.leader_cross) for item in episodes] == [
        (at(0), at(5)),
        (at(6), at(11)),
        (at(12), at(17)),
    ]


def test_a_follower_qualifying_after_the_deadline_is_no_episode():
    leader = series_of(ramp([0, 100], ["50", "55"]))
    follower = follower_of(ramp([0, 100, WINDOW_S + 50], ["50", "50", "60"]))
    crossing = leader_crossings(leader)[0]
    assert follower_response(follower_view(follower, crossing), crossing) is None
    assert pair_episodes(PAIRS[0], leader, follower, reverse=False) == []


def test_a_follower_moving_the_other_way_is_no_episode():
    leader = series_of(ramp([0, 100], ["50", "55"]))
    follower = follower_of(ramp([0, 100, 200], ["50", "50", "40"]))
    assert pair_episodes(PAIRS[0], leader, follower, reverse=False) == []


def test_a_follower_with_no_state_before_the_crossing_is_no_episode():
    leader = series_of(ramp([0, 100], ["50", "55"]))
    follower = follower_of([(50, "50", False), (150, "50", True), (200, "56", True)])
    assert follower_view(follower, leader_crossings(leader)[0]) is None
    assert pair_episodes(PAIRS[0], leader, follower, reverse=False) == []


def test_reverse_swaps_the_reading_but_not_the_pair():
    upstream = series_of(ramp([0, 100, 260], ["50", "50", "55"]))
    downstream = follower_of(ramp([0, 100, 200], ["50", "55", "60"]))
    episodes = pair_episodes(PAIRS[0], downstream, upstream, reverse=True)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.corridor == "gulf"
    assert episode.upstream == "DEN"
    assert episode.downstream == "OKC"
    assert episode.leader == "OKC"
    assert episode.follower == "DEN"
    assert episode.lead_s == Decimal("160")
    assert episode.evidence_span() == (at(0), at(260))


def test_episode_cents_and_direction_come_off_the_two_crossings():
    leader = series_of(ramp([0, 100], ["50", "44"]))
    follower = follower_of(ramp([0, 100, 160], ["50", "50", "43"]))
    episode = pair_episodes(PAIRS[0], leader, follower, reverse=False)[0]
    assert episode.direction == -1
    assert episode.leader_move_cents == Decimal("-6")
    assert episode.follower_move_cents == Decimal("-7")
    assert episode.event_date == EVENT_DATE
    assert isinstance(episode.lead_s, Decimal)
    assert episode.lead_s >= 0


def test_mismatched_event_dates_are_rejected():
    leader = series_of(ramp([0, 100], ["50", "55"]))
    follower = AtmSeries(
        series=OTHER_SERIES,
        event_date=date(2026, 7, 2),
        ticker="KXHIGHTOKC-26JUL02-B93.5",
        received_us=np.array([us(0)], dtype=np.int64),
        mid2=np.array([mid2("50")], dtype=np.int64),
        two_sided=np.array([True]),
    )
    with pytest.raises(ValueError, match="event date"):
        pair_episodes(PAIRS[0], leader, follower, reverse=False)


def test_series_that_do_not_match_the_requested_direction_are_rejected():
    leader = series_of(ramp([0, 100], ["50", "55"]))
    follower = follower_of(ramp([0, 100, 160], ["50", "50", "55"]))
    with pytest.raises(ValueError, match="DEN"):
        pair_episodes(PAIRS[0], follower, leader, reverse=False)
