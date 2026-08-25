from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.placement_grid import CloseSidecar, read_sidecar
from bot.lag.settlement_entry import EntryCounts, StraddleEntry, entry_counts, entry_of
from bot.lag.settlement_straddle import Straddle, cell_edges, straddle_of
from bot.markets.observation_window import observation_window
from bot.observations.metar import StationObservation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEN = REPO_ROOT / "data" / "tape_studies" / "closes_v2" / "KXHIGHDEN.json"

pytestmark = pytest.mark.skipif(not DEN.exists(), reason="the recorded tape is not on this host")

ROOT = "KXHIGHDEN"
STATION = "KDEN"
ZONE = "America/Denver"
EVENT_DATE = date(2026, 8, 1)
EVENT_TICKER = "KXHIGHDEN-26AUG01"
FIRST_STAMP = datetime(2026, 8, 1, 13, tzinfo=timezone.utc)
RISING = ("70", "74", "79", "83", "87", "90", "92", "94", "93", "88")
CAPPED = ("70", "74", "79", "83", "87", "90", "92", "93", "92", "88")


@pytest.fixture(scope="module")
def sidecar() -> CloseSidecar:
    return read_sidecar(DEN)


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


def walk(first: datetime, temps: Sequence[str]) -> tuple[StationObservation, ...]:
    return tuple(reading(first + timedelta(hours=index), temp) for index, temp in enumerate(temps))


def with_close(sidecar: CloseSidecar, ticker: str, close_time: datetime) -> CloseSidecar:
    moved = replace(sidecar.markets[ticker], close_time=close_time)
    return CloseSidecar(
        root=sidecar.root,
        markets={**sidecar.markets, ticker: moved},
        voided=sidecar.voided,
        sha256=sidecar.sha256,
    )


def four_stamp_straddle() -> Straddle:
    return Straddle(
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        extreme="max",
        timezone=ZONE,
        observed_f=Decimal("73"),
        acis_f=Decimal("71"),
        cell_edges=(71,),
        separating_strikes=(71,),
    )


def test_the_running_extreme_first_separates_at_the_third_stamp(sidecar: CloseSidecar) -> None:
    record = entry_of(four_stamp_straddle(), walk(FIRST_STAMP, ("70", "71", "72", "73")), sidecar)
    assert record.entry_instant == datetime(2026, 8, 1, 15, tzinfo=timezone.utc)
    assert record.entry_instant != datetime(2026, 8, 1, 14, tzinfo=timezone.utc)
    assert record.entry_instant != datetime(2026, 8, 1, 16, tzinfo=timezone.utc)
    assert record.instant_class == "crossing"
    assert record.strike == 71


def test_the_stamp_sitting_on_the_edge_has_not_crossed_it(sidecar: CloseSidecar) -> None:
    straddle = four_stamp_straddle()
    truncated = walk(FIRST_STAMP, ("70", "71"))
    with pytest.raises(ValueError) as refused:
        entry_of(straddle, truncated, sidecar)
    assert STATION in str(refused.value)
    assert "2026-08-01" in str(refused.value)


def test_the_walk_carries_the_running_extreme_not_the_raw_reading(sidecar: CloseSidecar) -> None:
    straddle = Straddle(
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        extreme="max",
        timezone=ZONE,
        observed_f=Decimal("92"),
        acis_f=Decimal("94"),
        cell_edges=cell_edges(sidecar, EVENT_TICKER),
        separating_strikes=(93,),
    )
    with pytest.raises(ValueError) as refused:
        entry_of(straddle, walk(FIRST_STAMP, ("94", "92")), sidecar)
    assert STATION in str(refused.value)
    assert "2026-08-01" in str(refused.value)


def test_a_rising_day_crosses_the_edge_at_the_eighth_stamp(sidecar: CloseSidecar) -> None:
    straddle = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("94"),
        acis_f=Decimal("93"),
    )
    assert straddle is not None
    assert straddle.separating_strikes == (93,)
    record = entry_of(straddle, walk(FIRST_STAMP, RISING), sidecar)
    assert record.instant_class == "crossing"
    assert record.entry_instant == datetime(2026, 8, 1, 20, tzinfo=timezone.utc)
    assert record.strike == 93


def test_a_day_that_opens_already_separated_is_not_a_crossing(sidecar: CloseSidecar) -> None:
    straddle = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("93"),
        acis_f=Decimal("94"),
    )
    assert straddle is not None
    assert straddle.separating_strikes == (93,)
    record = entry_of(straddle, walk(FIRST_STAMP, CAPPED), sidecar)
    assert record.instant_class == "open"
    assert record.entry_instant == FIRST_STAMP
    assert record.strike == 93


def test_a_close_after_the_instant_holds_the_comparison(sidecar: CloseSidecar) -> None:
    third = datetime(2026, 8, 1, 15, tzinfo=timezone.utc)
    moved = with_close(sidecar, "KXHIGHDEN-26AUG01-T88", third + timedelta(seconds=300))
    record = entry_of(four_stamp_straddle(), walk(FIRST_STAMP, ("70", "71", "72", "73")), moved)
    assert record.entry_instant == third
    assert record.close_time == third + timedelta(seconds=300)
    assert record.entry_at_or_before_close is True


def test_a_close_at_the_instant_exactly_holds_the_comparison(sidecar: CloseSidecar) -> None:
    third = datetime(2026, 8, 1, 15, tzinfo=timezone.utc)
    moved = with_close(sidecar, "KXHIGHDEN-26AUG01-T88", third)
    record = entry_of(four_stamp_straddle(), walk(FIRST_STAMP, ("70", "71", "72", "73")), moved)
    assert record.close_time == datetime(2026, 8, 1, 15, tzinfo=timezone.utc)
    assert record.entry_at_or_before_close is True


def test_a_close_a_minute_before_the_instant_fails_the_comparison(sidecar: CloseSidecar) -> None:
    third = datetime(2026, 8, 1, 15, tzinfo=timezone.utc)
    moved = with_close(sidecar, "KXHIGHDEN-26AUG01-T88", third - timedelta(seconds=60))
    record = entry_of(four_stamp_straddle(), walk(FIRST_STAMP, ("70", "71", "72", "73")), moved)
    assert record.close_time == datetime(2026, 8, 1, 14, 59, tzinfo=timezone.utc)
    assert record.entry_at_or_before_close is False


def test_the_side_is_the_settlement_sources_own(sidecar: CloseSidecar) -> None:
    straddle = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("93"),
        acis_f=Decimal("94"),
    )
    assert straddle is not None
    record = entry_of(straddle, walk(FIRST_STAMP, CAPPED), sidecar)
    assert record.ticker == "KXHIGHDEN-26AUG01-B94.5"
    assert record.settlement_side == "yes"
    assert record.settlement_side not in {"YES", "no"}
    assert not isinstance(record.settlement_side, bool)
    assert sidecar.markets[record.ticker].result == "yes"


def test_the_legacy_root_moves_its_close_inside_the_window(sidecar: CloseSidecar) -> None:
    earlier = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=date(2026, 8, 13),
        timezone=ZONE,
        observed_f=Decimal("84"),
        acis_f=Decimal("82"),
    )
    later = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=date(2026, 8, 14),
        timezone=ZONE,
        observed_f=Decimal("88"),
        acis_f=Decimal("86"),
    )
    assert earlier is not None
    assert later is not None
    earlier_record = entry_of(
        earlier, walk(datetime(2026, 8, 13, 13, tzinfo=timezone.utc), ("70", "84")), sidecar
    )
    later_record = entry_of(
        later, walk(datetime(2026, 8, 14, 13, tzinfo=timezone.utc), ("70", "88")), sidecar
    )
    assert earlier_record.close_time == datetime(2026, 8, 14, 6, 59, tzinfo=timezone.utc)
    assert later_record.close_time == datetime(2026, 8, 15, 7, tzinfo=timezone.utc)
    assert earlier_record.entry_at_or_before_close is True
    assert later_record.entry_at_or_before_close is True


def test_the_instant_is_never_identifiable_when_it_happens(sidecar: CloseSidecar) -> None:
    straddle = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("94"),
        acis_f=Decimal("93"),
    )
    assert straddle is not None
    rising = entry_of(straddle, walk(FIRST_STAMP, RISING), sidecar)
    opened = entry_of(four_stamp_straddle(), walk(FIRST_STAMP, ("70", "71", "72", "73")), sidecar)
    assert rising.identifiable_ex_ante is False
    assert opened.identifiable_ex_ante is False


def test_the_instant_carries_its_own_offset(sidecar: CloseSidecar) -> None:
    record = entry_of(four_stamp_straddle(), walk(FIRST_STAMP, ("70", "71", "72", "73")), sidecar)
    assert record.entry_instant.tzinfo is not None
    assert record.entry_instant.utcoffset() == timedelta(0)
    assert record.close_time.utcoffset() == timedelta(0)


def test_an_event_day_the_sidecar_does_not_name_is_refused(sidecar: CloseSidecar) -> None:
    straddle = replace(four_stamp_straddle(), event_date=date(2026, 12, 25))
    stamps = walk(datetime(2026, 12, 25, 18, tzinfo=timezone.utc), ("70", "71", "72", "73"))
    with pytest.raises(ValueError):
        entry_of(straddle, stamps, sidecar)


def test_a_root_the_sidecar_does_not_name_is_refused(sidecar: CloseSidecar) -> None:
    straddle = replace(four_stamp_straddle(), root="KXHIGHNY")
    stamps = walk(FIRST_STAMP, ("70", "71", "72", "73"))
    with pytest.raises(ValueError):
        entry_of(straddle, stamps, sidecar)


def test_a_step_across_two_edges_takes_the_lower_one(sidecar: CloseSidecar) -> None:
    straddle = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("92"),
        acis_f=Decimal("88"),
    )
    assert straddle is not None
    assert straddle.separating_strikes == (89, 91)
    record = entry_of(straddle, walk(FIRST_STAMP, ("80", "84", "92")), sidecar)
    assert record.instant_class == "crossing"
    assert record.entry_instant == datetime(2026, 8, 1, 15, tzinfo=timezone.utc)
    assert record.strike == 89


def test_the_edge_crossed_at_the_instant_is_not_the_lowest_the_straddle_carries(
    sidecar: CloseSidecar,
) -> None:
    straddle = Straddle(
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        extreme="max",
        timezone=ZONE,
        observed_f=Decimal("88"),
        acis_f=Decimal("92"),
        cell_edges=cell_edges(sidecar, EVENT_TICKER),
        separating_strikes=(89, 91),
    )
    record = entry_of(straddle, walk(FIRST_STAMP, ("90", "90")), sidecar)
    assert straddle.separating_strikes[0] == 89
    assert record.strike == 91


def test_the_window_opens_on_its_first_minute(sidecar: CloseSidecar) -> None:
    start, _ = observation_window(ZONE, EVENT_DATE)
    assert start == datetime(2026, 8, 1, 7, tzinfo=timezone.utc)
    stamps = (reading(start, "70"), reading(start + timedelta(hours=1), "73"))
    record = entry_of(four_stamp_straddle(), stamps, sidecar)
    assert record.instant_class == "crossing"
    assert record.entry_instant == start + timedelta(hours=1)


def test_the_window_closes_before_its_last_minute(sidecar: CloseSidecar) -> None:
    start, end = observation_window(ZONE, EVENT_DATE)
    assert end == datetime(2026, 8, 2, 7, tzinfo=timezone.utc)
    stamps = (reading(start, "70"), reading(end, "73"))
    with pytest.raises(ValueError):
        entry_of(four_stamp_straddle(), stamps, sidecar)


def test_readings_before_the_window_do_not_open_it(sidecar: CloseSidecar) -> None:
    start, _ = observation_window(ZONE, EVENT_DATE)
    stamps = (reading(start - timedelta(minutes=1), "73"), reading(start, "70"))
    with pytest.raises(ValueError):
        entry_of(four_stamp_straddle(), stamps, sidecar)


def test_a_window_with_no_readings_at_all_is_refused(sidecar: CloseSidecar) -> None:
    stamps = walk(datetime(2026, 7, 30, 13, tzinfo=timezone.utc), ("70", "71", "72", "73"))
    with pytest.raises(ValueError) as refused:
        entry_of(four_stamp_straddle(), stamps, sidecar)
    assert STATION in str(refused.value)
    assert "2026-08-01" in str(refused.value)


def test_a_min_side_straddle_is_refused(sidecar: CloseSidecar) -> None:
    straddle = replace(four_stamp_straddle(), extreme="min")
    stamps = walk(FIRST_STAMP, ("70", "71", "72", "73"))
    with pytest.raises(ValueError) as refused:
        entry_of(straddle, stamps, sidecar)
    assert "min" in str(refused.value)


def test_the_readings_are_walked_in_time_order(sidecar: CloseSidecar) -> None:
    stamps = walk(FIRST_STAMP, ("70", "71", "72", "73"))
    shuffled = (stamps[3], stamps[1], stamps[0], stamps[2])
    record = entry_of(four_stamp_straddle(), shuffled, sidecar)
    assert record.entry_instant == datetime(2026, 8, 1, 15, tzinfo=timezone.utc)
    assert record.instant_class == "crossing"


def entry(
    instant: datetime, close_time: datetime, instant_class: str = "crossing"
) -> StraddleEntry:
    return StraddleEntry(
        entry_instant=instant,
        strike=93,
        settlement_side="yes",
        ticker="KXHIGHDEN-26AUG01-B94.5",
        close_time=close_time,
        entry_at_or_before_close=instant <= close_time,
        instant_class=instant_class,
        identifiable_ex_ante=False,
    )


def test_the_counts_split_the_crossings_from_the_open_windows() -> None:
    instant = datetime(2026, 8, 1, 20, tzinfo=timezone.utc)
    counts = entry_counts(
        [
            entry(instant, instant),
            entry(instant, instant + timedelta(seconds=300)),
            entry(instant, instant - timedelta(seconds=60)),
            entry(instant, instant + timedelta(seconds=300), instant_class="open"),
        ]
    )
    assert isinstance(counts, EntryCounts)
    assert counts.n == 3
    assert counts.entry_at_close_n == 1
    assert counts.open_at_first_reading_n == 1
    assert counts.at_or_before_close_n == 3
    assert counts.close_minus_entry_s == {0: 1, 300: 2, -60: 1}


def test_an_empty_set_counts_nothing() -> None:
    counts = entry_counts([])
    assert counts.n == 0
    assert counts.entry_at_close_n == 0
    assert counts.open_at_first_reading_n == 0
    assert counts.at_or_before_close_n == 0
    assert counts.close_minus_entry_s == {}
