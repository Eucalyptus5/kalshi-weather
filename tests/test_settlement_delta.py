from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag import settlement_delta
from bot.lag.settlement_delta import (
    FALSIFIED,
    SURVIVES,
    coverage_of,
    delta_of,
    delta_partition,
    supported_reading,
)
from bot.lag.settlement_straddle import Straddle
from bot.markets.observation_window import observation_window
from bot.observations.metar import StationObservation


ZONE = "America/Denver"
WIDE_STATION = "KAUS"
WIDE_DATE = date(2026, 8, 4)


def straddle(station: str, event_date: date, observed: str, acis: str) -> Straddle:
    return Straddle(
        root="KXHIGHDEN",
        station=station,
        event_date=event_date,
        extreme="max",
        timezone=ZONE,
        observed_f=Decimal(observed),
        acis_f=Decimal(acis),
        cell_edges=(71, 89, 91, 93),
        separating_strikes=(93,),
    )


def reading(station: str, stamp: datetime, temp_f: str) -> StationObservation:
    return StationObservation(
        station=station,
        valid_time=stamp,
        publication_time=stamp,
        temp_f=Decimal(temp_f),
        is_special=False,
        raw=f"{station} {temp_f}F",
        source="tape",
    )


def mixed_rows() -> tuple[tuple[Straddle, tuple[StationObservation, ...]], ...]:
    return (
        (straddle("KDEN", date(2026, 8, 1), "94", "93"), coverage_walk()),
        (straddle("KNYC", date(2026, 8, 2), "88", "87"), ()),
        (straddle("KDEN", date(2026, 8, 3), "82", "83"), ()),
        (straddle(WIDE_STATION, WIDE_DATE, "79", "94"), ()),
    )


def two_degree_rows() -> tuple[tuple[Straddle, tuple[StationObservation, ...]], ...]:
    return ((straddle("KBOS", date(2026, 8, 5), "77", "75"), ()),)


def two_wide_rows() -> tuple[tuple[Straddle, tuple[StationObservation, ...]], ...]:
    return (
        (straddle("KDEN", date(2026, 8, 5), "60", "70"), ()),
        (straddle(WIDE_STATION, WIDE_DATE, "79", "94"), ()),
    )


def narrow_rows() -> tuple[tuple[Straddle, tuple[StationObservation, ...]], ...]:
    return (
        (straddle("KDEN", date(2026, 8, 1), "94", "93"), ()),
        (straddle("KNYC", date(2026, 8, 2), "82", "83"), ()),
    )


def coverage_walk() -> tuple[StationObservation, ...]:
    start, end = observation_window(ZONE, date(2026, 8, 1))
    return (
        reading("KDEN", start - timedelta(minutes=1), "70"),
        reading("KDEN", start, "71"),
        reading("KDEN", start + timedelta(hours=6), "80"),
        reading("KDEN", start + timedelta(hours=12), "94"),
        reading("KDEN", end, "72"),
        reading("KDEN", end + timedelta(minutes=1), "73"),
    )


def duplicated_minute_walk() -> tuple[StationObservation, ...]:
    start, _ = observation_window(ZONE, date(2026, 8, 1))
    return (
        reading("KDEN", start, "70"),
        reading("KDEN", start + timedelta(seconds=30), "71"),
        reading("KDEN", start + timedelta(minutes=1), "72"),
        reading("KDEN", start + timedelta(minutes=1, seconds=20, microseconds=500), "73"),
    )


def test_the_delta_runs_from_the_observer_to_the_settlement_source() -> None:
    assert delta_of(straddle("KDEN", date(2026, 8, 1), "94", "93")) == 1
    assert delta_of(straddle("KDEN", date(2026, 8, 3), "82", "83")) == -1
    assert delta_of(straddle(WIDE_STATION, WIDE_DATE, "79", "94")) == -15


def test_the_magnitudes_split_around_one_degree() -> None:
    partition = delta_partition(mixed_rows())
    assert partition.abs_delta_gt_1 == 1
    assert partition.abs_delta_le_1 == 3


def test_the_wide_row_is_named_rather_than_counted_anonymously() -> None:
    partition = delta_partition(mixed_rows())
    assert partition.abs_delta_gt_1_rows == (("KAUS", date(2026, 8, 4)),)
    assert partition.abs_delta_gt_1_rows[0][0] == "KAUS"
    assert partition.abs_delta_gt_1_rows[0][1] == date(2026, 8, 4)


def test_two_wide_rows_come_back_in_the_order_they_were_walked() -> None:
    partition = delta_partition(two_wide_rows())

    assert partition.abs_delta_gt_1 == 2
    assert partition.abs_delta_gt_1_rows == (
        ("KDEN", date(2026, 8, 5)),
        ("KAUS", date(2026, 8, 4)),
    )
    assert list(partition.abs_delta_gt_1_rows) != sorted(partition.abs_delta_gt_1_rows)


def test_a_two_degree_delta_sits_on_the_wide_side_of_the_split() -> None:
    rows = two_degree_rows()
    partition = delta_partition(rows)
    assert delta_of(rows[0][0]) == 2
    assert partition.abs_delta_gt_1 == 1
    assert partition.abs_delta_le_1 == 0
    assert partition.abs_delta_gt_1_rows == (("KBOS", date(2026, 8, 5)),)
    assert supported_reading(partition) == "rounding falsified"


def test_the_histogram_keeps_the_sign_of_every_delta() -> None:
    partition = delta_partition(mixed_rows())
    assert partition.delta_histogram == {1: 2, -1: 1, -15: 1}
    assert -15 in partition.delta_histogram
    assert 15 not in partition.delta_histogram
    assert -1 in partition.delta_histogram


def test_a_delta_past_one_degree_falsifies_the_rounding_reading() -> None:
    partition = delta_partition(mixed_rows())
    assert supported_reading(partition) == "rounding falsified"
    assert FALSIFIED == "rounding falsified"


def test_whole_degree_deltas_leave_the_rounding_reading_standing() -> None:
    partition = delta_partition(narrow_rows())
    assert partition.abs_delta_gt_1 == 0
    assert partition.abs_delta_le_1 == 2
    assert partition.abs_delta_gt_1_rows == ()
    assert supported_reading(partition) == "rounding survives the check"
    assert SURVIVES == "rounding survives the check"


def test_the_coverage_opens_on_its_first_minute_and_closes_before_its_last() -> None:
    start, end = observation_window(ZONE, date(2026, 8, 1))
    record = straddle("KDEN", date(2026, 8, 1), "94", "93")
    assert coverage_of(record, coverage_walk()) == 3
    assert coverage_of(record, (reading("KDEN", start, "71"),)) == 1
    assert coverage_of(record, (reading("KDEN", end, "72"),)) == 0


def test_the_coverage_is_a_count_of_minutes_under_the_station_and_the_day() -> None:
    partition = delta_partition(mixed_rows())
    count = partition.coverage_minutes[("KDEN", date(2026, 8, 1))]
    assert count == 3
    assert isinstance(count, int)
    assert not isinstance(count, bool)
    assert partition.coverage_minutes[(WIDE_STATION, WIDE_DATE)] == 0


def test_two_readings_on_one_minute_cover_that_minute_once() -> None:
    record = straddle("KDEN", date(2026, 8, 1), "94", "93")
    walk = duplicated_minute_walk()
    assert len(walk) == 4
    assert coverage_of(record, walk) == 2
    partition = delta_partition(((record, walk),))
    assert partition.coverage_minutes == {("KDEN", date(2026, 8, 1)): 2}


def test_a_fractional_delta_is_refused() -> None:
    record = straddle("KDEN", date(2026, 8, 1), "94.5", "93")
    with pytest.raises(ValueError) as refused:
        delta_of(record)
    assert "KDEN" in str(refused.value)
    assert "2026-08-01" in str(refused.value)


def test_an_empty_set_partitions_nothing() -> None:
    partition = delta_partition(())
    assert partition.abs_delta_gt_1 == 0
    assert partition.abs_delta_le_1 == 0
    assert partition.delta_histogram == {}
    assert partition.coverage_minutes == {}
    assert partition.abs_delta_gt_1_rows == ()
    assert supported_reading(partition) == SURVIVES


def test_the_reading_the_check_cannot_establish_is_never_claimed() -> None:
    source = Path(settlement_delta.__file__).read_text()
    assert "confirm" not in source.lower()
    assert SURVIVES == "rounding survives the check"
    assert FALSIFIED == "rounding falsified"
