import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.main import STATIONS
from bot.replay.analysis_stations import ANALYSIS_STATIONS
from bot.replay.blind_windows import BLIND_WINDOWS_SCHEMA
from bot.replay.run_scope import (
    D_EVAL,
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    OUTAGE_TOLERANCE_US,
    QUIET_BAND,
    RECORDED_GAP,
    RESUBSCRIBE_BLIND,
    SUBSCRIPTION_WIDE,
    DayInventory,
    EventDay,
    Exclusion,
    apply_spans,
    blind_exclusions,
    classify_window,
    event_day_inventory,
    freeze_digest,
    freeze_split,
    quiet_band_exclusions,
    read_blind_windows,
    read_coverage,
    split_lengths,
    split_payload,
    union_overlap_us,
    write_event_days,
    write_exclusions,
    write_split,
)


UTC = timezone.utc
TICK = timedelta(microseconds=1)
SECOND = 1_000_000
HOUR = 3_600_000_000
DAY_US = 86_400_000_000
BAND_US = 2 * HOUR

EAST = "KXHIGHNY"
WEST = "KXHIGHTSFO"
STRIKES = (80, 85)
DAYS = [date(2026, 7, 17) + timedelta(days=offset) for offset in range(17)]
FIRST_SCOPE_DAY = date(2026, 7, 19)
LAST_SCOPE_DAY = date(2026, 8, 1)

TAPE_FIRST = datetime(2026, 7, 16, tzinfo=UTC)
TAPE_LAST = datetime(2026, 8, 2, 12, tzinfo=UTC)
SCOPE_OPEN = datetime(2026, 7, 19, tzinfo=UTC)
SCOPE_START = datetime(2026, 7, 19, 5, tzinfo=UTC)
SCOPE_END = datetime(2026, 8, 2, 8, tzinfo=UTC)

INCIDENT = datetime(2026, 7, 23, 7, 43, 55, tzinfo=UTC)
INCIDENT_DETECTED = datetime(2026, 7, 23, 7, 43, 55, 776558, tzinfo=UTC)
INCIDENT_START = datetime(2026, 7, 23, 7, 24, 0, tzinfo=UTC)
INCIDENT_END = datetime(2026, 7, 23, 7, 43, 8, tzinfo=UTC)

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_INPUTS = REPO_ROOT / "data" / "tape_studies" / "inputs"
FROZEN_SCOPE = REPO_ROOT / "data" / "tape_studies" / "run_scope"
FROZEN_DIGEST = "12f7f9e25bb4c4eddb3ffa6f330f8b52156dee606ec64b5c843ff94946af4478"


def ticker(series: str, day: date, strike: int) -> str:
    return f"{series}-{day.strftime('%y%b%d').upper()}-T{strike}"


def cov(name: str, day: date, rows: int) -> dict[str, object]:
    return {
        "ticker": name,
        "rows": rows,
        "first_received_at": datetime(day.year, day.month, day.day, 9, tzinfo=UTC),
        "last_received_at": datetime(day.year, day.month, day.day, 10, tzinfo=UTC),
    }


def coverage_rows() -> list[dict[str, object]]:
    rows = [
        {
            "ticker": "KXLOWTDEN-26JUL19-T60",
            "rows": 7,
            "first_received_at": TAPE_FIRST,
            "last_received_at": TAPE_LAST,
        },
        cov(f"{EAST}-26JUL19-X85", FIRST_SCOPE_DAY, 3),
    ]
    for series in (EAST, WEST):
        for day in DAYS:
            rows.extend(cov(ticker(series, day, strike), day, strike) for strike in STRIKES)
    return rows


def spread_rows(days: list[date]) -> list[dict[str, object]]:
    return [cov(ticker(series, day, 80), day, 80) for series in (EAST, WEST) for day in days]


def inventory_of(rows: list[dict[str, object]], **overrides: object) -> DayInventory:
    settings: dict[str, object] = {
        "stations": STATIONS,
        "exclusions": (),
        "tape_first": TAPE_FIRST,
        "tape_last": TAPE_LAST,
        "scope_open": SCOPE_OPEN,
        "d_eval": D_EVAL,
    }
    return event_day_inventory(rows, **(settings | overrides))


def blind_row(
    boundary_id: int,
    start: datetime,
    end: datetime,
    *,
    gap_id: int | None = None,
    reason: str | None = None,
    detected_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "boundary_id": boundary_id,
        "prev_id": boundary_id - 1,
        "end_id": boundary_id + 1,
        "burst_messages": 1,
        "ticker": "",
        "start": start,
        "end": end,
        "blind_us": (end - start) // TICK,
        "prev_seq": 9,
        "seq": 1,
        "has_gap_row": gap_id is not None,
        "gap_id": gap_id,
        "gap_reason": reason,
        "gap_detected_at": detected_at,
        "in_frozen_window": True,
    }


BLIND_ROWS = [
    blind_row(
        11,
        datetime(2026, 7, 18, tzinfo=UTC),
        datetime(2026, 7, 18, 0, 0, 5, tzinfo=UTC),
        gap_id=1,
        reason="connection_reset",
        detected_at=datetime(2026, 7, 18, 0, 0, 6, tzinfo=UTC),
    ),
    blind_row(
        21,
        datetime(2026, 7, 20, 10, tzinfo=UTC),
        datetime(2026, 7, 20, 10, 0, 30, tzinfo=UTC),
        gap_id=2,
        reason="connection_reset",
        detected_at=datetime(2026, 7, 20, 10, 0, 35, tzinfo=UTC),
    ),
    blind_row(
        31,
        datetime(2026, 7, 21, 11, tzinfo=UTC),
        datetime(2026, 7, 21, 11, 0, 10, tzinfo=UTC),
        gap_id=3,
        reason="seq_skip",
        detected_at=datetime(2026, 7, 21, 11, 0, 15, tzinfo=UTC),
    ),
    blind_row(
        41, datetime(2026, 7, 22, 12, tzinfo=UTC), datetime(2026, 7, 22, 12, 0, 9, tzinfo=UTC)
    ),
    blind_row(
        51,
        INCIDENT_START,
        INCIDENT_END,
        gap_id=5,
        reason="connection_reset",
        detected_at=INCIDENT_DETECTED,
    ),
]

HOLE_DAY = date(2026, 7, 25)
HOLE_ROW = blind_row(
    71,
    datetime(2026, 7, 25, 12, tzinfo=UTC),
    datetime(2026, 7, 25, 14, tzinfo=UTC),
    gap_id=7,
    reason="connection_reset",
    detected_at=datetime(2026, 7, 25, 14, 0, 5, tzinfo=UTC),
)
LONG_TAPE_LAST = datetime(2026, 8, 25, tzinfo=UTC)
LONG_DAYS = [FIRST_SCOPE_DAY + timedelta(days=offset) for offset in range(34)]
RESTART_DAYS = [HOLE_DAY + timedelta(days=offset) for offset in range(1, D_EVAL + 1)]


def write_blind(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=BLIND_WINDOWS_SCHEMA), path)
    return path


def exclusion(start: datetime, end: datetime, kind: str = RECORDED_GAP) -> Exclusion:
    return Exclusion(
        exclusion_class=kind,
        start=start,
        end=end,
        boundary_id=None,
        gap_id=None,
        gap_reason=None,
        padded=False,
    )


@pytest.fixture
def inventory() -> list[EventDay]:
    return list(inventory_of(coverage_rows()).days)


def day_of(days: list[EventDay], series: str, event_date: date) -> EventDay:
    return next(day for day in days if day.series == series and day.event_date == event_date)


@pytest.mark.parametrize(("d_eval", "expected"), [(14, (9, 5)), (28, (18, 10))])
def test_the_split_is_the_forward_two_thirds_of_the_accrual(
    d_eval: int, expected: tuple[int, int]
) -> None:
    assert split_lengths(d_eval) == expected
    assert sum(split_lengths(d_eval)) == d_eval


def test_the_outage_tolerance_is_five_percent_of_an_event_day() -> None:
    assert OUTAGE_TOLERANCE_US == DAY_US // 20
    assert OUTAGE_TOLERANCE_US == 4_320_000_000


@pytest.mark.parametrize(
    ("has_gap_row", "reason", "expected"),
    [
        (False, None, RESUBSCRIBE_BLIND),
        (True, "seq_skip", SUBSCRIPTION_WIDE),
        (True, "terminal_error_10", SUBSCRIPTION_WIDE),
        (True, "terminal_error_17", SUBSCRIPTION_WIDE),
        (True, "terminal_error_25", SUBSCRIPTION_WIDE),
        (True, "connection_reset", RECORDED_GAP),
    ],
)
def test_every_boundary_lands_in_one_of_the_three_classes(
    has_gap_row: bool, reason: str | None, expected: str
) -> None:
    assert classify_window(has_gap_row, reason) == expected


def test_the_named_incident_is_widened_five_minutes_at_each_end() -> None:
    same_day = blind_row(
        61,
        datetime(2026, 7, 23, 12, tzinfo=UTC),
        datetime(2026, 7, 23, 12, 0, 20, tzinfo=UTC),
        gap_id=6,
        reason="connection_reset",
        detected_at=datetime(2026, 7, 23, 12, 0, 25, tzinfo=UTC),
    )
    rows = [BLIND_ROWS[4], same_day]

    exclusions = blind_exclusions(
        rows, scope_start=SCOPE_START, scope_end=SCOPE_END, incident=INCIDENT
    )

    incident, other = exclusions[0], exclusions[1]
    assert incident.padded is True
    assert incident.start == INCIDENT_START - timedelta(minutes=5)
    assert incident.end == INCIDENT_END + timedelta(minutes=5)
    assert incident.duration_us == (INCIDENT_END - INCIDENT_START) // TICK + 600 * SECOND
    assert incident.gap_id == 5
    assert other.padded is False
    assert (other.start, other.end) == (same_day["start"], same_day["end"])
    assert other.duration_us == 20 * SECOND


def test_an_unnamed_incident_is_left_at_its_recorded_width() -> None:
    exclusions = blind_exclusions(
        [BLIND_ROWS[4]],
        scope_start=SCOPE_START,
        scope_end=SCOPE_END,
        incident=datetime(2026, 8, 6, 7, 23, 9, tzinfo=UTC),
    )

    assert [item.padded for item in exclusions] == [False]
    assert (exclusions[0].start, exclusions[0].end) == (INCIDENT_START, INCIDENT_END)


def test_a_boundary_only_the_padding_brings_into_scope_is_kept_and_clipped() -> None:
    scope_start = INCIDENT_END + timedelta(minutes=2)
    rows = [BLIND_ROWS[4]]

    exclusions = blind_exclusions(
        rows, scope_start=scope_start, scope_end=SCOPE_END, incident=INCIDENT
    )

    assert len(exclusions) == 1
    assert exclusions[0].start == scope_start
    assert exclusions[0].end == INCIDENT_END + timedelta(minutes=5)
    assert (
        blind_exclusions(rows, scope_start=SCOPE_END, scope_end=SCOPE_END, incident=INCIDENT) == []
    )


def test_a_clipped_interval_never_escapes_the_scope() -> None:
    scope_start = datetime(2026, 7, 20, 10, 0, 10, tzinfo=UTC)
    scope_end = datetime(2026, 7, 20, 10, 0, 20, tzinfo=UTC)

    exclusions = blind_exclusions(
        BLIND_ROWS, scope_start=scope_start, scope_end=scope_end, incident=INCIDENT
    )

    assert [(e.start, e.end) for e in exclusions] == [(scope_start, scope_end)]
    assert exclusions[0].exclusion_class == RECORDED_GAP


def test_the_quiet_band_is_two_hours_of_every_utc_day_it_touches() -> None:
    exclusions = quiet_band_exclusions(
        datetime(2026, 7, 19, tzinfo=UTC), datetime(2026, 7, 22, tzinfo=UTC)
    )

    assert [e.exclusion_class for e in exclusions] == [QUIET_BAND] * 3
    assert [e.start.date().isoformat() for e in exclusions] == [
        "2026-07-19",
        "2026-07-20",
        "2026-07-21",
    ]
    assert [e.duration_us for e in exclusions] == [7_200_000_000] * 3
    assert all(e.boundary_id is None and e.gap_id is None for e in exclusions)
    assert all(e.gap_reason is None and e.padded is False for e in exclusions)


def test_a_quiet_band_is_clipped_to_the_scope_it_straddles() -> None:
    exclusions = quiet_band_exclusions(
        datetime(2026, 7, 19, 8, tzinfo=UTC), datetime(2026, 7, 20, 8, tzinfo=UTC)
    )

    assert [e.duration_us for e in exclusions] == [HOUR, HOUR]


def test_overlapping_exclusions_are_counted_once() -> None:
    base = datetime(2026, 7, 20, tzinfo=UTC)
    exclusions = [
        exclusion(base, base + timedelta(seconds=10)),
        exclusion(base + timedelta(seconds=5), base + timedelta(seconds=20)),
    ]

    assert union_overlap_us(exclusions, base, base + timedelta(minutes=1)) == 20 * SECOND


def test_the_union_is_clipped_to_the_query_range() -> None:
    base = datetime(2026, 7, 20, tzinfo=UTC)
    exclusions = [
        exclusion(base, base + timedelta(seconds=10)),
        exclusion(base + timedelta(seconds=5), base + timedelta(seconds=20)),
    ]

    overlap = union_overlap_us(exclusions, base + timedelta(seconds=2), base + timedelta(seconds=8))

    assert overlap == 6 * SECOND
    assert union_overlap_us(exclusions, base + timedelta(minutes=1), base + timedelta(hours=1)) == 0


@pytest.mark.parametrize("series", [EAST, WEST])
def test_the_cold_start_costs_both_coasts_the_same_two_event_days(
    inventory: list[EventDay], series: str
) -> None:
    evaluable = {day.event_date: day.evaluable for day in inventory if day.series == series}

    assert evaluable[date(2026, 7, 17)] is False
    assert evaluable[date(2026, 7, 18)] is False
    assert evaluable[FIRST_SCOPE_DAY] is True


def test_a_later_named_start_moves_the_frozen_window_forward() -> None:
    built = inventory_of(
        spread_rows(LONG_DAYS),
        scope_open=datetime(2026, 7, 22, tzinfo=UTC),
        tape_last=LONG_TAPE_LAST,
    )

    scoped = sorted({day.event_date for day in built.days if day.in_scope})

    assert scoped[0] == date(2026, 7, 22)
    assert len(scoped) == D_EVAL


def test_an_event_day_running_past_the_tape_is_not_covered(inventory: list[EventDay]) -> None:
    tail = day_of(inventory, EAST, date(2026, 8, 2))

    assert tail.window_end > TAPE_LAST
    assert (tail.covered, tail.evaluable, tail.in_scope) == (False, False, False)


def test_only_tickers_the_injected_map_names_reach_the_inventory(
    inventory: list[EventDay],
) -> None:
    first = day_of(inventory, EAST, FIRST_SCOPE_DAY)

    assert {day.series for day in inventory} == {EAST, WEST}
    assert (first.tickers, first.ladder_rows) == (2, sum(STRIKES))
    assert (first.station, first.timezone) == ("KNYC", "America/New_York")
    assert first.first_event_at == datetime(2026, 7, 19, 9, tzinfo=UTC)
    assert first.last_event_at == datetime(2026, 7, 19, 10, tzinfo=UTC)


def test_the_combined_map_builds_both_ladders_and_admits_no_rain_root() -> None:
    rows = [cov("KXRAINNYCM-26JUL19-T0.1", FIRST_SCOPE_DAY, 5)]
    for series in (EAST, WEST, "KXLOWTNYC", "KXLOWTSFO"):
        rows.extend(cov(ticker(series, day, 80), day, 80) for day in DAYS)

    built = inventory_of(rows, stations=ANALYSIS_STATIONS)

    assert len(ANALYSIS_STATIONS) == 40
    assert {day.series for day in built.days} == {EAST, WEST, "KXLOWTNYC", "KXLOWTSFO"}
    assert not [day for day in built.days if day.series.startswith("KXRAIN")]
    assert day_of(list(built.days), "KXLOWTNYC", FIRST_SCOPE_DAY).station == "KNYC"
    assert len([day for day in built.days if day.in_scope]) == 4 * D_EVAL


def test_the_quiet_band_alone_leaves_every_event_day_evaluable() -> None:
    bands = quiet_band_exclusions(TAPE_FIRST, TAPE_LAST)

    built = inventory_of(coverage_rows(), exclusions=bands)

    assert sum(band.duration_us for band in bands) > OUTAGE_TOLERANCE_US
    assert built.over_tolerance == 0
    assert len([day for day in built.days if day.evaluable]) == 2 * D_EVAL
    assert all(value == 0 for value in built.outage_us.values())


def test_an_outage_over_tolerance_costs_the_day_it_lands_on() -> None:
    outage = exclusion(datetime(2026, 7, 22, 12, tzinfo=UTC), datetime(2026, 7, 22, 14, tzinfo=UTC))

    built = inventory_of(spread_rows(LONG_DAYS), exclusions=[outage], tape_last=LONG_TAPE_LAST)

    assert built.over_tolerance == 2
    assert built.outage_us[(EAST, date(2026, 7, 22))] == 2 * HOUR
    assert not day_of(list(built.days), EAST, date(2026, 7, 22)).evaluable
    assert not day_of(list(built.days), WEST, date(2026, 7, 22)).evaluable


def test_a_hole_over_tolerance_restarts_the_count_at_the_next_evaluable_day() -> None:
    outage = exclusion(HOLE_ROW["start"], HOLE_ROW["end"])

    built = inventory_of(spread_rows(LONG_DAYS), exclusions=[outage], tape_last=LONG_TAPE_LAST)

    scoped = sorted({day.event_date for day in built.days if day.in_scope})
    assert scoped == RESTART_DAYS
    assert scoped[0] == HOLE_DAY + timedelta(days=1)
    assert HOLE_DAY not in scoped
    assert len(scoped) == D_EVAL


def test_a_series_short_of_the_accrual_cannot_be_frozen() -> None:
    days = [FIRST_SCOPE_DAY + timedelta(days=offset) for offset in range(D_EVAL - 1)]
    rows = [cov(ticker(EAST, day, 80), day, 80) for day in days]

    with pytest.raises(ValueError, match="2026-07-31") as raised:
        inventory_of(rows, tape_last=datetime(2026, 8, 5, tzinfo=UTC))

    assert str(D_EVAL) in str(raised.value)


def test_a_run_that_never_reaches_the_accrual_reports_its_day_sequence() -> None:
    days = [FIRST_SCOPE_DAY + timedelta(days=offset) for offset in range(6)]
    days += [date(2026, 7, 27) + timedelta(days=offset) for offset in range(10)]

    with pytest.raises(ValueError) as raised:
        inventory_of(spread_rows(days), tape_last=LONG_TAPE_LAST)

    reported = str(raised.value)
    assert all(day.isoformat() in reported for day in days)
    assert "2026-07-25" not in reported


def test_a_city_with_no_evaluable_day_is_named_in_the_refusal() -> None:
    rows = spread_rows(DAYS)
    rows += [cov(ticker("KXLOWTDEN", day, 60), day, 60) for day in DAYS[:2]]

    with pytest.raises(ValueError) as raised:
        inventory_of(rows, stations=ANALYSIS_STATIONS)

    reported = str(raised.value)
    assert "KXLOWTDEN=0" in reported
    assert f"{EAST}={D_EVAL}" in reported
    assert f"{WEST}={D_EVAL}" in reported
    assert "none" in reported
    assert reported == reported.rstrip(" ,:;")


def test_a_short_shared_run_reports_the_days_and_the_city_counts() -> None:
    days = [FIRST_SCOPE_DAY + timedelta(days=offset) for offset in range(D_EVAL - 1)]

    with pytest.raises(ValueError) as raised:
        inventory_of(spread_rows(days), tape_last=datetime(2026, 8, 5, tzinfo=UTC))

    reported = str(raised.value)
    assert all(day.isoformat() in reported for day in days)
    assert f"{EAST}={D_EVAL - 1}" in reported
    assert f"{WEST}={D_EVAL - 1}" in reported


@pytest.mark.parametrize("d_eval", [0, 1, 19, 22, -D_EVAL])
def test_an_accrual_that_is_not_a_whole_window_is_refused(d_eval: int) -> None:
    with pytest.raises(ValueError) as raised:
        inventory_of(coverage_rows(), d_eval=d_eval)

    reported = str(raised.value)
    assert f"{d_eval} event-days" in reported
    assert f"multiple of {D_EVAL}" in reported
    assert FIRST_SCOPE_DAY.isoformat() in reported
    assert LAST_SCOPE_DAY.isoformat() in reported
    assert "contiguous" not in reported


@pytest.mark.parametrize("d_eval", [D_EVAL, 2 * D_EVAL])
def test_a_whole_multiple_of_the_base_accrual_still_freezes(d_eval: int) -> None:
    built = inventory_of(spread_rows(LONG_DAYS), d_eval=d_eval, tape_last=LONG_TAPE_LAST)

    scoped = sorted({day.event_date for day in built.days if day.in_scope})

    assert len(scoped) == d_eval
    assert scoped[0] == FIRST_SCOPE_DAY


def test_the_frozen_split_runs_nine_days_then_five(inventory: list[EventDay]) -> None:
    split = freeze_split(inventory, d_eval=D_EVAL)
    d_disc, d_hold = split_lengths(D_EVAL)

    assert split.cities == (EAST, WEST)
    assert len(split.discovery_days) == d_disc
    assert len(split.holdout_days) == d_hold
    assert split.discovery_days[0] == FIRST_SCOPE_DAY
    assert split.holdout_days[-1] == LAST_SCOPE_DAY
    assert split.boundary_event_day == FIRST_SCOPE_DAY + timedelta(days=d_disc)
    assert split.boundary_event_day == split.holdout_days[0]
    assert (split.scope_start, split.scope_end) == (SCOPE_START, SCOPE_END)
    assert [day.split for day in inventory if day.in_scope].count(DISCOVERY) == 2 * d_disc
    assert [day.split for day in inventory if day.in_scope].count(HOLDOUT) == 2 * d_hold
    assert all(day.split == "" and day.day_index == 0 for day in inventory if not day.in_scope)


def test_a_longer_accrual_splits_eighteen_days_then_ten() -> None:
    built = inventory_of(spread_rows(LONG_DAYS), d_eval=28, tape_last=LONG_TAPE_LAST)

    split = freeze_split(built.days, d_eval=28)

    assert (len(split.discovery_days), len(split.holdout_days)) == (18, 10)
    assert split.discovery_days[0] == FIRST_SCOPE_DAY
    assert split.boundary_event_day == FIRST_SCOPE_DAY + timedelta(days=18)
    payload = split_payload(split)
    assert (payload["d_eval"], payload["d_disc"], payload["d_hold"]) == (28, 18, 10)


def test_two_cities_that_disagree_on_their_days_cannot_be_frozen(
    inventory: list[EventDay],
) -> None:
    shifted = [
        replace(day, event_date=date(2026, 9, 9))
        if day.series == WEST and day.event_date == LAST_SCOPE_DAY
        else day
        for day in inventory
    ]

    with pytest.raises(ValueError, match="cities"):
        freeze_split(shifted, d_eval=D_EVAL)


def test_the_digest_is_stable_and_moves_with_a_single_day(inventory: list[EventDay]) -> None:
    payload = split_payload(freeze_split(inventory, d_eval=D_EVAL))
    moved = dict(payload)
    moved["holdout_days"] = [*payload["holdout_days"][:-1], "2026-09-09"]

    assert freeze_digest(payload) == freeze_digest(dict(payload))
    assert freeze_digest(payload) != freeze_digest(moved)
    assert len(freeze_digest(payload)) == 64


def test_the_payload_names_the_frozen_boundary(inventory: list[EventDay]) -> None:
    payload = split_payload(freeze_split(inventory, d_eval=D_EVAL))

    assert payload["d_eval"] == D_EVAL
    assert (payload["d_disc"], payload["d_hold"]) == split_lengths(D_EVAL)
    assert payload["cities"] == [EAST, WEST]
    assert payload["first_evaluable_event_day"] == FIRST_SCOPE_DAY.isoformat()
    assert payload["last_evaluable_event_day"] == LAST_SCOPE_DAY.isoformat()
    assert payload["boundary_event_day"] == "2026-07-28"
    assert payload["scope_start"] == SCOPE_START.isoformat()
    assert payload["scope_end"] == SCOPE_END.isoformat()


def test_a_clean_event_day_measures_twenty_two_hours(inventory: list[EventDay]) -> None:
    split = freeze_split(inventory, d_eval=D_EVAL)
    bands = quiet_band_exclusions(split.scope_start, split.scope_end)

    spanned = apply_spans(inventory, bands)

    scoped = [day for day in spanned if day.in_scope]
    assert len(scoped) == 2 * D_EVAL
    assert all(day.excluded_us == BAND_US for day in scoped)
    assert all(day.span_us == 79_200_000_000 for day in scoped)
    assert all(day.excluded_us == 0 and day.span_us == 0 for day in spanned if not day.in_scope)


def test_a_pacific_day_pays_the_band_at_both_of_its_ends(inventory: list[EventDay]) -> None:
    split = freeze_split(inventory, d_eval=D_EVAL)
    bands = quiet_band_exclusions(split.scope_start, split.scope_end)
    pacific = day_of(apply_spans(inventory, bands), WEST, FIRST_SCOPE_DAY)

    opening = [band for band in bands if band.start.date() == FIRST_SCOPE_DAY]
    closing = [band for band in bands if band.start.date() == FIRST_SCOPE_DAY + timedelta(days=1)]

    assert pacific.excluded_us == BAND_US
    assert union_overlap_us(opening, pacific.window_start, pacific.window_end) == HOUR
    assert union_overlap_us(closing, pacific.window_start, pacific.window_end) == HOUR


def test_the_exclusions_table_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "exclusions.parquet"
    base = datetime(2026, 7, 20, tzinfo=UTC)
    exclusions = [
        Exclusion(
            exclusion_class=SUBSCRIPTION_WIDE,
            start=base + timedelta(seconds=30),
            end=base + timedelta(seconds=40),
            boundary_id=31,
            gap_id=3,
            gap_reason="seq_skip",
            padded=False,
        ),
        exclusion(base, base + timedelta(seconds=10), QUIET_BAND),
    ]

    write_exclusions(path, exclusions)
    table = pq.read_table(path)

    assert table.schema.equals(EXCLUSIONS_SCHEMA)
    assert table.to_pylist() == [
        {
            "exclusion_id": 0,
            "exclusion_class": QUIET_BAND,
            "start": base,
            "end": base + timedelta(seconds=10),
            "duration_us": 10 * SECOND,
            "boundary_id": None,
            "gap_id": None,
            "gap_reason": None,
            "padded": False,
        },
        {
            "exclusion_id": 1,
            "exclusion_class": SUBSCRIPTION_WIDE,
            "start": base + timedelta(seconds=30),
            "end": base + timedelta(seconds=40),
            "duration_us": 10 * SECOND,
            "boundary_id": 31,
            "gap_id": 3,
            "gap_reason": "seq_skip",
            "padded": False,
        },
    ]

    with pytest.raises(FileExistsError, match=str(path)):
        write_exclusions(path, exclusions)


def test_the_event_day_table_round_trips(tmp_path: Path, inventory: list[EventDay]) -> None:
    path = tmp_path / "event_days.parquet"
    split = freeze_split(inventory, d_eval=D_EVAL)
    spanned = apply_spans(inventory, quiet_band_exclusions(split.scope_start, split.scope_end))

    write_event_days(path, spanned)
    table = pq.read_table(path)

    assert table.schema.equals(EVENT_DAYS_SCHEMA)
    rows = table.to_pylist()
    assert len(rows) == len(spanned)
    first = rows[0]
    assert first["series"] == EAST
    assert first["event_date"] == date(2026, 7, 17)
    assert first["covered"] is True
    assert first["evaluable"] is False
    assert first["day_index"] == 0
    assert first["split"] == ""
    scoped = [row for row in rows if row["in_scope"]]
    assert len(scoped) == 2 * D_EVAL
    assert {row["span_us"] for row in scoped} == {79_200_000_000}

    with pytest.raises(FileExistsError, match=str(path)):
        write_event_days(path, spanned)


def test_the_frozen_tables_still_carry_the_columns_every_reader_expects() -> None:
    assert EXCLUSIONS_SCHEMA.equals(
        pa.schema(
            [
                ("exclusion_id", pa.int64()),
                ("exclusion_class", pa.string()),
                ("start", pa.timestamp("us", tz="UTC")),
                ("end", pa.timestamp("us", tz="UTC")),
                ("duration_us", pa.int64()),
                ("boundary_id", pa.int64()),
                ("gap_id", pa.int64()),
                ("gap_reason", pa.string()),
                ("padded", pa.bool_()),
            ]
        )
    )
    assert EVENT_DAYS_SCHEMA.equals(
        pa.schema(
            [
                ("series", pa.string()),
                ("station", pa.string()),
                ("timezone", pa.string()),
                ("event_date", pa.date32()),
                ("window_start", pa.timestamp("us", tz="UTC")),
                ("window_end", pa.timestamp("us", tz="UTC")),
                ("tickers", pa.int64()),
                ("ladder_rows", pa.int64()),
                ("first_event_at", pa.timestamp("us", tz="UTC")),
                ("last_event_at", pa.timestamp("us", tz="UTC")),
                ("covered", pa.bool_()),
                ("evaluable", pa.bool_()),
                ("in_scope", pa.bool_()),
                ("day_index", pa.int64()),
                ("split", pa.string()),
                ("excluded_us", pa.int64()),
                ("span_us", pa.int64()),
            ]
        )
    )


def test_the_split_file_carries_its_own_digest(tmp_path: Path, inventory: list[EventDay]) -> None:
    path = tmp_path / "split.json"
    split = freeze_split(inventory, d_eval=D_EVAL)

    digest = write_split(path, split)

    payload = json.loads(path.read_text())
    assert payload.pop("sha256") == digest
    assert payload == split_payload(split)
    assert freeze_digest(payload) == digest

    with pytest.raises(FileExistsError, match=str(path)):
        write_split(path, split)


def test_the_blind_windows_are_read_back_in_start_order(tmp_path: Path) -> None:
    path = write_blind(tmp_path / "blind.parquet", list(reversed(BLIND_ROWS)))

    rows = read_blind_windows(path)

    assert [row["boundary_id"] for row in rows] == [11, 21, 31, 41, 51]
    assert rows[0]["gap_reason"] == "connection_reset"
    assert rows[3]["has_gap_row"] is False


@pytest.mark.skipif(not FROZEN_SCOPE.exists(), reason="the recorded tape is not on this host")
def test_the_spent_window_rebuilds_off_the_recorded_tape() -> None:
    blind = read_blind_windows(FROZEN_INPUTS / "blind_windows-b000000.parquet")
    coverage = read_coverage(FROZEN_INPUTS / "coverage-b000000.parquet")
    tape_first = min(row["first_received_at"] for row in coverage)
    tape_last = max(row["last_received_at"] for row in coverage)
    outages = blind_exclusions(
        blind, scope_start=tape_first, scope_end=tape_last, incident=INCIDENT
    )

    built = event_day_inventory(
        coverage,
        stations=STATIONS,
        exclusions=outages,
        tape_first=tape_first,
        tape_last=tape_last,
        scope_open=SCOPE_OPEN,
        d_eval=D_EVAL,
    )
    split = freeze_split(built.days, d_eval=D_EVAL)
    payload = split_payload(split)
    stored = json.loads((FROZEN_SCOPE / "split.json").read_text())

    assert payload == {name: value for name, value in stored.items() if name != "sha256"}
    assert freeze_digest(payload) == FROZEN_DIGEST
    assert stored["sha256"] == FROZEN_DIGEST
    assert len(split.cities) == 20
    assert split.discovery_days[0] == date(2026, 7, 19)
    assert split.holdout_days[-1] == date(2026, 8, 1)
    assert split.boundary_event_day == date(2026, 7, 28)

    scoped = [day for day in built.days if day.in_scope]
    assert len(scoped) == 280
    assert built.over_tolerance == 0

    worst: dict[date, int] = {}
    for day in scoped:
        outage = built.outage_us[(day.series, day.event_date)]
        worst[day.event_date] = max(worst.get(day.event_date, 0), outage)
    assert max(worst.values()) <= OUTAGE_TOLERANCE_US
    assert worst[date(2026, 7, 30)] / DAY_US * 100 == pytest.approx(4.996, abs=0.001)
    assert worst[date(2026, 7, 23)] / DAY_US * 100 == pytest.approx(3.41, abs=0.001)

    bands = quiet_band_exclusions(split.scope_start, split.scope_end)
    folded = list(outages) + bands
    assert all(
        union_overlap_us(folded, day.window_start, day.window_end) > OUTAGE_TOLERANCE_US
        for day in scoped
    )
