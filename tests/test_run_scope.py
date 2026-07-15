import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.replay.blind_windows import BLIND_WINDOWS_SCHEMA
from bot.replay.run_scope import (
    D_DISC,
    D_EVAL,
    D_HOLD,
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    QUIET_BAND,
    RECORDED_GAP,
    RESUBSCRIBE_BLIND,
    SUBSCRIPTION_WIDE,
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
BAND_US = 2 * HOUR

EAST = "KXHIGHNY"
WEST = "KXHIGHTSFO"
STRIKES = (80, 85)
DAYS = [date(2026, 7, 17) + timedelta(days=offset) for offset in range(17)]
FIRST_SCOPE_DAY = date(2026, 7, 19)
LAST_SCOPE_DAY = date(2026, 8, 1)

TAPE_FIRST = datetime(2026, 7, 16, tzinfo=UTC)
TAPE_LAST = datetime(2026, 8, 2, 12, tzinfo=UTC)
SCOPE_START = datetime(2026, 7, 19, 5, tzinfo=UTC)
SCOPE_END = datetime(2026, 8, 2, 8, tzinfo=UTC)

INCIDENT_DETECTED = datetime(2026, 7, 23, 7, 43, 55, 776558, tzinfo=UTC)
INCIDENT_START = datetime(2026, 7, 23, 7, 24, 0, tzinfo=UTC)
INCIDENT_END = datetime(2026, 7, 23, 7, 43, 8, tzinfo=UTC)


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
    return event_day_inventory(coverage_rows(), tape_first=TAPE_FIRST, tape_last=TAPE_LAST)


def day_of(days: list[EventDay], series: str, event_date: date) -> EventDay:
    return next(day for day in days if day.series == series and day.event_date == event_date)


def test_the_split_is_the_forward_two_thirds_of_the_accrual() -> None:
    assert (D_EVAL, D_DISC, D_HOLD) == (14, 9, 5)
    assert D_DISC + D_HOLD == D_EVAL


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

    exclusions = blind_exclusions(rows, scope_start=SCOPE_START, scope_end=SCOPE_END)

    incident, other = exclusions[0], exclusions[1]
    assert incident.padded is True
    assert incident.start == INCIDENT_START - timedelta(minutes=5)
    assert incident.end == INCIDENT_END + timedelta(minutes=5)
    assert incident.duration_us == (INCIDENT_END - INCIDENT_START) // TICK + 600 * SECOND
    assert incident.gap_id == 5
    assert other.padded is False
    assert (other.start, other.end) == (same_day["start"], same_day["end"])
    assert other.duration_us == 20 * SECOND


def test_a_boundary_only_the_padding_brings_into_scope_is_kept_and_clipped() -> None:
    scope_start = INCIDENT_END + timedelta(minutes=2)
    rows = [BLIND_ROWS[4]]

    exclusions = blind_exclusions(rows, scope_start=scope_start, scope_end=SCOPE_END)

    assert len(exclusions) == 1
    assert exclusions[0].start == scope_start
    assert exclusions[0].end == INCIDENT_END + timedelta(minutes=5)
    assert blind_exclusions(rows, scope_start=SCOPE_END, scope_end=SCOPE_END) == []


def test_a_clipped_interval_never_escapes_the_scope() -> None:
    scope_start = datetime(2026, 7, 20, 10, 0, 10, tzinfo=UTC)
    scope_end = datetime(2026, 7, 20, 10, 0, 20, tzinfo=UTC)

    exclusions = blind_exclusions(BLIND_ROWS, scope_start=scope_start, scope_end=scope_end)

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


def test_an_event_day_running_past_the_tape_is_not_covered(inventory: list[EventDay]) -> None:
    tail = day_of(inventory, EAST, date(2026, 8, 2))

    assert tail.window_end > TAPE_LAST
    assert (tail.covered, tail.evaluable, tail.in_scope) == (False, False, False)


def test_only_parsed_kxhigh_tickers_reach_the_inventory(inventory: list[EventDay]) -> None:
    first = day_of(inventory, EAST, FIRST_SCOPE_DAY)

    assert {day.series for day in inventory} == {EAST, WEST}
    assert (first.tickers, first.ladder_rows) == (2, sum(STRIKES))
    assert (first.station, first.timezone) == ("KNYC", "America/New_York")
    assert first.first_event_at == datetime(2026, 7, 19, 9, tzinfo=UTC)
    assert first.last_event_at == datetime(2026, 7, 19, 10, tzinfo=UTC)


def test_a_series_short_of_the_accrual_cannot_be_frozen() -> None:
    days = [FIRST_SCOPE_DAY + timedelta(days=offset) for offset in range(D_EVAL - 1)]
    rows = [cov(ticker(EAST, day, 80), day, 80) for day in days]

    with pytest.raises(ValueError, match=EAST):
        event_day_inventory(rows, tape_first=TAPE_FIRST, tape_last=datetime(2026, 8, 5, tzinfo=UTC))


def test_a_hole_in_the_evaluable_run_cannot_be_frozen() -> None:
    days = [FIRST_SCOPE_DAY + timedelta(days=offset) for offset in range(16)]
    days.remove(date(2026, 7, 26))
    rows = [cov(ticker(EAST, day, 80), day, 80) for day in days]

    with pytest.raises(ValueError, match="2026-07-26"):
        event_day_inventory(rows, tape_first=TAPE_FIRST, tape_last=datetime(2026, 8, 5, tzinfo=UTC))


def test_the_frozen_split_runs_nine_days_then_five(inventory: list[EventDay]) -> None:
    split = freeze_split(inventory)

    assert split.cities == (EAST, WEST)
    assert len(split.discovery_days) == D_DISC
    assert len(split.holdout_days) == D_HOLD
    assert split.discovery_days[0] == FIRST_SCOPE_DAY
    assert split.holdout_days[-1] == LAST_SCOPE_DAY
    assert split.boundary_event_day == FIRST_SCOPE_DAY + timedelta(days=D_DISC)
    assert split.boundary_event_day == split.holdout_days[0]
    assert (split.scope_start, split.scope_end) == (SCOPE_START, SCOPE_END)
    assert [day.split for day in inventory if day.in_scope].count(DISCOVERY) == 2 * D_DISC
    assert [day.split for day in inventory if day.in_scope].count(HOLDOUT) == 2 * D_HOLD
    assert all(day.split == "" and day.day_index == 0 for day in inventory if not day.in_scope)


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
        freeze_split(shifted)


def test_the_digest_is_stable_and_moves_with_a_single_day(inventory: list[EventDay]) -> None:
    payload = split_payload(freeze_split(inventory))
    moved = dict(payload)
    moved["holdout_days"] = [*payload["holdout_days"][:-1], "2026-09-09"]

    assert freeze_digest(payload) == freeze_digest(dict(payload))
    assert freeze_digest(payload) != freeze_digest(moved)
    assert len(freeze_digest(payload)) == 64


def test_the_payload_names_the_frozen_boundary(inventory: list[EventDay]) -> None:
    payload = split_payload(freeze_split(inventory))

    assert payload["d_eval"] == D_EVAL
    assert payload["d_disc"] == D_DISC
    assert payload["d_hold"] == D_HOLD
    assert payload["cities"] == [EAST, WEST]
    assert payload["first_evaluable_event_day"] == FIRST_SCOPE_DAY.isoformat()
    assert payload["last_evaluable_event_day"] == LAST_SCOPE_DAY.isoformat()
    assert payload["boundary_event_day"] == "2026-07-28"
    assert payload["scope_start"] == SCOPE_START.isoformat()
    assert payload["scope_end"] == SCOPE_END.isoformat()


def test_a_clean_event_day_measures_twenty_two_hours(inventory: list[EventDay]) -> None:
    split = freeze_split(inventory)
    bands = quiet_band_exclusions(split.scope_start, split.scope_end)

    spanned = apply_spans(inventory, bands)

    scoped = [day for day in spanned if day.in_scope]
    assert len(scoped) == 2 * D_EVAL
    assert all(day.excluded_us == BAND_US for day in scoped)
    assert all(day.span_us == 79_200_000_000 for day in scoped)
    assert all(day.excluded_us == 0 and day.span_us == 0 for day in spanned if not day.in_scope)


def test_a_pacific_day_pays_the_band_at_both_of_its_ends(inventory: list[EventDay]) -> None:
    split = freeze_split(inventory)
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
    split = freeze_split(inventory)
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


def test_the_split_file_carries_its_own_digest(tmp_path: Path, inventory: list[EventDay]) -> None:
    path = tmp_path / "split.json"
    split = freeze_split(inventory)

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
