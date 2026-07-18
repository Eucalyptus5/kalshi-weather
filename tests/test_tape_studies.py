import json
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.r0_universe import (
    DISAGREE,
    LOCK_CARVE_OUT,
    Coverage,
    freeze_digest,
    freeze_universe,
    universe_payload,
    write_universe,
)
from bot.lag.tape_studies import (
    EvidenceWindow,
    RunScope,
    intersects_exclusion,
    load_run_scope,
    partition_files,
    partition_rows,
    read_inventory,
    read_window,
    screen_windows,
    split_of,
    window_dates,
)
from bot.replay.artifacts import (
    BOUNDARIES_SCHEMA,
    COVERAGE_SCHEMA,
    SCALARS_SCHEMA,
    TOUCH_SCHEMA,
    TRADES_SCHEMA,
    WINDOWS_SCHEMA,
)
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    QUIET_BAND,
    RECORDED_GAP,
    RESUBSCRIBE_BLIND,
    SUBSCRIPTION_WIDE,
    Split,
    write_split,
)


UTC = timezone.utc
MICROSECOND = timedelta(microseconds=1)

SERIES = "KXHIGHDEN"
TICKER = "KXHIGHDEN-26JUL18-B70"
UNCOVERED_DAY = date(2026, 7, 17)
DISCOVERY_DAY = date(2026, 7, 18)
HOLDOUT_DAY = date(2026, 7, 19)
STRANGER_DAY = date(2026, 7, 25)
SCOPE_START = datetime(2026, 7, 18, tzinfo=UTC)
SCOPE_END = datetime(2026, 7, 20, tzinfo=UTC)

QUIET_START = datetime(2026, 7, 18, 7, tzinfo=UTC)
QUIET_END = datetime(2026, 7, 18, 9, tzinfo=UTC)
GAP_START = datetime(2026, 7, 18, 8, tzinfo=UTC)
GAP_END = datetime(2026, 7, 18, 8, 10, tzinfo=UTC)
BLINK = datetime(2026, 7, 19, 12, tzinfo=UTC)
WIDE_START = datetime(2026, 7, 19, 18, tzinfo=UTC)
WIDE_END = datetime(2026, 7, 19, 18, 30, tzinfo=UTC)
ABUT_END = datetime(2026, 7, 19, 18, 40, tzinfo=UTC)
WIDE_BOUNDARY_ID = 41
WIDE_GAP_ID = 7


def touch_row(row_id: int, received_at: datetime) -> dict:
    return {
        "id": row_id,
        "ticker": TICKER,
        "received_at": received_at,
        "ts_ms": row_id * 1000,
        "yes_bid": "0.40",
        "yes_bid_depth": "10",
        "yes_ask": "0.42",
        "yes_ask_depth": "12",
        "no_bid": "0.58",
        "no_bid_depth": "12",
        "no_ask": "0.60",
        "no_ask_depth": "10",
    }


def write_partition(
    root: Path,
    day: date,
    barrier: int,
    rows: list[dict],
    *,
    kind: str = "touch",
    schema: pa.Schema = TOUCH_SCHEMA,
) -> Path:
    directory = root / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{SERIES}-{day.isoformat()}-b{barrier:06d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def exclusion_row(
    exclusion_id: int,
    exclusion_class: str,
    start: datetime,
    end: datetime,
    *,
    boundary_id: int | None = None,
    gap_id: int | None = None,
    gap_reason: str | None = None,
    padded: bool = False,
) -> dict:
    return {
        "exclusion_id": exclusion_id,
        "exclusion_class": exclusion_class,
        "start": start,
        "end": end,
        "duration_us": (end - start) // MICROSECOND,
        "boundary_id": boundary_id,
        "gap_id": gap_id,
        "gap_reason": gap_reason,
        "padded": padded,
    }


def exclusion_table() -> pa.Table:
    rows = [
        exclusion_row(0, QUIET_BAND, QUIET_START, QUIET_END),
        exclusion_row(1, RECORDED_GAP, GAP_START, GAP_END, gap_reason="seq_gap"),
        exclusion_row(2, RESUBSCRIBE_BLIND, BLINK, BLINK + MICROSECOND),
        exclusion_row(
            3,
            SUBSCRIPTION_WIDE,
            WIDE_START,
            WIDE_END,
            boundary_id=WIDE_BOUNDARY_ID,
            gap_id=WIDE_GAP_ID,
            gap_reason="seq_skip",
            padded=True,
        ),
        exclusion_row(4, RECORDED_GAP, WIDE_END, ABUT_END, gap_reason="seq_gap"),
    ]
    return pa.Table.from_pylist(rows, schema=EXCLUSIONS_SCHEMA)


def event_day_row(event_date: date, *, in_scope: bool, split: str, day_index: int = 0) -> dict:
    midnight = datetime(event_date.year, event_date.month, event_date.day, tzinfo=UTC)
    return {
        "series": SERIES,
        "station": "KDEN",
        "timezone": "America/Denver",
        "event_date": event_date,
        "window_start": midnight,
        "window_end": midnight + timedelta(days=1),
        "tickers": 6,
        "ladder_rows": 900,
        "first_event_at": midnight,
        "last_event_at": midnight + timedelta(days=1),
        "covered": in_scope,
        "evaluable": in_scope,
        "in_scope": in_scope,
        "day_index": day_index,
        "split": split,
        "excluded_us": 0,
        "span_us": 0,
    }


def event_day_table() -> pa.Table:
    rows = [
        event_day_row(UNCOVERED_DAY, in_scope=False, split=""),
        event_day_row(DISCOVERY_DAY, in_scope=True, split=DISCOVERY, day_index=1),
        event_day_row(HOLDOUT_DAY, in_scope=True, split=HOLDOUT, day_index=2),
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def scope_dir(
    tmp_path: Path, *, exclusions: pa.Table | None = None, days: pa.Table | None = None
) -> Path:
    directory = tmp_path / "scope"
    directory.mkdir()
    pq.write_table(
        exclusion_table() if exclusions is None else exclusions, directory / "exclusions.parquet"
    )
    pq.write_table(event_day_table() if days is None else days, directory / "event_days.parquet")
    write_split(
        directory / "split.json",
        Split(
            cities=(SERIES,),
            discovery_days=(DISCOVERY_DAY,),
            holdout_days=(HOLDOUT_DAY,),
            boundary_event_day=HOLDOUT_DAY,
            scope_start=SCOPE_START,
            scope_end=SCOPE_END,
        ),
    )
    write_universe(
        directory / "r0_universe.json",
        freeze_universe(
            fraction_invalid_max=Decimal("0.4"),
            passing=(SERIES, "KXHIGHLAX", LOCK_CARVE_OUT),
            coverage=Coverage(
                cities=("KXHIGHCHI", SERIES, "KXHIGHLAX"),
                ladder_widths=(6, 7),
                in_scope_city_days=28,
            ),
        ),
    )
    return directory


def inventory_root(tmp_path: Path, *, coverage: pa.Table | None = None) -> Path:
    directory = tmp_path / "inventory"
    directory.mkdir()
    midnight = datetime(2026, 7, 18, tzinfo=UTC)
    pq.write_table(
        pa.Table.from_pylist([], schema=WINDOWS_SCHEMA), directory / "windows-b000000.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": 7, "received_at": midnight, "prev_seq": 4, "seq": 9, "kind": "seq_skip"}],
            schema=BOUNDARIES_SCHEMA,
        ),
        directory / "boundaries-b000000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "ticker": TICKER,
                    "rows": 900,
                    "first_received_at": midnight,
                    "last_received_at": midnight,
                }
            ],
            schema=COVERAGE_SCHEMA,
        )
        if coverage is None
        else coverage,
        directory / "coverage-b000000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"name": "gap_rows", "value": "12"}, {"name": "tickers", "value": "20"}],
            schema=SCALARS_SCHEMA,
        ),
        directory / "scalars-b000000.parquet",
    )
    return tmp_path


def evidence(
    start: datetime, end: datetime, *, series: str = SERIES, event_date: date = DISCOVERY_DAY
) -> EvidenceWindow:
    return EvidenceWindow(series=series, event_date=event_date, start=start, end=end)


@pytest.fixture
def scope(tmp_path: Path) -> RunScope:
    return load_run_scope(scope_dir(tmp_path))


def test_a_window_across_utc_midnight_reads_both_partitions(tmp_path: Path) -> None:
    late = datetime(2026, 7, 18, 23, 59, tzinfo=UTC)
    early = datetime(2026, 7, 19, 0, 1, tzinfo=UTC)
    write_partition(tmp_path, DISCOVERY_DAY, 1, [touch_row(1, late)])
    write_partition(tmp_path, HOLDOUT_DAY, 1, [touch_row(2, early)])

    table = read_window(
        tmp_path, "touch", SERIES, late - timedelta(minutes=5), early + timedelta(minutes=5)
    )

    assert window_dates(late, early) == [DISCOVERY_DAY, HOLDOUT_DAY]
    assert [
        path.name
        for path in partition_files(tmp_path, "touch", SERIES, [HOLDOUT_DAY, DISCOVERY_DAY])
    ] == [
        f"{SERIES}-2026-07-18-b000001.parquet",
        f"{SERIES}-2026-07-19-b000001.parquet",
    ]
    assert table.column("id").to_pylist() == [1, 2]


def test_every_barrier_file_in_a_partition_is_read_in_id_order(tmp_path: Path) -> None:
    base = datetime(2026, 7, 18, 12, tzinfo=UTC)
    write_partition(
        tmp_path,
        DISCOVERY_DAY,
        2,
        [touch_row(5, base + timedelta(minutes=5)), touch_row(6, base + timedelta(minutes=6))],
    )
    write_partition(
        tmp_path,
        DISCOVERY_DAY,
        1,
        [touch_row(1, base), touch_row(2, base + timedelta(minutes=1))],
    )

    paths = partition_files(tmp_path, "touch", SERIES, [DISCOVERY_DAY])
    table = read_window(tmp_path, "touch", SERIES, base, base + timedelta(hours=1))

    assert [path.name for path in paths] == [
        f"{SERIES}-2026-07-18-b000001.parquet",
        f"{SERIES}-2026-07-18-b000002.parquet",
    ]
    assert table.column("id").to_pylist() == [1, 2, 5, 6]


def test_the_window_endpoints_are_kept_and_the_rows_outside_are_dropped(tmp_path: Path) -> None:
    start = datetime(2026, 7, 18, 12, tzinfo=UTC)
    end = start + timedelta(minutes=10)
    write_partition(
        tmp_path,
        DISCOVERY_DAY,
        1,
        [
            touch_row(1, start - MICROSECOND),
            touch_row(2, start),
            touch_row(3, start + timedelta(minutes=5)),
            touch_row(4, end),
            touch_row(5, end + MICROSECOND),
        ],
    )

    table = read_window(tmp_path, "touch", SERIES, start, end)

    assert table.column("id").to_pylist() == [2, 3, 4]


def test_a_partition_carrying_another_kinds_schema_raises_and_names_the_file(
    tmp_path: Path,
) -> None:
    path = write_partition(tmp_path, DISCOVERY_DAY, 1, [], schema=TRADES_SCHEMA)

    with pytest.raises(ValueError, match=re.escape(str(path))):
        read_window(tmp_path, "touch", SERIES, SCOPE_START, SCOPE_END)


def test_the_metadata_row_count_matches_a_full_read(tmp_path: Path) -> None:
    base = datetime(2026, 7, 18, 12, tzinfo=UTC)
    write_partition(
        tmp_path, DISCOVERY_DAY, 1, [touch_row(1, base), touch_row(2, base + timedelta(minutes=1))]
    )
    write_partition(tmp_path, DISCOVERY_DAY, 2, [touch_row(3, base + timedelta(minutes=2))])

    counted = partition_rows(tmp_path, "touch", SERIES, [DISCOVERY_DAY])
    table = read_window(
        tmp_path, "touch", SERIES, base - timedelta(hours=1), base + timedelta(hours=1)
    )

    assert counted == 3
    assert counted == table.num_rows


def test_the_inventory_scalars_round_trip_to_a_dict(tmp_path: Path) -> None:
    inventory = read_inventory(inventory_root(tmp_path))

    assert inventory.values == {"gap_rows": "12", "tickers": "20"}
    assert inventory.scalars.num_rows == 2
    assert inventory.windows.num_rows == 0
    assert inventory.boundaries.column("kind").to_pylist() == ["seq_skip"]
    assert inventory.coverage.column("ticker").to_pylist() == [TICKER]


def test_an_inventory_table_with_a_foreign_schema_raises_and_names_the_file(
    tmp_path: Path,
) -> None:
    root = inventory_root(tmp_path, coverage=pa.table({"ticker": [TICKER]}))
    path = root / "inventory" / "coverage-b000000.parquet"

    with pytest.raises(ValueError, match=re.escape(str(path))):
        read_inventory(root)


def test_the_frozen_scope_loads(scope: RunScope) -> None:
    assert scope.scope_start == SCOPE_START
    assert scope.scope_end == SCOPE_END
    assert scope.discovery_days == frozenset({DISCOVERY_DAY})
    assert scope.holdout_days == frozenset({HOLDOUT_DAY})
    assert set(scope.event_days) == {(SERIES, DISCOVERY_DAY), (SERIES, HOLDOUT_DAY)}
    assert [item.exclusion_class for item in scope.exclusions] == [
        QUIET_BAND,
        RECORDED_GAP,
        RESUBSCRIBE_BLIND,
        SUBSCRIPTION_WIDE,
        RECORDED_GAP,
    ]


def test_the_frozen_universe_round_trips_field_for_field(tmp_path: Path) -> None:
    directory = scope_dir(tmp_path)
    stored = json.loads((directory / "r0_universe.json").read_text())["sha256"]

    universe = load_run_scope(directory).universe

    assert universe.fraction_invalid_max == Decimal("0.4")
    assert universe.passing == (SERIES, "KXHIGHLAX", LOCK_CARVE_OUT)
    assert universe.lock_dependent == (SERIES, "KXHIGHLAX")
    assert universe.recorded == ("KXHIGHCHI", SERIES, "KXHIGHLAX")
    assert universe.ladder_widths == (6, 7)
    assert universe.in_scope_city_days == 28
    assert universe.reconciliation == DISAGREE
    assert universe.recorded_not_passing == ("KXHIGHCHI",)
    assert universe.passing_not_recorded == (LOCK_CARVE_OUT,)
    assert freeze_digest(universe_payload(universe)) == stored


def test_the_exclusion_columns_land_on_their_own_fields(scope: RunScope) -> None:
    wide = next(item for item in scope.exclusions if item.exclusion_class == SUBSCRIPTION_WIDE)

    assert wide.start == WIDE_START
    assert wide.end == WIDE_END
    assert wide.boundary_id == WIDE_BOUNDARY_ID
    assert wide.gap_id == WIDE_GAP_ID
    assert wide.gap_reason == "seq_skip"
    assert wide.padded is True


def test_the_overlapping_exclusions_merge_into_one_interval(scope: RunScope) -> None:
    assert scope.merged.starts == (QUIET_START, BLINK, WIDE_START)
    assert scope.merged.ends == (QUIET_END, BLINK + MICROSECOND, ABUT_END)
    assert len(scope.exclusions) == 5


def test_two_exclusions_meeting_end_to_start_merge_into_one_interval(scope: RunScope) -> None:
    abutting = next(item for item in scope.exclusions if item.end == ABUT_END)

    assert abutting.start == WIDE_END
    assert WIDE_END not in scope.merged.starts
    assert scope.merged.ends[scope.merged.starts.index(WIDE_START)] == ABUT_END


def test_a_class_with_no_exclusions_is_still_counted(tmp_path: Path) -> None:
    exclusions = pa.Table.from_pylist(
        [exclusion_row(0, QUIET_BAND, QUIET_START, QUIET_END)], schema=EXCLUSIONS_SCHEMA
    )
    scope = load_run_scope(scope_dir(tmp_path, exclusions=exclusions))

    screened = screen_windows(scope, [evidence(QUIET_START, QUIET_END)])

    assert set(scope.by_class) == {QUIET_BAND, RECORDED_GAP, RESUBSCRIBE_BLIND, SUBSCRIPTION_WIDE}
    assert screened.by_class == {
        QUIET_BAND: 1,
        RECORDED_GAP: 0,
        RESUBSCRIBE_BLIND: 0,
        SUBSCRIPTION_WIDE: 0,
    }


def test_an_exclusion_class_outside_the_frozen_set_raises(tmp_path: Path) -> None:
    exclusions = pa.Table.from_pylist(
        [exclusion_row(0, "venue_halt", QUIET_START, QUIET_END)], schema=EXCLUSIONS_SCHEMA
    )
    directory = scope_dir(tmp_path, exclusions=exclusions)

    with pytest.raises(ValueError, match="venue_halt"):
        load_run_scope(directory)


def test_a_tampered_split_payload_raises(tmp_path: Path) -> None:
    directory = scope_dir(tmp_path)
    path = directory / "split.json"
    payload = json.loads(path.read_text())
    payload["d_disc"] = payload["d_disc"] + 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="split.json"):
        load_run_scope(directory)


def test_a_tampered_universe_payload_raises(tmp_path: Path) -> None:
    directory = scope_dir(tmp_path)
    path = directory / "r0_universe.json"
    payload = json.loads(path.read_text())
    payload["fraction_invalid_max"] = "0.9"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="r0_universe.json"):
        load_run_scope(directory)


def test_a_frozen_payload_carrying_no_digest_raises(tmp_path: Path) -> None:
    directory = scope_dir(tmp_path)
    path = directory / "split.json"
    payload = json.loads(path.read_text())
    del payload["sha256"]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="split.json"):
        load_run_scope(directory)


def test_an_event_day_split_disagreeing_with_the_frozen_lists_raises(tmp_path: Path) -> None:
    days = pa.Table.from_pylist(
        [
            event_day_row(DISCOVERY_DAY, in_scope=True, split=HOLDOUT, day_index=1),
            event_day_row(HOLDOUT_DAY, in_scope=True, split=HOLDOUT, day_index=2),
        ],
        schema=EVENT_DAYS_SCHEMA,
    )
    directory = scope_dir(tmp_path, days=days)

    with pytest.raises(ValueError, match="event_days.parquet"):
        load_run_scope(directory)


def test_a_scope_parquet_with_a_foreign_schema_raises(tmp_path: Path) -> None:
    directory = scope_dir(tmp_path, exclusions=pa.table({"exclusion_id": [0]}))

    with pytest.raises(ValueError, match="exclusions.parquet"):
        load_run_scope(directory)


def test_a_window_inside_an_exclusion_drops(scope: RunScope) -> None:
    inside = evidence(GAP_START + timedelta(minutes=2), GAP_START + timedelta(minutes=5))

    screened = screen_windows(scope, [inside])

    assert intersects_exclusion(scope, inside.start, inside.end)
    assert screened.kept == ()
    assert screened.candidates == 1
    assert screened.excluded == 1
    assert screened.out_of_scope == 0


def test_a_window_straddling_either_edge_of_an_exclusion_drops(scope: RunScope) -> None:
    leading = evidence(QUIET_START - timedelta(minutes=30), QUIET_START + timedelta(minutes=30))
    trailing = evidence(QUIET_END - timedelta(minutes=30), QUIET_END + timedelta(minutes=30))

    screened = screen_windows(scope, [leading, trailing])

    assert screened.kept == ()
    assert screened.excluded == 2


def test_a_window_ending_on_an_exclusion_start_drops(scope: RunScope) -> None:
    touching = evidence(QUIET_START - timedelta(hours=1), QUIET_START)

    screened = screen_windows(scope, [touching])

    assert screened.kept == ()
    assert screened.excluded == 1


def test_a_window_starting_on_an_exclusion_end_drops(scope: RunScope) -> None:
    touching = evidence(QUIET_END, QUIET_END + timedelta(hours=1))
    clear = evidence(QUIET_END + MICROSECOND, QUIET_END + timedelta(hours=1))

    screened = screen_windows(scope, [touching, clear])

    assert screened.kept == (clear,)
    assert screened.excluded == 1


def test_a_window_ending_one_microsecond_short_is_kept_unchanged(scope: RunScope) -> None:
    clear = evidence(QUIET_START - timedelta(hours=1), QUIET_START - MICROSECOND)

    screened = screen_windows(scope, [clear])

    assert screened.kept == (clear,)
    assert screened.kept[0] is clear
    assert screened.excluded == 0
    assert screened.by_class == {
        QUIET_BAND: 0,
        RECORDED_GAP: 0,
        RESUBSCRIBE_BLIND: 0,
        SUBSCRIPTION_WIDE: 0,
    }


def test_a_one_microsecond_exclusion_drops_a_six_hour_window(scope: RunScope) -> None:
    spanning = evidence(
        BLINK - timedelta(hours=3), BLINK + timedelta(hours=3), event_date=HOLDOUT_DAY
    )

    screened = screen_windows(scope, [spanning])

    assert spanning.end - spanning.start == timedelta(hours=6)
    assert screened.excluded == 1
    assert screened.by_class[RESUBSCRIBE_BLIND] == 1


def test_an_out_of_scope_event_day_is_counted_apart_from_an_exclusion(scope: RunScope) -> None:
    stranger = evidence(
        datetime(2026, 7, 25, 12, tzinfo=UTC),
        datetime(2026, 7, 25, 13, tzinfo=UTC),
        event_date=STRANGER_DAY,
    )
    foreign = evidence(GAP_START, GAP_END, series="KXHIGHNY")
    excluded = evidence(GAP_START, GAP_END)

    screened = screen_windows(scope, [stranger, foreign, excluded])

    assert screened.candidates == 3
    assert screened.out_of_scope == 2
    assert screened.excluded == 1
    assert screened.kept == ()


def test_a_window_meeting_two_classes_at_once_counts_in_both(scope: RunScope) -> None:
    overlap = evidence(GAP_START + timedelta(minutes=2), GAP_START + timedelta(minutes=5))

    screened = screen_windows(scope, [overlap])

    assert screened.excluded == 1
    assert screened.by_class[QUIET_BAND] == 1
    assert screened.by_class[RECORDED_GAP] == 1
    assert sum(screened.by_class.values()) > screened.excluded


def test_the_lost_fraction_is_an_exact_decimal(scope: RunScope) -> None:
    base = datetime(2026, 7, 18, 10, tzinfo=UTC)
    clear = [
        evidence(base + timedelta(minutes=index), base + timedelta(minutes=index, seconds=30))
        for index in range(7)
    ]

    screened = screen_windows(scope, [*clear, evidence(GAP_START, GAP_END)])

    assert screened.candidates == 8
    assert screened.excluded == 1
    assert isinstance(screened.excluded_fraction, Decimal)
    assert screened.excluded_fraction == Decimal("0.125")


def test_the_lost_fraction_divides_by_every_window_the_question_offered(scope: RunScope) -> None:
    base = datetime(2026, 7, 18, 10, tzinfo=UTC)
    clear = [
        evidence(base + timedelta(minutes=index), base + timedelta(minutes=index, seconds=30))
        for index in range(2)
    ]
    stranger = evidence(base, base + timedelta(minutes=1), event_date=STRANGER_DAY)

    screened = screen_windows(scope, [*clear, stranger, evidence(GAP_START, GAP_END)])

    assert screened.candidates == 4
    assert screened.out_of_scope == 1
    assert screened.excluded == 1
    assert screened.excluded_fraction == Decimal("0.25")


def test_a_question_offering_no_windows_reports_no_lost_fraction(scope: RunScope) -> None:
    screened = screen_windows(scope, [])

    assert screened.candidates == 0
    assert screened.excluded == 0
    with pytest.raises(InvalidOperation):
        _ = screened.excluded_fraction


def test_the_split_of_an_in_scope_day_comes_from_the_frozen_lists(scope: RunScope) -> None:
    assert split_of(scope, SERIES, DISCOVERY_DAY) == DISCOVERY
    assert split_of(scope, SERIES, HOLDOUT_DAY) == HOLDOUT


def test_the_split_of_a_day_outside_the_scope_raises(scope: RunScope) -> None:
    with pytest.raises(ValueError, match=UNCOVERED_DAY.isoformat()):
        split_of(scope, SERIES, UNCOVERED_DAY)

    with pytest.raises(ValueError, match=STRANGER_DAY.isoformat()):
        split_of(scope, SERIES, STRANGER_DAY)
