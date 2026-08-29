import ast
import inspect
import json
import re
import subprocess
from dataclasses import fields
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag import ladder_run, lock_convergence, taker_flow_run
from bot.lag.fee_floor import (
    MAKER_RATE_SOURCE,
    PUBLISHED_MAKER_RATE,
    economic_bar_cents_per_contract,
    fee_source,
)
from bot.lag.r0_universe import (
    DISAGREE,
    LOCK_CARVE_OUT,
    Coverage,
    freeze_digest,
    freeze_universe,
    universe_payload,
    write_universe,
)
from bot.lag.read_rtt import FloorSource, ReadSample, encode_sample
from bot.lag.run_manifest import (
    MANIFEST_NAME,
    ManifestIncomplete,
    RunInputs,
    build_manifest,
    manifest_payload,
    write_manifest,
)
from bot.lag.tape_studies import (
    KIND_SCHEMAS,
    LADDER,
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    TOUCH,
    TRADES,
    EvidenceWindow,
    RunScope,
    assemble_run_inputs,
    intersects_exclusion,
    keep_mask,
    load_run_scope,
    partition_files,
    partition_rows,
    read_inventory,
    read_window,
    screen_windows,
    split_of,
    window_dates,
    within_event_day,
)
from bot.replay.analysis_stations import HIGH, LOW
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
LOW_SERIES = "KXLOWTDEN"
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

RUN_ID = "2026-08-12-q1"
SEED = 20260812
ADEQUATE_SAMPLES = 240
SHORT_SAMPLES = 58
STANDARD_OFFSET = timedelta(hours=7)
LATE_ARRIVAL = date(2026, 7, 20)
NOON = datetime(2026, 7, 18, 12, tzinfo=UTC)
ARTIFACT_PARTITIONS = (
    (TOUCH, UNCOVERED_DAY, 7),
    (TOUCH, DISCOVERY_DAY, 2),
    (TOUCH, HOLDOUT_DAY, 3),
    (TOUCH, LATE_ARRIVAL, 5),
    (LADDER, DISCOVERY_DAY, 1),
    (LADDER, LATE_ARRIVAL, 4),
    (TRADES, HOLDOUT_DAY, 6),
)
CONSUMED = {TOUCH: 10, LADDER: 5, TRADES: 6}
LOW_PARTITIONS = (
    (TOUCH, DISCOVERY_DAY, 4),
    (LADDER, HOLDOUT_DAY, 2),
    (TRADES, LATE_ARRIVAL, 1),
)
LOW_CONSUMED = {TOUCH: 4, LADDER: 2, TRADES: 1}
FRACTION_INVALID_MAX = Decimal("0.4")
CARVED_OUT = LOCK_CARVE_OUT[0]
EXCLUSION_ROWS = 5
EVENT_DAY_ROWS = 3

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
KNOWN_FAMILY_SCRIPTS = {
    "tape_report.py",
    "q1_report.py",
    "q2_report.py",
    "q3_report.py",
    "q3_near_lock.py",
    "q4_report.py",
    "depth_report.py",
    "depth_continuity_report.py",
    "f1_report.py",
}


def family_scripts() -> tuple[str, ...]:
    return tuple(
        sorted(path.name for path in SCRIPTS.glob("*.py") if "run_manifest" in path.read_text())
    )


FAMILY_SCRIPTS = family_scripts()

STATED_BAR = ("economic_bar_size", "economic_bar_price", "economic_bar_price_source")
DERIVED_BAR = "economic_bar_cents_per_contract"
STATED_REGIME = ("maker_rate", "maker_rate_source")


def argument_flags(source: str) -> set[str]:
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    named = [argument for call in calls for argument in call.args]
    named += [keyword.value for call in calls for keyword in call.keywords if keyword.arg == "dest"]
    return {
        item.value.lstrip("-").replace("-", "_")
        for item in named
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


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


def ladder_row(row_id: int, received_at: datetime) -> dict:
    return touch_row(row_id, received_at) | {
        "yes_prices": ["0.40"],
        "yes_sizes": ["10"],
        "yes_levels": 1,
        "no_prices": ["0.58"],
        "no_sizes": ["12"],
        "no_levels": 1,
    }


def trade_row(row_id: int, received_at: datetime) -> dict:
    return {
        "id": row_id,
        "ticker": TICKER,
        "received_at": received_at,
        "ts_ms": row_id * 1000,
        "yes_price": "0.41",
        "no_price": "0.59",
        "count": "3",
        "taker_side": "yes",
        "trade_id": f"t{row_id}",
    }


def write_partition(
    root: Path,
    day: date,
    barrier: int,
    rows: list[dict],
    *,
    kind: str = "touch",
    schema: pa.Schema = TOUCH_SCHEMA,
    series: str = SERIES,
) -> Path:
    directory = root / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{series}-{day.isoformat()}-b{barrier:06d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def artifact_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    builders = {TOUCH: touch_row, LADDER: ladder_row, TRADES: trade_row}
    for kind, day, rows in ARTIFACT_PARTITIONS:
        write_partition(
            root,
            day,
            1,
            [builders[kind](index, NOON) for index in range(rows)],
            kind=kind,
            schema=KIND_SCHEMAS[kind],
        )
    return root


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


def event_day_row(
    event_date: date,
    *,
    in_scope: bool,
    split: str,
    day_index: int = 0,
    opens: timedelta = timedelta(),
    series: str = SERIES,
) -> dict:
    midnight = datetime(event_date.year, event_date.month, event_date.day, tzinfo=UTC)
    return {
        "series": series,
        "station": "KDEN",
        "timezone": "America/Denver",
        "event_date": event_date,
        "window_start": midnight + opens,
        "window_end": midnight + opens + timedelta(days=1),
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


def local_event_day_table() -> pa.Table:
    rows = [
        event_day_row(UNCOVERED_DAY, in_scope=False, split="", opens=STANDARD_OFFSET),
        event_day_row(
            DISCOVERY_DAY, in_scope=True, split=DISCOVERY, day_index=1, opens=STANDARD_OFFSET
        ),
        event_day_row(
            HOLDOUT_DAY, in_scope=True, split=HOLDOUT, day_index=2, opens=STANDARD_OFFSET
        ),
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
            fraction_invalid_max=FRACTION_INVALID_MAX,
            passing=(SERIES, "KXHIGHLAX", CARVED_OUT),
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


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def seeded_repo(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "core.hooksPath", str(root / ".git" / "hooks"))
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "config", "user.name", "tape")
    _git(root, "config", "user.email", "tape@example.invalid")
    (root / "seed.txt").write_text("seed\n")
    _git(root, "add", "seed.txt")
    _git(root, "commit", "-q", "-m", "seed")
    return root


def write_rtt_samples(path: Path, count: int) -> Path:
    samples = [
        ReadSample(
            sequence=index,
            requested_at=datetime(2026, 8, 12, index % 24, index % 60, tzinfo=UTC),
            elapsed_s=0.24,
            ticker=TICKER,
            outcome="ok",
            status_code=200,
            api_host="api.elections.kalshi.com",
            endpoint=f"GET /trade-api/v2/markets/{TICKER}/orderbook",
            source_host="kalshi-ws",
        )
        for index in range(count)
    ]
    path.write_text("".join(encode_sample(sample) + "\n" for sample in samples))
    return path


def write_preregistration(path: Path) -> Path:
    path.write_text("preregistration\n")
    return path


def run_input_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path, days=local_event_day_table()),
        "artifacts": artifact_root(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
    }


def assemble(paths: dict[str, Path], cohort: str | None = None) -> RunInputs:
    return assemble_run_inputs(
        run_id=RUN_ID,
        floor_source=FloorSource.SIGNED_READ,
        maker_rate=PUBLISHED_MAKER_RATE,
        maker_rate_source=MAKER_RATE_SOURCE,
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        bootstrap_seed=SEED,
        cohort=cohort,
        **paths,
    )


def both_ladder_day_table() -> pa.Table:
    rows = [
        event_day_row(
            day,
            in_scope=True,
            split=split,
            day_index=index,
            opens=STANDARD_OFFSET,
            series=series,
        )
        for series in (SERIES, LOW_SERIES)
        for index, (day, split) in enumerate(
            ((DISCOVERY_DAY, DISCOVERY), (HOLDOUT_DAY, HOLDOUT)), start=1
        )
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def both_ladder_paths(tmp_path: Path) -> dict[str, Path]:
    root = artifact_root(tmp_path)
    builders = {TOUCH: touch_row, LADDER: ladder_row, TRADES: trade_row}
    for kind, day, rows in LOW_PARTITIONS:
        write_partition(
            root,
            day,
            1,
            [builders[kind](index, NOON) for index in range(rows)],
            kind=kind,
            schema=KIND_SCHEMAS[kind],
            series=LOW_SERIES,
        )
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path, days=both_ladder_day_table()),
        "artifacts": root,
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
    }


@pytest.fixture
def scope(tmp_path: Path) -> RunScope:
    return load_run_scope(scope_dir(tmp_path))


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return run_input_paths(tmp_path)


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
    assert universe.passing == (SERIES, "KXHIGHLAX", CARVED_OUT)
    assert universe.lock_dependent == (SERIES, "KXHIGHLAX")
    assert universe.recorded == ("KXHIGHCHI", SERIES, "KXHIGHLAX")
    assert universe.ladder_widths == (6, 7)
    assert universe.in_scope_city_days == 28
    assert universe.reconciliation == DISAGREE
    assert universe.recorded_not_passing == ("KXHIGHCHI",)
    assert universe.passing_not_recorded == (CARVED_OUT,)
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


def test_a_window_inside_its_event_day_is_kept(scope: RunScope) -> None:
    day = scope.event_days[(SERIES, DISCOVERY_DAY)]
    inside = evidence(day.window_start + timedelta(hours=1), day.window_start + timedelta(hours=2))

    screened = screen_windows(scope, [inside])

    assert within_event_day(scope, inside)
    assert screened.kept == (inside,)
    assert screened.candidates == 1
    assert screened.out_of_window == 0


def test_a_window_sitting_on_its_event_day_endpoints_is_kept(scope: RunScope) -> None:
    day = scope.event_days[(SERIES, DISCOVERY_DAY)]
    opening = evidence(day.window_start, day.window_start + timedelta(hours=6))
    closing = evidence(day.window_end - timedelta(hours=4), day.window_end)

    screened = screen_windows(scope, [opening, closing])

    assert within_event_day(scope, opening)
    assert within_event_day(scope, closing)
    assert screened.kept == (opening, closing)
    assert screened.out_of_window == 0


def test_a_window_opening_one_microsecond_before_its_event_day_is_out_of_window(
    scope: RunScope,
) -> None:
    day = scope.event_days[(SERIES, DISCOVERY_DAY)]
    early = evidence(day.window_start - MICROSECOND, day.window_start + timedelta(hours=6))

    screened = screen_windows(scope, [early])

    assert not within_event_day(scope, early)
    assert screened.candidates == 1
    assert screened.out_of_window == 1
    assert screened.excluded == 0
    assert screened.out_of_scope == 0
    assert screened.kept == ()


def test_a_window_closing_one_microsecond_after_its_event_day_is_out_of_window(
    scope: RunScope,
) -> None:
    day = scope.event_days[(SERIES, DISCOVERY_DAY)]
    late = evidence(day.window_end - timedelta(hours=4), day.window_end + MICROSECOND)

    screened = screen_windows(scope, [late])

    assert not within_event_day(scope, late)
    assert screened.out_of_window == 1
    assert screened.excluded == 0
    assert screened.kept == ()


def test_a_window_outside_its_event_day_is_never_charged_to_an_exclusion(scope: RunScope) -> None:
    early = evidence(
        QUIET_START + timedelta(minutes=30),
        QUIET_END - timedelta(minutes=30),
        event_date=HOLDOUT_DAY,
    )

    screened = screen_windows(scope, [early])

    assert intersects_exclusion(scope, early.start, early.end)
    assert early.end < scope.event_days[(SERIES, HOLDOUT_DAY)].window_start
    assert screened.out_of_window == 1
    assert screened.excluded == 0
    assert set(screened.by_class.values()) == {0}
    assert screened.kept == ()


def test_an_event_day_the_scope_never_froze_is_out_of_scope_not_out_of_window(
    scope: RunScope,
) -> None:
    stranger = evidence(
        datetime(2026, 7, 25, 12, tzinfo=UTC),
        datetime(2026, 7, 25, 13, tzinfo=UTC),
        event_date=STRANGER_DAY,
    )

    screened = screen_windows(scope, [stranger])

    assert screened.out_of_scope == 1
    assert screened.out_of_window == 0
    with pytest.raises(ValueError, match=STRANGER_DAY.isoformat()):
        within_event_day(scope, stranger)


def test_the_kept_windows_hold_the_order_the_question_offered(scope: RunScope) -> None:
    day = scope.event_days[(SERIES, DISCOVERY_DAY)]
    first = evidence(day.window_start + timedelta(hours=1), day.window_start + timedelta(hours=2))
    second = evidence(day.window_start + timedelta(hours=3), day.window_start + timedelta(hours=4))
    early = evidence(day.window_start - MICROSECOND, day.window_start + timedelta(minutes=1))
    stranger = evidence(GAP_START, GAP_END, event_date=STRANGER_DAY)

    screened = screen_windows(scope, [early, first, stranger, evidence(GAP_START, GAP_END), second])

    assert screened.kept == (first, second)
    assert screened.candidates == 5
    assert screened.out_of_window == 1
    assert screened.out_of_scope == 1
    assert screened.excluded == 1


def test_the_keep_mask_recovers_the_windows_the_screen_kept(scope: RunScope) -> None:
    clear = evidence(NOON, NOON + timedelta(minutes=1))
    hit = evidence(GAP_START + timedelta(minutes=2), GAP_START + timedelta(minutes=5))
    offered = [clear, hit, clear]

    screened = screen_windows(scope, offered)

    assert screened.excluded == 1
    assert keep_mask(offered, screened.kept).tolist() == [True, False, True]


def test_two_windows_sharing_a_span_are_kept_or_dropped_together(scope: RunScope) -> None:
    clear = evidence(NOON, NOON + timedelta(minutes=1))
    hit = evidence(GAP_START + timedelta(minutes=2), GAP_START + timedelta(minutes=5))

    kept_twice = screen_windows(scope, [clear, clear])
    dropped_twice = screen_windows(scope, [hit, hit])

    assert keep_mask([clear, clear], kept_twice.kept).tolist() == [True, True]
    assert keep_mask([hit, hit], dropped_twice.kept).tolist() == [False, False]


def test_the_keep_mask_survives_a_dropped_first_window(scope: RunScope) -> None:
    hit = evidence(GAP_START + timedelta(minutes=2), GAP_START + timedelta(minutes=5))
    clear = evidence(NOON, NOON + timedelta(minutes=1))
    offered = [hit, clear]

    screened = screen_windows(scope, offered)

    assert keep_mask(offered, screened.kept).tolist() == [False, True]


def test_a_kept_list_the_offer_never_carried_raises() -> None:
    offered = [evidence(NOON, NOON + timedelta(minutes=1))]
    stranger = [evidence(NOON + timedelta(hours=1), NOON + timedelta(hours=1, minutes=1))]

    with pytest.raises(ValueError):
        keep_mask(offered, stranger)


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


def test_the_assembled_inputs_carry_every_field_the_manifest_names(paths: dict[str, Path]) -> None:
    inputs = assemble(paths)

    assert [field.name for field in fields(RunInputs) if getattr(inputs, field.name) is None] == [
        "cohort",
        "fee_type_check",
        "settlement",
    ]
    manifest = build_manifest(inputs)
    assert manifest.run_id == RUN_ID
    assert manifest.preregistration == paths["preregistration"]
    assert manifest.fee == fee_source(
        maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE
    )
    assert manifest.floor.source is FloorSource.SIGNED_READ
    assert manifest.floor.n_usable == ADEQUATE_SAMPLES
    assert manifest.bootstrap_seed == SEED


def test_the_accrual_window_is_the_one_the_scope_froze(paths: dict[str, Path]) -> None:
    days = load_run_scope(paths["run_scope"]).event_days.values()

    inputs = assemble(paths)

    assert inputs.accrual_start == SCOPE_START
    assert inputs.accrual_end == SCOPE_END
    assert inputs.accrual_start != min(day.window_start for day in days)
    assert inputs.accrual_end != max(day.window_end for day in days)


def test_the_row_counts_name_each_artifact_kind_and_both_scope_tables(
    paths: dict[str, Path],
) -> None:
    inputs = assemble(paths)

    assert inputs.row_counts == {
        TOUCH: CONSUMED[TOUCH],
        LADDER: CONSUMED[LADDER],
        TRADES: CONSUMED[TRADES],
        "exclusions": EXCLUSION_ROWS,
        "event_days": EVENT_DAY_ROWS,
    }
    assert inputs.row_counts["exclusions"] == len(load_run_scope(paths["run_scope"]).exclusions)


def test_an_event_day_window_across_utc_midnight_reaches_the_later_partition(
    paths: dict[str, Path],
) -> None:
    root = paths["artifacts"]
    frozen = load_run_scope(paths["run_scope"])
    day = frozen.event_days[(SERIES, HOLDOUT_DAY)]

    inputs = assemble(paths)

    assert window_dates(day.window_start, day.window_end) == [HOLDOUT_DAY, LATE_ARRIVAL]
    assert LATE_ARRIVAL not in {event_date for _, event_date in frozen.event_days}
    assert inputs.row_counts[TOUCH] > partition_rows(
        root, TOUCH, SERIES, [DISCOVERY_DAY, HOLDOUT_DAY]
    )
    assert partition_rows(root, TOUCH, SERIES, [LATE_ARRIVAL]) == 5


def test_a_partition_before_the_first_in_scope_event_day_is_not_counted(
    paths: dict[str, Path],
) -> None:
    inputs = assemble(paths)

    assert partition_rows(paths["artifacts"], TOUCH, SERIES, [UNCOVERED_DAY]) == 7
    assert inputs.row_counts[TOUCH] == CONSUMED[TOUCH]


def test_the_row_counts_hold_only_the_ladder_the_run_names(tmp_path: Path) -> None:
    paths = both_ladder_paths(tmp_path)

    high = assemble(paths, HIGH)
    low = assemble(paths, LOW)

    assert high.cohort == HIGH
    assert {kind: high.row_counts[kind] for kind in KIND_SCHEMAS} == CONSUMED
    assert {kind: low.row_counts[kind] for kind in KIND_SCHEMAS} == LOW_CONSUMED


def test_a_two_ladder_scope_counts_no_rows_until_the_run_names_a_ladder(tmp_path: Path) -> None:
    paths = both_ladder_paths(tmp_path)

    with pytest.raises(ValueError, match="names no cohort"):
        assemble(paths)


def test_a_partition_outside_the_scopes_city_set_is_not_counted(paths: dict[str, Path]) -> None:
    before = assemble(paths).row_counts
    write_partition(paths["artifacts"], DISCOVERY_DAY, 2, [touch_row(9, NOON)], series="KXHIGHLAX")

    after = assemble(paths).row_counts

    assert "KXHIGHLAX" in load_run_scope(paths["run_scope"]).universe.passing
    assert after == before


def test_a_short_read_rtt_sample_set_aborts_naming_the_latency_floor(
    paths: dict[str, Path],
) -> None:
    write_rtt_samples(paths["rtt_samples"], SHORT_SAMPLES)

    with pytest.raises(ManifestIncomplete) as excinfo:
        assemble(paths)

    assert excinfo.value.fields == ("latency_floor",)
    assert f"usable samples {SHORT_SAMPLES} short of" in str(excinfo.value)


def test_the_universe_the_manifest_records_is_the_frozen_one(paths: dict[str, Path]) -> None:
    stored = json.loads((paths["run_scope"] / "r0_universe.json").read_text())["sha256"]

    inputs = assemble(paths)

    assert inputs.universe.fraction_invalid_max == FRACTION_INVALID_MAX
    assert freeze_digest(universe_payload(inputs.universe)) == stored
    assert manifest_payload(build_manifest(inputs))["r0_universe_sha256"] == stored


def test_the_assembler_states_no_default_bar_or_regime(paths: dict[str, Path]) -> None:
    parameters = inspect.signature(assemble_run_inputs).parameters

    for name in (*STATED_BAR, *STATED_REGIME):
        assert parameters[name].default is inspect.Parameter.empty
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="economic_bar_size"):
        assemble_run_inputs(
            run_id=RUN_ID,
            floor_source=FloorSource.SIGNED_READ,
            maker_rate=PUBLISHED_MAKER_RATE,
            maker_rate_source=MAKER_RATE_SOURCE,
            bootstrap_seed=SEED,
            **paths,
        )
    with pytest.raises(TypeError, match="maker_rate"):
        assemble_run_inputs(
            run_id=RUN_ID,
            floor_source=FloorSource.SIGNED_READ,
            economic_bar_size=SELF_CHARGED_BAR,
            economic_bar_price=SELF_CHARGED_BAR,
            economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
            bootstrap_seed=SEED,
            **paths,
        )


def test_an_ungated_study_states_a_zero_bar_and_still_writes_its_manifest(
    tmp_path: Path, paths: dict[str, Path]
) -> None:
    root = tmp_path / "tape_studies"

    write_manifest(root, assemble(paths))

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert payload["economic_bar_size"] == "0"
    assert payload["economic_bar_price"] == "0"
    assert payload["economic_bar_price_source"] == SELF_CHARGED_BAR_SOURCE
    assert payload["economic_bar_cents_per_contract"] == "0"


@pytest.mark.parametrize("module", [ladder_run, lock_convergence, taker_flow_run])
def test_every_gated_family_states_the_bar_and_regime_it_is_read_under(module: ModuleType) -> None:
    bar = economic_bar_cents_per_contract(module.ECONOMIC_BAR_SIZE, module.ECONOMIC_BAR_PRICE)

    assert module.ECONOMIC_BAR_SIZE == Decimal("26")
    assert module.ECONOMIC_BAR_PRICE == Decimal("0.50")
    assert module.ECONOMIC_BAR_PRICE_SOURCE == "preregistration"
    assert bar.quantize(Decimal("0.0001")) == Decimal("2.7692")
    assert module.MAKER_RATE == PUBLISHED_MAKER_RATE
    assert module.MAKER_RATE_SOURCE == MAKER_RATE_SOURCE


@pytest.mark.parametrize("module", [ladder_run, lock_convergence, taker_flow_run])
def test_a_gated_family_will_not_run_without_the_bar_and_regime_it_is_read_under(
    module: ModuleType,
) -> None:
    parameters = inspect.signature(module.execute).parameters

    for name in (*STATED_BAR, *STATED_REGIME):
        assert parameters[name].default is inspect.Parameter.empty
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_every_known_family_script_is_discovered() -> None:
    assert FAMILY_SCRIPTS, f"no scripts under {SCRIPTS} reference run_manifest"
    assert set(FAMILY_SCRIPTS).issuperset(KNOWN_FAMILY_SCRIPTS), FAMILY_SCRIPTS


@pytest.mark.parametrize("name", FAMILY_SCRIPTS)
def test_no_family_script_takes_the_bar_or_the_regime_off_the_command_line(name: str) -> None:
    flags = argument_flags((SCRIPTS / name).read_text())

    assert flags
    assert flags.isdisjoint({*STATED_BAR, DERIVED_BAR, *STATED_REGIME})
    assert [flag for flag in flags if "bar" in flag] == []
    assert [flag for flag in flags if "rate" in flag] == []
