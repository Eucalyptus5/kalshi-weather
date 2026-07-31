import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from bot.lag.depth_continuity import (
    ERA_START,
    SAMPLE_MAX,
    SNAPSHOT_ROWS,
    Continuity,
    RowTally,
    Tally,
    agreement,
    compare_leg,
    day_books,
    execute,
    read_snapshots,
    result_payload,
    sweep_continuity,
)
from bot.lag.depth_map_run import NO, YES
from bot.lag.r0_universe import Coverage, freeze_universe, write_universe
from bot.lag.read_rtt import FloorSource
from bot.lag.tape_studies import LADDER, load_run_scope
from bot.replay.artifacts import LADDER_SCHEMA
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    RESUBSCRIBE_BLIND,
    Split,
    write_split,
)
from bot.storage.sqlite import OrderbookSnapshot
from tests.test_depth_map_run import (
    EVENT_DATE,
    HOLDOUT_DAY,
    LEG_A,
    LEG_B,
    SCOPE_END,
    SERIES,
    WINDOW_END,
    WINDOW_START,
    exclusion_table,
    flat,
    ladder_artifacts,
    scope_dir,
    stamp,
)
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    event_day_row,
    seeded_repo,
    write_partition,
    write_preregistration,
    write_rtt_samples,
)


OTHER = "KXHIGHTOKC"
OTHER_LEG = "KXHIGHTOKC-26JUL18-B70.5"
RUN_ID = "2026-08-13-d3"
SEED = 20260813
DEEP = 500


def snapshot(
    ticker: str,
    at: datetime,
    *,
    yes_bid: str = "0.40",
    no_bid: str = "0.58",
    yes_depth: int | None = 5,
    no_depth: int | None = DEEP,
) -> dict:
    return {
        "ticker": ticker,
        "snapshot_at": at,
        "yes_bid": Decimal(yes_bid),
        "no_bid": Decimal(no_bid),
        "yes_bid_depth": yes_depth,
        "no_bid_depth": no_depth,
    }


def snapshots_db(tmp_path: Path, rows: Sequence[dict], *, name: str = "snapshots.db") -> Path:
    path = tmp_path / name
    engine = create_engine(f"sqlite:///{path}")
    OrderbookSnapshot.__table__.create(engine)
    with Session(engine) as session:
        for row in rows:
            session.add(
                OrderbookSnapshot(
                    ticker=row["ticker"],
                    snapshot_at=row["snapshot_at"],
                    yes_bid=row["yes_bid"],
                    no_bid=row["no_bid"],
                    yes_ask=Decimal("1") - row["no_bid"],
                    no_ask=Decimal("1") - row["yes_bid"],
                    yes_bid_depth=row["yes_bid_depth"],
                    no_bid_depth=row["no_bid_depth"],
                    yes_ask_depth=row["no_bid_depth"],
                    no_ask_depth=row["yes_bid_depth"],
                )
            )
        session.commit()
    engine.dispose()
    return path


def pair_artifacts(tmp_path: Path, legs: Mapping[str, Sequence[dict]], *, name: str) -> Path:
    root = tmp_path / name
    for series, rows in legs.items():
        write_partition(
            root, EVENT_DATE, 1, list(rows), kind=LADDER, schema=LADDER_SCHEMA, series=series
        )
    return root


def pair_scope_dir(tmp_path: Path, *, name: str = "pair_scope") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    pq.write_table(exclusion_table(), directory / "exclusions.parquet")
    rows = [
        event_day_row(EVENT_DATE, in_scope=True, split=DISCOVERY, day_index=1) | {"series": series}
        for series in (SERIES, OTHER)
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA), directory / "event_days.parquet"
    )
    write_split(
        directory / "split.json",
        Split(
            cities=(SERIES, OTHER),
            discovery_days=(EVENT_DATE,),
            holdout_days=(HOLDOUT_DAY,),
            boundary_event_day=HOLDOUT_DAY,
            scope_start=WINDOW_START,
            scope_end=SCOPE_END,
        ),
    )
    write_universe(
        directory / "r0_universe.json",
        freeze_universe(
            fraction_invalid_max=Decimal("0.4"),
            passing=(SERIES, OTHER),
            coverage=Coverage(cities=(SERIES, OTHER), ladder_widths=(6,), in_scope_city_days=2),
        ),
    )
    return directory


def swept(
    tmp_path: Path,
    rows: Sequence[dict],
    snapshots: Sequence[dict],
    *,
    name: str = "artifacts",
    scope: Path | None = None,
) -> Continuity:
    frozen = load_run_scope(scope_dir(tmp_path) if scope is None else scope)
    artifacts = ladder_artifacts(tmp_path, rows, name=name)
    return sweep_continuity(frozen, artifacts, snapshots_db(tmp_path, snapshots, name=f"{name}.db"))


def opened(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    return conn


def run_paths(tmp_path: Path, snapshots: Sequence[dict]) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
        "db": snapshots_db(tmp_path, snapshots),
    }


def payload_keys(payload: object) -> Iterator[str]:
    if isinstance(payload, dict):
        for name, value in payload.items():
            yield name
            yield from payload_keys(value)
    if isinstance(payload, list):
        for value in payload:
            yield from payload_keys(value)


AGREEING_ROWS = (flat(1, LEG_A, stamp(10), "5"),)
AGREEING_SNAPSHOTS = (snapshot(LEG_A, stamp(11)),)


def test_the_snapshot_seek_rides_the_ticker_snapshot_index(tmp_path: Path) -> None:
    conn = opened(snapshots_db(tmp_path, [snapshot(LEG_A, stamp(10))]))
    try:
        plan = conn.execute(f"EXPLAIN QUERY PLAN {SNAPSHOT_ROWS}", (LEG_A, "a", "b")).fetchall()
    finally:
        conn.close()

    assert len(plan) == 1
    assert plan[0][3].startswith(
        "SEARCH orderbook_snapshots USING INDEX ix_orderbook_snapshots_ticker_snapshot_at"
    )


def test_a_snapshot_between_two_states_reads_the_earlier_one(tmp_path: Path) -> None:
    rows = (flat(1, LEG_A, stamp(10), "5"), flat(2, LEG_A, stamp(12), "9"))

    swept_run = swept(tmp_path, rows, [snapshot(LEG_A, stamp(11), yes_depth=5)])

    tally = swept_run.tallies[SERIES]
    assert swept_run.rows.compared == 1
    assert tally.depth_agree == 1
    assert tally.differences[YES] == {0: 1}
    assert swept_run.disagreements == ()


def test_a_snapshot_before_the_first_ws_state_carries_no_coverage(tmp_path: Path) -> None:
    swept_run = swept(tmp_path, AGREEING_ROWS, [snapshot(LEG_A, stamp(9))])

    assert swept_run.rows.returned == 1
    assert swept_run.rows.no_coverage == 1
    assert swept_run.rows.compared == 0
    assert swept_run.tallies[SERIES].compared == 0


def test_a_snapshot_inside_an_excluded_interval_is_dropped_and_counted_by_class(
    tmp_path: Path,
) -> None:
    scope = scope_dir(
        tmp_path, name="blink", exclusions=exclusion_table([(stamp(10, 50), stamp(11, 10))])
    )

    swept_run = swept(tmp_path, AGREEING_ROWS, AGREEING_SNAPSHOTS, scope=scope)

    assert swept_run.rows.excluded == 1
    assert swept_run.rows.by_class[RESUBSCRIBE_BLIND] == 1
    assert swept_run.rows.compared == 0


def test_a_snapshot_outside_the_event_day_window_is_counted_separately(tmp_path: Path) -> None:
    scope = load_run_scope(scope_dir(tmp_path))
    day = scope.event_days[(SERIES, EVENT_DATE)]
    book = day_books(ladder_artifacts(tmp_path, AGREEING_ROWS), SERIES, EVENT_DATE, day)[LEG_A]
    conn = opened(snapshots_db(tmp_path, [snapshot(LEG_A, WINDOW_END + timedelta(hours=1))]))
    rows = RowTally()
    tally = Tally()

    try:
        found = read_snapshots(conn, LEG_A, WINDOW_START, SCOPE_END, rows=rows)
    finally:
        conn.close()
    compare_leg(scope, SERIES, EVENT_DATE, LEG_A, found, book, rows=rows, tally=tally)

    assert rows.returned == 1
    assert rows.out_of_window == 1
    assert rows.excluded == 0
    assert rows.compared == 0
    assert tally.compared == 0


def test_a_snapshot_before_the_era_boundary_stops_the_run(tmp_path: Path) -> None:
    stale = ERA_START - timedelta(seconds=1)
    conn = opened(snapshots_db(tmp_path, [snapshot(LEG_A, stale), snapshot(LEG_A, stamp(11))]))
    rows = RowTally()

    try:
        with pytest.raises(ValueError, match=LEG_A):
            read_snapshots(conn, LEG_A, ERA_START - timedelta(days=1), WINDOW_END, rows=rows)
    finally:
        conn.close()

    assert rows.returned == 2
    assert rows.era == 1


def test_a_null_depth_leaves_the_price_comparison_standing(tmp_path: Path) -> None:
    snapshots = [snapshot(LEG_A, stamp(11), yes_depth=None)]

    swept_run = swept(tmp_path, AGREEING_ROWS, snapshots)

    tally = swept_run.tallies[SERIES]
    assert swept_run.rows.null_depth == 1
    assert swept_run.rows.compared == 1
    assert tally.compared == 1
    assert tally.price_agree == 1
    assert tally.depth_compared == 0
    assert tally.differences == {}


def test_a_fractional_ws_depth_against_a_zero_rest_depth_matches_only_truncated(
    tmp_path: Path,
) -> None:
    rows = (flat(1, LEG_A, stamp(10), "0.39", "7"),)
    snapshots = [snapshot(LEG_A, stamp(11), yes_depth=0, no_depth=7)]

    swept_run = swept(tmp_path, rows, snapshots)

    tally = swept_run.tallies[SERIES]
    assert tally.depth_compared == 1
    assert tally.depth_agree == 0
    assert tally.depth_truncated == 1
    assert tally.fractional == 1
    assert tally.differences[YES] == {-39: 1}
    assert tally.within_one[YES] == 1
    assert swept_run.disagreements == ()


def test_a_snapshot_batch_is_read_as_its_final_book(tmp_path: Path) -> None:
    rows = (
        flat(1, LEG_A, stamp(10), "5"),
        flat(2, LEG_A, stamp(10), "20"),
        flat(3, LEG_A, stamp(10), "300"),
    )

    swept_run = swept(tmp_path, rows, [snapshot(LEG_A, stamp(11), yes_depth=300)])

    tally = swept_run.tallies[SERIES]
    assert tally.depth_agree == 1
    assert tally.differences[YES] == {0: 1}


def test_the_agreement_rates_split_by_city(tmp_path: Path) -> None:
    scope = load_run_scope(pair_scope_dir(tmp_path))
    artifacts = pair_artifacts(
        tmp_path,
        {SERIES: (flat(1, LEG_A, stamp(10), "5"),), OTHER: (flat(1, OTHER_LEG, stamp(10), "9"),)},
        name="pair",
    )
    snapshots = [snapshot(LEG_A, stamp(11)), snapshot(OTHER_LEG, stamp(11))]

    swept_run = sweep_continuity(scope, artifacts, snapshots_db(tmp_path, snapshots))

    pooled = agreement(swept_run.pooled())
    den = agreement(swept_run.tallies[SERIES])
    okc = agreement(swept_run.tallies[OTHER])
    assert swept_run.cities == (SERIES, OTHER)
    assert pooled.compared == 2
    assert pooled.price_fraction == Decimal(1)
    assert pooled.depth_fraction == Decimal("0.5")
    assert den.depth_fraction == Decimal(1)
    assert okc.depth_fraction == Decimal(0)
    assert den.sides[YES].p50 == Decimal(0)
    assert okc.sides[YES].p50 == Decimal(-4)
    assert okc.sides[YES].within_one_contract == 0
    assert okc.sides[NO].within_one_contract == 1
    assert [item.ticker for item in swept_run.disagreements] == [OTHER_LEG]


def test_the_disagreement_sample_is_capped_and_deterministic(tmp_path: Path) -> None:
    rows = (flat(1, LEG_A, stamp(9), "5"), flat(2, LEG_B, stamp(9), "5"))
    snapshots = [
        snapshot(leg, stamp(10) + timedelta(minutes=index), yes_depth=index + 100)
        for leg in (LEG_B, LEG_A)
        for index in range(SAMPLE_MAX)
    ]

    scope = scope_dir(tmp_path)
    first = swept(tmp_path, rows, snapshots, name="first", scope=scope)
    second = swept(tmp_path, rows, snapshots, name="second", scope=scope)

    sample = first.disagreements
    assert first.rows.compared == 2 * SAMPLE_MAX
    assert len(sample) == SAMPLE_MAX
    assert {item.ticker for item in sample} == {LEG_A}
    assert [item.snapshot_at for item in sample] == sorted(item.snapshot_at for item in sample)
    assert sample == second.disagreements
    assert sample[0].rest_yes_bid_depth == Decimal(100)
    assert sample[0].ws_yes_bid_depth == Decimal(5)


def test_the_row_accounting_closes(tmp_path: Path) -> None:
    scope = scope_dir(tmp_path, name="mixed", exclusions=exclusion_table([(stamp(13), stamp(14))]))
    rows = (flat(1, LEG_A, stamp(10), "5"), flat(2, LEG_B, stamp(10), "5"))
    snapshots = [
        snapshot(LEG_A, stamp(9)),
        snapshot(LEG_A, stamp(11)),
        snapshot(LEG_A, stamp(13, 30)),
        snapshot(LEG_A, stamp(15), yes_depth=None),
    ]

    accounting = swept(tmp_path, rows, snapshots, scope=scope).rows

    assert accounting.returned == 4
    assert accounting.era == 0
    assert accounting.out_of_window == 0
    assert accounting.excluded == 1
    assert accounting.no_coverage == 1
    assert accounting.null_depth == 1
    assert accounting.compared == 2
    assert accounting.compared == accounting.returned - (
        accounting.era + accounting.out_of_window + accounting.excluded + accounting.no_coverage
    )


def test_a_complete_run_reports_the_pooled_rates(tmp_path: Path) -> None:
    run = execute(
        run_id=RUN_ID,
        artifacts=ladder_artifacts(tmp_path, AGREEING_ROWS),
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **run_paths(tmp_path, AGREEING_SNAPSHOTS),
    )

    payload = result_payload(run)

    assert payload["run_id"] == RUN_ID
    assert payload["cities"] == [SERIES]
    assert payload["era_start"] == ERA_START.isoformat()
    assert payload["rows"]["compared"] == 1
    assert payload["pooled"]["price_agree_fraction"] == "1"
    assert payload["pooled"]["depth_agree_fraction"] == "1"
    assert payload["by_city"][SERIES]["depth_agree_truncated_fraction"] == "1"
    assert payload["disagreements"] == []
    assert json.loads(json.dumps(payload)) == payload


def test_the_reading_carries_no_inference_field(tmp_path: Path) -> None:
    run = execute(
        run_id=RUN_ID,
        artifacts=ladder_artifacts(tmp_path, AGREEING_ROWS),
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **run_paths(tmp_path, AGREEING_SNAPSHOTS),
    )

    names = set(payload_keys(result_payload(run)))

    assert names.isdisjoint({"verdict", "interval", "ci_level", "alpha", "resamples"})
    assert not any("bootstrap" in name or "p_value" in name for name in names)
