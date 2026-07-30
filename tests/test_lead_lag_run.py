import json
from collections.abc import Iterator, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag import lead_lag_run
from bot.lag.lead_lag import CITY_SERIES, CORRIDORS, PAIRS, Episode
from bot.lag.lead_lag_run import (
    ATM_COLUMNS,
    CI_LEVEL,
    CORRIDOR_DAY_MIN_REPORT,
    FORWARD,
    REVERSE,
    Sweep,
    execute,
    median_interval,
    read_city_day,
    readout,
    result_payload,
    sign_observations,
    sweep_pairs,
)
from bot.lag.r0_universe import Coverage, freeze_universe, write_universe
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import MANIFEST_NAME
from bot.lag.tape_stats import BLOCK_DAYS, CorridorDayAggregate
from bot.lag.tape_studies import RunScope, load_run_scope
from bot.replay.artifacts import TOUCH_SCHEMA
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    QUIET_BAND,
    RESUBSCRIBE_BLIND,
    Split,
    write_split,
)
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    event_day_row,
    exclusion_row,
    seeded_repo,
    write_partition,
    write_preregistration,
    write_rtt_samples,
)


UTC = timezone.utc
MICROSECOND = timedelta(microseconds=1)
MICROS_PER_S = 1_000_000

DEN = "KXHIGHDEN"
OKC = "KXHIGHTOKC"
PAIR_ROOTS = (DEN, OKC)
RECORDED = tuple(sorted(set(CITY_SERIES.values()) | {"KXHIGHMIA", "KXHIGHTSEA"}))

DISCOVERY_DAY = date(2026, 7, 18)
HOLDOUT_DAY = date(2026, 7, 19)
SCOPE_START = datetime(2026, 7, 18, tzinfo=UTC)
SCOPE_END = datetime(2026, 7, 20, tzinfo=UTC)

QUIET_START = datetime(2026, 7, 18, 7, tzinfo=UTC)
QUIET_END = datetime(2026, 7, 18, 9, tzinfo=UTC)
BLINK = datetime(2026, 7, 18, 12, 8, tzinfo=UTC)

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

RUN_ID = "2026-08-13-q2"
SEED = 20260813
FIXTURE_RESAMPLES = 399
LEAD_S = Decimal("120")
LEAD_GRID = ("60", "90", "120", "150", "180")

UPSTREAM_POINTS = ((0, "0.50"), (300, "0.55"), (540, "0.60"))
DOWNSTREAM_POINTS = ((240, "0.50"), (420, "0.55"))
DEAD_POINTS = ((240, None), (420, None))

FLOOR_DAYS = (date(2026, 7, 18), date(2026, 7, 19), date(2026, 7, 20))
FLOOR_HOLDOUT_DAY = date(2026, 7, 21)
FLOOR_SCOPE_END = datetime(2026, 7, 22, tzinfo=UTC)
FIRST_PAIRS = {
    corridor: next(pair for pair in PAIRS if pair.corridor == corridor) for corridor in CORRIDORS
}
FLOOR_ROOTS = tuple(
    sorted(
        CITY_SERIES[city]
        for pair in FIRST_PAIRS.values()
        for city in (pair.upstream, pair.downstream)
    )
)
NINE_DAYS = tuple(DISCOVERY_DAY + timedelta(days=index) for index in range(9))


def stamp(event_date: date) -> str:
    return f"{event_date.year % 100:02d}{MONTHS[event_date.month - 1]}{event_date.day:02d}"


def leg(root: str, event_date: date) -> str:
    return f"{root}-{stamp(event_date)}-B70.5"


def noon(event_date: date) -> datetime:
    return datetime(event_date.year, event_date.month, event_date.day, 12, tzinfo=UTC)


def touch(row_id: int, ticker: str, at: datetime, dollars: str | None) -> dict:
    yes_bid = Decimal("0") if dollars is None else Decimal(dollars) - Decimal("0.01")
    no_bid = Decimal("0") if dollars is None else Decimal("0.99") - Decimal(dollars)
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at,
        "ts_ms": None,
        "yes_bid": str(yes_bid),
        "yes_bid_depth": "50",
        "yes_ask": str(Decimal("1") - no_bid),
        "yes_ask_depth": "50",
        "no_bid": str(no_bid),
        "no_bid_depth": "50",
        "no_ask": str(Decimal("1") - yes_bid),
        "no_ask_depth": "50",
    }


def city_rows(
    root: str, event_date: date, first_id: int, points: Sequence[tuple[int, str | None]]
) -> list[dict]:
    ticker = leg(root, event_date)
    return [
        touch(first_id + index, ticker, noon(event_date) + timedelta(seconds=offset), dollars)
        for index, (offset, dollars) in enumerate(points)
    ]


def days_table(
    roots: Sequence[str],
    *,
    discovery: Sequence[date] = (DISCOVERY_DAY,),
    holdout: Sequence[date] = (HOLDOUT_DAY,),
    opens: dict[str, timedelta] | None = None,
) -> pa.Table:
    offsets = {} if opens is None else opens
    frozen = [(day, DISCOVERY) for day in discovery] + [(day, HOLDOUT) for day in holdout]
    rows = [
        event_day_row(
            day,
            in_scope=True,
            split=split,
            day_index=index + 1,
            opens=offsets.get(root, timedelta()),
        )
        | {"series": root}
        for root in roots
        for index, (day, split) in enumerate(frozen)
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def exclusion_table(*, blink: bool = False) -> pa.Table:
    rows = [exclusion_row(0, QUIET_BAND, QUIET_START, QUIET_END)]
    if blink:
        rows.append(exclusion_row(1, RESUBSCRIBE_BLIND, BLINK, BLINK + MICROSECOND))
    return pa.Table.from_pylist(rows, schema=EXCLUSIONS_SCHEMA)


def scope_dir(
    tmp_path: Path,
    *,
    name: str = "scope",
    roots: Sequence[str] = PAIR_ROOTS,
    cities: Sequence[str] | None = None,
    days: pa.Table | None = None,
    exclusions: pa.Table | None = None,
    discovery: Sequence[date] = (DISCOVERY_DAY,),
    holdout: Sequence[date] = (HOLDOUT_DAY,),
    scope_end: datetime = SCOPE_END,
) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    pq.write_table(
        exclusion_table() if exclusions is None else exclusions, directory / "exclusions.parquet"
    )
    pq.write_table(
        days_table(roots, discovery=discovery, holdout=holdout) if days is None else days,
        directory / "event_days.parquet",
    )
    write_split(
        directory / "split.json",
        Split(
            cities=tuple(roots),
            discovery_days=tuple(discovery),
            holdout_days=tuple(holdout),
            boundary_event_day=holdout[0],
            scope_start=SCOPE_START,
            scope_end=scope_end,
        ),
    )
    write_universe(
        directory / "r0_universe.json",
        freeze_universe(
            fraction_invalid_max=Decimal("0.4"),
            passing=tuple(roots),
            coverage=Coverage(
                cities=tuple(roots if cities is None else cities),
                ladder_widths=(6,),
                in_scope_city_days=len(roots) * (len(discovery) + len(holdout)),
            ),
        ),
    )
    return directory


def artifacts_dir(
    tmp_path: Path,
    *,
    name: str = "artifacts",
    day: date = DISCOVERY_DAY,
    downstream: Sequence[tuple[int, str | None]] = DOWNSTREAM_POINTS,
) -> Path:
    root = tmp_path / name
    write_partition(root, day, 1, city_rows(DEN, day, 1, UPSTREAM_POINTS), series=DEN)
    write_partition(root, day, 1, city_rows(OKC, day, 100, downstream), series=OKC)
    return root


def floor_scope_dir(tmp_path: Path, *, name: str = "floor_scope") -> Path:
    return scope_dir(
        tmp_path,
        name=name,
        roots=FLOOR_ROOTS,
        discovery=FLOOR_DAYS,
        holdout=(FLOOR_HOLDOUT_DAY,),
        scope_end=FLOOR_SCOPE_END,
    )


def floor_artifacts(tmp_path: Path, *, name: str = "floor_artifacts") -> Path:
    root = tmp_path / name
    row_id = 1
    for day in FLOOR_DAYS:
        for pair in FIRST_PAIRS.values():
            for city, points in (
                (pair.upstream, UPSTREAM_POINTS),
                (pair.downstream, DOWNSTREAM_POINTS),
            ):
                series = CITY_SERIES[city]
                write_partition(root, day, 1, city_rows(series, day, row_id, points), series=series)
                row_id += len(points)
    return root


def run_paths(tmp_path: Path, samples: int = ADEQUATE_SAMPLES) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", samples),
    }


def episode_at(pair_key: tuple[str, str, str], event_date: date, lead_s: str) -> Episode:
    corridor, upstream, downstream = pair_key
    anchor = noon(event_date)
    leader_cross = anchor + timedelta(seconds=60)
    return Episode(
        corridor=corridor,
        upstream=upstream,
        downstream=downstream,
        leader=upstream,
        follower=downstream,
        event_date=event_date,
        direction=1,
        anchor=anchor,
        leader_cross=leader_cross,
        follower_cross=leader_cross + timedelta(microseconds=int(Decimal(lead_s) * MICROS_PER_S)),
        lead_s=Decimal(lead_s),
        leader_move_cents=Decimal("5"),
        follower_move_cents=Decimal("5"),
    )


def panel(
    corridors: Sequence[str], days: Sequence[date], leads: Sequence[str] = ("120",)
) -> list[Episode]:
    out: list[Episode] = []
    for corridor in corridors:
        pair = FIRST_PAIRS[corridor]
        for day in days:
            out.append(
                episode_at(
                    (corridor, pair.upstream, pair.downstream), day, leads[len(out) % len(leads)]
                )
            )
    return out


def payload_keys(payload: dict) -> Iterator[str]:
    for name, value in payload.items():
        yield name
        if isinstance(value, dict):
            yield from payload_keys(value)


@pytest.fixture
def scope(tmp_path: Path) -> RunScope:
    return load_run_scope(scope_dir(tmp_path))


@pytest.fixture
def swept(tmp_path: Path, scope: RunScope) -> Sweep:
    return sweep_pairs(scope, artifacts_dir(tmp_path))


def test_a_planted_pair_reads_one_episode_each_way(swept: Sweep) -> None:
    forward = swept.episodes[FORWARD]
    reverse = swept.episodes[REVERSE]

    assert len(forward) == len(reverse) == 1
    assert forward[0].leader == "DEN"
    assert forward[0].follower == "OKC"
    assert forward[0].corridor == "gulf"
    assert forward[0].lead_s == LEAD_S
    assert reverse[0].leader == "OKC"
    assert reverse[0].follower == "DEN"
    assert reverse[0].lead_s == LEAD_S
    assert swept.offered == 2
    assert swept.kept == 2
    assert swept.rows == len(UPSTREAM_POINTS) + len(DOWNSTREAM_POINTS)
    assert swept.in_scope == 2
    assert swept.silent == ()
    assert swept.tickers == {
        (DEN, DISCOVERY_DAY): leg(DEN, DISCOVERY_DAY),
        (OKC, DISCOVERY_DAY): leg(OKC, DISCOVERY_DAY),
    }


def test_every_episode_offers_one_window_per_city(swept: Sweep) -> None:
    assert swept.screened.candidates == 2 * swept.offered
    assert swept.screened.excluded == 0
    assert swept.screened.out_of_window == 0
    assert swept.screened.out_of_scope == 0
    assert swept.screened.excluded_fraction == Decimal("0")


def test_read_city_day_drops_a_leg_of_another_event_date_in_the_same_partition(
    tmp_path: Path, scope: RunScope
) -> None:
    root = tmp_path / "shared"
    rows = city_rows(DEN, DISCOVERY_DAY, 1, UPSTREAM_POINTS) + city_rows(
        DEN, HOLDOUT_DAY, 10, UPSTREAM_POINTS
    )
    path = write_partition(root, DISCOVERY_DAY, 1, rows, series=DEN)

    table = read_city_day(root, DEN, DISCOVERY_DAY, scope.event_days[(DEN, DISCOVERY_DAY)])

    assert pq.read_table(path).num_rows == len(rows)
    assert set(pq.read_table(path).column("ticker").to_pylist()) == {
        leg(DEN, DISCOVERY_DAY),
        leg(DEN, HOLDOUT_DAY),
    }
    assert table.column_names == list(ATM_COLUMNS)
    assert set(table.column("ticker").to_pylist()) == {leg(DEN, DISCOVERY_DAY)}
    assert table.num_rows == len(UPSTREAM_POINTS)


def test_an_episode_over_an_exclusion_is_dropped_and_charged_to_its_class(
    tmp_path: Path,
) -> None:
    scope = load_run_scope(
        scope_dir(tmp_path, exclusions=exclusion_table(blink=True), name="blink_scope")
    )

    swept = sweep_pairs(scope, artifacts_dir(tmp_path))

    assert swept.offered == 2
    assert swept.kept == 1
    assert swept.episodes[FORWARD][0].lead_s == LEAD_S
    assert swept.episodes[REVERSE] == ()
    assert swept.screened.excluded == 2
    assert swept.screened.by_class[RESUBSCRIBE_BLIND] == 2
    assert swept.screened.by_class[QUIET_BAND] == 0


def test_an_episode_leaving_the_follower_window_alone_is_dropped(tmp_path: Path) -> None:
    scope = load_run_scope(
        scope_dir(
            tmp_path,
            name="skewed_scope",
            days=days_table(PAIR_ROOTS, opens={OKC: timedelta(hours=12, minutes=2)}),
        )
    )

    swept = sweep_pairs(scope, artifacts_dir(tmp_path))
    forward = swept.episodes[FORWARD]
    reverse = swept.episodes[REVERSE]

    den = scope.event_days[(DEN, DISCOVERY_DAY)]
    okc = scope.event_days[(OKC, DISCOVERY_DAY)]
    assert den.window_start < okc.window_start
    assert swept.offered == 2
    assert forward == ()
    assert len(reverse) == 1
    assert swept.screened.out_of_window == 1
    assert swept.screened.excluded == 0


def test_a_city_with_no_two_sided_quote_is_recorded_and_the_sweep_runs_on(
    tmp_path: Path, scope: RunScope
) -> None:
    swept = sweep_pairs(scope, artifacts_dir(tmp_path, name="dead", downstream=DEAD_POINTS))

    assert swept.silent == ((OKC, DISCOVERY_DAY),)
    assert swept.tickers == {(DEN, DISCOVERY_DAY): leg(DEN, DISCOVERY_DAY)}
    assert swept.in_scope == 2
    assert swept.offered == 0
    assert swept.episodes == {FORWARD: (), REVERSE: ()}


def test_a_series_set_disagreeing_with_the_recorded_universe_raises(tmp_path: Path) -> None:
    scope = load_run_scope(scope_dir(tmp_path, cities=(DEN, OKC, "KXHIGHLAX")))

    with pytest.raises(ValueError, match="KXHIGHLAX"):
        sweep_pairs(scope, artifacts_dir(tmp_path))


def test_an_episode_rich_holdout_day_contributes_nothing(tmp_path: Path, scope: RunScope) -> None:
    swept = sweep_pairs(scope, artifacts_dir(tmp_path, name="holdout_only", day=HOLDOUT_DAY))

    assert HOLDOUT_DAY in scope.holdout_days
    assert swept.rows == 0
    assert swept.offered == 0
    assert swept.in_scope == 2
    assert swept.episodes == {FORWARD: (), REVERSE: ()}


def test_a_touch_partition_carrying_another_schema_is_refused(
    tmp_path: Path, scope: RunScope
) -> None:
    root = tmp_path / "foreign"
    path = write_partition(
        root,
        DISCOVERY_DAY,
        1,
        [],
        schema=pa.schema([("id", pa.int64()), ("ticker", pa.string())]),
        series=DEN,
    )

    with pytest.raises(ValueError, match=path.name):
        sweep_pairs(scope, root)

    assert TOUCH_SCHEMA.names != ["id", "ticker"]


def test_sign_observations_aggregate_one_row_per_corridor_day() -> None:
    leads = np.array([60, 120, 180, 240], dtype=np.int64) * MICROS_PER_S
    groups = np.array([0, 0, 0, 1], dtype=np.int64)
    keys = (("gulf", DISCOVERY_DAY), ("northeast", DISCOVERY_DAY))

    observations = sign_observations(leads, groups, keys, Decimal("120"))

    assert observations == [
        CorridorDayAggregate("gulf", DISCOVERY_DAY, Decimal("0"), Decimal("3")),
        CorridorDayAggregate("northeast", DISCOVERY_DAY, Decimal("1"), Decimal("1")),
    ]


def test_a_lead_equal_to_theta_contributes_nothing_to_the_total() -> None:
    leads = np.array([120, 120], dtype=np.int64) * MICROS_PER_S
    groups = np.array([0, 0], dtype=np.int64)
    keys = (("gulf", DISCOVERY_DAY),)

    observations = sign_observations(leads, groups, keys, Decimal("120"))

    assert observations == [CorridorDayAggregate("gulf", DISCOVERY_DAY, Decimal("0"), Decimal("2"))]


def test_the_interval_covers_a_planted_median() -> None:
    interval = median_interval(
        panel(tuple(CORRIDORS), NINE_DAYS, LEAD_GRID),
        ci_level=CI_LEVEL,
        resamples=FIXTURE_RESAMPLES,
        seed=SEED,
        block_days=BLOCK_DAYS,
    )

    assert interval.low <= LEAD_S <= interval.high
    assert interval.ci_level == CI_LEVEL
    assert interval.tail == (1 - CI_LEVEL) / 2
    assert interval.tested >= 2


def test_the_same_seed_walks_the_same_interval_twice() -> None:
    episodes = panel(tuple(CORRIDORS), NINE_DAYS, LEAD_GRID)
    kwargs = {
        "ci_level": CI_LEVEL,
        "resamples": FIXTURE_RESAMPLES,
        "seed": SEED,
        "block_days": BLOCK_DAYS,
    }

    assert median_interval(episodes, **kwargs) == median_interval(episodes, **kwargs)


def test_a_wider_ci_level_never_narrows_the_interval() -> None:
    episodes = panel(tuple(CORRIDORS), NINE_DAYS, LEAD_GRID)
    narrow = median_interval(
        episodes, ci_level=0.80, resamples=FIXTURE_RESAMPLES, seed=SEED, block_days=BLOCK_DAYS
    )
    wide = median_interval(
        episodes, ci_level=0.99, resamples=FIXTURE_RESAMPLES, seed=SEED, block_days=BLOCK_DAYS
    )

    assert wide.low <= narrow.low
    assert wide.high >= narrow.high


def test_nine_consecutive_days_across_four_corridors_carry_twelve_blocks() -> None:
    result = readout(panel(tuple(CORRIDORS), NINE_DAYS), reading=FORWARD, seed=SEED)

    assert result.corridor_days == len(CORRIDORS) * len(NINE_DAYS)
    assert result.n_blocks == len(CORRIDORS) * 3
    assert result.median_lead_s == LEAD_S


def test_under_the_reporting_floor_every_median_is_declined() -> None:
    episodes = panel(tuple(CORRIDORS), FLOOR_DAYS)[:-1]

    result = readout(episodes, reading=FORWARD, seed=SEED)

    assert result.corridor_days == CORRIDOR_DAY_MIN_REPORT - 1
    assert result.episodes == CORRIDOR_DAY_MIN_REPORT - 1
    assert result.n_blocks == len(CORRIDORS)
    assert result.median_lead_s is None
    assert result.interval is None
    assert [item.median_lead_s for item in result.per_pair.values()] == [None] * len(PAIRS)
    assert [item.median_lead_s for item in result.per_corridor.values()] == [None] * len(CORRIDORS)
    assert result.per_pair["DEN->OKC"].episodes == len(FLOOR_DAYS)
    assert result.per_corridor["gulf"].corridor_days == len(FLOOR_DAYS)
    assert len(result.episodes_per_corridor_day) == CORRIDOR_DAY_MIN_REPORT - 1


def test_at_the_reporting_floor_the_estimate_is_reported() -> None:
    result = readout(panel(tuple(CORRIDORS), FLOOR_DAYS), reading=REVERSE, seed=SEED)

    assert result.reading == REVERSE
    assert result.corridor_days == CORRIDOR_DAY_MIN_REPORT
    assert result.median_lead_s == LEAD_S
    assert result.interval.low == result.interval.high == LEAD_S
    assert result.per_pair["DEN->OKC"].median_lead_s == LEAD_S
    assert result.per_pair["OKC->DFW"].median_lead_s is None
    assert result.per_corridor["gulf"].median_lead_s == LEAD_S
    assert result.episodes_per_corridor_day[f"gulf {FLOOR_DAYS[0].isoformat()}"] == 1


def test_every_pair_and_corridor_appears_even_with_no_episode() -> None:
    result = readout([], reading=FORWARD, seed=SEED)

    assert len(result.per_pair) == len(PAIRS)
    assert len(result.per_corridor) == len(CORRIDORS)
    assert result.episodes == 0
    assert result.corridor_days == 0
    assert result.n_blocks == 0
    assert result.episodes_per_corridor_day == {}


def test_the_manifest_lands_before_any_statistic_is_read(tmp_path: Path) -> None:
    paths = run_paths(tmp_path)
    run_root = tmp_path / "tape_studies"
    root = tmp_path / "disordered"
    rows = city_rows(DEN, DISCOVERY_DAY, 1, UPSTREAM_POINTS)
    write_partition(
        root,
        DISCOVERY_DAY,
        1,
        [row | {"id": len(rows) - index} for index, row in enumerate(rows)],
        series=DEN,
    )
    write_partition(
        root, DISCOVERY_DAY, 1, city_rows(OKC, DISCOVERY_DAY, 100, DOWNSTREAM_POINTS), series=OKC
    )

    with pytest.raises(ValueError, match="id order"):
        execute(
            run_id=RUN_ID,
            artifacts=root,
            floor_source=FloorSource.SIGNED_READ,
            seed=SEED,
            run_root=run_root,
            **paths,
        )

    assert (run_root / RUN_ID / MANIFEST_NAME).exists()


def test_the_payload_round_trips_through_json_without_an_encoder(tmp_path: Path) -> None:
    paths = run_paths(tmp_path)
    run = execute(
        run_id=RUN_ID,
        artifacts=artifacts_dir(tmp_path),
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **paths,
    )

    payload = result_payload(run)

    assert json.loads(json.dumps(payload)) == payload
    assert payload["run_id"] == RUN_ID
    assert payload["split"] == DISCOVERY
    assert payload["ci_level"] == CI_LEVEL
    assert payload["block_days"] == BLOCK_DAYS
    assert Decimal(payload["move_bar_cents"]) == Decimal("5")
    assert Decimal(payload["round_trip_fee_cents_at_mid"]) == Decimal("4")
    assert payload["window_s"] == 600
    assert payload["corridor_day_min_report"] == CORRIDOR_DAY_MIN_REPORT
    assert payload["discovery_days"] == [DISCOVERY_DAY.isoformat()]
    assert payload["rows"] == len(UPSTREAM_POINTS) + len(DOWNSTREAM_POINTS)
    assert payload["atm_ticker_per_city_day"] == {
        f"{DEN} {DISCOVERY_DAY.isoformat()}": leg(DEN, DISCOVERY_DAY),
        f"{OKC} {DISCOVERY_DAY.isoformat()}": leg(OKC, DISCOVERY_DAY),
    }
    assert payload["episodes"] == {"offered": 2, "kept": 2, FORWARD: 1, REVERSE: 1}


def test_the_study_names_no_verdict_anywhere(tmp_path: Path) -> None:
    source = Path(lead_lag_run.__file__).read_text()
    paths = run_paths(tmp_path)
    run = execute(
        run_id=RUN_ID,
        artifacts=artifacts_dir(tmp_path),
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **paths,
    )

    for token in ("evaluate_gate", "evaluate_holdout", "PASS", "CLOSED", "verdict"):
        assert token not in source
    assert {"verdict", "gate", "holdout", "replication"}.isdisjoint(
        set(payload_keys(result_payload(run)))
    )


def test_the_pair_table_names_every_pair_and_the_roots_outside_them(tmp_path: Path) -> None:
    paths = run_paths(tmp_path) | {"run_scope": scope_dir(tmp_path, name="full", roots=RECORDED)}
    run = execute(
        run_id=RUN_ID,
        artifacts=artifacts_dir(tmp_path),
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **paths,
    )

    payload = result_payload(run)

    assert len(payload["pairs"]) == len(PAIRS)
    assert payload["pairs"]["DEN->OKC"] == {
        "corridor": "gulf",
        "upstream_series": DEN,
        "downstream_series": OKC,
    }
    assert payload["unpaired_roots"] == ["KXHIGHMIA", "KXHIGHTSEA"]
    assert payload["corridor_day_ceiling"] == len(CORRIDORS) * len(payload["discovery_days"])
    assert payload["city_event_days"] == len(CITY_SERIES)
    assert len(payload["no_atm_series"]) == len(CITY_SERIES) - 2


def test_a_full_corridor_sweep_reaches_the_reporting_floor(tmp_path: Path) -> None:
    scope = load_run_scope(floor_scope_dir(tmp_path))

    swept = sweep_pairs(scope, floor_artifacts(tmp_path))
    result = readout(swept.episodes[FORWARD], reading=FORWARD, seed=SEED)

    assert swept.offered == 2 * len(CORRIDORS) * len(FLOOR_DAYS)
    assert swept.kept == swept.offered
    assert result.corridor_days == CORRIDOR_DAY_MIN_REPORT
    assert result.median_lead_s == LEAD_S
    assert result.interval.low == LEAD_S
