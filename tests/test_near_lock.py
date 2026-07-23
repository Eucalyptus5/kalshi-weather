import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.near_lock import (
    LOCK_HALF_WIDTH_S,
    PRINT_MIN_STRATUM,
    REPORTED,
    NearLockRun,
    read_observations,
    reading_payload,
    result_payload,
    scan_locks,
)
from bot.lag.r0_universe import Coverage, freeze_universe, write_universe
from bot.lag.taker_flow import HORIZONS_S, PRIMARY_HORIZON_S
from bot.lag.taker_flow_run import UNDERPOWERED, readout, sweep_prints
from bot.lag.tape_studies import RunScope, load_run_scope
from bot.replay.artifacts import TRADES_SCHEMA
from bot.replay.run_scope import DISCOVERY, EVENT_DAYS_SCHEMA, HOLDOUT, Split, write_split
from tests.test_taker_flow_run import (
    DAY_TICKER,
    DISCOVERY_DAY,
    HOLDOUT_DAY,
    NEXT_TICKER,
    OPENS,
    SCOPE_END,
    SCOPE_START,
    SEED,
    SERIES,
    TOUCH_DISCOVERY_DAY,
    TRADES_DISCOVERY_DAY,
    artifacts_dir,
    clusters_of,
    exclusion_table,
    readout_of,
    scope_dir,
    trade,
    when,
)
from tests.test_tape_studies import event_day_row, write_partition


STATION = "KDEN"
MIA = "KXHIGHMIA"
LAX = "KXHIGHLAX"
MIA_TICKER = "KXHIGHMIA-26JUL18-T70"
LAX_TICKER = "KXHIGHLAX-26JUL18-T70"
WIDE_CITIES = (
    (SERIES, STATION, "America/Denver"),
    (MIA, "KMIA", "America/New_York"),
    (LAX, "KLAX", "America/Los_Angeles"),
)

HALF_WIDTH = timedelta(seconds=LOCK_HALF_WIDTH_S)
CROSSING_VALID = when(DISCOVERY_DAY, 17, 0, 0)
CROSSING_PUBLISHED = when(DISCOVERY_DAY, 18, 0, 0)
NEXT_VALID = when(HOLDOUT_DAY, 17, 0, 0)
NEXT_PUBLISHED = when(HOLDOUT_DAY, 18, 0, 0)

BOUNDARY_TRADES = [
    trade(150, DAY_TICKER, CROSSING_PUBLISHED + HALF_WIDTH, 18500),
    trade(151, DAY_TICKER, CROSSING_PUBLISHED + HALF_WIDTH + timedelta(seconds=1), 18600),
]


def observation(station: str, valid: datetime, published: datetime, temp_f: str) -> dict:
    return {
        "station": station,
        "source": "metar",
        "obs_time": valid.isoformat(),
        "tmpf": temp_f,
        "received_at": published.isoformat(),
    }


def write_observations(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def crossing_observations(tmp_path: Path) -> Path:
    return write_observations(
        tmp_path / "crossing.jsonl",
        [
            observation(
                STATION, when(DISCOVERY_DAY, 12, 0, 0), when(DISCOVERY_DAY, 12, 1, 0), "60"
            ),
            observation(STATION, CROSSING_VALID, CROSSING_PUBLISHED, "72"),
            observation(STATION, NEXT_VALID, NEXT_PUBLISHED, "72"),
        ],
    )


def cool_observations(tmp_path: Path) -> Path:
    return write_observations(
        tmp_path / "cool.jsonl",
        [
            observation(
                STATION, when(DISCOVERY_DAY, 12, 0, 0), when(DISCOVERY_DAY, 12, 1, 0), "60"
            ),
            observation(STATION, CROSSING_VALID, CROSSING_PUBLISHED, "65"),
            observation(STATION, NEXT_VALID, NEXT_PUBLISHED, "65"),
        ],
    )


def ambiguous_observations(tmp_path: Path) -> Path:
    return write_observations(
        tmp_path / "ambiguous.jsonl",
        [observation(STATION, CROSSING_VALID, CROSSING_PUBLISHED, "70")],
    )


def wide_observations(tmp_path: Path) -> Path:
    return write_observations(
        tmp_path / "wide.jsonl",
        [
            observation(station, CROSSING_VALID, CROSSING_PUBLISHED, "72")
            for _, station, _ in WIDE_CITIES
        ],
    )


def day_row(series: str, station: str, zone: str, event_date: date, split: str, index: int) -> dict:
    return event_day_row(event_date, in_scope=True, split=split, day_index=index, opens=OPENS) | {
        "series": series,
        "station": station,
        "timezone": zone,
    }


def wide_scope_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "wide_scope"
    directory.mkdir()
    pq.write_table(exclusion_table(), directory / "exclusions.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                day_row(series, station, zone, event_date, split, index)
                for series, station, zone in WIDE_CITIES
                for index, (event_date, split) in enumerate(
                    ((DISCOVERY_DAY, DISCOVERY), (HOLDOUT_DAY, HOLDOUT)), start=1
                )
            ],
            schema=EVENT_DAYS_SCHEMA,
        ),
        directory / "event_days.parquet",
    )
    cities = tuple(series for series, _, _ in WIDE_CITIES)
    write_split(
        directory / "split.json",
        Split(
            cities=cities,
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
            passing=(SERIES, MIA),
            coverage=Coverage(cities=cities, ladder_widths=(6,), in_scope_city_days=6),
        ),
    )
    return directory


def wide_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "wide_artifacts"
    write_partition(root, DISCOVERY_DAY, 1, TOUCH_DISCOVERY_DAY)
    write_partition(
        root, DISCOVERY_DAY, 1, TRADES_DISCOVERY_DAY, kind="trades", schema=TRADES_SCHEMA
    )
    for series, ticker in ((MIA, MIA_TICKER), (LAX, LAX_TICKER)):
        write_partition(
            root,
            DISCOVERY_DAY,
            1,
            [trade(300, ticker, when(DISCOVERY_DAY, 18, 0, 1), 12500)],
            kind="trades",
            schema=TRADES_SCHEMA,
            series=series,
        )
    return root


def boundary_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "boundary"
    write_partition(root, DISCOVERY_DAY, 1, TOUCH_DISCOVERY_DAY)
    write_partition(root, DISCOVERY_DAY, 1, BOUNDARY_TRADES, kind="trades", schema=TRADES_SCHEMA)
    return root


@pytest.fixture
def scope(tmp_path: Path) -> RunScope:
    return load_run_scope(scope_dir(tmp_path))


def test_the_reader_builds_observations_the_detector_can_read(tmp_path: Path) -> None:
    path = crossing_observations(tmp_path)

    grouped = read_observations(path)

    assert sorted(grouped) == [STATION]
    first = grouped[STATION][1]
    assert first.valid_time == CROSSING_VALID
    assert first.publication_time == CROSSING_PUBLISHED
    assert first.temp_f == Decimal("72")
    assert type(first.temp_f) is Decimal
    assert first.source == "metar"
    assert first.is_special is False
    assert first.raw == ""


def test_a_crossing_centres_its_window_on_when_the_observation_was_published(
    tmp_path: Path, scope: RunScope
) -> None:
    found = scan_locks(
        scope, artifacts_dir(tmp_path), read_observations(crossing_observations(tmp_path))
    )

    assert found.windows[DAY_TICKER] == (
        CROSSING_PUBLISHED - HALF_WIDTH,
        CROSSING_PUBLISHED + HALF_WIDTH,
    )
    assert found.windows[DAY_TICKER] != (CROSSING_VALID - HALF_WIDTH, CROSSING_VALID + HALF_WIDTH)
    assert found.locked == 2
    assert found.ambiguous == 0
    assert found.no_lock == 0


def test_a_market_that_never_locks_carries_no_window_and_no_prints(
    tmp_path: Path, scope: RunScope
) -> None:
    root = artifacts_dir(tmp_path)
    found = scan_locks(scope, root, read_observations(cool_observations(tmp_path)))

    sweep = sweep_prints(scope, root, lock_windows=found.windows)

    assert found.windows == {}
    assert found.no_lock == 2
    assert found.markets == 2
    assert sweep.no_lock_prints == 6
    assert sweep.off_universe_prints == 0
    assert sweep.outside_lock_window == 0
    assert sweep.in_scope == {DISCOVERY: 0, HOLDOUT: 0}
    assert sweep.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].n_prints == 0


def test_an_ambiguous_lock_is_counted_apart_and_still_carries_its_window(
    tmp_path: Path, scope: RunScope
) -> None:
    found = scan_locks(
        scope, artifacts_dir(tmp_path), read_observations(ambiguous_observations(tmp_path))
    )

    assert found.ambiguous == 1
    assert found.locked == 1
    assert found.no_lock == 1
    assert DAY_TICKER in found.windows


def test_the_prints_the_sweep_reads_are_the_ones_inside_the_lock_window(
    tmp_path: Path, scope: RunScope
) -> None:
    root = artifacts_dir(tmp_path)
    found = scan_locks(scope, root, read_observations(crossing_observations(tmp_path)))

    sweep = sweep_prints(scope, root, lock_windows=found.windows)

    assert sweep.in_scope == {DISCOVERY: 1, HOLDOUT: 1}
    assert sweep.outside_lock_window == 4
    assert sweep.no_lock_prints == 0
    assert [item.cluster for item in sweep.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].clusters()] == [
        DAY_TICKER
    ]


def test_a_print_on_the_window_edge_is_read_and_the_next_second_is_not(
    tmp_path: Path, scope: RunScope
) -> None:
    root = boundary_artifacts(tmp_path)
    windows = {DAY_TICKER: (CROSSING_PUBLISHED - HALF_WIDTH, CROSSING_PUBLISHED + HALF_WIDTH)}

    sweep = sweep_prints(scope, root, lock_windows=windows)

    assert sweep.in_scope == {DISCOVERY: 1, HOLDOUT: 0}
    assert sweep.outside_lock_window == 1


def test_the_carved_out_city_and_a_series_off_the_passing_set_never_appear(
    tmp_path: Path,
) -> None:
    scope = load_run_scope(wide_scope_dir(tmp_path))
    root = wide_artifacts(tmp_path)

    found = scan_locks(scope, root, read_observations(wide_observations(tmp_path)))
    sweep = sweep_prints(scope, root, lock_windows=found.windows)
    payload = result_payload(
        NearLockRun(
            run_id="wide",
            manifest=tmp_path / "manifest.json",
            manifest_sha256="",
            seed=SEED,
            locks=found,
            sweep=sweep,
            discovery=readout(
                sweep.tallies[(DISCOVERY, PRIMARY_HORIZON_S)],
                split=DISCOVERY,
                horizon_s=PRIMARY_HORIZON_S,
                seed=SEED,
            ),
            holdout=readout(
                sweep.tallies[(HOLDOUT, PRIMARY_HORIZON_S)],
                split=HOLDOUT,
                horizon_s=PRIMARY_HORIZON_S,
                seed=SEED,
            ),
        )
    )

    assert MIA not in scope.universe.lock_dependent
    assert LAX not in scope.universe.lock_dependent
    assert sorted(found.windows) == [DAY_TICKER]
    assert found.markets == 1
    assert sweep.no_lock_prints == 0
    assert sweep.off_universe_prints == 2
    assert payload["cities"] == [SERIES]
    assert list(payload["tickers_per_city_day"]) == [f"{SERIES} {DISCOVERY_DAY.isoformat()}"]
    assert [item.cluster for item in sweep.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].clusters()] == [
        DAY_TICKER
    ]


def test_a_sweep_with_no_lock_windows_reads_every_series_the_scope_carries(
    tmp_path: Path,
) -> None:
    scope = load_run_scope(wide_scope_dir(tmp_path))

    sweep = sweep_prints(scope, wide_artifacts(tmp_path))

    assert sweep.off_universe_prints == 0
    assert sweep.no_lock_prints == 0
    assert sorted({series for series, _ in sweep.tickers}) == [SERIES, LAX, MIA]


def test_a_station_with_no_recorded_arrivals_locks_nothing_and_is_counted(
    tmp_path: Path, scope: RunScope
) -> None:
    elsewhere = write_observations(
        tmp_path / "elsewhere.jsonl",
        [observation("KBOI", CROSSING_VALID, CROSSING_PUBLISHED, "72")],
    )

    found = scan_locks(scope, artifacts_dir(tmp_path), read_observations(elsewhere))

    assert found.windows == {}
    assert found.markets == 2
    assert found.no_observations == 2
    assert found.no_lock == 0
    assert found.locked == 0


def test_the_gating_sweep_does_not_move_when_no_lock_windows_are_supplied(
    tmp_path: Path, scope: RunScope
) -> None:
    root = artifacts_dir(tmp_path)

    sweep = sweep_prints(scope, root)

    assert sweep == sweep_prints(scope, root, lock_windows=None)
    assert sweep.empty_side == 1
    assert sweep.duplicates == 1
    assert sweep.out_of_scope == 1
    assert sweep.in_scope == {DISCOVERY: 4, HOLDOUT: 2}
    assert sweep.read_ts_violations == 0
    assert sweep.fractional_size_prints == 0
    assert sweep.outside_lock_window == 0
    assert sweep.no_lock_prints == 0
    assert sweep.tickers == {
        (SERIES, DISCOVERY_DAY): frozenset({DAY_TICKER}),
        (SERIES, HOLDOUT_DAY): frozenset({NEXT_TICKER}),
    }
    assert sweep.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].candidates == 4
    assert sweep.tallies[(DISCOVERY, PRIMARY_HORIZON_S)].excluded == 1
    assert {
        horizon_s: readout(
            sweep.tallies[(DISCOVERY, horizon_s)],
            split=DISCOVERY,
            horizon_s=horizon_s,
            seed=SEED,
        ).bootstrap.estimate
        for horizon_s in HORIZONS_S
    } == {1: Decimal("-2.6"), 10: Decimal("-1.6"), 60: Decimal("1.4"), 300: Decimal("6.4")}


def test_a_reading_under_the_stratum_minimum_reports_underpowered_and_no_estimate() -> None:
    thin = readout_of(
        clusters_of(("a", "20", "10"), ("b", "20", "10")),
        split=DISCOVERY,
        n_prints=PRINT_MIN_STRATUM - 1,
    )

    payload = reading_payload(thin)

    assert payload["status"] == UNDERPOWERED
    assert payload["mean_net_cents"] is None
    assert payload["ci_low"] is None
    assert payload["p_value"] is None
    assert payload["n_prints"] == PRINT_MIN_STRATUM - 1
    assert payload["n_min"] == PRINT_MIN_STRATUM


def test_a_reading_at_the_stratum_minimum_carries_its_estimate_and_interval() -> None:
    thick = readout_of(
        clusters_of(("a", "20", "10"), ("b", "20", "10")),
        split=DISCOVERY,
        n_prints=PRINT_MIN_STRATUM,
    )

    payload = reading_payload(thick)

    assert payload["status"] == REPORTED
    assert Decimal(payload["mean_net_cents"]) == Decimal("2")
    assert payload["ci_low"] is not None
    assert payload["p_value"] is not None
