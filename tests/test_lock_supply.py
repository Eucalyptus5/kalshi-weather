from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest

from bot.backtest.normalize import CanonicalSnapshot
from bot.lag.lock_supply import (
    DISCOVERY_STATION_DAY_MIN,
    HOLDOUT_STATION_DAY_MIN,
    POOLED,
    read_ladders,
    read_observations,
    summarize,
    survey_lock_supply,
)
from bot.lag.observation_freeze import read_observation_index
from bot.lag.placement_grid import CloseSidecar, read_sidecar, write_sidecar
from bot.lag.settlement_straddle import event_ticker_of
from bot.lag.tape_studies import RunScope, load_run_scope
from bot.markets.observation_window import observation_window
from bot.observations.metar import StationObservation
from bot.replay.run_scope import DISCOVERY, EVENT_DAYS_SCHEMA, HOLDOUT
from tests.test_tape_studies import (
    DISCOVERY_DAY,
    HOLDOUT_DAY,
    UNCOVERED_DAY,
    event_day_row,
    scope_dir,
)


UTC = timezone.utc
SERIES = "KXLOWTDEN"
HIGH_SERIES = "KXHIGHDEN"
STRANGER = "KXRAINDEN"
STATION = "KDEN"
ZONE = "America/Denver"
ZERO = Decimal("0")

PAIR = ("T58", "T65")
RICH = ("T50", "B57.5", "B61.5", "T65")
TAIL = ("T50", "T65")

WINDOW_OPEN_TEMPS = ("62", "70")
MID_DAY_TEMPS = ("70", "62")
RICH_TEMPS = ("63", "60.5", "55")
TAIL_TEMPS = ("60", "49")
AMBIGUOUS_TEMPS = ("64.5",)

PUBLICATION_LAG = timedelta(minutes=12)

SPLITS = {DISCOVERY_DAY: DISCOVERY, HOLDOUT_DAY: HOLDOUT}

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCOPE = REPO_ROOT / "data" / "tape_studies" / "run_scope_v2"
FROZEN_CLOSES = REPO_ROOT / "data" / "tape_studies" / "f3_low_closes"
FROZEN_OBSERVATIONS = REPO_ROOT / "data" / "tape_studies" / "f2_observations"
FROZEN_ROOTS = 20
FROZEN_STATION_DAYS = 280
FROZEN_MARKETS = 1680

needs_tape = pytest.mark.skipif(
    not (FROZEN_SCOPE.exists() and FROZEN_CLOSES.exists() and FROZEN_OBSERVATIONS.exists()),
    reason="the recorded tape is not on this host",
)


def reading(stamp: datetime, temp_f: str) -> StationObservation:
    return StationObservation(
        station=STATION,
        valid_time=stamp,
        publication_time=stamp,
        temp_f=Decimal(temp_f),
        is_special=False,
        raw="",
        source="tape",
    )


def readings_for(event_date: date, temps: Sequence[str]) -> list[StationObservation]:
    start, _ = observation_window(ZONE, event_date)
    return [reading(start + timedelta(hours=index + 1), temp) for index, temp in enumerate(temps)]


def scope_path(tmp_path: Path, *, days: Sequence[date] = (DISCOVERY_DAY, HOLDOUT_DAY)) -> Path:
    rows = [
        event_day_row(
            event_date, in_scope=True, split=SPLITS[event_date], day_index=index + 1, series=SERIES
        )
        for index, event_date in enumerate(days)
    ]
    return scope_dir(tmp_path, days=pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA))


def scope_for(tmp_path: Path, *, days: Sequence[date] = (DISCOVERY_DAY, HOLDOUT_DAY)) -> RunScope:
    return load_run_scope(scope_path(tmp_path, days=days))


def snapshot(root: str, event_date: date, suffix: str) -> CanonicalSnapshot:
    event = event_ticker_of(root, event_date)
    return CanonicalSnapshot(
        ticker=f"{event}-{suffix}",
        event_ticker=event,
        series_ticker=root,
        status="finalized",
        result="no",
        yes_ask=ZERO,
        yes_bid=ZERO,
        no_ask=ZERO,
        no_bid=ZERO,
        last_price=ZERO,
        volume=ZERO,
        volume_24h=ZERO,
        open_interest=ZERO,
        close_time=datetime.combine(event_date + timedelta(days=1), time(6, 59), tzinfo=UTC),
    )


def closes_dir(
    tmp_path: Path,
    ladders: Mapping[str, Mapping[date, Sequence[str]]],
    *,
    name: str = "closes",
) -> Path:
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    for root, ladder in ladders.items():
        write_sidecar(
            directory / f"{root}.json",
            root,
            [
                snapshot(root, event_date, suffix)
                for event_date, suffixes in ladder.items()
                for suffix in suffixes
            ],
        )
    return directory


def sidecars_for(
    tmp_path: Path, ladders: Mapping[str, Mapping[date, Sequence[str]]]
) -> dict[str, CloseSidecar]:
    directory = closes_dir(tmp_path, ladders)
    return {root: read_sidecar(directory / f"{root}.json") for root in ladders}


def test_a_lock_at_the_first_in_window_reading_is_a_window_open_lock(tmp_path: Path) -> None:
    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: PAIR}}),
        {(STATION, DISCOVERY_DAY): readings_for(DISCOVERY_DAY, WINDOW_OPEN_TEMPS)},
    )

    (row,) = supply.station_days
    assert (row.clean, row.window_open, row.mid_day) == (1, 1, 0)


def test_a_lock_at_a_later_reading_is_a_mid_day_lock(tmp_path: Path) -> None:
    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: PAIR}}),
        {(STATION, DISCOVERY_DAY): readings_for(DISCOVERY_DAY, MID_DAY_TEMPS)},
    )

    (row,) = supply.station_days
    assert (row.clean, row.window_open, row.mid_day) == (1, 0, 1)


def test_the_window_open_anchor_is_the_publication_time_not_the_valid_time(tmp_path: Path) -> None:
    start, _ = observation_window(ZONE, DISCOVERY_DAY)
    lagged = [
        StationObservation(
            station=STATION,
            valid_time=start + timedelta(hours=index + 1),
            publication_time=start + timedelta(hours=index + 1) + PUBLICATION_LAG,
            temp_f=Decimal(temp),
            is_special=False,
            raw="",
            source="tape",
        )
        for index, temp in enumerate(WINDOW_OPEN_TEMPS)
    ]

    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: PAIR}}),
        {(STATION, DISCOVERY_DAY): lagged},
    )

    (row,) = supply.station_days
    assert (row.clean, row.window_open, row.mid_day) == (1, 1, 0)


def test_the_lowest_tail_leg_locks_on_the_kinds_settled_across_the_ladder(tmp_path: Path) -> None:
    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: TAIL}}),
        {(STATION, DISCOVERY_DAY): readings_for(DISCOVERY_DAY, TAIL_TEMPS)},
    )

    (row,) = supply.station_days
    assert (row.clean, row.ambiguous, row.no_lock) == (2, 0, 0)
    assert (row.window_open, row.mid_day) == (1, 1)


def test_a_clean_lock_and_an_ambiguous_lock_are_counted_apart(tmp_path: Path) -> None:
    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: RICH}}),
        {(STATION, DISCOVERY_DAY): readings_for(DISCOVERY_DAY, RICH_TEMPS)},
    )

    pooled = summarize(supply, None)

    assert (pooled.markets, pooled.clean, pooled.ambiguous, pooled.no_lock) == (4, 2, 1, 1)
    assert (pooled.window_open, pooled.mid_day) == (1, 1)
    assert pooled.ambiguous_to_clean == Decimal("0.5")
    assert pooled.ambiguous_to_mid_day == Decimal("1")


def test_a_series_on_neither_ladder_stops_the_sweep(tmp_path: Path) -> None:
    ladders = sidecars_for(tmp_path, {STRANGER: {DISCOVERY_DAY: PAIR}})

    with pytest.raises(ValueError, match="neither a high nor a low temperature ladder"):
        survey_lock_supply(scope_for(tmp_path), ladders, {})


def test_a_sweep_handed_both_ladders_stops(tmp_path: Path) -> None:
    ladders = sidecars_for(
        tmp_path, {SERIES: {DISCOVERY_DAY: PAIR}, HIGH_SERIES: {DISCOVERY_DAY: PAIR}}
    )

    with pytest.raises(ValueError, match="span both ladders"):
        survey_lock_supply(scope_for(tmp_path), ladders, {})


def test_a_station_day_with_no_in_window_readings_is_reported_not_dropped(tmp_path: Path) -> None:
    start, _ = observation_window(ZONE, DISCOVERY_DAY)

    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: RICH}}),
        {(STATION, DISCOVERY_DAY): [reading(start - timedelta(hours=1), "55")]},
    )

    (row,) = supply.station_days
    assert row.readings == 0
    assert (row.markets, row.clean, row.ambiguous, row.no_lock) == (4, 0, 0, 4)


def test_an_event_day_outside_the_frozen_scope_is_dropped(tmp_path: Path) -> None:
    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: PAIR, UNCOVERED_DAY: PAIR}}),
        {
            (STATION, DISCOVERY_DAY): readings_for(DISCOVERY_DAY, WINDOW_OPEN_TEMPS),
            (STATION, UNCOVERED_DAY): readings_for(UNCOVERED_DAY, WINDOW_OPEN_TEMPS),
        },
    )

    assert [row.event_date for row in supply.station_days] == [DISCOVERY_DAY]
    assert supply.markets == 2


def test_each_station_day_lands_in_the_split_the_freeze_names(tmp_path: Path) -> None:
    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: RICH, HOLDOUT_DAY: PAIR}}),
        {
            (STATION, DISCOVERY_DAY): readings_for(DISCOVERY_DAY, RICH_TEMPS),
            (STATION, HOLDOUT_DAY): readings_for(HOLDOUT_DAY, MID_DAY_TEMPS),
        },
    )

    discovery = summarize(supply, DISCOVERY)
    holdout = summarize(supply, HOLDOUT)
    pooled = summarize(supply, None)

    assert {row.event_date: row.split for row in supply.station_days} == {
        DISCOVERY_DAY: DISCOVERY,
        HOLDOUT_DAY: HOLDOUT,
    }
    assert pooled.split == POOLED
    assert pooled.station_days == discovery.station_days + holdout.station_days == 2
    assert pooled.markets == discovery.markets + holdout.markets
    assert pooled.clean == discovery.clean + holdout.clean
    assert pooled.ambiguous == discovery.ambiguous + holdout.ambiguous
    assert pooled.no_lock == discovery.no_lock + holdout.no_lock
    assert pooled.window_open == discovery.window_open + holdout.window_open
    assert pooled.mid_day == discovery.mid_day + holdout.mid_day
    assert pooled.mid_day_station_days == 2


def test_the_holdout_floor_is_the_discovery_floor_halved_and_rounded_up() -> None:
    assert DISCOVERY_STATION_DAY_MIN == 30
    assert HOLDOUT_STATION_DAY_MIN == 15
    assert (DISCOVERY_STATION_DAY_MIN + 1) // 2 == HOLDOUT_STATION_DAY_MIN


def test_a_ratio_whose_denominator_is_zero_is_reported_as_missing(tmp_path: Path) -> None:
    supply = survey_lock_supply(
        scope_for(tmp_path),
        sidecars_for(tmp_path, {SERIES: {DISCOVERY_DAY: PAIR, HOLDOUT_DAY: PAIR}}),
        {
            (STATION, DISCOVERY_DAY): readings_for(DISCOVERY_DAY, AMBIGUOUS_TEMPS),
            (STATION, HOLDOUT_DAY): readings_for(HOLDOUT_DAY, WINDOW_OPEN_TEMPS),
        },
    )

    no_clean = summarize(supply, DISCOVERY)
    no_mid_day = summarize(supply, HOLDOUT)

    assert (no_clean.clean, no_clean.ambiguous) == (0, 1)
    assert no_clean.ambiguous_to_clean is None
    assert no_clean.ambiguous_to_mid_day is None
    assert (no_mid_day.clean, no_mid_day.window_open, no_mid_day.mid_day) == (1, 1, 0)
    assert no_mid_day.ambiguous_to_clean == Decimal("0")
    assert no_mid_day.ambiguous_to_mid_day is None


@needs_tape
def test_the_frozen_low_ladders_sweep_every_station_day_the_scope_carries() -> None:
    supply = survey_lock_supply(
        load_run_scope(FROZEN_SCOPE),
        read_ladders(FROZEN_CLOSES),
        read_observations(FROZEN_OBSERVATIONS, read_observation_index(FROZEN_OBSERVATIONS)),
    )

    assert len(supply.roots) == FROZEN_ROOTS
    assert len(supply.station_days) == FROZEN_STATION_DAYS
    assert supply.markets == FROZEN_MARKETS
