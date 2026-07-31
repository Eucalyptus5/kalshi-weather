from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.ladder_consistency import (
    CITY_DAY_MIN_DISCOVERY,
    CITY_DAY_MIN_HOLDOUT,
    EXCESS_BAR,
    MONOTONICITY,
    SUM_BUY,
    SUM_SELL,
)
from bot.lag.ladder_run import (
    CLOSED,
    MICROS_PER_S,
    NO_ESTIMATE,
    PASS,
    QUANTILES,
    TICKS_PER_CENT,
    UNDECIDABLE,
    ZERO_ESTIMATE,
    Decision,
    SplitReadout,
    Sweep,
    census,
    decide,
    execute,
    readout,
    summarise,
    sweep_ladders,
)
from bot.lag.r0_universe import Coverage, freeze_universe, write_universe
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import MANIFEST_NAME
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
OPENS = timedelta(hours=6)

SERIES = "KXHIGHDEN"
DISCOVERY_DAY = date(2026, 7, 18)
HOLDOUT_DAY = date(2026, 7, 19)
LATE_DAY = date(2026, 7, 20)
SCOPE_START = datetime(2026, 7, 18, 6, tzinfo=UTC)
SCOPE_END = datetime(2026, 7, 20, 6, tzinfo=UTC)

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
STRIKES = ("T70", "B70.5", "B72.5", "B74.5", "B76.5", "T77")
FIVE_LEGS = STRIKES[:5]
NO_UPPER_TAIL = (*STRIKES[:5], "B78.5")

QUIET_START = datetime(2026, 7, 18, 7, tzinfo=UTC)
QUIET_END = datetime(2026, 7, 18, 9, tzinfo=UTC)
BLINK = datetime(2026, 7, 18, 20, tzinfo=UTC)

RUN_ID = "2026-08-13-q1"
SEED = 20260812
T_PERSIST = Decimal("10")
EXCESS = Decimal("7.12")
MAGNITUDE = Decimal("10")
DEPTH = Decimal("100")
DURATION = Decimal("60")

Quote = tuple[str, str, str, str]

DEAD: Quote = ("0.0000", "0.00", "1.0000", "0.00")
DEEP_BELOW: Quote = ("0.98", "100", "0.99", "100")
WIDE_ABOVE: Quote = ("0.12", "100", "0.14", "100")
QUIET_ABOVE: Quote = ("0.01", "100", "0.03", "100")


def stamp(event_date: date) -> str:
    return f"{event_date.year % 100:02d}{MONTHS[event_date.month - 1]}{event_date.day:02d}"


def legs(event_date: date, strikes: Sequence[str] = STRIKES) -> tuple[str, ...]:
    return tuple(f"{SERIES}-{stamp(event_date)}-{strike}" for strike in strikes)


def when(day: date, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=UTC)


def touch(row_id: int, ticker: str, received_at: datetime, quote: Quote) -> dict:
    yes_bid, yes_bid_depth, yes_ask, yes_ask_depth = quote
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": received_at,
        "ts_ms": None,
        "yes_bid": yes_bid,
        "yes_bid_depth": yes_bid_depth,
        "yes_ask": yes_ask,
        "yes_ask_depth": yes_ask_depth,
        "no_bid": str(Decimal("1") - Decimal(yes_ask)),
        "no_bid_depth": yes_ask_depth,
        "no_ask": str(Decimal("1") - Decimal(yes_bid)),
        "no_ask_depth": yes_bid_depth,
    }


def opening(tickers: Sequence[str], first_id: int, at: datetime) -> list[dict]:
    book = (DEEP_BELOW, DEAD, DEAD, DEAD, DEAD, QUIET_ABOVE)
    return [
        touch(first_id + seat, tickers[seat], at + timedelta(seconds=seat), book[seat])
        for seat in range(len(tickers))
    ]


def violation_day(
    event_date: date,
    first_id: int,
    *,
    warm: datetime,
    opens: datetime,
    closes: datetime,
    strikes: Sequence[str] = STRIKES,
) -> list[dict]:
    tickers = legs(event_date, strikes)
    rows = opening(tickers, first_id, warm)
    rows.append(touch(first_id + len(tickers), tickers[-1], opens, WIDE_ABOVE))
    rows.append(touch(first_id + len(tickers) + 1, tickers[-1], closes, QUIET_ABOVE))
    return rows


DISCOVERY_ROWS = violation_day(
    DISCOVERY_DAY,
    1,
    warm=when(DISCOVERY_DAY, 5, 0),
    opens=when(DISCOVERY_DAY, 12, 0),
    closes=when(DISCOVERY_DAY, 12, 1),
)
HOLDOUT_ROWS = violation_day(
    HOLDOUT_DAY,
    101,
    warm=when(HOLDOUT_DAY, 10, 0),
    opens=when(HOLDOUT_DAY, 12, 0),
    closes=when(HOLDOUT_DAY, 12, 1),
)
EXCLUDED_ROWS = violation_day(
    DISCOVERY_DAY,
    1,
    warm=when(DISCOVERY_DAY, 10, 0),
    opens=when(DISCOVERY_DAY, 19, 59, 30),
    closes=when(DISCOVERY_DAY, 20, 0, 30),
)
EARLY_ROWS = violation_day(
    DISCOVERY_DAY,
    1,
    warm=when(DISCOVERY_DAY, 3, 0),
    opens=when(DISCOVERY_DAY, 4, 0),
    closes=when(DISCOVERY_DAY, 4, 1),
)
FIVE_LEG_ROWS = violation_day(
    DISCOVERY_DAY,
    1,
    warm=when(DISCOVERY_DAY, 10, 0),
    opens=when(DISCOVERY_DAY, 12, 0),
    closes=when(DISCOVERY_DAY, 12, 1),
    strikes=FIVE_LEGS,
)
NO_TAIL_ROWS = violation_day(
    DISCOVERY_DAY,
    1,
    warm=when(DISCOVERY_DAY, 10, 0),
    opens=when(DISCOVERY_DAY, 12, 0),
    closes=when(DISCOVERY_DAY, 12, 1),
    strikes=NO_UPPER_TAIL,
)
BRIEF_ROWS = violation_day(
    DISCOVERY_DAY,
    1,
    warm=when(DISCOVERY_DAY, 10, 0),
    opens=when(DISCOVERY_DAY, 12, 0),
    closes=when(DISCOVERY_DAY, 12, 0, 1),
)
LATE_CLOSE_ROWS = violation_day(
    HOLDOUT_DAY,
    101,
    warm=when(HOLDOUT_DAY, 10, 0),
    opens=when(HOLDOUT_DAY, 23, 0),
    closes=when(LATE_DAY, 0, 30),
)
QUIET_DISCOVERY_ROWS = opening(legs(DISCOVERY_DAY), 1, when(DISCOVERY_DAY, 12, 0))
QUIET_HOLDOUT_ROWS = opening(legs(HOLDOUT_DAY), 101, when(HOLDOUT_DAY, 12, 0))


def exclusion_table() -> pa.Table:
    rows = [
        exclusion_row(0, QUIET_BAND, QUIET_START, QUIET_END),
        exclusion_row(1, RESUBSCRIBE_BLIND, BLINK, BLINK + MICROSECOND),
    ]
    return pa.Table.from_pylist(rows, schema=EXCLUSIONS_SCHEMA)


def event_day_table() -> pa.Table:
    rows = [
        event_day_row(DISCOVERY_DAY, in_scope=True, split=DISCOVERY, day_index=1, opens=OPENS),
        event_day_row(HOLDOUT_DAY, in_scope=True, split=HOLDOUT, day_index=2, opens=OPENS),
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def scope_dir(tmp_path: Path, *, cities: tuple[str, ...] = (SERIES,)) -> Path:
    directory = tmp_path / "scope"
    directory.mkdir()
    pq.write_table(exclusion_table(), directory / "exclusions.parquet")
    pq.write_table(event_day_table(), directory / "event_days.parquet")
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
            passing=(SERIES,),
            coverage=Coverage(cities=cities, ladder_widths=(6,), in_scope_city_days=2),
        ),
    )
    return directory


def artifacts_dir(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    write_partition(root, DISCOVERY_DAY, 1, DISCOVERY_ROWS)
    write_partition(root, HOLDOUT_DAY, 1, HOLDOUT_ROWS)
    return root


def quiet_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "quiet"
    write_partition(root, DISCOVERY_DAY, 1, QUIET_DISCOVERY_ROWS)
    write_partition(root, HOLDOUT_DAY, 1, QUIET_HOLDOUT_ROWS)
    return root


def excluded_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "excluded"
    write_partition(root, DISCOVERY_DAY, 1, EXCLUDED_ROWS)
    write_partition(root, HOLDOUT_DAY, 1, QUIET_HOLDOUT_ROWS)
    return root


def early_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "early"
    write_partition(root, DISCOVERY_DAY, 1, EARLY_ROWS)
    write_partition(root, HOLDOUT_DAY, 1, QUIET_HOLDOUT_ROWS)
    return root


def five_leg_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "five_legs"
    write_partition(root, DISCOVERY_DAY, 1, FIVE_LEG_ROWS)
    write_partition(root, HOLDOUT_DAY, 1, QUIET_HOLDOUT_ROWS)
    return root


def no_tail_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "no_tail"
    write_partition(root, DISCOVERY_DAY, 1, NO_TAIL_ROWS)
    write_partition(root, HOLDOUT_DAY, 1, QUIET_HOLDOUT_ROWS)
    return root


def brief_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "brief"
    write_partition(root, DISCOVERY_DAY, 1, BRIEF_ROWS)
    write_partition(root, HOLDOUT_DAY, 1, QUIET_HOLDOUT_ROWS)
    return root


def shared_partition_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "shared"
    write_partition(root, HOLDOUT_DAY, 1, DISCOVERY_ROWS + HOLDOUT_ROWS)
    return root


def late_close_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "late_close"
    write_partition(root, DISCOVERY_DAY, 1, QUIET_DISCOVERY_ROWS)
    write_partition(root, HOLDOUT_DAY, 1, LATE_CLOSE_ROWS[:-1])
    write_partition(root, LATE_DAY, 1, LATE_CLOSE_ROWS[-1:])
    return root


def disordered_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "disordered"
    write_partition(
        root,
        DISCOVERY_DAY,
        1,
        [row | {"id": len(DISCOVERY_ROWS) - index} for index, row in enumerate(DISCOVERY_ROWS)],
    )
    return root


def run_paths(tmp_path: Path, samples: int = ADEQUATE_SAMPLES) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", samples),
    }


def spread(value: str, count: int, *, split: str) -> SplitReadout:
    return panel(
        {
            f"{SERIES} city{index:02d}": [Decimal(value) + Decimal(2 * index - (count - 1)) / 20]
            for index in range(count)
        },
        split=split,
    )


def flat(value: str, count: int, *, split: str) -> SplitReadout:
    return panel(
        {f"{SERIES} city{index:02d}": [Decimal(value)] for index in range(count)}, split=split
    )


def panel(values: dict[str, list[Decimal]], *, split: str) -> SplitReadout:
    return readout(values, split=split, population=len(values), seed=SEED)


def decided(swept: Sweep) -> Decision:
    return decide(
        readout(swept.values[DISCOVERY], split=DISCOVERY, population=1, seed=SEED),
        readout(swept.values[HOLDOUT], split=HOLDOUT, population=1, seed=SEED),
    )


@pytest.fixture
def scope(tmp_path: Path) -> RunScope:
    return load_run_scope(scope_dir(tmp_path))


@pytest.fixture
def swept(tmp_path: Path, scope: RunScope) -> Sweep:
    return sweep_ladders(scope, artifacts_dir(tmp_path), t_persist_s=T_PERSIST)


def test_a_planted_violation_is_one_tradeable_episode_on_each_split(swept: Sweep) -> None:
    tally = swept.tallies[MONOTONICITY]

    assert tally.found == 2
    assert tally.tradeable == 2
    assert tally.kept == 2
    assert tally.censored == 0
    assert swept.tallies[SUM_BUY].found == 0
    assert swept.tallies[SUM_SELL].found == 0
    assert [episode.excess_cents for episode in swept.admitted] == [EXCESS, EXCESS]
    assert {episode.magnitude_cents for episode in swept.admitted} == {MAGNITUDE}
    assert {episode.depth for episode in swept.admitted} == {DEPTH}
    assert {episode.duration_s for episode in swept.admitted} == {DURATION}


def test_each_surviving_episode_lands_in_its_own_city_event_day_cluster(swept: Sweep) -> None:
    assert swept.values[DISCOVERY] == {f"{SERIES} {DISCOVERY_DAY.isoformat()}": [EXCESS]}
    assert swept.values[HOLDOUT] == {f"{SERIES} {HOLDOUT_DAY.isoformat()}": [EXCESS]}


def test_the_discovery_median_comes_off_the_surviving_episodes(swept: Sweep) -> None:
    result = readout(swept.values[DISCOVERY], split=DISCOVERY, population=1, seed=SEED)

    assert result.bootstrap.estimate == EXCESS
    assert result.n_city_days == 1
    assert result.episodes == 1
    assert result.bootstrap.ci_low == float(EXCESS)
    assert result.bootstrap.ci_high == float(EXCESS)


def test_a_holdout_violation_feeds_the_replication(swept: Sweep) -> None:
    decision = decided(swept)

    assert decision.gate.estimate == EXCESS
    assert decision.gate.economic
    assert not decision.gate.powered
    assert decision.replication.holdout_estimate == EXCESS
    assert decision.replication.same_sign
    assert decision.replication.magnitude
    assert decision.replication.holdout_n_min == CITY_DAY_MIN_HOLDOUT
    # One city event-day per split resamples to itself every time, so the tape says nothing about
    # the edge either way and the question stays open.
    assert decision.gate.undecidable
    assert decision.replication.undecidable
    assert decision.verdict == UNDECIDABLE


def test_rarity_closes_the_question_rather_than_underpowering_it() -> None:
    thin = spread("7.12", CITY_DAY_MIN_DISCOVERY - 1, split=DISCOVERY)
    holdout = spread("7.12", CITY_DAY_MIN_HOLDOUT, split=HOLDOUT)

    decision = decide(thin, holdout)

    assert not decision.gate.powered
    assert decision.gate.economic
    assert decision.gate.significant
    assert decision.verdict == CLOSED


def test_a_gate_that_clears_and_replicates_passes() -> None:
    discovery = spread("7.12", CITY_DAY_MIN_DISCOVERY, split=DISCOVERY)
    holdout = spread("7.12", CITY_DAY_MIN_HOLDOUT, split=HOLDOUT)

    decision = decide(discovery, holdout)

    assert decision.gate.passed
    assert decision.gate.n == CITY_DAY_MIN_DISCOVERY
    assert decision.replication.replicated
    assert decision.verdict == PASS


def test_a_median_under_the_bar_closes_the_question() -> None:
    discovery = spread("1", CITY_DAY_MIN_DISCOVERY, split=DISCOVERY)
    holdout = spread("1", CITY_DAY_MIN_HOLDOUT, split=HOLDOUT)

    decision = decide(discovery, holdout)

    assert decision.gate.estimate < EXCESS_BAR
    assert not decision.gate.economic
    assert decision.replication.replicated
    assert decision.verdict == CLOSED


def test_a_median_under_the_bar_closes_the_question_however_the_resamples_landed() -> None:
    decision = decide(
        flat("1", CITY_DAY_MIN_DISCOVERY, split=DISCOVERY),
        flat("1", CITY_DAY_MIN_HOLDOUT, split=HOLDOUT),
    )

    assert decision.gate.undecidable
    assert not decision.gate.economic
    assert decision.verdict == CLOSED


def test_a_median_over_the_bar_that_no_resample_moved_refuses_a_verdict() -> None:
    decision = decide(
        flat("7.12", CITY_DAY_MIN_DISCOVERY, split=DISCOVERY),
        flat("7.12", CITY_DAY_MIN_HOLDOUT, split=HOLDOUT),
    )

    assert decision.gate.economic
    assert decision.gate.powered
    assert decision.gate.undecidable
    assert not decision.gate.significant
    assert not decision.gate.passed
    assert decision.verdict == UNDECIDABLE


def test_a_holdout_no_resample_moved_refuses_a_verdict_the_discovery_alone_would_pass() -> None:
    decision = decide(
        spread("7.12", CITY_DAY_MIN_DISCOVERY, split=DISCOVERY),
        flat("7.12", CITY_DAY_MIN_HOLDOUT, split=HOLDOUT),
    )

    assert decision.gate.passed
    assert not decision.gate.undecidable
    assert decision.replication.undecidable
    assert decision.replication.same_sign
    assert decision.replication.magnitude
    assert decision.replication.powered
    assert not decision.replication.replicated
    assert decision.verdict == UNDECIDABLE


def test_a_degenerate_median_with_no_holdout_estimate_closes_the_question() -> None:
    decision = decide(
        flat("7.12", CITY_DAY_MIN_DISCOVERY, split=DISCOVERY), panel({}, split=HOLDOUT)
    )

    assert decision.gate.economic
    assert decision.gate.undecidable
    assert decision.replication is None
    assert decision.skipped == NO_ESTIMATE
    assert decision.verdict == CLOSED


def test_a_discovery_median_of_exactly_zero_skips_the_replication() -> None:
    discovery = spread("0", CITY_DAY_MIN_DISCOVERY, split=DISCOVERY)
    holdout = spread("7.12", CITY_DAY_MIN_HOLDOUT, split=HOLDOUT)

    decision = decide(discovery, holdout)

    assert decision.gate.estimate == Decimal("0")
    assert decision.replication is None
    assert decision.skipped == ZERO_ESTIMATE
    assert decision.verdict == CLOSED


def test_no_tradeable_episode_anywhere_closes_without_a_bootstrap(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_ladders(scope, quiet_artifacts(tmp_path), t_persist_s=T_PERSIST)
    discovery = readout(sweep.values[DISCOVERY], split=DISCOVERY, population=1, seed=SEED)

    decision = decided(sweep)

    assert sweep.values == {DISCOVERY: {}, HOLDOUT: {}}
    assert discovery.bootstrap is None
    assert discovery.n_city_days == 0
    assert decision.gate is None
    assert decision.replication is None
    assert decision.skipped == NO_ESTIMATE
    assert decision.verdict == CLOSED


def test_an_episode_before_its_event_day_opens_never_reaches_a_cluster(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_ladders(scope, early_artifacts(tmp_path), t_persist_s=T_PERSIST)

    assert when(DISCOVERY_DAY, 4, 1) < scope.event_days[(SERIES, DISCOVERY_DAY)].window_start
    assert sweep.tallies[MONOTONICITY].tradeable == 1
    assert sweep.tallies[MONOTONICITY].kept == 0
    assert sweep.screened.candidates == 1
    assert sweep.screened.out_of_window == 1
    assert sweep.screened.excluded == 0
    assert sweep.values[DISCOVERY] == {}


def test_an_episode_over_an_exclusion_is_charged_to_its_class(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_ladders(scope, excluded_artifacts(tmp_path), t_persist_s=T_PERSIST)

    assert sweep.tallies[MONOTONICITY].tradeable == 1
    assert sweep.tallies[MONOTONICITY].kept == 0
    assert sweep.screened.excluded == 1
    assert sweep.screened.by_class[RESUBSCRIBE_BLIND] == 1
    assert sweep.screened.by_class[QUIET_BAND] == 0
    assert sweep.screened.excluded_fraction == Decimal("1")
    assert sweep.values[DISCOVERY] == {}


def test_an_episode_short_of_persistence_is_never_offered_to_the_screen(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_ladders(scope, brief_artifacts(tmp_path), t_persist_s=T_PERSIST)
    tally = sweep.tallies[MONOTONICITY]

    assert tally.found == 1
    assert tally.tradeable == 0
    assert tally.kept == 0
    assert sweep.screened.candidates == 0
    assert sweep.screened.excluded_fraction is None
    assert summarise(tally.all_episodes["duration_s"], MICROS_PER_S)["count"] == 1
    assert summarise(tally.kept_episodes["duration_s"], MICROS_PER_S)["count"] == 0
    assert Decimal(summarise(tally.all_episodes["duration_s"], MICROS_PER_S)["max"]) < T_PERSIST


def test_the_excluded_fraction_divides_by_the_qualifying_episodes(swept: Sweep) -> None:
    assert swept.screened.candidates == 2
    assert swept.screened.excluded == 0
    assert swept.screened.excluded_fraction == Decimal("0")


@pytest.mark.parametrize(
    ("builder", "tickers"),
    [
        pytest.param(five_leg_artifacts, 5, id="five_legs"),
        pytest.param(no_tail_artifacts, 6, id="two_brackets_where_a_tail_belongs"),
    ],
)
def test_an_incomplete_ladder_drops_whole_and_is_recorded(
    tmp_path: Path, scope: RunScope, builder: Callable[[Path], Path], tickers: int
) -> None:
    sweep = sweep_ladders(scope, builder(tmp_path), t_persist_s=T_PERSIST)

    assert sweep.incomplete == ((SERIES, DISCOVERY_DAY),)
    assert sweep.in_scope == 2
    assert sweep.complete == 1
    assert sweep.tickers[(SERIES, DISCOVERY_DAY)] == tickers
    assert sweep.tallies[MONOTONICITY].found == 0
    assert sweep.values[DISCOVERY] == {}


def test_the_census_reads_a_partition_holding_two_event_days(
    tmp_path: Path, scope: RunScope
) -> None:
    root = shared_partition_artifacts(tmp_path)

    groups = census(root, scope, SERIES)
    sweep = sweep_ladders(scope, root, t_persist_s=T_PERSIST)

    assert groups[DISCOVERY_DAY] == tuple(sorted(legs(DISCOVERY_DAY)))
    assert groups[HOLDOUT_DAY] == tuple(sorted(legs(HOLDOUT_DAY)))
    assert sweep.tickers == {(SERIES, DISCOVERY_DAY): 6, (SERIES, HOLDOUT_DAY): 6}
    assert sweep.incomplete == ()
    assert sweep.tallies[MONOTONICITY].kept == 2


def test_the_warm_up_rows_open_a_leg_without_becoming_an_episode(
    swept: Sweep, scope: RunScope
) -> None:
    admitted = next(item for item in swept.admitted if item.event_date == DISCOVERY_DAY)

    assert DISCOVERY_ROWS[0]["received_at"] < scope.event_days[(SERIES, DISCOVERY_DAY)].window_start
    assert admitted.start == when(DISCOVERY_DAY, 12, 0)
    assert admitted.end == when(DISCOVERY_DAY, 12, 1)
    assert swept.rows == 2 * len(DISCOVERY_ROWS)
    assert swept.screened.out_of_window == 0
    assert swept.tallies[MONOTONICITY].incomplete_states == 10


def test_a_row_arriving_after_the_window_closes_the_episode_it_opened(
    tmp_path: Path, scope: RunScope
) -> None:
    sweep = sweep_ladders(scope, late_close_artifacts(tmp_path), t_persist_s=T_PERSIST)

    assert LATE_DAY not in {event_date for _, event_date in scope.event_days}
    assert sweep.tallies[MONOTONICITY].censored == 0
    assert sweep.tallies[MONOTONICITY].kept == 1
    assert sweep.admitted[0].end == when(LATE_DAY, 0, 30)
    assert sweep.rows == len(QUIET_DISCOVERY_ROWS) + len(LATE_CLOSE_ROWS)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("min", "1", id="min"),
        pytest.param("p25", "1", id="p25"),
        pytest.param("median", "2", id="median"),
        pytest.param("p75", "3", id="p75"),
        pytest.param("p90", "4", id="p90"),
        pytest.param("max", "4", id="max"),
    ],
)
def test_the_quantiles_are_nearest_rank_over_an_even_count(name: str, expected: str) -> None:
    values = [np.array([400, 100, 300, 200], dtype=np.int64)]

    summary = summarise(values, TICKS_PER_CENT)

    assert summary["count"] == 4
    assert summary[name] == expected
    assert Decimal(summary["median"]) != (Decimal("2") + Decimal("3")) / 2


def test_every_quantile_label_reports_the_fraction_it_names() -> None:
    values = [np.arange(1, 101, dtype=np.int64) * TICKS_PER_CENT]

    summary = summarise(values, TICKS_PER_CENT)

    assert list(summary) == ["count", "min", *(name for name, _, _ in QUANTILES), "max"]
    assert {name: Decimal(summary[name]) for name, _, _ in QUANTILES} == {
        name: Decimal(numerator * 100 // denominator) for name, numerator, denominator in QUANTILES
    }


def test_the_quantiles_of_a_single_value_are_that_value() -> None:
    summary = summarise([np.array([712], dtype=np.int64)], TICKS_PER_CENT)

    assert summary == {
        "count": 1,
        "min": "7.12",
        "p25": "7.12",
        "median": "7.12",
        "p75": "7.12",
        "p90": "7.12",
        "max": "7.12",
    }


def test_a_stream_with_no_episode_reports_no_quantiles() -> None:
    summary = summarise([], TICKS_PER_CENT)

    assert summary == {
        "count": 0,
        "min": None,
        "p25": None,
        "median": None,
        "p75": None,
        "p90": None,
        "max": None,
    }


def test_a_series_set_disagreeing_with_the_recorded_universe_raises(tmp_path: Path) -> None:
    scope = load_run_scope(scope_dir(tmp_path, cities=(SERIES, "KXHIGHLAX")))

    with pytest.raises(ValueError, match="KXHIGHLAX"):
        sweep_ladders(scope, artifacts_dir(tmp_path), t_persist_s=T_PERSIST)


def test_the_manifest_lands_before_any_statistic_is_read(tmp_path: Path) -> None:
    paths = run_paths(tmp_path)
    run_root = tmp_path / "tape_studies"

    with pytest.raises(ValueError, match="id order"):
        execute(
            run_id=RUN_ID,
            artifacts=disordered_artifacts(tmp_path),
            floor_source=FloorSource.SIGNED_READ,
            seed=SEED,
            run_root=run_root,
            **paths,
        )

    assert (run_root / RUN_ID / MANIFEST_NAME).exists()


def test_a_touch_partition_carrying_another_schema_is_refused(
    tmp_path: Path, scope: RunScope
) -> None:
    root = tmp_path / "foreign"
    path = write_partition(
        root, DISCOVERY_DAY, 1, [], schema=pa.schema([("id", pa.int64()), ("ticker", pa.string())])
    )

    with pytest.raises(ValueError, match=path.name):
        sweep_ladders(scope, root, t_persist_s=T_PERSIST)

    assert TOUCH_SCHEMA.names != ["id", "ticker"]
