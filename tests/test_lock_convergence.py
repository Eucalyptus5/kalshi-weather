import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.lock_convergence import (
    CLOSED,
    HALF_LIFE_THRESHOLD_S,
    LOCK_BAND,
    NO_ESTIMATE,
    PASS,
    PERSIST_S,
    PROVIDING,
    ROUNDING_MARGIN_F,
    STATION_DAY_MIN,
    TAKING,
    UNDECIDABLE,
    UNDERPOWERED,
    SplitReadout,
    Sweep,
    clears_strike,
    converged,
    decide,
    execute,
    readout,
    result_payload,
    scan_locks,
    sweep_convergence,
)
from bot.lag.lock_events import detect_lock_events
from bot.lag.r0_universe import Coverage, freeze_universe, write_universe
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import MANIFEST_NAME
from bot.lag.tape_studies import RunScope, load_run_scope
from bot.markets.parser import parse_ticker
from bot.observations.metar import StationObservation
from bot.replay.artifacts import TRADES_SCHEMA
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
OPENS = timedelta(hours=7)

SERIES = "KXHIGHDEN"
STATION = "KDEN"
ZONE = "America/Denver"
DISCOVERY_DAY = date(2026, 7, 18)
HOLDOUT_DAY = date(2026, 7, 19)
SCOPE_START = datetime(2026, 7, 18, 7, tzinfo=UTC)
SCOPE_END = datetime(2026, 7, 20, 7, tzinfo=UTC)
WINDOW_END = datetime(2026, 7, 19, 7, tzinfo=UTC)

ABOVE = f"{SERIES}-26JUL18-T70"
LOWER = f"{SERIES}-26JUL18-T60"

LOCK_AT = datetime(2026, 7, 18, 18, tzinfo=UTC)
BLINK = datetime(2026, 7, 18, 12, tzinfo=UTC)
QUIET_START = datetime(2026, 7, 18, 8, tzinfo=UTC)
QUIET_END = datetime(2026, 7, 18, 9, tzinfo=UTC)
BITE_START = datetime(2026, 7, 18, 18, 2, tzinfo=UTC)
BITE_END = datetime(2026, 7, 18, 18, 5, tzinfo=UTC)

SEED = 20260813
RUN_ID = "2026-08-13-q4"
SETTLES = {(STATION, DISCOVERY_DAY): Decimal("72")}


def at(hour: int, minute: int, second: int = 0, *, day: date = DISCOVERY_DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=UTC)


def reading(valid: datetime, temp: str, *, published: datetime | None = None) -> StationObservation:
    return StationObservation(
        station=STATION,
        valid_time=valid,
        publication_time=valid if published is None else published,
        temp_f=Decimal(temp),
        is_special=False,
        raw="",
        source="iem_1min_asos_archive",
    )


ARCHIVE = {(STATION, DISCOVERY_DAY): [reading(at(12, 0), "55"), reading(at(18, 0), "72")]}
AMBIGUOUS_ARCHIVE = {(STATION, DISCOVERY_DAY): [reading(at(18, 0), "70.5")]}
LATE_ARCHIVE = {
    (STATION, DISCOVERY_DAY): [reading(at(6, 59, 30, day=HOLDOUT_DAY), "72")],
}


def touch(
    row_id: int,
    ticker: str,
    received_at: datetime,
    yes_bid: str,
    yes_ask: str,
    *,
    bid_depth: str = "10",
    ask_depth: str = "10",
) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": received_at,
        "ts_ms": row_id * 1000,
        "yes_bid": yes_bid,
        "yes_bid_depth": bid_depth,
        "yes_ask": yes_ask,
        "yes_ask_depth": ask_depth,
        "no_bid": str(Decimal("1") - Decimal(yes_ask)),
        "no_bid_depth": ask_depth,
        "no_ask": str(Decimal("1") - Decimal(yes_bid)),
        "no_ask_depth": bid_depth,
    }


def trade(
    row_id: int,
    ticker: str,
    received_at: datetime,
    *,
    yes_price: str = "0.97",
    side: str = "yes",
    count: str = "10",
) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": received_at,
        "ts_ms": row_id * 1000,
        "yes_price": yes_price,
        "no_price": str(Decimal("1") - Decimal(yes_price)),
        "count": count,
        "taker_side": side,
        "trade_id": f"t{row_id}",
    }


LADDER_ROWS = [
    touch(1, ABOVE, at(17, 59), "0.60", "0.62"),
    touch(2, ABOVE, at(18, 0), "0.60", "0.62"),
    touch(3, ABOVE, at(18, 2), "0.94", "0.96"),
    touch(4, ABOVE, at(18, 10), "0.94", "0.96"),
    touch(11, LOWER, at(17, 59), "0.40", "0.42"),
    touch(12, LOWER, at(18, 0, 30), "0.04", "0.06"),
    touch(13, LOWER, at(18, 5), "0.04", "0.06"),
]

ZERO_ROWS = [
    touch(1, ABOVE, at(17, 59), "0.94", "0.96"),
    touch(2, ABOVE, at(18, 5), "0.94", "0.96"),
]

ONE_SIDED_ROWS = [
    touch(1, ABOVE, at(17, 59), "0.94", "1.00", ask_depth="0"),
    touch(2, ABOVE, at(18, 3), "0.94", "0.96"),
    touch(3, ABOVE, at(18, 10), "0.94", "0.96"),
]

EMPTY_YES_ROWS = [
    touch(1, ABOVE, at(17, 59), "0.94", "0.96"),
    touch(11, LOWER, at(17, 59), "0.40", "0.42"),
    touch(12, LOWER, at(18, 0, 30), "0.00", "0.06", bid_depth="0"),
    touch(13, LOWER, at(18, 2), "0.04", "0.06"),
    touch(14, LOWER, at(18, 10), "0.04", "0.06"),
]

INTERRUPTED_ROWS = [
    touch(1, ABOVE, at(17, 59), "0.94", "0.96"),
    touch(2, ABOVE, at(18, 0, 30), "0.94", "1.00", ask_depth="0"),
    touch(3, ABOVE, at(18, 1), "0.94", "0.96"),
    touch(4, ABOVE, at(18, 10), "0.94", "0.96"),
]

BOUNCE_ROWS = [
    touch(1, ABOVE, at(17, 59), "0.94", "0.96"),
    touch(2, ABOVE, at(18, 0, 30), "0.60", "0.62"),
    touch(3, ABOVE, at(18, 1), "0.94", "0.96"),
    touch(4, ABOVE, at(18, 10), "0.94", "0.96"),
]

CENSORED_ROWS = [touch(1, ABOVE, at(6, 59, day=HOLDOUT_DAY), "0.94", "0.96")]


def exclusion_table(*, bites: bool = False) -> pa.Table:
    rows = [
        exclusion_row(0, QUIET_BAND, QUIET_START, QUIET_END),
        exclusion_row(1, RESUBSCRIBE_BLIND, BLINK, BLINK + MICROSECOND),
    ]
    if bites:
        rows.append(exclusion_row(2, RESUBSCRIBE_BLIND, BITE_START, BITE_END))
    return pa.Table.from_pylist(rows, schema=EXCLUSIONS_SCHEMA)


def event_day_table() -> pa.Table:
    rows = [
        event_day_row(DISCOVERY_DAY, in_scope=True, split=DISCOVERY, day_index=1, opens=OPENS),
        event_day_row(HOLDOUT_DAY, in_scope=True, split=HOLDOUT, day_index=2, opens=OPENS),
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def scope_dir(tmp_path: Path, *, bites: bool = False) -> Path:
    directory = tmp_path / ("bitten_scope" if bites else "scope")
    directory.mkdir()
    pq.write_table(exclusion_table(bites=bites), directory / "exclusions.parquet")
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
            coverage=Coverage(cities=(SERIES,), ladder_widths=(6,), in_scope_city_days=2),
        ),
    )
    return directory


def artifacts_dir(
    tmp_path: Path,
    rows: Sequence[dict],
    *,
    name: str = "artifacts",
    trades: Sequence[dict] = (),
    arrival_day: date = DISCOVERY_DAY,
) -> Path:
    root = tmp_path / name
    write_partition(root, arrival_day, 1, list(rows))
    write_partition(
        root, arrival_day, 1, list(trades), kind="trades", schema=TRADES_SCHEMA, series=SERIES
    )
    return root


def swept(
    tmp_path: Path,
    rows: Sequence[dict],
    *,
    name: str = "artifacts",
    trades: Sequence[dict] = (),
    arrival_day: date = DISCOVERY_DAY,
    observations: Mapping[tuple[str, date], list[StationObservation]] | None = None,
    arrivals: Mapping[str, list[StationObservation]] | None = None,
    settles: Mapping[tuple[str, date], Decimal] | None = None,
    bites: bool = False,
) -> Sweep:
    scope = load_run_scope(scope_dir(tmp_path, bites=bites))
    return sweep_convergence(
        scope,
        artifacts_dir(tmp_path, rows, name=name, trades=trades, arrival_day=arrival_day),
        observations=ARCHIVE if observations is None else observations,
        arrivals={} if arrivals is None else arrivals,
        settles=SETTLES if settles is None else settles,
    )


def half_lives(sweep: Sweep, split: str = DISCOVERY) -> dict[str, list[Decimal]]:
    return {name: list(values) for name, values in sweep.values[split].items()}


def spread(value: str, count: int, *, split: str) -> SplitReadout:
    return panel(
        {
            f"ST{index:02d} {DISCOVERY_DAY.isoformat()}": [
                Decimal(value) + Decimal(2 * index - (count - 1)) / 20
            ]
            for index in range(count)
        },
        split=split,
    )


def flat(value: str, count: int, *, split: str) -> SplitReadout:
    return panel(
        {f"ST{index:02d} {DISCOVERY_DAY.isoformat()}": [Decimal(value)] for index in range(count)},
        split=split,
    )


def panel(values: dict[str, list[Decimal]], *, split: str) -> SplitReadout:
    return readout(values, split=split, seed=SEED, powered=True)


def run_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
    }


@pytest.fixture
def scope(tmp_path: Path) -> RunScope:
    return load_run_scope(scope_dir(tmp_path))


def test_a_one_sided_book_carries_no_mid_and_never_counts_as_in_band(tmp_path: Path) -> None:
    sweep = swept(tmp_path, ONE_SIDED_ROWS)

    values = half_lives(sweep)

    assert sweep.one_sided_rows == 1
    assert values == {f"{STATION} {DISCOVERY_DAY.isoformat()}": [Decimal(180)]}
    assert values[f"{STATION} {DISCOVERY_DAY.isoformat()}"] != [Decimal(0)]


def test_an_empty_yes_book_carries_no_mid_though_its_stored_prices_read_inside_the_band(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, EMPTY_YES_ROWS)

    scored = {item.ticker: item.half_life_s for item in sweep.kept}

    assert sweep.one_sided_rows == 1
    assert scored[LOWER] == Decimal(120)
    assert scored[LOWER] != Decimal(30)
    assert scored[ABOVE] == Decimal(0)


def test_a_one_sided_row_inside_the_persistence_window_breaks_the_entry_it_interrupts(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, INTERRUPTED_ROWS)

    values = half_lives(sweep)

    assert sweep.one_sided_rows == 1
    assert values == {f"{STATION} {DISCOVERY_DAY.isoformat()}": [Decimal(60)]}
    assert values[f"{STATION} {DISCOVERY_DAY.isoformat()}"] != [Decimal(0)]


def test_a_leg_already_in_band_at_the_lock_scores_zero_and_stays_in_the_denominator(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, ZERO_ROWS)

    assert half_lives(sweep) == {f"{STATION} {DISCOVERY_DAY.isoformat()}": [Decimal(0)]}
    assert sweep.censored == 0


def test_a_leg_that_bounces_out_before_the_persistence_window_closes_scores_the_later_entry(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, BOUNCE_ROWS)

    assert half_lives(sweep) == {f"{STATION} {DISCOVERY_DAY.isoformat()}": [Decimal(60)]}


def test_a_leg_with_less_runway_than_the_persistence_window_is_censored_at_the_window_end(
    tmp_path: Path,
) -> None:
    sweep = swept(
        tmp_path,
        CENSORED_ROWS,
        arrival_day=HOLDOUT_DAY,
        observations=LATE_ARCHIVE,
    )

    assert half_lives(sweep) == {f"{STATION} {DISCOVERY_DAY.isoformat()}": [Decimal(30)]}
    assert sweep.censored == 1
    assert [item.censored for item in sweep.kept] == [True]
    assert [item.t_end for item in sweep.kept] == [WINDOW_END]


def test_both_bands_are_read_off_the_yes_contract_in_the_direction_the_lock_fixes(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, LADDER_ROWS)

    assert [(item.ticker, item.side_locked) for item in sweep.kept] == [
        (LOWER, "no"),
        (ABOVE, "yes"),
    ]
    assert sorted(half_lives(sweep)[f"{STATION} {DISCOVERY_DAY.isoformat()}"]) == [
        Decimal(30),
        Decimal(120),
    ]


def test_two_locks_on_one_station_day_make_one_cluster_carrying_two_values(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, LADDER_ROWS)

    values = half_lives(sweep)

    assert list(values) == [f"{STATION} {DISCOVERY_DAY.isoformat()}"]
    assert len(values[f"{STATION} {DISCOVERY_DAY.isoformat()}"]) == 2
    assert sweep.clean_station_days[DISCOVERY] == 1


def test_an_ambiguous_crossing_leaves_the_denominator_and_is_counted_apart(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, ONE_SIDED_ROWS, observations=AMBIGUOUS_ARCHIVE)

    assert sweep.scan.ambiguous == 1
    assert sweep.scan.clean == ()
    assert half_lives(sweep) == {}


def test_a_station_day_with_no_published_settle_is_dropped_and_counted(tmp_path: Path) -> None:
    sweep = swept(tmp_path, LADDER_ROWS, settles={})

    assert sweep.unsettled_station_days == 1
    assert sweep.usable_event_days == 0
    assert sweep.last_settled_event_day is None
    assert half_lives(sweep) == {}


def test_a_settle_that_does_not_confirm_the_crossing_is_reported_and_gates_nothing(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, ZERO_ROWS, settles={(STATION, DISCOVERY_DAY): Decimal("70")})

    assert sweep.settle_contradicts == 1
    assert half_lives(sweep) == {f"{STATION} {DISCOVERY_DAY.isoformat()}": [Decimal(0)]}


def test_a_station_day_with_no_arrival_rows_in_its_window_is_counted(tmp_path: Path) -> None:
    sweep = swept(tmp_path, ZERO_ROWS, arrivals={STATION: []})

    assert sweep.arrivals.no_arrival_rows == 1
    assert sweep.arrivals.never_clears == 0
    assert sweep.arrivals.arrival_precedes_lock == 0
    assert sweep.arrivals.deltas == []


def test_arrivals_whose_running_max_never_clears_the_strike_are_counted(tmp_path: Path) -> None:
    sweep = swept(tmp_path, ZERO_ROWS, arrivals={STATION: [reading(at(15, 0), "65")]})

    assert sweep.arrivals.never_clears == 1
    assert sweep.arrivals.no_arrival_rows == 0
    assert sweep.arrivals.deltas == []


def test_an_arrival_anchor_before_the_lock_is_counted_and_never_measured(tmp_path: Path) -> None:
    sweep = swept(tmp_path, ZERO_ROWS, arrivals={STATION: [reading(at(17, 0), "72")]})

    assert sweep.arrivals.arrival_precedes_lock == 1
    assert sweep.arrivals.deltas == []


def test_the_arrival_anchor_reads_the_running_max_and_reports_its_offset_from_the_lock(
    tmp_path: Path,
) -> None:
    sweep = swept(
        tmp_path,
        ZERO_ROWS,
        arrivals={
            STATION: [
                reading(at(18, 2), "65"),
                reading(at(18, 4), "72", published=at(18, 5)),
                reading(at(18, 6), "60"),
            ]
        },
    )

    assert sweep.arrivals.deltas == [Decimal(300)]
    assert sweep.arrivals.never_clears == 0
    assert sweep.arrivals.arrival_precedes_lock == 0


def test_fractional_fill_sizes_survive_into_contracts_and_notional(tmp_path: Path) -> None:
    sweep = swept(
        tmp_path,
        ZERO_ROWS,
        trades=[
            trade(200, ABOVE, at(18, 1), count="0.01", side="no", yes_price="0.97"),
            trade(201, ABOVE, at(18, 2), count="1.24", side="no", yes_price="0.97"),
        ],
    )

    tally = sweep.fills[TAKING]

    assert tally.contracts == Decimal("1.25")
    assert tally.notional == Decimal("0.0375")
    assert tally.fills == 2


def test_the_invalidated_side_split_names_who_lifted_and_who_rested(tmp_path: Path) -> None:
    sweep = swept(
        tmp_path,
        LADDER_ROWS,
        trades=[
            trade(200, ABOVE, at(18, 1), count="3", side="no"),
            trade(201, ABOVE, at(18, 1), count="4", side="yes"),
            trade(202, LOWER, at(18, 1), count="5", side="yes", yes_price="0.03"),
            trade(203, LOWER, at(18, 1), count="6", side="no", yes_price="0.03"),
            trade(204, ABOVE, at(18, 1), count="7", side=""),
        ],
    )

    assert sweep.fills[TAKING].contracts == Decimal(8)
    assert sweep.fills[PROVIDING].contracts == Decimal(10)
    assert sweep.fills_unclassified == 1


def test_an_evidence_window_meeting_an_exclusion_is_dropped_whole_and_counted_by_class(
    tmp_path: Path,
) -> None:
    sweep = swept(tmp_path, LADDER_ROWS, bites=True)

    assert [item.ticker for item in sweep.kept] == [LOWER]
    assert sweep.screened.excluded == 1
    assert sweep.screened.by_class[RESUBSCRIBE_BLIND] == 1
    assert half_lives(sweep) == {f"{STATION} {DISCOVERY_DAY.isoformat()}": [Decimal(30)]}


def test_a_reading_under_the_station_day_minimum_reports_no_statistic_but_keeps_the_lock_rate(
    tmp_path: Path,
) -> None:
    run = execute(
        run_id=RUN_ID,
        artifacts=artifacts_dir(tmp_path, LADDER_ROWS),
        observations=ARCHIVE,
        arrivals={},
        settles=SETTLES,
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **run_paths(tmp_path),
    )

    payload = result_payload(run)

    assert run.decision.verdict == UNDERPOWERED
    assert run.decision.gate is None
    assert run.discovery.n_station_days == 1
    assert run.discovery.bootstrap is None
    assert run.holdout.bootstrap is None
    assert payload["discovery"]["median_half_life_s"] is None
    assert payload["locks"]["lock_rate"] == "2"
    assert payload["locks"]["clean_station_days"] == 1
    assert payload["locks"]["population_ceiling"] == 1
    assert payload["half_life"]["gate_ran"] is False
    assert payload["half_life"]["pooled"]["count"] == 2
    assert payload["half_life"]["pooled"]["median"] == "75"
    assert json.dumps(payload)


def test_a_median_above_the_threshold_that_replicates_passes() -> None:
    decision = decide(
        spread("600", STATION_DAY_MIN, split=DISCOVERY),
        spread("600", (STATION_DAY_MIN + 1) // 2, split=HOLDOUT),
    )

    assert decision.verdict == PASS
    assert decision.gate.threshold == HALF_LIFE_THRESHOLD_S
    assert decision.gate.n_min == STATION_DAY_MIN
    assert decision.replication.holdout_n_min == (STATION_DAY_MIN + 1) // 2


def test_a_median_under_the_threshold_closes_the_question() -> None:
    decision = decide(
        spread("30", STATION_DAY_MIN, split=DISCOVERY),
        spread("30", (STATION_DAY_MIN + 1) // 2, split=HOLDOUT),
    )

    assert decision.verdict == CLOSED
    assert decision.gate.economic is False


def test_a_median_under_the_threshold_closes_it_however_the_resamples_landed() -> None:
    decision = decide(
        flat("30", STATION_DAY_MIN, split=DISCOVERY),
        flat("30", (STATION_DAY_MIN + 1) // 2, split=HOLDOUT),
    )

    assert decision.gate.undecidable
    assert decision.gate.economic is False
    assert decision.verdict == CLOSED


def test_a_median_over_the_threshold_that_no_resample_moved_refuses_a_verdict() -> None:
    decision = decide(
        flat("600", STATION_DAY_MIN, split=DISCOVERY),
        flat("600", (STATION_DAY_MIN + 1) // 2, split=HOLDOUT),
    )

    assert decision.gate.economic
    assert decision.gate.undecidable
    assert not decision.gate.significant
    assert not decision.gate.passed
    assert decision.verdict == UNDECIDABLE


def test_a_holdout_no_resample_moved_refuses_a_verdict_the_discovery_alone_would_pass() -> None:
    decision = decide(
        spread("600", STATION_DAY_MIN, split=DISCOVERY),
        flat("600", (STATION_DAY_MIN + 1) // 2, split=HOLDOUT),
    )

    assert decision.gate.passed
    assert decision.replication.undecidable
    assert decision.replication.powered
    assert not decision.replication.replicated
    assert decision.verdict == UNDECIDABLE


def test_a_degenerate_median_with_no_holdout_estimate_closes_the_question() -> None:
    decision = decide(flat("600", STATION_DAY_MIN, split=DISCOVERY), panel({}, split=HOLDOUT))

    assert decision.gate.economic
    assert decision.gate.undecidable
    assert decision.replication is None
    assert decision.skipped == NO_ESTIMATE
    assert decision.verdict == CLOSED


def test_a_discovery_split_under_the_minimum_is_underpowered_even_carrying_an_estimate() -> None:
    discovery = spread("600", STATION_DAY_MIN - 1, split=DISCOVERY)
    holdout = spread("600", (STATION_DAY_MIN + 1) // 2, split=HOLDOUT)

    decision = decide(discovery, holdout)

    assert decision.verdict == UNDERPOWERED
    assert decision.gate is None
    assert decision.replication is None
    assert discovery.bootstrap is not None


def test_the_convergence_instant_is_the_first_durable_entry_into_the_band() -> None:
    stamps = [at(17, 59), at(18, 0, 30), at(18, 1)]
    mids = [Decimal("0.96"), Decimal("0.60"), Decimal("0.96")]

    found = converged(stamps, mids, side_locked="yes", t_lock=LOCK_AT, window_end=WINDOW_END)

    assert found == at(18, 1)
    assert converged(stamps, mids, side_locked="no", t_lock=LOCK_AT, window_end=WINDOW_END) is None
    assert PERSIST_S == 60
    assert LOCK_BAND == Decimal("0.95")


def test_the_scan_reads_every_leg_the_touch_artifact_carries(tmp_path: Path) -> None:
    scope = load_run_scope(scope_dir(tmp_path))

    scan = scan_locks(scope, artifacts_dir(tmp_path, LADDER_ROWS), ARCHIVE)

    assert scan.markets == 2
    assert scan.cities == (SERIES,)
    assert sorted(event.ticker for event, _ in scan.clean) == [LOWER, ABOVE]
    assert scan.no_observations == 0
    assert scan.no_lock == 0
    assert all(clears_strike(event.crossing_temp_f, event) for event, _ in scan.clean)


# 70.7 separates a 1.0 margin from a smaller one and 71.5 from a larger one, so a detector default
# drifting either way off the margin clears_strike measures against shows up here.
@pytest.mark.parametrize(("temp_f", "ambiguous"), [("70.7", True), ("71.5", False)])
def test_the_margin_the_detector_locks_on_is_the_margin_the_anchor_measures_against(
    temp_f: str, ambiguous: bool
) -> None:
    market = parse_ticker(ABOVE)
    recorded = [reading(at(18, 0), temp_f)]

    carried = detect_lock_events(market, recorded, tz_name=ZONE)
    pinned = detect_lock_events(market, recorded, tz_name=ZONE, rounding_margin_f=ROUNDING_MARGIN_F)

    assert carried == pinned
    assert [event.lock_ambiguous for event in pinned] == [ambiguous]


def test_a_station_day_with_no_recorded_archive_locks_nothing_and_is_counted(
    tmp_path: Path,
) -> None:
    scope = load_run_scope(scope_dir(tmp_path))

    scan = scan_locks(scope, artifacts_dir(tmp_path, LADDER_ROWS), {})

    assert scan.no_observations == 2
    assert scan.clean == ()


def test_the_manifest_lands_before_any_statistic_is_read(tmp_path: Path) -> None:
    paths = run_paths(tmp_path)
    root = tmp_path / "foreign"
    path = write_partition(
        root,
        DISCOVERY_DAY,
        1,
        [],
        schema=pa.schema([("id", pa.int64()), ("ticker", pa.string())]),
    )
    run_root = tmp_path / "tape_studies"

    with pytest.raises(ValueError, match=path.name):
        execute(
            run_id=RUN_ID,
            artifacts=root,
            observations=ARCHIVE,
            arrivals={},
            settles=SETTLES,
            floor_source=FloorSource.SIGNED_READ,
            seed=SEED,
            run_root=run_root,
            **paths,
        )

    assert (run_root / RUN_ID / MANIFEST_NAME).exists()


def test_a_full_run_writes_its_manifest_and_reports_a_json_safe_payload(tmp_path: Path) -> None:
    paths = run_paths(tmp_path)
    run_root = tmp_path / "tape_studies"

    run = execute(
        run_id=RUN_ID,
        artifacts=artifacts_dir(tmp_path, LADDER_ROWS),
        observations=ARCHIVE,
        arrivals={STATION: [reading(at(18, 4), "72", published=at(18, 5))]},
        settles=SETTLES,
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=run_root,
        **paths,
    )

    payload = result_payload(run)

    assert (run_root / RUN_ID / MANIFEST_NAME).exists()
    assert run.decision.verdict == UNDERPOWERED
    assert payload["verdict"] == UNDERPOWERED
    assert payload["arrival_anchor"]["count"] == 2
    assert payload["arrival_anchor"]["gating"] is False
    assert json.loads(json.dumps(payload))["run_id"] == RUN_ID
