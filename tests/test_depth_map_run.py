import json
import sqlite3
from collections.abc import Iterator, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from bot.lag.depth_map import (
    BUCKET_HOURS,
    FINAL_HOURS,
    REPLENISH_FRACTION,
    SLIPPAGE_BAR_CENTS,
    TOUCH_DEPTH_BAR,
)
from bot.lag.depth_map_run import (
    ALL_LEGS,
    ATM,
    CLOSE_TIMES_QUERY,
    CONFIRMED,
    NO,
    REFUTED,
    REPLENISH_DENOMINATOR,
    REPLENISH_NUMERATOR,
    SLIPPAGE_BAR_UNITS,
    YES,
    DepthSweep,
    PrintTally,
    ResilienceSweep,
    _prints_payload,
    execute,
    load_close_times,
    replenish_median_s,
    result_payload,
    sweep_depth,
    sweep_resilience,
    universe_readout,
)
from bot.lag.ladder_consistency import PRICE_TICKS
from bot.lag.r0_universe import Coverage, freeze_universe, write_universe
from bot.lag.read_rtt import FloorSource
from bot.lag.tape_studies import load_run_scope
from bot.replay.artifacts import LADDER_SCHEMA, TRADES_SCHEMA
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    RESUBSCRIBE_BLIND,
    Split,
    write_split,
)
from bot.storage.sqlite import Market
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
PRICE = Decimal("0.0001")
SIZE = Decimal("0.01")

SERIES = "KXHIGHDEN"
EVENT_DATE = date(2026, 7, 18)
HOLDOUT_DAY = date(2026, 7, 19)
LEG_A = "KXHIGHDEN-26JUL18-B70.5"
LEG_B = "KXHIGHDEN-26JUL18-B72.5"
LEGS = (LEG_A, LEG_B)
WINDOW_START = datetime(2026, 7, 18, tzinfo=UTC)
WINDOW_END = WINDOW_START + timedelta(days=1)
SCOPE_END = datetime(2026, 7, 20, tzinfo=UTC)
CLOSE = WINDOW_END
EARLY_CLOSE = WINDOW_START + timedelta(hours=6)

RUN_ID = "2026-08-13-d2"
SEED = 20260813

YES_BID = "0.40"
NO_BID = "0.58"
DEEP = "500"
THIN = "1"
FINAL_HOUR = 20
SIX_LEVELS = (
    ("0.58", "1000"),
    ("0.57", "1"),
    ("0.56", "1"),
    ("0.55", "1"),
    ("0.54", "1"),
    ("0.53", "1"),
)


def stamp(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 7, 18, hour, minute, second, tzinfo=UTC)


def price(value: str) -> str:
    return str(Decimal(value).quantize(PRICE))


def size(value: str) -> str:
    return str(Decimal(value).quantize(SIZE))


def ladder_row(
    row_id: int,
    ticker: str,
    at: datetime,
    yes: Sequence[tuple[str, str]],
    no: Sequence[tuple[str, str]],
    *,
    yes_levels: int | None = None,
    no_levels: int | None = None,
) -> dict:
    yes_bid, yes_depth = yes[0] if yes else ("0", "0")
    no_bid, no_depth = no[0] if no else ("0", "0")
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at,
        "ts_ms": None,
        "yes_bid": price(yes_bid),
        "yes_bid_depth": size(yes_depth),
        "yes_ask": price(str(Decimal("1") - Decimal(no_bid))),
        "yes_ask_depth": size(no_depth),
        "no_bid": price(no_bid),
        "no_bid_depth": size(no_depth),
        "no_ask": price(str(Decimal("1") - Decimal(yes_bid))),
        "no_ask_depth": size(yes_depth),
        "yes_prices": [price(level) for level, _ in yes],
        "yes_sizes": [size(depth) for _, depth in yes],
        "yes_levels": len(yes) if yes_levels is None else yes_levels,
        "no_prices": [price(level) for level, _ in no],
        "no_sizes": [size(depth) for _, depth in no],
        "no_levels": len(no) if no_levels is None else no_levels,
    }


def flat(row_id: int, ticker: str, at: datetime, yes_depth: str, no_depth: str = DEEP) -> dict:
    return ladder_row(row_id, ticker, at, [(YES_BID, yes_depth)], [(NO_BID, no_depth)])


def trade_row(row_id: int, ticker: str, at: datetime, yes_price: str) -> dict:
    return {
        "id": row_id,
        "ticker": ticker,
        "received_at": at,
        "ts_ms": None,
        "yes_price": price(yes_price),
        "no_price": price(str(Decimal("1") - Decimal(yes_price))),
        "count": size("3"),
        "taker_side": "yes",
        "trade_id": f"t{row_id}",
    }


def ladder_artifacts(
    tmp_path: Path,
    rows: Sequence[dict],
    *,
    name: str = "artifacts",
    trades: Sequence[dict] = (),
    barriers: int = 1,
) -> Path:
    root = tmp_path / name
    step = -(-len(rows) // barriers)
    for barrier in range(barriers):
        write_partition(
            root,
            EVENT_DATE,
            barrier + 1,
            list(rows[barrier * step : (barrier + 1) * step]),
            kind="ladder",
            schema=LADDER_SCHEMA,
            series=SERIES,
        )
    if trades:
        write_partition(
            root, EVENT_DATE, 1, list(trades), kind="trades", schema=TRADES_SCHEMA, series=SERIES
        )
    return root


def days_table() -> pa.Table:
    rows = [
        event_day_row(EVENT_DATE, in_scope=True, split=DISCOVERY, day_index=1) | {"series": SERIES},
        event_day_row(HOLDOUT_DAY, in_scope=True, split=HOLDOUT, day_index=2) | {"series": SERIES},
    ]
    return pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA)


def exclusion_table(spans: Sequence[tuple[datetime, datetime]] = ()) -> pa.Table:
    rows = [
        exclusion_row(index, RESUBSCRIBE_BLIND, start, end)
        for index, (start, end) in enumerate(spans)
    ]
    return pa.Table.from_pylist(rows, schema=EXCLUSIONS_SCHEMA)


def scope_dir(tmp_path: Path, *, name: str = "scope", exclusions: pa.Table | None = None) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    pq.write_table(
        exclusion_table() if exclusions is None else exclusions, directory / "exclusions.parquet"
    )
    pq.write_table(days_table(), directory / "event_days.parquet")
    write_split(
        directory / "split.json",
        Split(
            cities=(SERIES,),
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
            passing=(SERIES,),
            coverage=Coverage(cities=(SERIES,), ladder_widths=(6,), in_scope_city_days=2),
        ),
    )
    return directory


def markets_db(
    tmp_path: Path,
    *,
    name: str = "state.db",
    tickers: Sequence[str] = LEGS,
    close: datetime | None = CLOSE,
) -> Path:
    path = tmp_path / name
    engine = create_engine(f"sqlite:///{path}")
    Market.__table__.create(engine)
    with Session(engine) as session:
        for ticker in tickers:
            session.add(
                Market(
                    ticker=ticker,
                    series=SERIES,
                    event_date=EVENT_DATE,
                    is_monthly=False,
                    is_tail=False,
                    strike_low=Decimal("70.5"),
                    strike_high=None,
                    close_time=close,
                    status="active",
                    last_seen_at=WINDOW_START,
                )
            )
        session.commit()
    engine.dispose()
    return path


def run_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
        "db": markets_db(tmp_path),
    }


def swept(tmp_path: Path, rows: Sequence[dict], *, name: str = "artifacts") -> DepthSweep:
    scope = load_run_scope(scope_dir(tmp_path))
    artifacts = ladder_artifacts(tmp_path, rows, name=name)
    return sweep_depth(scope, artifacts, load_close_times(markets_db(tmp_path), scope))


def resilience_of(
    tmp_path: Path, rows: Sequence[dict], trades: Sequence[dict], name: str
) -> ResilienceSweep:
    scope = load_run_scope(scope_dir(tmp_path))
    artifacts = ladder_artifacts(tmp_path, rows, name=name, trades=trades)
    closes = load_close_times(markets_db(tmp_path), scope)
    sweep = sweep_depth(scope, artifacts, closes)
    return sweep_resilience(scope, artifacts, closes, sweep.picks)


def cube_of(sweep: DepthSweep) -> dict:
    return {
        key: (cell.kept_s, cell.touch.p25, cell.touch.p50, cell.touch.p75)
        for key, cell in universe_readout(sweep, ALL_LEGS).cube.items()
    }


def payload_keys(payload: object) -> Iterator[str]:
    if isinstance(payload, dict):
        for name, value in payload.items():
            yield name
            yield from payload_keys(value)
    if isinstance(payload, list):
        for value in payload:
            yield from payload_keys(value)


THIN_ROWS = (
    flat(1, LEG_A, stamp(10), THIN),
    flat(2, LEG_B, stamp(10), THIN),
    flat(3, LEG_A, stamp(11), "200"),
    flat(4, LEG_A, stamp(11, 0, 1), THIN),
)

DEEP_ROWS = (
    flat(1, LEG_A, stamp(FINAL_HOUR - 2), DEEP),
    flat(2, LEG_A, stamp(FINAL_HOUR, 0, 5), "100"),
    flat(3, LEG_A, stamp(FINAL_HOUR, 0, 35), "300"),
    flat(4, LEG_A, stamp(FINAL_HOUR, 1), DEEP),
)
DEEP_TRADES = (trade_row(1, LEG_A, stamp(FINAL_HOUR), YES_BID),)


def test_the_applied_bars_are_the_preregistered_ones() -> None:
    assert SLIPPAGE_BAR_UNITS == 100
    assert (REPLENISH_NUMERATOR, REPLENISH_DENOMINATOR) == (1, 2)
    assert SLIPPAGE_BAR_CENTS * PRICE_TICKS == SLIPPAGE_BAR_UNITS * 100
    assert REPLENISH_FRACTION * REPLENISH_DENOMINATOR == REPLENISH_NUMERATOR


def test_load_close_times_reads_every_in_scope_leg(tmp_path: Path) -> None:
    scope = load_run_scope(scope_dir(tmp_path))

    closes = load_close_times(markets_db(tmp_path), scope)

    assert closes == {LEG_A: CLOSE, LEG_B: CLOSE}


def test_the_close_time_seek_rides_the_series_event_date_index(tmp_path: Path) -> None:
    db = markets_db(tmp_path)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        plan = conn.execute(
            f"EXPLAIN QUERY PLAN {CLOSE_TIMES_QUERY}", (SERIES, EVENT_DATE.isoformat())
        ).fetchall()
    finally:
        conn.close()

    assert len(plan) == 1
    assert "SEARCH markets USING INDEX ix_markets_series_event_date" in plan[0][3]


def test_a_null_close_time_is_refused(tmp_path: Path) -> None:
    scope = load_run_scope(scope_dir(tmp_path))
    db = markets_db(tmp_path, name="null.db", tickers=(LEG_A,), close=None)

    with pytest.raises(ValueError, match=LEG_A):
        load_close_times(db, scope)


def test_a_leg_with_no_market_row_stops_the_sweep(tmp_path: Path) -> None:
    scope = load_run_scope(scope_dir(tmp_path))
    artifacts = ladder_artifacts(tmp_path, THIN_ROWS)
    closes = load_close_times(markets_db(tmp_path, name="partial.db", tickers=(LEG_A,)), scope)

    with pytest.raises(ValueError, match=LEG_B):
        sweep_depth(scope, artifacts, closes)


def test_hours_at_one_contract_outweigh_more_states_at_two_hundred(tmp_path: Path) -> None:
    rows = (
        flat(1, LEG_A, stamp(21), THIN),
        flat(2, LEG_A, stamp(22, 0, 0), "200"),
        flat(3, LEG_A, stamp(22, 0, 1), "200"),
        flat(4, LEG_A, stamp(22, 0, 2), "200"),
        flat(5, LEG_A, stamp(22, 0, 3), "200"),
        flat(6, LEG_A, stamp(22, 0, 4), THIN),
    )
    sweep = swept(tmp_path, rows)

    cell = universe_readout(sweep, ALL_LEGS).by_city[(SERIES, YES)]

    assert sweep.cells == 7
    assert cell.kept_s == Decimal("10800")
    assert cell.touch.p50 == Decimal("1")
    assert cell.touch.p75 == Decimal("1")


def test_the_at_the_money_universe_is_not_every_leg(tmp_path: Path) -> None:
    rows = (flat(1, LEG_A, stamp(10), THIN), flat(2, LEG_B, stamp(10), "300"))
    sweep = swept(tmp_path, rows)

    all_legs = universe_readout(sweep, ALL_LEGS).by_city[(SERIES, YES)]
    atm = universe_readout(sweep, ATM).by_city[(SERIES, YES)]

    assert sweep.picks == {(SERIES, EVENT_DATE): LEG_A}
    assert all_legs.kept_s == atm.kept_s * 2 == Decimal("100800")
    assert all_legs.touch.p75 == Decimal("300")
    assert atm.touch.p75 == Decimal("1")


def test_a_snapshot_batch_contributes_only_its_final_book(tmp_path: Path) -> None:
    rows = (
        flat(1, LEG_A, stamp(10), "5"),
        flat(2, LEG_A, stamp(10), "20"),
        flat(3, LEG_A, stamp(10), "300"),
        flat(4, LEG_A, stamp(11), THIN),
    )

    readout = universe_readout(swept(tmp_path, rows), ALL_LEGS)

    assert readout.by_hour[(10, YES)].touch.p50 == Decimal("300")
    assert readout.by_hour[(10, YES)].kept_s == Decimal("3600")


def test_a_state_outliving_an_hour_boundary_is_split_across_both_hours(tmp_path: Path) -> None:
    rows = (flat(1, LEG_A, stamp(10, 30), "7"), flat(2, LEG_A, stamp(12, 15), "9"))

    readout = universe_readout(swept(tmp_path, rows), ALL_LEGS)

    assert readout.by_hour[(10, YES)].kept_s == Decimal("1800")
    assert readout.by_hour[(11, YES)].kept_s == Decimal("3600")
    assert readout.by_hour[(11, YES)].touch.p50 == Decimal("7")


def test_a_cell_opening_on_a_bucket_break_lands_in_the_newer_bucket(tmp_path: Path) -> None:
    rows = (flat(1, LEG_A, stamp(17, 30), "5"), flat(2, LEG_A, stamp(18), "9"))

    readout = universe_readout(swept(tmp_path, rows), ALL_LEGS)

    assert readout.by_bucket[(1, YES)].kept_s == Decimal("1800")
    assert readout.by_bucket[(0, YES)].kept_s == Decimal("21600")
    assert readout.by_bucket[(0, YES)].touch.p50 == Decimal("9")


def test_a_cell_straddling_an_exclusion_is_dropped_whole(tmp_path: Path) -> None:
    rows = (flat(1, LEG_A, stamp(10), "3"), flat(2, LEG_A, stamp(11), "9"))
    scope = load_run_scope(
        scope_dir(
            tmp_path, name="blink", exclusions=exclusion_table([(stamp(10, 20), stamp(10, 40))])
        )
    )
    artifacts = ladder_artifacts(tmp_path, rows)

    sweep = sweep_depth(scope, artifacts, load_close_times(markets_db(tmp_path), scope))

    assert universe_readout(sweep, ALL_LEGS).by_hour.get((10, YES)) is None
    assert sweep.screened.excluded == 1
    assert sweep.screened.by_class[RESUBSCRIBE_BLIND] == 1
    assert sweep.screened.excluded_fraction == Decimal(1) / Decimal(sweep.screened.candidates)


def test_a_cell_abutting_an_exclusion_endpoint_is_dropped(tmp_path: Path) -> None:
    rows = (flat(1, LEG_A, stamp(10), "3"), flat(2, LEG_A, stamp(11), "9"))
    scope = load_run_scope(
        scope_dir(
            tmp_path,
            name="abut",
            exclusions=exclusion_table([(stamp(11), stamp(11) + MICROSECOND)]),
        )
    )
    artifacts = ladder_artifacts(tmp_path, rows)

    sweep = sweep_depth(scope, artifacts, load_close_times(markets_db(tmp_path), scope))

    readout = universe_readout(sweep, ALL_LEGS)
    assert readout.by_hour.get((10, YES)) is None
    assert readout.by_hour.get((11, YES)) is None
    assert sweep.screened.excluded == 2


def test_a_state_running_past_the_close_is_counted_past_the_close(tmp_path: Path) -> None:
    rows = (flat(1, LEG_A, stamp(5), "4"), flat(2, LEG_A, stamp(7), "8"))
    scope = load_run_scope(scope_dir(tmp_path))
    artifacts = ladder_artifacts(tmp_path, rows)
    closes = load_close_times(markets_db(tmp_path, name="early.db", close=EARLY_CLOSE), scope)

    sweep = sweep_depth(scope, artifacts, closes)

    readout = universe_readout(sweep, ALL_LEGS)
    assert readout.by_hour[(5, YES)].kept_s == Decimal("3600")
    assert readout.by_hour.get((6, YES)) is None
    assert sweep.screened.past_close == 1


def test_the_tallies_do_not_move_when_the_rows_are_split_across_files(tmp_path: Path) -> None:
    rows = [
        flat(index + 1, LEG_A, stamp(10) + timedelta(minutes=17 * index), str(index + 1))
        for index in range(9)
    ]
    scope = load_run_scope(scope_dir(tmp_path))
    closes = load_close_times(markets_db(tmp_path), scope)

    one = sweep_depth(scope, ladder_artifacts(tmp_path, rows, name="one"), closes)
    three = sweep_depth(scope, ladder_artifacts(tmp_path, rows, name="three", barriers=3), closes)

    assert len(list((tmp_path / "three" / "ladder").glob("*.parquet"))) == 3
    assert cube_of(one) == cube_of(three)
    assert one.screened.kept_us == three.screened.kept_us
    assert one.screened.candidates == three.screened.candidates


def test_a_one_sided_book_is_not_a_two_sided_quote(tmp_path: Path) -> None:
    rows = (
        ladder_row(1, LEG_A, stamp(10), [(YES_BID, "5")], []),
        ladder_row(2, LEG_A, stamp(11), [(YES_BID, "5")], []),
    )

    readout = universe_readout(swept(tmp_path, rows), ALL_LEGS)

    assert rows[0]["yes_ask"] == "1.0000"
    assert rows[0]["yes_ask_depth"] == "0.00"
    assert readout.by_hour[(10, YES)].two_sided_fraction == Decimal(0)
    assert readout.by_hour[(10, YES)].spread_p50_cents is None


def test_a_ladder_deeper_than_the_stored_width_is_censored(tmp_path: Path) -> None:
    rows = (
        ladder_row(1, LEG_A, stamp(10), [(YES_BID, "5")], SIX_LEVELS, no_levels=8),
        ladder_row(2, LEG_A, stamp(11), [(YES_BID, "5")], SIX_LEVELS, no_levels=8),
    )

    cell = universe_readout(swept(tmp_path, rows), ALL_LEGS).by_hour[(10, NO)]

    assert cell.censored_s == Decimal("3600")
    assert cell.censored_p50 == Decimal("1005")


def test_a_ladder_inside_the_stored_width_is_not_censored(tmp_path: Path) -> None:
    rows = (
        ladder_row(1, LEG_A, stamp(10), [(YES_BID, "5")], SIX_LEVELS),
        ladder_row(2, LEG_A, stamp(11), [(YES_BID, "5")], SIX_LEVELS),
    )

    cell = universe_readout(swept(tmp_path, rows), ALL_LEGS).by_hour[(10, NO)]

    assert cell.censored_s == Decimal(0)
    assert cell.censored_p50 is None
    assert cell.capacity.p50 == Decimal("1005")


def test_a_side_shallower_than_the_stored_width_is_not_censored(tmp_path: Path) -> None:
    rows = (
        ladder_row(1, LEG_A, stamp(10), [(YES_BID, "5")], SIX_LEVELS[:5]),
        ladder_row(2, LEG_A, stamp(11), [(YES_BID, "5")], SIX_LEVELS[:5]),
    )

    cell = universe_readout(swept(tmp_path, rows), ALL_LEGS).by_hour[(10, NO)]

    assert cell.censored_s == Decimal(0)
    assert cell.censored_p50 is None
    assert cell.capacity.p50 == Decimal("1004")


def test_a_print_that_halves_the_touch_and_refills_inside_the_window_replenishes(
    tmp_path: Path,
) -> None:
    result = resilience_of(tmp_path, DEEP_ROWS, DEEP_TRADES, "refill")

    tally = result.universes[ALL_LEGS].tallies[(SERIES, 0)]
    assert tally.matched == 1
    assert tally.events == 1
    assert tally.replenished == 1
    assert tally.times_us == [35_000_000]


def test_a_refill_after_the_window_does_not_replenish(tmp_path: Path) -> None:
    rows = (
        flat(1, LEG_A, stamp(FINAL_HOUR - 2), DEEP),
        flat(2, LEG_A, stamp(FINAL_HOUR, 0, 5), "100"),
        flat(3, LEG_A, stamp(FINAL_HOUR, 1, 35), "300"),
    )

    result = resilience_of(tmp_path, rows, DEEP_TRADES, "late")

    tally = result.universes[ALL_LEGS].tallies[(SERIES, 0)]
    assert tally.events == 1
    assert tally.replenished == 0
    assert tally.times_us == []


def test_a_print_on_the_bucket_break_lands_in_the_final_bucket(tmp_path: Path) -> None:
    rows = (
        flat(1, LEG_A, stamp(FINAL_HOUR - 2), DEEP),
        flat(2, LEG_A, stamp(FINAL_HOUR - 2, 0, 5), "100"),
    )
    trades = (trade_row(1, LEG_A, stamp(FINAL_HOUR - 2), YES_BID),)

    result = resilience_of(tmp_path, rows, trades, "break")

    assert stamp(FINAL_HOUR - 2) == CLOSE - timedelta(hours=FINAL_HOURS)
    assert result.universes[ALL_LEGS].tallies[(SERIES, 0)].events == 1
    assert (SERIES, 1) not in result.universes[ALL_LEGS].tallies


def test_a_print_that_leaves_the_touch_above_half_is_no_resilience_event(tmp_path: Path) -> None:
    rows = (
        flat(1, LEG_A, stamp(FINAL_HOUR - 2), DEEP),
        flat(2, LEG_A, stamp(FINAL_HOUR, 0, 5), "300"),
    )

    result = resilience_of(tmp_path, rows, DEEP_TRADES, "shallow")

    tally = result.universes[ALL_LEGS].tallies[(SERIES, 0)]
    assert tally.matched == 1
    assert tally.events == 0


def test_a_print_at_neither_touch_is_counted_and_dropped(tmp_path: Path) -> None:
    trades = (trade_row(1, LEG_A, stamp(FINAL_HOUR), "0.45"),)

    result = resilience_of(tmp_path, DEEP_ROWS, trades, "stray")

    assert result.universes[ALL_LEGS].accounting.prints == 1
    assert result.universes[ALL_LEGS].accounting.no_match == 1
    assert result.universes[ALL_LEGS].tallies == {}


def test_several_trades_between_two_states_are_one_print(tmp_path: Path) -> None:
    trades = (
        trade_row(1, LEG_A, stamp(FINAL_HOUR), YES_BID),
        trade_row(2, LEG_A, stamp(FINAL_HOUR, 0, 1), YES_BID),
        trade_row(3, LEG_A, stamp(FINAL_HOUR, 0, 2), YES_BID),
    )

    result = resilience_of(tmp_path, DEEP_ROWS, trades, "burst")

    assert result.universes[ALL_LEGS].accounting.prints == 1
    assert result.universes[ALL_LEGS].tallies[(SERIES, 0)].events == 1


def test_the_replenish_median_is_the_crossing_and_is_declined_under_censoring() -> None:
    crossed = PrintTally(matched=3, events=3, replenished=2, times_us=[10_000_000, 40_000_000])
    censored = PrintTally(matched=4, events=4, replenished=2, times_us=[10_000_000, 40_000_000])

    assert replenish_median_s(crossed) == Decimal("40")
    assert _prints_payload(crossed)["median_replenish_s"] == "40"
    assert replenish_median_s(censored) is None
    assert _prints_payload(censored)["median_replenish_s"] is None
    assert _prints_payload(censored)["replenished_fraction"] == "0.5"


def test_a_thin_book_confirms_the_small_capacity_reading(tmp_path: Path) -> None:
    run = execute(
        run_id=RUN_ID,
        artifacts=ladder_artifacts(tmp_path, THIN_ROWS),
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **run_paths(tmp_path),
    )

    payload = result_payload(run)

    assert run.verdict == CONFIRMED
    assert payload["verdict"] == CONFIRMED
    assert json.loads(json.dumps(payload)) == payload
    assert payload["run_id"] == RUN_ID
    assert payload["cities"] == [SERIES]
    assert payload["atm_ticker_per_city_day"] == {f"{SERIES} {EVENT_DATE.isoformat()}": LEG_A}
    assert payload["no_atm_leg"] == [f"{SERIES} {HOLDOUT_DAY.isoformat()}"]
    assert Decimal(payload["bars"]["touch_depth_bar_contracts"]) == TOUCH_DEPTH_BAR
    assert payload["bars"]["bucket_hours"] == BUCKET_HOURS


def test_a_deep_book_that_replenishes_refutes_the_small_capacity_reading(tmp_path: Path) -> None:
    run = execute(
        run_id=RUN_ID,
        artifacts=ladder_artifacts(tmp_path, DEEP_ROWS, trades=DEEP_TRADES),
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **run_paths(tmp_path),
    )

    payload = result_payload(run)

    assert run.verdict == REFUTED
    assert payload["headline"]["universe"] == ATM
    assert payload["headline"]["median_touch_depth_contracts"] == {YES: "500", NO: "500"}
    assert payload["headline"]["resilience"]["events"] == 1
    assert payload["headline"]["resilience"]["replenished"] == 1
    assert payload["headline"]["resilience"]["median_replenish_s"] == "35"


def test_the_reading_carries_no_inference_field(tmp_path: Path) -> None:
    run = execute(
        run_id=RUN_ID,
        artifacts=ladder_artifacts(tmp_path, THIN_ROWS),
        floor_source=FloorSource.SIGNED_READ,
        seed=SEED,
        run_root=tmp_path / "tape_studies",
        **run_paths(tmp_path),
    )

    names = set(payload_keys(result_payload(run)))

    assert names.isdisjoint({"interval", "ci_level", "holdout", "replication", "resamples"})
    assert not any("bootstrap" in name or "p_value" in name for name in names)
