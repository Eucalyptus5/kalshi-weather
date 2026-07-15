import gzip
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.taker_side import (
    ABSENT,
    DISCOVERY,
    EMPTY,
    HOLDOUT,
    LEGACY_KEY,
    OUTCOME_KEY,
    PRESENT,
    TRADES_QUERY,
    ExcludedTime,
    Interval,
    ScopeDay,
    Split,
    TradeRow,
    count_trades,
    is_trade_frame,
    key_state,
    read_exclusions,
    read_scope_days,
    read_split,
    read_trade_frames,
    read_trades,
    scoped_tickers,
    split_of,
    taker_direction,
    tally_frames,
)


UTC = timezone.utc
T0 = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)
DB_TS = "%Y-%m-%d %H:%M:%S.%f"

TRADE_SCHEMA = """
CREATE TABLE ws_trades (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, trade_id VARCHAR NOT NULL,
    received_at DATETIME NOT NULL, yes_price VARCHAR NOT NULL, count VARCHAR NOT NULL,
    taker_side VARCHAR(8) NOT NULL, ts_ms INTEGER NOT NULL, PRIMARY KEY (id))
"""
TRADE_INDEX = "CREATE INDEX ix_ws_trades_ticker_received_at ON ws_trades (ticker, received_at)"

ACK_FRAME = {"type": "subscribed", "id": 15, "msg": {"channel": "trade", "sid": 2}}
DELTA_FRAME = {
    "type": "orderbook_delta",
    "sid": 1,
    "seq": 17239090,
    "msg": {
        "market_ticker": "KXHIGHLAX-26JUL25-B79.5",
        "price_dollars": "0.4000",
        "delta_fp": "39.00",
        "side": "no",
        "ts_ms": 1784937599973,
    },
}


def trade_frame(**overrides: object) -> dict:
    msg = {
        "trade_id": "b6a2eb34-e8b7-69b3-93b5-a4aac093205b",
        "market_ticker": "KXHIGHTDEN-26JUL24-B64.5",
        "yes_price_dollars": "0.0100",
        "no_price_dollars": "0.9900",
        "count_fp": "14.76",
        "taker_side": "no",
        "taker_outcome_side": "no",
        "taker_book_side": "ask",
        "ts": 1784937600,
        "ts_ms": 1784937600228,
    }
    msg.update(overrides)
    for key in [k for k, v in msg.items() if v is ...]:
        del msg[key]
    return {"type": "trade", "sid": 2, "seq": 81589, "msg": msg}


def write_tape(path: Path, records: list[tuple[datetime, dict]]) -> Path:
    with gzip.open(path, "wt") as handle:
        for received_at, frame in records:
            raw = json.dumps(frame, separators=(",", ":")) + "\n"
            handle.write(json.dumps({"received_at": received_at.isoformat(), "raw": raw}) + "\n")
    return path


def build_db(path: Path, rows: list[tuple[object, ...]]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(TRADE_SCHEMA)
    conn.execute(TRADE_INDEX)
    conn.executemany("INSERT INTO ws_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def trade_row(row_id: int, ticker: str, trade_id: str, at: datetime, side: str) -> tuple:
    return (row_id, ticker, trade_id, at.strftime(DB_TS), "0.5000", "1.00", side, 0)


def side_row(
    trade_id: str, at: datetime, side: str, ticker: str = "KXHIGHNY-26JUL25-B79.5"
) -> TradeRow:
    return TradeRow(ticker=ticker, trade_id=trade_id, received_at=at, taker_side=side)


def test_subscribe_ack_naming_the_trade_channel_is_not_a_trade_frame() -> None:
    assert "trade" in json.dumps(ACK_FRAME)
    assert is_trade_frame(ACK_FRAME) is False


def test_trade_frame_is_screened_on_its_own_type_field() -> None:
    assert is_trade_frame(trade_frame()) is True
    assert is_trade_frame(DELTA_FRAME) is False


def test_tape_screen_keeps_the_trade_and_drops_the_ack(tmp_path: Path) -> None:
    path = write_tape(
        tmp_path / "day.jsonl.gz",
        [(T0, ACK_FRAME), (T0, DELTA_FRAME), (T0, trade_frame())],
    )

    frames = list(read_trade_frames(path, start=T0, end=T0 + timedelta(hours=1)))

    assert len(frames) == 1
    assert frames[0]["msg"]["market_ticker"] == "KXHIGHTDEN-26JUL24-B64.5"


def test_tape_screen_bounds_are_half_open(tmp_path: Path) -> None:
    end = T0 + timedelta(minutes=10)
    path = write_tape(
        tmp_path / "day.jsonl.gz",
        [
            (T0 - timedelta(microseconds=1), trade_frame(trade_id="before")),
            (T0, trade_frame(trade_id="at-start")),
            (end - timedelta(microseconds=1), trade_frame(trade_id="inside")),
            (end, trade_frame(trade_id="at-end")),
        ],
    )

    frames = list(read_trade_frames(path, start=T0, end=end))

    assert [f["msg"]["trade_id"] for f in frames] == ["at-start", "inside"]


def test_key_state_separates_present_empty_and_absent() -> None:
    assert key_state({OUTCOME_KEY: "no"}, OUTCOME_KEY) == PRESENT
    assert key_state({OUTCOME_KEY: None}, OUTCOME_KEY) == EMPTY
    assert key_state({OUTCOME_KEY: ""}, OUTCOME_KEY) == EMPTY
    assert key_state({LEGACY_KEY: "no"}, OUTCOME_KEY) == ABSENT


def test_tally_separates_key_presence_from_agreement() -> None:
    frames = [
        trade_frame(),
        trade_frame(taker_side="yes", taker_outcome_side="no"),
        trade_frame(taker_outcome_side=...),
        trade_frame(taker_side=...),
        trade_frame(taker_outcome_side=None),
    ]

    tally = tally_frames(frames)

    assert tally.frames == 5
    assert tally.outcome_state == {PRESENT: 3, ABSENT: 1, EMPTY: 1}
    assert tally.legacy_state == {PRESENT: 4, ABSENT: 1}
    assert tally.both_present == 2
    assert tally.disagreed == 1
    assert tally.outcome_values == {"no": 3}
    assert tally.legacy_values == {"no": 3, "yes": 1}
    assert tally.key_sets[",".join(sorted(trade_frame()["msg"]))] == 3


def test_tally_counts_the_kxhigh_root_separately_from_all_frames() -> None:
    tally = tally_frames([trade_frame(), trade_frame(market_ticker="KXLOWTDEN-26JUL24-B64.5")])

    assert tally.frames == 2
    assert tally.high_frames == 1


@pytest.mark.parametrize("value,pressure", [("yes", 1), ("no", -1)])
def test_each_observed_taker_side_maps_to_one_direction(value: str, pressure: int) -> None:
    assert taker_direction(value).yes_pressure == pressure


def test_empty_taker_side_maps_to_no_direction() -> None:
    assert taker_direction("") is None


def test_excluded_time_is_half_open_on_each_interval() -> None:
    start = T0
    end = T0 + timedelta(seconds=10)
    excluded = ExcludedTime([Interval(start=start, end=end)])

    assert excluded.covers(start) is True
    assert excluded.covers(start + timedelta(seconds=5)) is True
    assert excluded.covers(end - timedelta(microseconds=1)) is True
    assert excluded.covers(end) is False
    assert excluded.covers(start - timedelta(microseconds=1)) is False


def test_excluded_time_merges_overlapping_intervals() -> None:
    excluded = ExcludedTime(
        [
            Interval(start=T0 + timedelta(seconds=5), end=T0 + timedelta(seconds=20)),
            Interval(start=T0, end=T0 + timedelta(seconds=10)),
            Interval(start=T0 + timedelta(seconds=40), end=T0 + timedelta(seconds=50)),
        ]
    )

    assert excluded.intervals == (
        Interval(start=T0, end=T0 + timedelta(seconds=20)),
        Interval(start=T0 + timedelta(seconds=40), end=T0 + timedelta(seconds=50)),
    )
    assert excluded.covers(T0 + timedelta(seconds=15)) is True
    assert excluded.covers(T0 + timedelta(seconds=30)) is False


def test_counts_drop_rows_inside_exclusions() -> None:
    excluded = ExcludedTime([Interval(start=T0, end=T0 + timedelta(seconds=10))])
    rows = [
        side_row("a", T0 - timedelta(microseconds=1), "yes"),
        side_row("b", T0, "yes"),
        side_row("c", T0 + timedelta(seconds=5), "no"),
        side_row("d", T0 + timedelta(seconds=10), "no"),
    ]

    counts = count_trades(rows, excluded)

    assert counts.rows == 2
    assert counts.dropped_excluded == 2
    assert counts.side_values == {"yes": 1, "no": 1}


def test_counts_separate_empty_taker_sides() -> None:
    counts = count_trades(
        [side_row("a", T0, ""), side_row("b", T0, "yes"), side_row("c", T0, "no")], ExcludedTime([])
    )

    assert counts.rows == 3
    assert counts.empty_side == 1
    assert counts.side_values == {"": 1, "yes": 1, "no": 1}


def test_duplicate_trade_ids_separate_surplus_rows_from_duplicated_values() -> None:
    counts = count_trades(
        [
            side_row("a", T0, "yes"),
            side_row("a", T0, "yes"),
            side_row("a", T0, "yes"),
            side_row("b", T0, "no"),
            side_row("b", T0, "no"),
            side_row("c", T0, "no"),
        ],
        ExcludedTime([]),
    )

    assert counts.rows == 6
    assert counts.distinct_trade_ids == 3
    assert counts.duplicated_values == 2
    assert counts.surplus_rows == 3


def write_split(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "cities": ["KXHIGHNY", "KXHIGHDEN"],
                "discovery_days": ["2026-07-19", "2026-07-20"],
                "holdout_days": ["2026-07-21"],
                "boundary_event_day": "2026-07-21",
                "scope_start": "2026-07-19T05:00:00+00:00",
                "scope_end": "2026-07-22T08:00:00+00:00",
            }
        )
    )
    return path


def test_read_split_carries_the_frozen_days_and_scope_bounds(tmp_path: Path) -> None:
    split = read_split(write_split(tmp_path / "split.json"))

    assert split.cities == ("KXHIGHNY", "KXHIGHDEN")
    assert split.discovery_days == (date(2026, 7, 19), date(2026, 7, 20))
    assert split.holdout_days == (date(2026, 7, 21),)
    assert split.boundary_event_day == date(2026, 7, 21)
    assert split.scope_start == datetime(2026, 7, 19, 5, tzinfo=UTC)
    assert split.scope_end == datetime(2026, 7, 22, 8, tzinfo=UTC)


def test_split_assignment_matches_the_frozen_day_lists(tmp_path: Path) -> None:
    split = read_split(write_split(tmp_path / "split.json"))

    assert split_of(date(2026, 7, 19), split) == DISCOVERY
    assert split_of(date(2026, 7, 20), split) == DISCOVERY
    assert split_of(date(2026, 7, 21), split) == HOLDOUT
    assert split_of(date(2026, 7, 22), split) == ""


def test_read_scope_days_keeps_only_in_scope_rows(tmp_path: Path) -> None:
    path = tmp_path / "event_days.parquet"
    pq.write_table(
        pa.table(
            {
                "series": ["KXHIGHNY", "KXHIGHNY", "KXHIGHDEN"],
                "event_date": [date(2026, 7, 18), date(2026, 7, 19), date(2026, 7, 19)],
                "window_start": [
                    datetime(2026, 7, 18, 4, tzinfo=UTC),
                    datetime(2026, 7, 19, 4, tzinfo=UTC),
                    datetime(2026, 7, 19, 6, tzinfo=UTC),
                ],
                "window_end": [
                    datetime(2026, 7, 19, 4, tzinfo=UTC),
                    datetime(2026, 7, 20, 4, tzinfo=UTC),
                    datetime(2026, 7, 20, 6, tzinfo=UTC),
                ],
                "tickers": [6, 6, 6],
                "in_scope": [False, True, True],
            }
        ),
        path,
    )

    days = read_scope_days(path)

    assert days == (
        ScopeDay(
            series="KXHIGHDEN",
            event_date=date(2026, 7, 19),
            window_start=datetime(2026, 7, 19, 6, tzinfo=UTC),
            window_end=datetime(2026, 7, 20, 6, tzinfo=UTC),
        ),
        ScopeDay(
            series="KXHIGHNY",
            event_date=date(2026, 7, 19),
            window_start=datetime(2026, 7, 19, 4, tzinfo=UTC),
            window_end=datetime(2026, 7, 20, 4, tzinfo=UTC),
        ),
    )


def test_read_exclusions_merges_the_frozen_intervals(tmp_path: Path) -> None:
    path = tmp_path / "exclusions.parquet"
    pq.write_table(
        pa.table(
            {
                "exclusion_id": [0, 1],
                "exclusion_class": ["resubscribe_blind", "quiet_band"],
                "start": [T0 + timedelta(seconds=5), T0],
                "end": [T0 + timedelta(seconds=20), T0 + timedelta(seconds=10)],
            }
        ),
        path,
    )

    excluded = read_exclusions(path)

    assert excluded.intervals == (Interval(start=T0, end=T0 + timedelta(seconds=20)),)


def test_scoped_tickers_drop_event_days_outside_the_frozen_scope(tmp_path: Path) -> None:
    split = read_split(write_split(tmp_path / "split.json"))
    days = (
        ScopeDay(
            series="KXHIGHNY",
            event_date=date(2026, 7, 19),
            window_start=datetime(2026, 7, 19, 4, tzinfo=UTC),
            window_end=datetime(2026, 7, 20, 4, tzinfo=UTC),
        ),
        ScopeDay(
            series="KXHIGHNY",
            event_date=date(2026, 7, 21),
            window_start=datetime(2026, 7, 21, 4, tzinfo=UTC),
            window_end=datetime(2026, 7, 22, 4, tzinfo=UTC),
        ),
    )

    scoped = scoped_tickers(
        [
            "KXHIGHNY-26JUL19-B79.5",
            "KXHIGHNY-26JUL21-T90",
            "KXHIGHNY-26JUL25-B79.5",
            "KXHIGHDEN-26JUL19-B79.5",
        ],
        days,
        split,
    )

    assert [(s.ticker, s.split) for s in scoped] == [
        ("KXHIGHNY-26JUL19-B79.5", DISCOVERY),
        ("KXHIGHNY-26JUL21-T90", HOLDOUT),
    ]
    assert scoped[0].window_start == datetime(2026, 7, 19, 4, tzinfo=UTC)
    assert scoped[0].window_end == datetime(2026, 7, 20, 4, tzinfo=UTC)


def test_trades_query_is_driven_by_the_ticker_received_at_index(tmp_path: Path) -> None:
    path = build_db(tmp_path / "state.db", [trade_row(1, "KXHIGHNY-26JUL25-B79.5", "a", T0, "yes")])
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    plan = conn.execute("EXPLAIN QUERY PLAN " + TRADES_QUERY, ("x", "a", "b")).fetchall()
    conn.close()

    assert len(plan) == 1
    assert "USING INDEX ix_ws_trades_ticker_received_at" in plan[0][3]
    assert "SCAN ws_trades" not in plan[0][3]


def test_read_trades_bounds_rows_to_the_event_day_window(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-26JUL25-B79.5"
    window_start = datetime(2026, 7, 25, 4, tzinfo=UTC)
    window_end = datetime(2026, 7, 26, 4, tzinfo=UTC)
    path = build_db(
        tmp_path / "state.db",
        [
            trade_row(1, ticker, "before", window_start - timedelta(microseconds=1), "yes"),
            trade_row(2, ticker, "at-start", window_start, "yes"),
            trade_row(3, ticker, "inside", T0, "no"),
            trade_row(4, ticker, "at-end", window_end, "no"),
            trade_row(5, "KXHIGHDEN-26JUL25-B79.5", "other", T0, "no"),
        ],
    )
    scoped = scoped_tickers(
        [ticker],
        (
            ScopeDay(
                series="KXHIGHNY",
                event_date=date(2026, 7, 25),
                window_start=window_start,
                window_end=window_end,
            ),
        ),
        Split(
            cities=("KXHIGHNY",),
            discovery_days=(date(2026, 7, 25),),
            holdout_days=(date(2026, 7, 26),),
            boundary_event_day=date(2026, 7, 26),
            scope_start=window_start,
            scope_end=window_end,
        ),
    )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    rows = list(read_trades(conn, scoped[0]))
    conn.close()

    assert [row.trade_id for row in rows] == ["at-start", "inside"]
    assert rows[1].received_at == T0
    assert rows[1].taker_side == "no"
    assert rows[1].ticker == ticker
