import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.lag.ws_book import WsGapError, book_state_at
from bot.replay.inventory import check_excluded
from bot.replay.parity import SamplePoint, compare_points, gap_windows, sample_points


UTC = timezone.utc
T0 = datetime(2026, 7, 19, 4, 59, 0, 155692, tzinfo=UTC)
DB_TS = "%Y-%m-%d %H:%M:%S.%f"
DEN = "KXHIGHDEN-26JUL19-B85"
CHI = "KXHIGHCHI-26JUL19-B75"
BOS = "KXHIGHTBOS-26JUL19-B70"

BOOK_SCHEMA = """
CREATE TABLE ws_book_events (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, received_at DATETIME NOT NULL,
    seq INTEGER NOT NULL, side VARCHAR(8) NOT NULL, price VARCHAR NOT NULL,
    size VARCHAR NOT NULL, is_snapshot BOOLEAN NOT NULL, created_at DATETIME NOT NULL,
    ts_ms INTEGER, PRIMARY KEY (id))
"""

GAP_SCHEMA = """
CREATE TABLE ws_gaps (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, detected_at DATETIME NOT NULL,
    last_seq INTEGER NOT NULL, reason VARCHAR(64) NOT NULL, created_at DATETIME NOT NULL,
    PRIMARY KEY (id))
"""


def at(offset_s: float) -> datetime:
    return T0 + timedelta(seconds=offset_s)


def book(
    row_id: int,
    ticker: str,
    offset_s: float,
    seq: int,
    side: str,
    price: str,
    size: str,
    is_snapshot: bool = False,
) -> tuple[object, ...]:
    stamp = at(offset_s).strftime(DB_TS)
    ts_ms = None if is_snapshot else 1_753_000_000_000 + row_id
    return (row_id, ticker, stamp, seq, side, price, size, int(is_snapshot), stamp, ts_ms)


def gap(row_id: int, ticker: str, offset_s: float) -> tuple[object, ...]:
    stamp = at(offset_s).strftime(DB_TS)
    return (row_id, ticker, stamp, 0, "connection_reset", stamp)


def build_db(
    path: Path,
    rows: list[tuple[object, ...]],
    gaps: list[tuple[object, ...]] | None = None,
) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(BOOK_SCHEMA)
    conn.execute(GAP_SCHEMA)
    conn.execute(
        "CREATE INDEX ix_ws_book_events_ticker_received_at ON ws_book_events (ticker, received_at)"
    )
    conn.executemany("INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.executemany("INSERT INTO ws_gaps VALUES (?, ?, ?, ?, ?, ?)", gaps or [])
    conn.commit()
    conn.close()
    return path


COMPOSED = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
    book(2, DEN, 0.0, 1, "yes", "0.3900", "5.00", True),
    book(3, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
    book(4, DEN, 0.0, 1, "no", "0.5400", "3.00", True),
    book(5, DEN, 2.0, 2, "yes", "0.4100", "2.00"),
    book(6, DEN, 3.0, 3, "yes", "0.3900", "-5.00"),
    book(7, DEN, 4.0, 4, "no", "0.5500", "-2.00"),
    book(8, DEN, 5.0, 5, "no", "0.5300", "8.00"),
]


def point(ticker: str, offset_s: float, kind: str = "interior") -> SamplePoint:
    return SamplePoint(ticker=ticker, t=at(offset_s), kind=kind, cohort="spread")


def test_snapshot_batch_plus_deltas_agrees_on_touch_and_full_ladder(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", COMPOSED)

    result = compare_points(db_path, [point(DEN, 6.0)], [])

    outcome = result.results[0]
    assert outcome.status == "agreed"
    assert outcome.detail == ""
    assert dict(outcome.pass_view.touch) == {
        "yes_bid": "0.4100",
        "yes_ask": "0.4500",
        "no_bid": "0.5500",
        "no_ask": "0.5900",
    }
    assert outcome.pass_view.ladder == outcome.oracle_view.ladder
    assert dict(outcome.pass_view.ladder)["no"] == (
        ("0.5500", "5.00"),
        ("0.5400", "3.00"),
        ("0.5300", "8.00"),
    )
    assert result.counts()["agreed"] == 1


TIED_AT_THE_TOUCH = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
    book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
    book(3, DEN, 0.0, 1, "yes", "0.4000", "2.00"),
]


def test_a_tied_delta_the_oracle_drops_is_reported_with_its_ticker_and_t(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", TIED_AT_THE_TOUCH)

    result = compare_points(db_path, [point(DEN, 0.0)], [], exclude_tied_deltas=False)

    outcome = result.results[0]
    assert outcome.status == "disagreed"
    assert result.disagreements() == (outcome,)
    assert DEN in outcome.detail
    assert at(0.0).isoformat() in outcome.detail
    assert "yes_bid_depth pass=12 oracle=10" in outcome.detail


TIED_BELOW_THE_TOUCH = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
    book(2, DEN, 0.0, 1, "yes", "0.3900", "5.00", True),
    book(3, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
    book(4, DEN, 0.0, 1, "yes", "0.3900", "3.00"),
]


def test_a_deeper_level_disagreement_survives_an_identical_touch(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", TIED_BELOW_THE_TOUCH)

    result = compare_points(db_path, [point(DEN, 0.0)], [], exclude_tied_deltas=False)

    outcome = result.results[0]
    assert outcome.pass_view.touch == outcome.oracle_view.touch
    assert outcome.pass_view.depth == outcome.oracle_view.depth
    assert outcome.status == "disagreed"
    assert dict(outcome.pass_view.ladder)["yes"] == (("0.4000", "10.00"), ("0.3900", "8.00"))
    assert dict(outcome.oracle_view.ladder)["yes"] == (("0.4000", "10.00"), ("0.3900", "5.00"))
    assert "yes_ladder" in outcome.detail


TIED_AND_CLEAN = TIED_AT_THE_TOUCH + [
    book(4, CHI, 1.0, 2, "yes", "0.2000", "4.00", True),
    book(5, CHI, 1.0, 2, "no", "0.7000", "6.00", True),
    book(6, CHI, 2.0, 3, "yes", "0.2100", "1.00"),
]


def test_the_tie_rule_skips_only_the_tied_point_and_counts_it(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", TIED_AND_CLEAN)

    result = compare_points(db_path, [point(DEN, 0.0), point(CHI, 3.0)], [])

    assert [outcome.status for outcome in result.results] == ["excluded_tied_delta", "agreed"]
    assert result.disagreements() == ()
    scalars = result.scalars()
    assert scalars["parity_excluded_tied_delta"] == "1"
    assert scalars["parity_agreed"] == "1"
    assert scalars["parity_points"] == "2"
    assert scalars["parity_compared"] == "1"
    assert all(name.startswith("parity_") for name in scalars)
    assert all(type(value) is str for value in scalars.values())


def test_a_point_past_the_last_recorded_row_is_flagged_blind_and_still_compared(
    tmp_path: Path,
) -> None:
    db_path = build_db(tmp_path / "state.db", COMPOSED)

    result = compare_points(db_path, [point(DEN, 4.5), point(DEN, 5.5, "blind")], [])

    assert [outcome.blind for outcome in result.results] == [False, True]
    assert [outcome.status for outcome in result.results] == ["agreed", "agreed"]
    assert result.scalars()["parity_blind"] == "1"
    assert result.scalars()["parity_compared"] == "2"


GAPPED = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
    book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
    book(3, DEN, 2.0, 2, "yes", "0.4100", "2.00"),
    book(4, DEN, 20.0, 3, "yes", "0.3000", "1.00", True),
    book(5, DEN, 20.0, 3, "no", "0.6000", "4.00", True),
]


def test_both_paths_raise_inside_a_connection_wide_gap(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", GAPPED, [gap(1, "", 11.0)])
    windows = gap_windows(db_path)
    assert [(w.gap_id, w.ticker, w.start, w.end) for w in windows] == [(1, "", at(2.0), at(20.0))]

    with pytest.raises(WsGapError):
        book_state_at(db_path, DEN, at(11.0))
    with pytest.raises(WsGapError):
        check_excluded(windows, DEN, at(11.0))

    result = compare_points(db_path, [point(DEN, 11.0, "post_gap")], windows)

    assert result.results[0].status == "agreed_on_raise"
    assert result.disagreements() == ()
    assert result.scalars()["parity_agreed_on_raise"] == "1"
    assert result.scalars()["parity_compared"] == "0"


def test_a_point_outside_the_gap_window_has_neither_path_raising(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", GAPPED, [gap(1, "", 11.0)])
    windows = gap_windows(db_path)

    assert book_state_at(db_path, DEN, at(2.0)) is not None
    assert book_state_at(db_path, DEN, at(20.0)) is not None
    check_excluded(windows, DEN, at(2.0))
    check_excluded(windows, DEN, at(20.0))

    result = compare_points(db_path, [point(DEN, 2.0), point(DEN, 20.0)], windows)

    assert [outcome.status for outcome in result.results] == ["agreed", "agreed"]


def _tape() -> list[tuple[object, ...]]:
    rows = [
        book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
        book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
        book(3, CHI, 0.5, 2, "yes", "0.2000", "4.00", True),
        book(4, CHI, 0.5, 2, "no", "0.7000", "6.00", True),
        book(5, BOS, 0.8, 3, "yes", "0.1000", "3.00", True),
        book(6, BOS, 0.9, 4, "yes", "0.1000", "1.00"),
    ]
    row_id = 7
    for step in range(40):
        rows.append(book(row_id, DEN, 1.0 + step * 0.1, 5 + step, "yes", "0.4000", "1.00"))
        row_id += 1
        rows.append(book(row_id, CHI, 1.05 + step * 0.1, 45 + step, "yes", "0.2000", "1.00"))
        row_id += 1
    rows.append(book(row_id, DEN, 20.0, 200, "yes", "0.3000", "5.00", True))
    rows.append(book(row_id + 1, CHI, 20.1, 201, "yes", "0.2500", "5.00", True))
    for step in range(10):
        rows.append(book(row_id + 2 + step, DEN, 21.0 + step, 210 + step, "yes", "0.3000", "1.00"))
    return rows


def test_the_sample_is_deterministic_and_covers_every_class(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", _tape(), [gap(1, "", 11.0)])
    windows = gap_windows(db_path)

    first = sample_points(db_path, 40, windows)
    second = sample_points(db_path, 40, windows)

    assert first == second
    assert len(first) == len(set(first))
    assert first == sorted(first, key=lambda p: (p.ticker, p.t, p.kind))
    kinds = {p.kind for p in first}
    assert {"interior", "snapshot", "gap_edge", "post_gap", "blind"} <= kinds
    cohorts = {p.cohort for p in first}
    assert {"chicago", "quiet", "spread", "gap"} <= cohorts
    assert [p.t for p in first if p.kind == "post_gap"] == [at(11.0)]
    assert {p.ticker for p in first if p.cohort == "chicago"} == {CHI}


def test_the_sampled_slice_agrees_everywhere(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", _tape(), [gap(1, "", 11.0)])
    windows = gap_windows(db_path)

    result = compare_points(db_path, sample_points(db_path, 40, windows), windows)

    assert result.disagreements() == ()
    assert result.counts()["agreed_on_raise"] == 1
    assert result.counts()["agreed"] > 20


UNSUBSCRIBED_AT_THE_GAP = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
    book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
    book(3, CHI, 0.5, 2, "yes", "0.2000", "4.00", True),
    book(4, CHI, 0.5, 2, "no", "0.7000", "6.00", True),
    book(5, DEN, 2.0, 3, "yes", "0.4100", "2.00"),
    book(6, BOS, 20.0, 4, "yes", "0.1000", "3.00", True),
    book(7, BOS, 21.0, 5, "yes", "0.1000", "1.00"),
]


def test_a_ticker_scoped_gap_with_nothing_to_probe_names_the_gap(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", UNSUBSCRIBED_AT_THE_GAP, [gap(1, BOS, 11.0)])
    windows = gap_windows(db_path)
    assert [(w.gap_id, w.ticker) for w in windows] == [(1, BOS)]

    with pytest.raises(ValueError) as excinfo:
        sample_points(db_path, 40, windows)

    assert str(excinfo.value) == (
        f"ws gap 1 has no ticker with a snapshot at or before it: ticker={BOS} "
        f"detected_at={at(11.0).strftime(DB_TS)} reason=connection_reset"
    )


FRACTIONAL = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "9.00", True),
    book(2, DEN, 0.0, 1, "no", "0.5500", "7.24", True),
]


def test_depth_is_compared_as_int_and_never_quantized(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", FRACTIONAL)

    result = compare_points(db_path, [point(DEN, 1.0)], [])

    outcome = result.results[0]
    assert outcome.status == "agreed"
    depths = dict(outcome.pass_view.depth)
    assert depths == dict(outcome.oracle_view.depth)
    assert depths["yes_bid_depth"] == 9
    assert depths["no_bid_depth"] == 7
    assert [type(value) for value in depths.values()] == [int, int, int, int]
    assert str(depths["yes_bid_depth"]) == "9"
    assert dict(outcome.pass_view.ladder)["yes"] == (("0.4000", "9.00"),)


def _two_snapshot_tape() -> list[tuple[object, ...]]:
    rows = [
        book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
        book(2, DEN, 0.0, 1, "yes", "0.3900", "5.00", True),
        book(3, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
        book(4, DEN, 0.0, 1, "no", "0.5400", "3.00", True),
    ]
    row_id = 5
    for step in range(30):
        rows.append(book(row_id, DEN, 1.0 + step * 0.1, 10 + step, "yes", "0.4000", "1.00"))
        row_id += 1
    rows.extend(
        [
            book(row_id, DEN, 10.0, 100, "yes", "0.3000", "6.00", True),
            book(row_id + 1, DEN, 10.0, 100, "yes", "0.2900", "2.00", True),
            book(row_id + 2, DEN, 10.0, 100, "no", "0.6000", "4.00", True),
            book(row_id + 3, DEN, 11.0, 101, "yes", "0.3000", "1.00"),
            book(row_id + 4, DEN, 12.0, 102, "no", "0.6000", "-1.00"),
            book(row_id + 5, DEN, 13.0, 103, "no", "0.5900", "3.00"),
            book(row_id + 6, DEN, 20.0, 104, "yes", "0.3100", "2.00"),
        ]
    )
    return rows


def test_the_fold_replays_the_whole_ticker_stream_not_just_the_governing_snapshot(
    tmp_path: Path,
) -> None:
    rows = _two_snapshot_tape()
    db_path = build_db(tmp_path / "state.db", rows)
    stamps = [row[2] for row in rows]
    cutoff = at(15.0).strftime(DB_TS)
    governing = at(10.0).strftime(DB_TS)
    whole_stream = sum(1 for stamp in stamps if stamp <= cutoff)
    from_governing = sum(1 for stamp in stamps if governing <= stamp <= cutoff)
    assert (whole_stream, from_governing) == (40, 6)

    result = compare_points(db_path, [point(DEN, 15.0)], [])

    outcome = result.results[0]
    assert result.rows_read == whole_stream
    assert outcome.blind is False
    assert outcome.status == "agreed"
    assert outcome.pass_view == outcome.oracle_view
    assert dict(outcome.pass_view.ladder)["yes"] == (("0.3000", "7.00"), ("0.2900", "2.00"))


IDS_OUT_OF_ORDER = [
    book(1, DEN, 0.0, 1, "yes", "0.4000", "10.00", True),
    book(2, DEN, 0.0, 1, "no", "0.5500", "7.00", True),
    book(9, DEN, 1.0, 2, "yes", "0.4100", "2.00"),
    book(5, DEN, 2.0, 3, "yes", "0.4200", "1.00"),
]


def test_the_fold_refuses_a_row_whose_id_falls_while_received_at_rises(tmp_path: Path) -> None:
    db_path = build_db(tmp_path / "state.db", IDS_OUT_OF_ORDER)

    with pytest.raises(ValueError) as excinfo:
        compare_points(db_path, [point(DEN, 3.0)], [])

    assert str(excinfo.value) == f"{DEN} id 5 follows id 9 in received_at order"
