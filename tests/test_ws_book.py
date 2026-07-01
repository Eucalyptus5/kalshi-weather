from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from bot.lag.event_study import OrderbookSnapshotRow, median_int
from bot.lag.ws_book import WsGapError, book_state_at, open_book_db, probe_event
from bot.storage.sqlite import Base, WsBookEvent, WsGap, make_engine, make_session_factory


UTC = timezone.utc
T0 = datetime(2026, 6, 20, 15, 0, 0, tzinfo=UTC)
TICKER = "KXHIGHDEN-26JUN20-T85"
OFFSETS = (30, 60, 90, 120)


@pytest.fixture
def db(tmp_path: Path) -> tuple[Path, sessionmaker[Session]]:
    path = tmp_path / "state.db"
    engine = make_engine(path)
    Base.metadata.create_all(engine)
    return path, make_session_factory(engine)


def _snapshot(
    factory: sessionmaker[Session],
    *,
    received_at: datetime,
    seq: int,
    yes: list[tuple[str, str]],
    no: list[tuple[str, str]],
    ticker: str = TICKER,
) -> None:
    rows = [
        WsBookEvent(
            ticker=ticker,
            received_at=received_at,
            seq=seq,
            side=side,
            price=Decimal(price),
            size=Decimal(size),
            is_snapshot=True,
        )
        for side, levels in (("yes", yes), ("no", no))
        for price, size in levels
    ]
    with factory() as session:
        session.add_all(rows)
        session.commit()


def _delta(
    factory: sessionmaker[Session],
    *,
    received_at: datetime,
    seq: int,
    side: str,
    price: str,
    delta: str,
    ticker: str = TICKER,
) -> None:
    with factory() as session:
        session.add(
            WsBookEvent(
                ticker=ticker,
                received_at=received_at,
                seq=seq,
                side=side,
                price=Decimal(price),
                size=Decimal(delta),
                is_snapshot=False,
            )
        )
        session.commit()


def _gap(
    factory: sessionmaker[Session],
    *,
    detected_at: datetime,
    ticker: str,
    reason: str = "seq_skip",
) -> None:
    with factory() as session:
        session.add(WsGap(ticker=ticker, detected_at=detected_at, last_seq=0, reason=reason))
        session.commit()


def _seed_base_book(factory: sessionmaker[Session]) -> None:
    _snapshot(
        factory,
        received_at=T0,
        seq=10,
        yes=[("0.400000", "50"), ("0.350000", "20")],
        no=[("0.500000", "30"), ("0.450000", "10")],
    )


def test_snapshot_plus_deltas_compose_to_pinned_book(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=10),
        seq=11,
        side="yes",
        price="0.400000",
        delta="25",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=20),
        seq=12,
        side="no",
        price="0.500000",
        delta="-30",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=30),
        seq=13,
        side="no",
        price="0.480000",
        delta="40",
    )

    t = T0 + timedelta(seconds=60)
    row = book_state_at(path, TICKER, t)

    assert row is not None
    assert row.ticker == TICKER
    assert row.snapshot_at == t
    assert row.yes_bid == Decimal("0.400000")
    assert row.yes_bid_depth == 75
    assert row.no_bid == Decimal("0.480000")
    assert row.no_bid_depth == 40
    assert row.yes_ask == Decimal("0.520000")
    assert row.yes_ask_depth == 40
    assert row.no_ask == Decimal("0.600000")
    assert row.no_ask_depth == 75


def test_delta_after_t_not_applied(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=30),
        seq=11,
        side="yes",
        price="0.400000",
        delta="25",
    )

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=10))

    assert row is not None
    assert row.yes_bid_depth == 50


def test_delta_at_exactly_t_applied(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=30),
        seq=11,
        side="yes",
        price="0.400000",
        delta="25",
    )

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert row is not None
    assert row.yes_bid_depth == 75


def test_later_snapshot_supersedes_earlier(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=10),
        seq=11,
        side="yes",
        price="0.400000",
        delta="25",
    )
    _snapshot(
        factory,
        received_at=T0 + timedelta(seconds=20),
        seq=20,
        yes=[("0.300000", "5")],
        no=[("0.600000", "7")],
    )

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert row is not None
    assert row.yes_bid == Decimal("0.300000")
    assert row.yes_bid_depth == 5
    assert row.no_bid == Decimal("0.600000")
    assert row.no_bid_depth == 7


def test_gap_for_ticker_in_window_raises(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=10), ticker=TICKER)

    with pytest.raises(ValueError, match=TICKER):
        book_state_at(path, TICKER, T0 + timedelta(seconds=30))


def test_connection_level_gap_in_window_raises(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=10), ticker="", reason="connection_reset")

    with pytest.raises(ValueError, match=TICKER):
        book_state_at(path, TICKER, T0 + timedelta(seconds=30))


def test_gap_before_snapshot_does_not_raise(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _gap(factory, detected_at=T0 - timedelta(seconds=10), ticker=TICKER)

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert row is not None
    assert row.yes_bid == Decimal("0.400000")


def test_gap_for_other_ticker_does_not_raise(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=10), ticker="KXHIGHNY-26JUN20-T90")

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert row is not None
    assert row.yes_bid == Decimal("0.400000")


def test_no_snapshot_returns_none(db) -> None:
    path, factory = db
    _delta(factory, received_at=T0, seq=1, side="yes", price="0.400000", delta="25")

    assert book_state_at(path, TICKER, T0 + timedelta(seconds=30)) is None


def test_negative_level_raises(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=10),
        seq=11,
        side="yes",
        price="0.400000",
        delta="-60",
    )

    with pytest.raises(ValueError, match=TICKER):
        book_state_at(path, TICKER, T0 + timedelta(seconds=30))


def test_row_shape_inversion_identities(db) -> None:
    path, factory = db
    _seed_base_book(factory)

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert row is not None
    assert row.yes_ask == Decimal("1") - row.no_bid
    assert row.no_ask == Decimal("1") - row.yes_bid
    assert row.yes_ask_depth == row.no_bid_depth
    assert row.no_ask_depth == row.yes_bid_depth


def test_empty_no_side_maps_to_yes_ask_one_zero_depth(db) -> None:
    path, factory = db
    _snapshot(factory, received_at=T0, seq=10, yes=[("0.400000", "50")], no=[])

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert row is not None
    assert row.no_bid == Decimal("0")
    assert row.no_bid_depth == 0
    assert row.yes_ask == Decimal("1")
    assert row.yes_ask_depth == 0


def test_empty_yes_side_maps_to_no_ask_one_zero_depth(db) -> None:
    path, factory = db
    _snapshot(factory, received_at=T0, seq=10, yes=[], no=[("0.500000", "30")])

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert row is not None
    assert row.yes_bid == Decimal("0")
    assert row.yes_bid_depth == 0
    assert row.no_ask == Decimal("1")
    assert row.no_ask_depth == 0


def test_decimal_scale_mirrored_field_wise(db) -> None:
    path, factory = db
    _snapshot(
        factory,
        received_at=T0,
        seq=10,
        yes=[("0.500000", "12")],
        no=[("0.500000", "9")],
    )

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert row is not None
    assert str(row.yes_bid) == "0.500000"
    assert str(row.no_bid) == "0.500000"
    assert str(row.yes_ask) == str(Decimal("1") - Decimal("0.500000"))
    assert str(row.yes_ask) == "0.500000"
    assert str(row.no_ask) == "0.500000"


def test_returned_row_feeds_consumers_unmodified(db) -> None:
    path, factory = db
    _seed_base_book(factory)

    row = book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert isinstance(row, OrderbookSnapshotRow)
    assert row.ticker is not None
    assert row.snapshot_at is not None
    assert row.yes_bid is not None
    assert row.yes_ask is not None
    assert row.no_bid is not None
    assert row.no_ask is not None
    assert row.yes_ask_depth is not None
    assert row.yes_bid_depth is not None
    assert row.no_ask_depth is not None
    assert row.no_bid_depth is not None


def _seed_single_level_book(factory: sessionmaker[Session]) -> None:
    _snapshot(
        factory,
        received_at=T0,
        seq=10,
        yes=[("0.400000", "50")],
        no=[("0.500000", "30")],
    )


def _seed_yes_lock_deltas(factory: sessionmaker[Session]) -> None:
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=10),
        seq=11,
        side="yes",
        price="0.950000",
        delta="10",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=50),
        seq=12,
        side="no",
        price="0.040000",
        delta="12",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=83),
        seq=13,
        side="no",
        price="0.500000",
        delta="-30",
    )


def test_probe_resolves_lag_from_the_delta_stream_not_a_sampling_grid(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _seed_yes_lock_deltas(factory)

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.lag_s == 83


def test_probe_no_side_band_crossing(db) -> None:
    path, factory = db
    _snapshot(factory, received_at=T0, seq=10, yes=[("0.500000", "30")], no=[("0.400000", "50")])
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=47),
        seq=11,
        side="no",
        price="0.950000",
        delta="10",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=61),
        seq=12,
        side="yes",
        price="0.040000",
        delta="12",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=74),
        seq=13,
        side="yes",
        price="0.500000",
        delta="-30",
    )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "no", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.lag_s == 74


def test_probe_crosses_on_a_mid_exactly_at_the_band(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=20),
        seq=11,
        side="yes",
        price="0.940000",
        delta="10",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=40),
        seq=12,
        side="no",
        price="0.040000",
        delta="12",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=55),
        seq=13,
        side="no",
        price="0.500000",
        delta="-30",
    )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert (probe.at_decision[60].yes_bid + probe.at_decision[60].yes_ask) / 2 == Decimal("0.95")
    assert probe.lag_s == 55


def test_probe_holds_off_just_inside_the_band(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=20),
        seq=11,
        side="yes",
        price="0.930000",
        delta="10",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=40),
        seq=12,
        side="no",
        price="0.040000",
        delta="12",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=55),
        seq=13,
        side="no",
        price="0.500000",
        delta="-30",
    )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.lag_s is None


def test_probe_lag_is_none_when_band_never_crossed(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=10),
        seq=11,
        side="yes",
        price="0.400000",
        delta="5",
    )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.lag_s is None


def test_probe_yields_t0_and_every_decision_state_in_one_pass(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _seed_yes_lock_deltas(factory)

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.at_t0.snapshot_at == T0
    assert probe.at_t0.yes_bid == Decimal("0.400000")
    assert probe.at_t0.yes_ask == Decimal("0.500000")

    assert sorted(probe.at_decision) == list(OFFSETS)
    assert probe.at_decision[30].snapshot_at == T0 + timedelta(seconds=30)
    assert probe.at_decision[30].yes_bid == Decimal("0.950000")
    assert probe.at_decision[30].yes_bid_depth == 10
    assert probe.at_decision[60].no_bid == Decimal("0.500000")
    assert probe.at_decision[90].no_bid == Decimal("0.040000")
    assert probe.at_decision[90].yes_ask == Decimal("0.960000")
    assert probe.at_decision[120].no_bid == Decimal("0.040000")


def test_probe_decision_state_includes_a_delta_landing_exactly_on_the_offset(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=60),
        seq=11,
        side="yes",
        price="0.400000",
        delta="7",
    )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.at_decision[30].yes_bid_depth == 50
    assert probe.at_decision[60].yes_bid_depth == 57


def test_probe_replays_a_resubscribe_snapshot_inside_the_window(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _snapshot(
        factory,
        received_at=T0 + timedelta(seconds=45),
        seq=90,
        yes=[("0.300000", "5")],
        no=[("0.600000", "7")],
    )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.at_decision[30].yes_bid == Decimal("0.400000")
    assert probe.at_decision[60].yes_bid == Decimal("0.300000")
    assert probe.at_decision[60].yes_bid_depth == 5
    assert probe.at_decision[60].no_bid == Decimal("0.600000")


def test_probe_cadence_is_zero_for_sub_second_deltas(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    for i in range(1, 8):
        _delta(
            factory,
            received_at=T0 + timedelta(milliseconds=100 * i),
            seq=10 + i,
            side="yes",
            price="0.400000",
            delta="1",
        )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.cadence_s == 0


def test_probe_cadence_uses_the_same_median_as_the_rest_path(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    for i, offset in enumerate((1, 3, 6, 16)):
        _delta(
            factory,
            received_at=T0 + timedelta(seconds=offset),
            seq=11 + i,
            side="yes",
            price="0.400000",
            delta="1",
        )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.cadence_s == median_int([1, 2, 3, 10])
    assert probe.cadence_s == 3


def test_open_book_db_closes_the_connection_on_exit(db) -> None:
    path, _ = db

    with open_book_db(path) as conn:
        conn.execute("SELECT 1").fetchone()

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_book_state_at_closes_its_connection(db, monkeypatch: pytest.MonkeyPatch) -> None:
    path, factory = db
    _seed_base_book(factory)
    opened: list[sqlite3.Connection] = []
    real = sqlite3.connect

    def _tracking(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _tracking)
    book_state_at(path, TICKER, T0 + timedelta(seconds=30))

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


def test_probe_cadence_is_none_below_three_arrivals(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=10),
        seq=11,
        side="yes",
        price="0.400000",
        delta="1",
    )

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.cadence_s is None


def test_probe_returns_none_without_ws_coverage(db) -> None:
    path, factory = db
    _delta(factory, received_at=T0, seq=1, side="yes", price="0.400000", delta="25")

    with open_book_db(path) as conn:
        assert probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS) is None


def test_probe_raises_when_a_gap_precedes_t0(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=5), ticker=TICKER)
    _seed_yes_lock_deltas(factory)

    with open_book_db(path) as conn:
        with pytest.raises(WsGapError, match=TICKER):
            probe_event(conn, TICKER, T0 + timedelta(seconds=30), "yes", decision_offsets_s=OFFSETS)


def test_probe_raises_when_a_gap_lands_inside_the_decision_horizon(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _seed_yes_lock_deltas(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=100), ticker=TICKER)

    with open_book_db(path) as conn:
        with pytest.raises(WsGapError, match=TICKER):
            probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)


def test_probe_raises_when_a_gap_precedes_an_unresolved_band_crossing(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _gap(factory, detected_at=T0 + timedelta(hours=3), ticker=TICKER)

    with open_book_db(path) as conn:
        with pytest.raises(WsGapError, match=TICKER):
            probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)


def test_probe_tolerates_a_gap_after_everything_it_needed(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _seed_yes_lock_deltas(factory)
    _gap(factory, detected_at=T0 + timedelta(hours=3), ticker=TICKER)

    with open_book_db(path) as conn:
        probe = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)

    assert probe is not None
    assert probe.lag_s == 83


def test_probe_never_replays_deltas_recorded_after_a_gap(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=40), ticker="", reason="connection_reset")
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=50),
        seq=99,
        side="yes",
        price="0.400000",
        delta="-999",
    )

    with open_book_db(path) as conn:
        with pytest.raises(WsGapError, match=TICKER):
            probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS)


def test_ws_gap_error_stays_a_value_error_for_existing_callers(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=10), ticker=TICKER)

    with pytest.raises(ValueError):
        book_state_at(path, TICKER, T0 + timedelta(seconds=30))


def test_probe_window_bounds_the_forward_scan(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=10),
        seq=11,
        side="yes",
        price="0.950000",
        delta="10",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=50),
        seq=12,
        side="no",
        price="0.040000",
        delta="12",
    )
    _delta(
        factory,
        received_at=T0 + timedelta(seconds=900),
        seq=13,
        side="no",
        price="0.500000",
        delta="-30",
    )

    with open_book_db(path) as conn:
        inside = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS, window_s=3600)
        outside = probe_event(conn, TICKER, T0, "yes", decision_offsets_s=OFFSETS, window_s=600)

    assert inside is not None and inside.lag_s == 900
    assert outside is not None and outside.lag_s is None


def _plan(conn, sql: str, params: tuple) -> str:
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return " | ".join(str(r[3]) for r in rows)


def test_book_event_queries_seek_an_index_and_need_no_sort(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _seed_yes_lock_deltas(factory)
    t_db = "2026-06-20 15:10:00.000000"

    from bot.lag.ws_book import _DELTAS, _FORWARD, _LATEST_SNAPSHOT, _SNAPSHOT_BATCH

    with open_book_db(path) as conn:
        plans = [
            _plan(conn, _LATEST_SNAPSHOT, (TICKER, t_db)),
            _plan(conn, _SNAPSHOT_BATCH, (TICKER, t_db, 10)),
            _plan(conn, _DELTAS, (TICKER, t_db, t_db)),
            _plan(conn, _FORWARD, (TICKER, t_db, t_db)),
        ]

    for plan in plans:
        assert "SEARCH ws_book_events USING INDEX ix_ws_book_events_ticker_received_at" in plan
        assert "SCAN" not in plan, plan
        assert "TEMP B-TREE" not in plan, plan


def test_gap_lookup_seeks_its_index(db) -> None:
    path, factory = db
    _seed_single_level_book(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=10), ticker=TICKER)
    t_db = "2026-06-20 15:10:00.000000"

    from bot.lag.ws_book import _GAP

    with open_book_db(path) as conn:
        plan = _plan(conn, _GAP, (TICKER, t_db, t_db))

    assert "SEARCH ws_gaps USING INDEX ix_ws_gaps_ticker_detected_at" in plan
    assert "SCAN" not in plan, plan
