from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from bot.lag.event_study import OrderbookSnapshotRow
from bot.lag.ws_book import book_state_at
from bot.storage.sqlite import Base, WsBookEvent, WsGap, make_engine, make_session_factory


UTC = timezone.utc
T0 = datetime(2026, 6, 20, 15, 0, 0, tzinfo=UTC)
TICKER = "KXHIGHDEN-26JUN20-T85"


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
