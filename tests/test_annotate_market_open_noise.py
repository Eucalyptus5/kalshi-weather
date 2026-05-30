from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select

from bot.storage.sqlite import Base, GateFailure, make_engine, make_session_factory
from scripts.annotate_market_open_noise import FIX_TIMESTAMP, NOTE_LABEL, annotate


def _setup() -> tuple[object, object]:
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    return engine, factory


def _insert(factory, **kwargs) -> int:
    with factory() as session:
        row = GateFailure(**kwargs)
        session.add(row)
        session.commit()
        return row.id


def test_annotates_pre_fix_market_open_rows() -> None:
    engine, factory = _setup()
    pre_fix = FIX_TIMESTAMP - timedelta(hours=1)
    row_id = _insert(
        factory,
        evaluated_at=pre_fix,
        gate_name="market_open",
        reason="market_status=active",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY07-T70-75",
        last_seen_at=pre_fix,
    )

    rows_annotated = annotate(factory)

    assert rows_annotated == 1
    with factory() as session:
        got = session.scalars(select(GateFailure).where(GateFailure.id == row_id)).one()
    assert got.notes == NOTE_LABEL
    engine.dispose()


def test_skips_post_fix_rows() -> None:
    engine, factory = _setup()
    post_fix = FIX_TIMESTAMP + timedelta(hours=1)
    row_id = _insert(
        factory,
        evaluated_at=post_fix,
        gate_name="market_open",
        reason="market_status=active",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY07-T70-75",
        last_seen_at=post_fix,
    )

    rows_annotated = annotate(factory)

    assert rows_annotated == 0
    with factory() as session:
        got = session.scalars(select(GateFailure).where(GateFailure.id == row_id)).one()
    assert got.notes is None
    engine.dispose()


def test_skips_other_gate_rows() -> None:
    engine, factory = _setup()
    pre_fix = FIX_TIMESTAMP - timedelta(hours=1)
    row_id = _insert(
        factory,
        evaluated_at=pre_fix,
        gate_name="edge_threshold",
        reason="edge=0.01 < 0.05",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY07-T70-75",
        last_seen_at=pre_fix,
    )

    rows_annotated = annotate(factory)

    assert rows_annotated == 0
    with factory() as session:
        got = session.scalars(select(GateFailure).where(GateFailure.id == row_id)).one()
    assert got.notes is None
    engine.dispose()


def test_idempotent_rerun() -> None:
    engine, factory = _setup()
    pre_fix = FIX_TIMESTAMP - timedelta(hours=1)
    _insert(
        factory,
        evaluated_at=pre_fix,
        gate_name="market_open",
        reason="market_status=active",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY07-T70-75",
        last_seen_at=pre_fix,
    )

    first = annotate(factory)
    second = annotate(factory)

    assert first == 1
    assert second == 0
    engine.dispose()


def test_order_independent_against_running_bot() -> None:
    engine, factory = _setup()
    pre_fix = FIX_TIMESTAMP - timedelta(hours=1)
    _insert(
        factory,
        evaluated_at=pre_fix,
        gate_name="market_open",
        reason="market_status=active",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY07-T70-75",
        last_seen_at=pre_fix,
    )

    first = annotate(factory)

    post_fix = FIX_TIMESTAMP + timedelta(minutes=30)
    _insert(
        factory,
        evaluated_at=post_fix,
        gate_name="market_open",
        reason="market_status=closed",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY07-T70-75",
        last_seen_at=post_fix,
    )

    second = annotate(factory)

    assert first == 1
    assert second == 0
    with factory() as session:
        rows = session.scalars(select(GateFailure).order_by(GateFailure.id)).all()
    assert rows[0].notes == NOTE_LABEL
    assert rows[1].notes is None
    engine.dispose()
