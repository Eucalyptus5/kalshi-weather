from __future__ import annotations

from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal

import pytest
from sqlalchemy import select, text

from bot.storage.sqlite import (
    Base,
    DemoOrder,
    PaperTradeRow,
    SimulatedPnl,
    make_engine,
    make_session_factory,
)


def _now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, tzinfo=_timezone.utc)


INITIAL_BALANCE = Decimal("500.000000")

NATURAL_KEY_FORMULA = text("SELECT COALESCE(SUM(realized_pnl), 0) FROM simulated_pnl")

DOUBLE_COUNT_FORMULA = text(
    "SELECT COALESCE((SELECT SUM(realized_pnl) FROM simulated_pnl), 0) "
    "+ COALESCE((SELECT SUM(realized_pnl_dollars) FROM demo_orders), 0)"
)


@pytest.fixture
def session():
    eng = make_engine(":memory:")
    Base.metadata.create_all(eng)
    factory = make_session_factory(eng)
    with factory() as s:
        yield s
    eng.dispose()


def _seed_natural_key_trade(session, *, cid: str, realized: Decimal) -> None:
    session.add(
        DemoOrder(
            client_order_id=cid,
            exchange_order_id="EX-" + cid,
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy="edge",
            side="yes",
            requested_contracts=10,
            filled_contracts=10,
            requested_yes_price_dollars=Decimal("0.58"),
            fair_at_entry=Decimal("0.62"),
            intended_at=_now(),
            avg_fill_price=Decimal("0.205"),
            fee_dollars=Decimal("0.07"),
            status="executed",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )
    trade = PaperTradeRow(
        intended_at=_now(),
        market_ticker="KXHIGHDEN-26MAY22-T70",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        strategy="edge",
        demo_order_client_id=cid,
    )
    session.add(trade)
    session.flush()
    session.add(
        SimulatedPnl(
            paper_trade_id=trade.id,
            settled_at=_now(),
            outcome="won",
            realized_pnl=realized,
        )
    )


def _seed_backfill_order(session, *, eid: str) -> None:
    session.add(
        DemoOrder(
            client_order_id=f"kw-backfill-{eid}",
            exchange_order_id=eid,
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy=None,
            side="yes",
            requested_contracts=10,
            filled_contracts=10,
            requested_yes_price_dollars=None,
            fair_at_entry=None,
            intended_at=None,
            avg_fill_price=Decimal("0.300000"),
            fee_dollars=Decimal("0.05"),
            realized_pnl_dollars=None,
            status="executed",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )


def test_two_cohort_sum_double_counts_paper_mirrored_demo_trades(session):
    cid = "kw-edge-yes-A"
    _seed_natural_key_trade(session, cid=cid, realized=Decimal("9.270000"))
    row = session.scalars(select(DemoOrder).where(DemoOrder.client_order_id == cid)).one()
    row.realized_pnl_dollars = Decimal("1.530000")
    session.commit()

    sim_sum = Decimal(str(session.execute(NATURAL_KEY_FORMULA).scalar()))
    assert sim_sum == Decimal("9.270000")
    demo_sum = Decimal(
        str(session.execute(text("SELECT SUM(realized_pnl_dollars) FROM demo_orders")).scalar())
    )
    assert demo_sum == Decimal("1.530000")

    double = session.execute(DOUBLE_COUNT_FORMULA).scalar()
    assert abs(Decimal(str(double)) - Decimal("10.800000")) < Decimal("0.000001")
    assert Decimal(str(double)) != sim_sum
    assert Decimal(str(double)) != demo_sum


def test_natural_key_formula_returns_expected_sum(session):
    _seed_natural_key_trade(session, cid="kw-edge-yes-A", realized=Decimal("3.000000"))
    _seed_natural_key_trade(session, cid="kw-edge-yes-B", realized=Decimal("-1.500000"))
    _seed_backfill_order(session, eid="EXB1")
    session.commit()

    nat = Decimal(str(session.execute(NATURAL_KEY_FORMULA).scalar()))
    assert INITIAL_BALANCE + nat == INITIAL_BALANCE + Decimal("1.500000")


def test_residual_equals_backfill_cohort_settled_pnl(session):
    _seed_natural_key_trade(session, cid="kw-edge-yes-A", realized=Decimal("3.000000"))
    _seed_backfill_order(session, eid="EXB1")
    session.commit()

    backfill_settled_on_exchange = Decimal("4.000000")
    exchange_balance = INITIAL_BALANCE + Decimal("3.000000") + backfill_settled_on_exchange

    nat = Decimal(str(session.execute(NATURAL_KEY_FORMULA).scalar()))
    residual = exchange_balance - (INITIAL_BALANCE + nat)
    assert residual == backfill_settled_on_exchange
