from __future__ import annotations

from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from bot.execution.demo_row_translator import (
    paper_trade_row_from_demo_order,
    paper_trade_row_values,
    yes_frame_price,
    _paper_trade_row_from_demo_row,
)
from bot.execution.order_placer import DemoOrder
from bot.execution.paper import TradeIntent, TradeSide
from bot.storage.sqlite import (
    Base,
    DemoOrder as DemoOrderRow,
    PaperTradeRow,
    make_engine,
    make_session_factory,
)
from bot.validation.scoring import realized_pnl_for_trade


def _now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, tzinfo=_timezone.utc)


@pytest.fixture
def session():
    eng = make_engine(":memory:")
    Base.metadata.create_all(eng)
    factory = make_session_factory(eng)
    with factory() as s:
        yield s
    eng.dispose()


def _demo_order(
    *,
    side_kalshi: str = "yes",
    status: str = "executed",
    filled_contracts: int = 10,
    avg: Decimal | None = Decimal("0.205"),
    fee: Decimal = Decimal("0.07"),
    client_order_id: str = "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22",
) -> DemoOrder:
    return DemoOrder(
        client_order_id=client_order_id,
        exchange_order_id="EX1",
        ticker="KXHIGHDEN-26MAY22-T70",
        side_kalshi=side_kalshi,
        requested_contracts=10,
        filled_contracts=filled_contracts,
        requested_yes_price_dollars=Decimal("0.58"),
        avg_yes_fill_price_dollars=avg,
        fee_dollars=fee,
        status=status,
        placed_at=_now(),
    )


def _intent(*, strategy: str = "edge", fair_yes: Decimal = Decimal("0.62")) -> TradeIntent:
    return TradeIntent(
        market_ticker="KXHIGHDEN-26MAY22-T70",
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=fair_yes,
        strategy=strategy,
    )


def _demo_row(**overrides) -> DemoOrderRow:
    base = dict(
        client_order_id="kw-tails-no-KXHIGHDEN-26MAY22-T70-2026-05-22",
        exchange_order_id="EX2",
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy="tails",
        side="no",
        requested_contracts=10,
        filled_contracts=10,
        requested_yes_price_dollars=Decimal("0.20"),
        fair_at_entry=Decimal("0.18"),
        intended_at=_now(),
        avg_fill_price=Decimal("0.80"),
        fee_dollars=Decimal("0.05"),
        status="executed",
        placed_at=_now(),
        last_status_at=_now(),
    )
    base.update(overrides)
    return DemoOrderRow(**base)


def test_yes_frame_translates_no_fill_to_yes_bid():
    assert yes_frame_price("no", Decimal("0.80")) == Decimal("0.20")
    assert yes_frame_price("yes", Decimal("0.85")) == Decimal("0.85")


def test_paper_trade_row_pnl_round_trips_through_sell_yes_no_ask_translation():
    row = _paper_trade_row_from_demo_row(
        _demo_row(
            side="no", filled_contracts=10, avg_fill_price=Decimal("0.80"), fee_dollars=Decimal("0")
        ),
        _now(),
    )
    assert row is not None
    assert row.simulated_price == Decimal("0.20")
    assert row.side == "sell_yes"

    loss = realized_pnl_for_trade(
        TradeSide(row.side), row.simulated_price, row.contracts, row.fee_dollars, won=False
    )
    win = realized_pnl_for_trade(
        TradeSide(row.side), row.simulated_price, row.contracts, row.fee_dollars, won=True
    )
    assert loss == Decimal("-8.00")
    assert win == Decimal("2.00")


def test_buggy_plan_writes_no_side_price_inverts_pnl():
    row = _demo_row(side="no", filled_contracts=10, avg_fill_price=Decimal("0.80"))
    buggy_price = row.avg_fill_price
    loss = realized_pnl_for_trade(TradeSide.SELL_YES, buggy_price, 10, Decimal("0"), won=False)
    assert loss == Decimal("-2.00")


def test_buy_yes_fill_passes_through_unchanged():
    row = _paper_trade_row_from_demo_row(
        _demo_row(
            client_order_id="kw-edge-yes-X",
            side="yes",
            strategy="edge",
            filled_contracts=10,
            avg_fill_price=Decimal("0.65"),
            fee_dollars=Decimal("0"),
        ),
        _now(),
    )
    assert row is not None
    assert row.simulated_price == Decimal("0.65")
    assert row.side == "buy_yes"

    win = realized_pnl_for_trade(
        TradeSide(row.side), row.simulated_price, row.contracts, row.fee_dollars, won=True
    )
    assert win == Decimal("3.50")


def test_returns_none_when_intent_carry_columns_are_null():
    assert _paper_trade_row_from_demo_row(_demo_row(strategy=None), _now()) is None
    assert _paper_trade_row_from_demo_row(_demo_row(fair_at_entry=None), _now()) is None
    assert _paper_trade_row_from_demo_row(_demo_row(intended_at=None), _now()) is None
    assert (
        _paper_trade_row_from_demo_row(_demo_row(requested_yes_price_dollars=None), _now()) is None
    )


def test_returns_none_when_filled_contracts_zero():
    assert _paper_trade_row_from_demo_row(_demo_row(filled_contracts=0), _now()) is None


def test_reconciler_side_translator_returns_none_on_zero_price():
    row = _demo_row(filled_contracts=10, avg_fill_price=Decimal("0.0000"))
    assert _paper_trade_row_from_demo_row(row, _now()) is None


def test_reconciler_side_translator_returns_none_on_null_price():
    row = _demo_row(filled_contracts=10, avg_fill_price=None)
    assert _paper_trade_row_from_demo_row(row, _now()) is None


def test_placer_side_translator_returns_none_on_resting_order():
    order = _demo_order(status="resting", filled_contracts=0, avg=None)
    assert paper_trade_row_from_demo_order(order, _intent(), intended_at=_now()) is None


def test_placer_side_translator_returns_none_on_partial_fill_with_no_avg_price():
    order = _demo_order(status="resting", filled_contracts=3, avg=None)
    assert paper_trade_row_from_demo_order(order, _intent(), intended_at=_now()) is None


def test_placer_side_translator_builds_row_on_executed_with_avg_price():
    order = _demo_order(status="executed", filled_contracts=10, avg=Decimal("0.205"))
    row = paper_trade_row_from_demo_order(order, _intent(strategy="edge"), intended_at=_now())
    assert row is not None
    assert row.simulated_price == Decimal("0.205")
    assert row.fee_dollars == Decimal("0.07")
    assert row.contracts == 10
    assert row.strategy == "edge"
    assert row.fair_at_entry == Decimal("0.62")
    assert row.intended_at == _now()
    assert row.demo_order_client_id == order.client_order_id
    assert row.side == "buy_yes"
    assert row.market_ticker == "KXHIGHDEN-26MAY22-T70"
    assert row.attempted_contracts == 10
    assert row.ensemble_spread_sigma_t is None
    assert row.lead_time_hours is None
    assert row.nbm_divergence is None


def test_placer_side_translator_sell_yes_rebases_to_yes_frame():
    order = _demo_order(
        side_kalshi="no", status="executed", filled_contracts=10, avg=Decimal("0.80")
    )
    intent = TradeIntent(
        market_ticker="KXHIGHDEN-26MAY22-T70",
        side=TradeSide.SELL_YES,
        contracts=10,
        fair_yes=Decimal("0.18"),
        strategy="tails",
    )
    row = paper_trade_row_from_demo_order(order, intent, intended_at=_now())
    assert row is not None
    assert row.side == "sell_yes"
    assert row.simulated_price == Decimal("0.20")


def test_placer_side_translator_returns_none_on_zero_price_partial_fill():
    order = _demo_order(status="executed", filled_contracts=10, avg=Decimal("0.0000"))
    assert paper_trade_row_from_demo_order(order, _intent(), intended_at=_now()) is None


def test_paper_trade_row_values_excludes_id_and_created_at():
    row = PaperTradeRow(
        intended_at=_now(),
        market_ticker="KXHIGHDEN-26MAY22-T70",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.205000"),
        fee_dollars=Decimal("0.070000"),
        fair_at_entry=Decimal("0.620000"),
        strategy="edge",
        attempted_contracts=10,
        ensemble_spread_sigma_t=None,
        lead_time_hours=None,
        nbm_divergence=None,
        demo_order_client_id="kw-edge-yes-X",
    )
    values = paper_trade_row_values(row)
    expected_keys = {c.name for c in PaperTradeRow.__table__.columns} - {"id", "created_at"}
    assert set(values.keys()) == expected_keys
    assert values["simulated_price"] == Decimal("0.205000")
    assert values["demo_order_client_id"] == "kw-edge-yes-X"
    assert values["strategy"] == "edge"


def test_paper_trade_row_has_no_to_dict_method():
    row = PaperTradeRow(
        intended_at=_now(),
        market_ticker="KXHIGHDEN-26MAY22-T70",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.01"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
    )
    with pytest.raises(AttributeError):
        row.to_dict()


def test_paper_trade_row_values_round_trips_through_sqlite_insert(session):
    row = PaperTradeRow(
        intended_at=_now(),
        market_ticker="KXHIGHDEN-26MAY22-T70",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.205000"),
        fee_dollars=Decimal("0.070000"),
        fair_at_entry=Decimal("0.620000"),
        strategy="edge",
        demo_order_client_id="kw-edge-yes-X",
    )
    session.execute(sqlite_insert(PaperTradeRow).values(**paper_trade_row_values(row)))
    session.commit()

    got = session.scalars(select(PaperTradeRow)).one()
    assert got.id is not None
    assert got.created_at is not None
    assert got.simulated_price == Decimal("0.205000")
    assert got.demo_order_client_id == "kw-edge-yes-X"
