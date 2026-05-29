from __future__ import annotations

from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from bot.config import Settings
from bot.execution.order_placer import DemoOrder
from bot.execution.order_reconciler import (
    DemoFill,
    _parse_fill,
    idempotent_insert_demo_order_row,
    poll_fills,
    poll_open_orders,
    reconcile_fills_into_demo_orders,
    stitch_natural_key_order,
    upsert_exchange_record,
)
from bot.kalshi_client import KalshiDemoClient
from bot.storage.sqlite import (
    Base,
    DemoOrder as DemoOrderRow,
    PaperTradeRow,
    make_engine,
    make_session_factory,
)


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


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("kalshi-reconciler") / "demo.pem"
    path.write_bytes(pem)
    return path


def _settings(pem_path: Path) -> Settings:
    return Settings(
        mode="paper",
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=pem_path,
    )


async def _client_with_handler(rsa_pem: Path, handler) -> KalshiDemoClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    )
    client = KalshiDemoClient(_settings(rsa_pem), http_client=http)
    await client.aopen()
    return client


def _exchange_order(
    *,
    client_order_id: str = "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22",
    exchange_order_id: str = "EX1",
    side_kalshi: str = "yes",
    status: str = "executed",
    filled_contracts: int = 10,
    avg: Decimal | None = Decimal("0.205"),
    fee: Decimal = Decimal("0.07"),
) -> DemoOrder:
    return DemoOrder(
        client_order_id=client_order_id,
        exchange_order_id=exchange_order_id,
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


def _seed_row(session, **overrides) -> DemoOrderRow:
    base = dict(
        client_order_id="kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22",
        exchange_order_id="EX1",
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy="edge",
        side="yes",
        requested_contracts=10,
        filled_contracts=0,
        requested_yes_price_dollars=Decimal("0.58"),
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        intended_at=_now(),
        avg_fill_price=None,
        fee_dollars=None,
        status="resting",
        placed_at=_now(),
        last_status_at=_now(),
    )
    base.update(overrides)
    row = DemoOrderRow(**base)
    session.add(row)
    session.commit()
    return row


def _fill(
    *,
    fill_id: str = "F1",
    order_id: str = "EX1",
    yes_price: str = "0.205",
    no_price: str = "0.795",
    count: int = 10,
    fee: str = "0.07",
) -> DemoFill:
    return DemoFill(
        fill_id=fill_id,
        order_id=order_id,
        ticker="KXHIGHDEN-26MAY22-T70",
        outcome_side="yes",
        book_side="yes",
        count=count,
        yes_price_dollars=Decimal(yes_price),
        no_price_dollars=Decimal(no_price),
        is_taker=True,
        created_time="2026-05-22T12:00:05Z",
        fee_cost=Decimal(fee),
    )


def test_parse_fill_handles_count_fp() -> None:
    raw = {
        "fill_id": "F1",
        "order_id": "EX1",
        "ticker": "KXHIGHNY-26MAY31-T79",
        "outcome_side": "yes",
        "book_side": "bid",
        "count_fp": "1.00",
        "yes_price_dollars": "0.9900",
        "no_price_dollars": "0.0100",
        "is_taker": True,
        "created_time": "2026-05-31T00:43:47.462584Z",
        "fee_cost": "0.000700",
    }
    fill = _parse_fill(raw)
    assert fill.count == 1
    assert fill.yes_price_dollars == Decimal("0.9900")
    assert fill.fee_cost == Decimal("0.000700")


async def test_poll_fills_paginates_via_cursor(rsa_pem: Path) -> None:
    pages = [
        {
            "fills": [
                {
                    "fill_id": "F1",
                    "order_id": "EX1",
                    "ticker": "KXHIGHDEN-26MAY22-T70",
                    "outcome_side": "yes",
                    "book_side": "yes",
                    "count_fp": 5,
                    "yes_price_dollars": "0.205",
                    "no_price_dollars": "0.795",
                    "is_taker": True,
                    "created_time": "2026-05-22T12:00:05Z",
                    "fee_cost": "0.04",
                }
            ],
            "cursor": "page2",
        },
        {
            "fills": [
                {
                    "fill_id": "F2",
                    "order_id": "EX1",
                    "ticker": "KXHIGHDEN-26MAY22-T70",
                    "outcome_side": "yes",
                    "book_side": "yes",
                    "count_fp": 5,
                    "yes_price_dollars": "0.205",
                    "no_price_dollars": "0.795",
                    "is_taker": True,
                    "created_time": "2026-05-22T12:00:06Z",
                    "fee_cost": "0.03",
                }
            ],
            "cursor": "",
        },
    ]
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        page = pages[len(calls) - 1]
        return httpx.Response(200, json=page)

    client = await _client_with_handler(rsa_pem, handler)
    try:
        fills = await poll_fills(client, _now())
    finally:
        await client.aclose()

    assert [f.fill_id for f in fills] == ["F1", "F2"]
    assert all(f.order_id == "EX1" for f in fills)
    assert calls[1].get("cursor") == "page2"


async def test_poll_open_orders_paginates_via_cursor(rsa_pem: Path) -> None:
    pages = [
        {
            "orders": [
                {
                    "order_id": "EX1",
                    "client_order_id": "kw-edge-yes-A",
                    "ticker": "KXHIGHDEN-26MAY22-T70",
                    "side": "yes",
                    "status": "executed",
                    "initial_count_fp": "10.00",
                    "fill_count_fp": "10.00",
                    "remaining_count_fp": "0.00",
                    "yes_price_dollars": "0.58",
                    "taker_fees_dollars": "0.070000",
                    "maker_fees_dollars": "0.000000",
                    "taker_fill_cost_dollars": "2.050000",
                    "maker_fill_cost_dollars": "0.000000",
                }
            ],
            "cursor": "page2",
        },
        {
            "orders": [
                {
                    "order_id": "EX2",
                    "client_order_id": "kw-edge-no-B",
                    "ticker": "KXHIGHDEN-26MAY22-T70",
                    "side": "no",
                    "status": "canceled",
                    "initial_count_fp": "5.00",
                    "fill_count_fp": "0.00",
                    "remaining_count_fp": "5.00",
                    "no_price_dollars": "0.42",
                    "taker_fees_dollars": "0.000000",
                    "maker_fees_dollars": "0.000000",
                    "taker_fill_cost_dollars": "0.000000",
                    "maker_fill_cost_dollars": "0.000000",
                }
            ],
            "cursor": "",
        },
    ]
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        page = pages[len(calls) - 1]
        return httpx.Response(200, json=page)

    client = await _client_with_handler(rsa_pem, handler)
    try:
        orders = await poll_open_orders(client, _now())
    finally:
        await client.aclose()

    assert [o.exchange_order_id for o in orders] == ["EX1", "EX2"]
    assert [o.status for o in orders] == ["executed", "canceled"]
    assert calls[0].get("status") == "executed,canceled"
    assert calls[0].get("min_ts") == str(int(_now().timestamp()))
    assert "cursor" not in calls[0]
    assert calls[1].get("cursor") == "page2"
    assert orders[0].requested_contracts == 10
    assert orders[0].filled_contracts == 10
    assert orders[0].fee_dollars == Decimal("0.07")
    assert orders[0].avg_yes_fill_price_dollars == Decimal("0.2050")
    assert orders[1].requested_contracts == 5
    assert orders[1].filled_contracts == 0
    assert orders[1].fee_dollars == Decimal("0")
    assert orders[1].avg_yes_fill_price_dollars is None


def test_upsert_exchange_record_inserts_backfill_row_with_null_intent(session):
    upsert_exchange_record(
        session, _exchange_order(client_order_id="kw-edge-yes-A", exchange_order_id="EX123")
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.client_order_id == "kw-backfill-EX123"
    assert row.strategy is None
    assert row.fair_at_entry is None
    assert row.intended_at is None
    assert row.requested_yes_price_dollars is None
    assert row.exchange_order_id == "EX123"
    assert row.status == "executed"
    assert row.filled_contracts == 10
    assert row.avg_fill_price == Decimal("0.205")


def test_upsert_exchange_record_is_idempotent(session):
    record = _exchange_order(exchange_order_id="EX123")
    upsert_exchange_record(session, record)
    session.commit()
    upsert_exchange_record(session, record)
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    assert rows[0].client_order_id == "kw-backfill-EX123"


def test_upsert_exchange_record_updates_existing_natural_key_row(session):
    _seed_row(session, status="resting", filled_contracts=0, avg_fill_price=None)
    upsert_exchange_record(
        session,
        _exchange_order(
            exchange_order_id="EX1", status="executed", filled_contracts=10, avg=Decimal("0.205")
        ),
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    assert rows[0].client_order_id == "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22"
    assert rows[0].status == "executed"
    assert rows[0].filled_contracts == 10
    assert rows[0].avg_fill_price == Decimal("0.205")


def test_backfill_synthesizes_kw_backfill_client_order_id(session):
    upsert_exchange_record(session, _exchange_order(exchange_order_id="EXNEW"))
    session.commit()
    row = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == "EXNEW")
    ).one()
    assert row.client_order_id == "kw-backfill-EXNEW"
    assert row.strategy is None
    assert row.fair_at_entry is None
    assert row.intended_at is None
    assert row.requested_yes_price_dollars is None


def test_reconcile_fills_updates_demo_order_row(session):
    _seed_row(session, status="resting", filled_contracts=0, avg_fill_price=None)
    n = reconcile_fills_into_demo_orders(session, [_fill(count=10)], [])
    session.commit()

    row = session.scalars(select(DemoOrderRow)).one()
    assert row.status == "executed"
    assert row.filled_contracts == 10
    assert row.avg_fill_price == Decimal("0.205")
    assert n >= 1


def test_reconcile_fills_is_idempotent(session):
    _seed_row(session, status="resting", filled_contracts=0, avg_fill_price=None)
    fills = [_fill(count=10)]
    reconcile_fills_into_demo_orders(session, fills, [])
    session.commit()
    reconcile_fills_into_demo_orders(session, fills, [])
    session.commit()

    paper_rows = session.scalars(select(PaperTradeRow)).all()
    assert len(paper_rows) == 1


def test_reconcile_late_fill_inserts_paper_trade_with_strategy_attribution(session):
    _seed_row(
        session,
        status="resting",
        filled_contracts=0,
        avg_fill_price=None,
        strategy="edge",
        fair_at_entry=Decimal("0.62"),
        intended_at=_now(),
    )
    reconcile_fills_into_demo_orders(session, [_fill(count=10)], [])
    session.commit()

    pt = session.scalars(select(PaperTradeRow)).one()
    assert pt.strategy == "edge"
    assert pt.fair_at_entry == Decimal("0.62")
    assert pt.intended_at == _now()
    assert pt.demo_order_client_id == "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22"
    assert pt.simulated_price == Decimal("0.205")


def test_late_fill_from_backfill_row_skips_paper_trade_insertion(session):
    _seed_row(
        session,
        client_order_id="kw-backfill-EX1",
        strategy=None,
        fair_at_entry=None,
        intended_at=None,
        requested_yes_price_dollars=None,
        status="resting",
        filled_contracts=0,
        avg_fill_price=None,
    )
    reconcile_fills_into_demo_orders(session, [_fill(count=10)], [])
    session.commit()

    assert session.scalars(select(PaperTradeRow)).all() == []
    row = session.scalars(select(DemoOrderRow)).one()
    assert row.filled_contracts == 10


def test_stitch_natural_key_order_case1_no_holder_overwrites_in_place(session):
    _seed_row(
        session,
        client_order_id="kw-backfill-EX123",
        exchange_order_id="EX123",
        strategy=None,
        fair_at_entry=None,
        intended_at=None,
        requested_yes_price_dollars=None,
        status="executed",
        filled_contracts=10,
        avg_fill_price=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
    )
    stitch_natural_key_order(
        session,
        exchange_order_id="EX123",
        client_order_id="kw-edge-yes-T1",
        strategy="edge",
        side="yes",
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        intended_at=_now(),
        requested_yes_price_dollars=Decimal("0.58"),
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.client_order_id == "kw-edge-yes-T1"
    assert row.strategy == "edge"
    assert row.fair_at_entry == Decimal("0.62")
    assert row.intended_at == _now()
    assert row.requested_yes_price_dollars == Decimal("0.58")
    assert row.exchange_order_id == "EX123"
    assert row.status == "executed"
    assert row.avg_fill_price == Decimal("0.205")


def test_stitch_case2_flushes_delete_before_update_no_integrity_error(session):
    session.add(
        DemoOrderRow(
            client_order_id="kw-backfill-EX2",
            exchange_order_id="EX2",
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy=None,
            side="yes",
            requested_contracts=10,
            filled_contracts=10,
            requested_yes_price_dollars=None,
            fair_at_entry=None,
            intended_at=None,
            avg_fill_price=Decimal("0.205"),
            fee_dollars=Decimal("0.07"),
            status="executed",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )
    session.add(
        DemoOrderRow(
            client_order_id="kw-edge-yes-T1",
            exchange_order_id=None,
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy=None,
            side="yes",
            requested_contracts=10,
            filled_contracts=0,
            requested_yes_price_dollars=None,
            fair_at_entry=None,
            intended_at=None,
            avg_fill_price=None,
            fee_dollars=None,
            status="resting",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )
    session.commit()

    stitch_natural_key_order(
        session,
        exchange_order_id="EX2",
        client_order_id="kw-edge-yes-T1",
        strategy="edge",
        side="yes",
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        intended_at=_now(),
        requested_yes_price_dollars=Decimal("0.58"),
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.client_order_id == "kw-edge-yes-T1"
    assert row.exchange_order_id == "EX2"
    assert row.status == "executed"
    assert row.filled_contracts == 10
    assert row.avg_fill_price == Decimal("0.205")
    assert row.strategy == "edge"
    assert row.fair_at_entry == Decimal("0.62")
    assert row.intended_at == _now()
    assert row.requested_yes_price_dollars == Decimal("0.58")


def test_naive_case2_pattern_hits_integrity_error_under_default_flush_order(session):
    matched = DemoOrderRow(
        client_order_id="kw-backfill-EX2",
        exchange_order_id="EX2",
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy=None,
        side="yes",
        requested_contracts=10,
        filled_contracts=10,
        requested_yes_price_dollars=None,
        fair_at_entry=None,
        intended_at=None,
        avg_fill_price=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
        status="executed",
        placed_at=_now(),
        last_status_at=_now(),
    )
    holder = DemoOrderRow(
        client_order_id="kw-edge-yes-T1",
        exchange_order_id=None,
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy=None,
        side="yes",
        requested_contracts=10,
        filled_contracts=0,
        requested_yes_price_dollars=None,
        fair_at_entry=None,
        intended_at=None,
        avg_fill_price=None,
        fee_dollars=None,
        status="resting",
        placed_at=_now(),
        last_status_at=_now(),
    )
    session.add_all([matched, holder])
    session.commit()

    holder.exchange_order_id = matched.exchange_order_id
    holder.filled_contracts = matched.filled_contracts
    holder.status = matched.status
    holder.avg_fill_price = matched.avg_fill_price
    holder.fee_dollars = matched.fee_dollars
    session.delete(matched)
    with pytest.raises(IntegrityError):
        session.flush()


def test_stitch_natural_key_order_case3_idempotent_when_holder_has_same_exchange_id(session):
    session.add(
        DemoOrderRow(
            client_order_id="kw-edge-yes-T1",
            exchange_order_id="EX2",
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
    session.commit()

    stitch_natural_key_order(
        session,
        exchange_order_id="EX2",
        client_order_id="kw-edge-yes-T1",
        strategy="edge",
        side="yes",
        fair_at_entry=Decimal("0.99"),
        q_raw=Decimal("0.99"),
        intended_at=_now(),
        requested_yes_price_dollars=Decimal("0.58"),
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.client_order_id == "kw-edge-yes-T1"
    assert row.fair_at_entry == Decimal("0.99")
    assert row.status == "executed"
    assert row.avg_fill_price == Decimal("0.205")
    assert row.filled_contracts == 10


def test_stitch_natural_key_order_case4_raises_on_different_exchange_id(session):
    session.add(
        DemoOrderRow(
            client_order_id="kw-edge-yes-T1",
            exchange_order_id="EX_OTHER",
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
    _seed_row(
        session,
        client_order_id="kw-backfill-EX2",
        exchange_order_id="EX2",
        strategy=None,
        fair_at_entry=None,
        intended_at=None,
        requested_yes_price_dollars=None,
        status="executed",
        filled_contracts=10,
        avg_fill_price=Decimal("0.50"),
    )

    with pytest.raises(ValueError, match="natural-key collision"):
        stitch_natural_key_order(
            session,
            exchange_order_id="EX2",
            client_order_id="kw-edge-yes-T1",
            strategy="edge",
            side="yes",
            fair_at_entry=Decimal("0.62"),
            q_raw=Decimal("0.62"),
            intended_at=_now(),
            requested_yes_price_dollars=Decimal("0.58"),
        )
    session.rollback()

    other = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == "EX_OTHER")
    ).one()
    assert other.client_order_id == "kw-edge-yes-T1"
    backfill = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == "EX2")
    ).one()
    assert backfill.client_order_id == "kw-backfill-EX2"


def test_naive_blind_insert_under_natural_key_id_produces_two_rows(session):
    _seed_row(
        session,
        client_order_id="kw-backfill-EX123",
        exchange_order_id="EX123",
        strategy=None,
        fair_at_entry=None,
        intended_at=None,
        requested_yes_price_dollars=None,
        status="executed",
        filled_contracts=10,
        avg_fill_price=Decimal("0.205"),
    )
    session.add(
        DemoOrderRow(
            client_order_id="kw-edge-yes-T1",
            exchange_order_id=None,
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy="edge",
            side="yes",
            requested_contracts=10,
            filled_contracts=0,
            requested_yes_price_dollars=Decimal("0.58"),
            fair_at_entry=Decimal("0.62"),
            intended_at=_now(),
            avg_fill_price=None,
            fee_dollars=None,
            status="resting",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )
    session.commit()

    count = session.scalar(
        select(func.count())
        .select_from(DemoOrderRow)
        .where(DemoOrderRow.market_ticker == "KXHIGHDEN-26MAY22-T70")
    )
    assert count == 2
    eid_rows = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == "EX123")
    ).all()
    assert len(eid_rows) == 1
    assert eid_rows[0].client_order_id == "kw-backfill-EX123"


def test_stitch_natural_key_order_raises_on_missing_exchange_id(session):
    with pytest.raises(ValueError):
        stitch_natural_key_order(
            session,
            exchange_order_id="NOPE",
            client_order_id="kw-edge-yes-T1",
            strategy="edge",
            side="yes",
            fair_at_entry=Decimal("0.62"),
            q_raw=Decimal("0.62"),
            intended_at=_now(),
            requested_yes_price_dollars=Decimal("0.58"),
        )


def test_late_fill_then_stitch_materializes_paper_trade_row(session):
    session.add(
        DemoOrderRow(
            client_order_id="kw-backfill-EX2",
            exchange_order_id="EX2",
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy=None,
            side="yes",
            requested_contracts=10,
            filled_contracts=10,
            requested_yes_price_dollars=None,
            fair_at_entry=None,
            intended_at=None,
            avg_fill_price=Decimal("0.205"),
            fee_dollars=Decimal("0.07"),
            status="executed",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )
    session.add(
        DemoOrderRow(
            client_order_id="kw-edge-yes-T1",
            exchange_order_id=None,
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy=None,
            side="yes",
            requested_contracts=10,
            filled_contracts=0,
            requested_yes_price_dollars=None,
            fair_at_entry=None,
            intended_at=None,
            avg_fill_price=None,
            fee_dollars=None,
            status="resting",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )
    session.commit()

    stitch_natural_key_order(
        session,
        exchange_order_id="EX2",
        client_order_id="kw-edge-yes-T1",
        strategy="edge",
        side="yes",
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        intended_at=_now(),
        requested_yes_price_dollars=Decimal("0.58"),
    )
    session.commit()

    pt = session.scalars(select(PaperTradeRow)).one()
    assert pt.strategy == "edge"
    assert pt.fair_at_entry == Decimal("0.62")
    assert pt.intended_at == _now()
    assert pt.demo_order_client_id == "kw-edge-yes-T1"
    assert pt.simulated_price == Decimal("0.205")


def test_late_fill_then_stitch_without_followup_loses_paper_trade_row(session):
    matched = DemoOrderRow(
        client_order_id="kw-backfill-EX2",
        exchange_order_id="EX2",
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy=None,
        side="yes",
        requested_contracts=10,
        filled_contracts=10,
        requested_yes_price_dollars=None,
        fair_at_entry=None,
        intended_at=None,
        avg_fill_price=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
        status="executed",
        placed_at=_now(),
        last_status_at=_now(),
    )
    holder = DemoOrderRow(
        client_order_id="kw-edge-yes-T1",
        exchange_order_id=None,
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy=None,
        side="yes",
        requested_contracts=10,
        filled_contracts=0,
        requested_yes_price_dollars=None,
        fair_at_entry=None,
        intended_at=None,
        avg_fill_price=None,
        fee_dollars=None,
        status="resting",
        placed_at=_now(),
        last_status_at=_now(),
    )
    session.add_all([matched, holder])
    session.commit()

    eid = matched.exchange_order_id
    filled = matched.filled_contracts
    status = matched.status
    avg = matched.avg_fill_price
    fee = matched.fee_dollars
    session.delete(matched)
    session.flush()
    holder.exchange_order_id = eid
    holder.filled_contracts = filled
    holder.status = status
    holder.avg_fill_price = avg
    holder.fee_dollars = fee
    holder.strategy = "edge"
    holder.fair_at_entry = Decimal("0.62")
    holder.intended_at = _now()
    holder.requested_yes_price_dollars = Decimal("0.58")
    session.commit()

    assert session.scalars(select(PaperTradeRow)).all() == []


def test_stitch_case3_post_insert_paper_trade_is_idempotent_against_existing_row(session):
    session.add(
        DemoOrderRow(
            client_order_id="kw-edge-yes-T1",
            exchange_order_id="EX2",
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
    session.add(
        PaperTradeRow(
            intended_at=_now(),
            market_ticker="KXHIGHDEN-26MAY22-T70",
            side="buy_yes",
            contracts=10,
            simulated_price=Decimal("0.205"),
            fee_dollars=Decimal("0.07"),
            fair_at_entry=Decimal("0.62"),
            q_raw=Decimal("0.62"),
            strategy="edge",
            demo_order_client_id="kw-edge-yes-T1",
        )
    )
    session.commit()

    stitch_natural_key_order(
        session,
        exchange_order_id="EX2",
        client_order_id="kw-edge-yes-T1",
        strategy="edge",
        side="yes",
        fair_at_entry=Decimal("0.77"),
        q_raw=Decimal("0.77"),
        intended_at=_now(),
        requested_yes_price_dollars=Decimal("0.58"),
    )
    session.commit()

    pts = session.scalars(
        select(PaperTradeRow).where(PaperTradeRow.demo_order_client_id == "kw-edge-yes-T1")
    ).all()
    assert len(pts) == 1
    row = session.scalars(select(DemoOrderRow)).one()
    assert row.fair_at_entry == Decimal("0.77")
    assert row.avg_fill_price == Decimal("0.205")


def test_naive_session_add_in_stitch_raises_integrity_error_on_duplicate_cid(session):
    session.add(
        DemoOrderRow(
            client_order_id="kw-edge-yes-T1",
            exchange_order_id="EX2",
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
    session.add(
        PaperTradeRow(
            intended_at=_now(),
            market_ticker="KXHIGHDEN-26MAY22-T70",
            side="buy_yes",
            contracts=10,
            simulated_price=Decimal("0.205"),
            fee_dollars=Decimal("0.07"),
            fair_at_entry=Decimal("0.62"),
            q_raw=Decimal("0.62"),
            strategy="edge",
            demo_order_client_id="kw-edge-yes-T1",
        )
    )
    session.commit()

    session.add(
        PaperTradeRow(
            intended_at=_now(),
            market_ticker="KXHIGHDEN-26MAY22-T70",
            side="buy_yes",
            contracts=10,
            simulated_price=Decimal("0.205"),
            fee_dollars=Decimal("0.07"),
            fair_at_entry=Decimal("0.62"),
            q_raw=Decimal("0.62"),
            strategy="edge",
            demo_order_client_id="kw-edge-yes-T1",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_stitch_natural_key_order_writes_q_raw(session):
    _seed_row(
        session,
        client_order_id="kw-backfill-EX300",
        exchange_order_id="EX300",
        strategy=None,
        fair_at_entry=None,
        q_raw=None,
        intended_at=None,
        requested_yes_price_dollars=None,
        status="executed",
        filled_contracts=10,
        avg_fill_price=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
    )
    stitch_natural_key_order(
        session,
        exchange_order_id="EX300",
        client_order_id="kw-edge-yes-Q1",
        strategy="edge",
        side="yes",
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.005"),
        intended_at=_now(),
        requested_yes_price_dollars=Decimal("0.58"),
    )
    session.commit()
    row = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == "EX300")
    ).one()
    assert row.q_raw == Decimal("0.005")


def test_stitch_natural_key_order_writes_q_raw_holder_branch(session):
    session.add(
        DemoOrderRow(
            client_order_id="kw-backfill-EX301",
            exchange_order_id="EX301",
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy=None,
            side="yes",
            requested_contracts=10,
            filled_contracts=10,
            requested_yes_price_dollars=None,
            fair_at_entry=None,
            q_raw=None,
            intended_at=None,
            avg_fill_price=Decimal("0.205"),
            fee_dollars=Decimal("0.07"),
            status="executed",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )
    session.add(
        DemoOrderRow(
            client_order_id="kw-edge-yes-Q2",
            exchange_order_id=None,
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy=None,
            side="yes",
            requested_contracts=10,
            filled_contracts=0,
            requested_yes_price_dollars=None,
            fair_at_entry=None,
            q_raw=None,
            intended_at=None,
            avg_fill_price=None,
            fee_dollars=None,
            status="resting",
            placed_at=_now(),
            last_status_at=_now(),
        )
    )
    session.commit()
    stitch_natural_key_order(
        session,
        exchange_order_id="EX301",
        client_order_id="kw-edge-yes-Q2",
        strategy="edge",
        side="yes",
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.005"),
        intended_at=_now(),
        requested_yes_price_dollars=Decimal("0.58"),
    )
    session.commit()
    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    assert rows[0].q_raw == Decimal("0.005")
    assert rows[0].client_order_id == "kw-edge-yes-Q2"


def test_upsert_exchange_record_orphan_row_has_null_q_raw(session):
    record = DemoOrder(
        client_order_id="kw-edge-yes-ORPHAN",
        exchange_order_id="EX-ORPHAN",
        ticker="KXHIGHDEN-26MAY22-T70",
        side_kalshi="yes",
        requested_contracts=10,
        filled_contracts=10,
        requested_yes_price_dollars=None,
        avg_yes_fill_price_dollars=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
        status="executed",
        placed_at=_now(),
    )
    upsert_exchange_record(session, record)
    session.commit()
    row = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == "EX-ORPHAN")
    ).one()
    assert row.q_raw is None
    assert row.fair_at_entry is None


def test_fills_win_over_parsed_order_values_after_full_reconcile_cycle(session):
    _seed_row(session, status="resting", filled_contracts=0, avg_fill_price=None, fee_dollars=None)
    parsed = _exchange_order(
        exchange_order_id="EX1",
        status="executed",
        filled_contracts=10,
        avg=Decimal("0.999"),
        fee=Decimal("9.99"),
    )
    upsert_exchange_record(session, parsed)
    session.commit()

    fills = [
        _fill(fill_id="F1", count=4, yes_price="0.205", no_price="0.795", fee="0.04"),
        _fill(fill_id="F2", count=6, yes_price="0.205", no_price="0.795", fee="0.03"),
    ]
    reconcile_fills_into_demo_orders(session, fills, [parsed])
    session.commit()

    row = session.scalars(select(DemoOrderRow)).one()
    assert row.filled_contracts == 10
    assert row.avg_fill_price == Decimal("0.205")
    assert row.fee_dollars == Decimal("0.07")


async def test_poll_open_orders_inverts_no_side_avg_fill_from_dollars(
    rsa_pem: Path, session
) -> None:
    pages = [
        {
            "orders": [
                {
                    "order_id": "EX-NO",
                    "client_order_id": "kw-edge-no-A",
                    "ticker": "KXHIGHDEN-26MAY22-T70",
                    "side": "no",
                    "status": "executed",
                    "initial_count_fp": "4.00",
                    "fill_count_fp": "4.00",
                    "remaining_count_fp": "0.00",
                    "no_price_dollars": "0.0750",
                    "taker_fees_dollars": "0.010000",
                    "maker_fees_dollars": "0.000000",
                    "taker_fill_cost_dollars": "0.300000",
                    "maker_fill_cost_dollars": "0.000000",
                }
            ],
            "cursor": "",
        },
    ]
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        return httpx.Response(200, json=pages[len(calls) - 1])

    client = await _client_with_handler(rsa_pem, handler)
    try:
        orders = await poll_open_orders(client, _now())
    finally:
        await client.aclose()

    assert len(orders) == 1
    parsed = orders[0]
    assert parsed.side_kalshi == "no"
    assert parsed.avg_yes_fill_price_dollars == Decimal("0.9250")
    assert parsed.fee_dollars == Decimal("0.01")

    upsert_exchange_record(session, parsed)
    session.commit()
    row = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == "EX-NO")
    ).one()
    assert row.avg_fill_price == Decimal("0.9250")
    assert row.fee_dollars == Decimal("0.01")
    assert row.side == "no"


def test_upsert_exchange_record_backfills_empty_eid_on_existing_cid_row(session):
    cid = "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22"
    idempotent_insert_demo_order_row(
        session,
        client_order_id=cid,
        exchange_order_id=None,
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy="edge",
        side="yes",
        requested_contracts=10,
        filled_contracts=5,
        requested_yes_price_dollars=Decimal("0.58"),
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        intended_at=_now(),
        status="resting",
        placed_at=_now(),
    )
    session.commit()

    upsert_exchange_record(
        session,
        _exchange_order(
            client_order_id=cid,
            exchange_order_id="EX-NEW",
            status="executed",
            filled_contracts=10,
            avg=Decimal("0.205"),
        ),
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.client_order_id == cid
    assert row.exchange_order_id == "EX-NEW"
    assert row.strategy == "edge"
    assert row.fair_at_entry == Decimal("0.62")
    assert row.status == "executed"
    assert row.filled_contracts == 10
    backfill = session.scalars(
        select(DemoOrderRow).where(DemoOrderRow.client_order_id.like("kw-backfill-%"))
    ).all()
    assert backfill == []


def test_upsert_exchange_record_raises_on_cid_eid_mismatch(session):
    cid = "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22"
    _seed_row(session, client_order_id=cid, exchange_order_id="EX-OLD")
    with pytest.raises(ValueError):
        upsert_exchange_record(
            session,
            _exchange_order(
                client_order_id=cid,
                exchange_order_id="EX-DIFFERENT",
                status="executed",
                filled_contracts=10,
            ),
        )
    session.rollback()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    assert rows[0].exchange_order_id == "EX-OLD"


def test_idempotent_insert_helper_cid_update_branch_backfills_eid(session):
    cid = "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22"
    _seed_row(
        session,
        client_order_id=cid,
        exchange_order_id=None,
        status="resting",
        filled_contracts=0,
    )

    idempotent_insert_demo_order_row(
        session,
        client_order_id=cid,
        exchange_order_id="EX-1",
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy="edge",
        side="yes",
        requested_contracts=10,
        filled_contracts=5,
        requested_yes_price_dollars=Decimal("0.58"),
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        intended_at=_now(),
        status="executed",
        placed_at=_now(),
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.client_order_id == cid
    assert row.exchange_order_id == "EX-1"
    assert row.status == "executed"
    assert row.strategy == "edge"
    assert row.fair_at_entry == Decimal("0.62")


def test_idempotent_insert_helper_renames_backfill_row_and_lifts_intent_fields(session):
    cid = "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22"
    _seed_row(
        session,
        client_order_id="kw-backfill-EX-1",
        exchange_order_id="EX-1",
        strategy=None,
        fair_at_entry=None,
        q_raw=None,
        intended_at=None,
        requested_yes_price_dollars=None,
        status="executed",
        filled_contracts=10,
        avg_fill_price=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
    )

    idempotent_insert_demo_order_row(
        session,
        client_order_id=cid,
        exchange_order_id="EX-1",
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy="edge",
        side="yes",
        requested_contracts=10,
        filled_contracts=10,
        requested_yes_price_dollars=Decimal("0.58"),
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        intended_at=_now(),
        status="executed",
        placed_at=_now(),
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.client_order_id == cid
    assert row.exchange_order_id == "EX-1"
    assert row.strategy == "edge"
    assert row.fair_at_entry == Decimal("0.62")
    assert row.requested_yes_price_dollars == Decimal("0.58")
    assert row.side == "yes"
    assert row.status == "executed"


def test_idempotent_insert_helper_raises_on_eid_collision_with_unrelated_cid(session):
    _seed_row(
        session,
        client_order_id="kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22-other",
        exchange_order_id="EX-1",
        strategy="edge",
        fair_at_entry=Decimal("0.62"),
        status="executed",
        filled_contracts=10,
    )

    with pytest.raises(ValueError):
        idempotent_insert_demo_order_row(
            session,
            client_order_id="kw-edge-yes-DIFFERENT-CID",
            exchange_order_id="EX-1",
            market_ticker="KXHIGHDEN-26MAY22-T70",
            strategy="edge",
            side="yes",
            requested_contracts=10,
            filled_contracts=10,
            requested_yes_price_dollars=Decimal("0.58"),
            fair_at_entry=Decimal("0.62"),
            q_raw=Decimal("0.62"),
            intended_at=_now(),
            status="executed",
            placed_at=_now(),
        )
    session.rollback()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    assert rows[0].client_order_id == "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22-other"


def test_idempotent_insert_helper_fresh_insert(session):
    cid = "kw-edge-yes-NEW"
    idempotent_insert_demo_order_row(
        session,
        client_order_id=cid,
        exchange_order_id="EX-NEW",
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy="edge",
        side="yes",
        requested_contracts=10,
        filled_contracts=3,
        requested_yes_price_dollars=Decimal("0.58"),
        fair_at_entry=Decimal("0.62"),
        q_raw=Decimal("0.62"),
        intended_at=_now(),
        status="executed",
        placed_at=_now(),
    )
    session.commit()

    rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.client_order_id == cid
    assert row.exchange_order_id == "EX-NEW"
    assert row.strategy == "edge"
    assert row.side == "yes"
    assert row.requested_contracts == 10
    assert row.filled_contracts == 3
    assert row.fair_at_entry == Decimal("0.62")
    assert row.status == "executed"
