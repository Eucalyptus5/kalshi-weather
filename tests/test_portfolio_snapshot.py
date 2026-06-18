from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from sqlalchemy import select

from bot.config import Settings
from bot.execution.portfolio_snapshot import aggregate_positions
from bot.kalshi_client import KalshiDemoClient
from bot.main import App, _portfolio_snapshot_once
from bot.storage.sqlite import (
    Base,
    DemoOrder as DemoOrderRow,
    PortfolioSnapshot,
    make_engine,
    make_session_factory,
)


def _now() -> datetime:
    return datetime(2026, 6, 2, 12, 0, tzinfo=_timezone.utc)


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("portfolio-snapshot") / "demo.pem"
    path.write_bytes(pem)
    return path


def _settings(pem_path: Path) -> Settings:
    return Settings(
        mode="demo",
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=pem_path,
    )


async def _client_with_handler(pem_path: Path, handler) -> KalshiDemoClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    )
    client = KalshiDemoClient(_settings(pem_path), http_client=http)
    await client.aopen()
    return client


def _make_app(pem_path: Path, kalshi: KalshiDemoClient) -> App:
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    return App(
        settings=_settings(pem_path),
        engine=engine,
        session_factory=sf,
        meteo=None,  # type: ignore[arg-type]
        kalshi=kalshi,
        kalshi_read=kalshi,
        acis=None,  # type: ignore[arg-type]
        series_list=("KXHIGHDEN",),
    )


def _balance_json(balance_dollars: str = "788.3901", balance: int = 78839) -> dict:
    return {
        "balance": balance,
        "balance_breakdown": [{"balance": balance_dollars, "exchange_index": 0}],
        "balance_dollars": balance_dollars,
        "portfolio_value": 13434,
        "updated_ts": 1780428732,
    }


def _position(
    *,
    ticker: str,
    market_exposure: str,
    realized_pnl: str = "0.000000",
    fees_paid: str = "0.000000",
    position_fp: str = "-1.00",
) -> dict:
    return {
        "fees_paid_dollars": fees_paid,
        "last_updated_ts": "2026-06-02T11:00:00Z",
        "market_exposure_dollars": market_exposure,
        "position_fp": position_fp,
        "realized_pnl_dollars": realized_pnl,
        "resting_orders_count": 0,
        "ticker": ticker,
        "total_traded_dollars": market_exposure,
    }


def _seed_demo_order(
    session,
    *,
    cid: str,
    ticker: str,
    status: str = "executed",
    filled: int = 10,
    placed_at: datetime | None = None,
    last_status_at: datetime | None = None,
    realized_pnl_dollars: Decimal | None = None,
) -> DemoOrderRow:
    placed = placed_at if placed_at is not None else _now()
    last = last_status_at if last_status_at is not None else placed
    row = DemoOrderRow(
        client_order_id=cid,
        exchange_order_id="EX-" + cid,
        market_ticker=ticker,
        strategy="edge" if not cid.startswith("kw-backfill-") else None,
        side="no",
        requested_contracts=filled if filled else 10,
        filled_contracts=filled,
        requested_yes_price_dollars=Decimal("0.58"),
        fair_at_entry=Decimal("0.62"),
        intended_at=placed,
        avg_fill_price=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
        realized_pnl_dollars=realized_pnl_dollars,
        status=status,
        placed_at=placed,
        last_status_at=last,
    )
    session.add(row)
    session.commit()
    return row


async def test_snapshot_writes_one_row_with_aggregated_values(rsa_pem: Path) -> None:
    positions_payload = {
        "market_positions": [
            _position(
                ticker="KXHIGHDEN-26JUN01-T70",
                market_exposure="0.830000",
                realized_pnl="0.000000",
                fees_paid="0.009900",
                position_fp="-1.00",
            ),
            _position(
                ticker="KXHIGHCHI-26JUN01-T74",
                market_exposure="1.620000",
                realized_pnl="0.500000",
                fees_paid="0.021600",
                position_fp="-2.00",
            ),
            _position(
                ticker="KXHIGHNY-26JUN01-T70",
                market_exposure="0.000000",
                realized_pnl="1.230000",
                fees_paid="0.040000",
                position_fp="0.00",
            ),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json=positions_payload)
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    try:
        await _portfolio_snapshot_once(app)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        rows = session.scalars(select(PortfolioSnapshot)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.cash_dollars == Decimal("788.3901")
    assert row.total_exposure_dollars == Decimal("2.450000")
    assert row.total_collateral_dollars == Decimal("790.840100")
    assert row.realized_pnl_dollars == Decimal("1.730000")
    assert row.fees_paid_dollars == Decimal("0.071500")
    assert row.open_positions_count == 2


async def test_snapshot_writes_mtm_from_wire_portfolio_value(
    rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    positions_payload = {
        "market_positions": [
            _position(
                ticker="KXHIGHDEN-26JUN01-T70",
                market_exposure="0.830000",
                realized_pnl="0.000000",
                fees_paid="0.009900",
                position_fp="-1.00",
            ),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json=positions_payload)
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    caplog.set_level(logging.INFO, logger="bot.main")
    try:
        await _portfolio_snapshot_once(app)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        row = session.scalars(select(PortfolioSnapshot)).one()
    assert row.portfolio_value_mtm_dollars == Decimal("922.7301")
    assert row.total_collateral_dollars == Decimal("789.220100")
    assert row.total_collateral_dollars != row.portfolio_value_mtm_dollars

    snapshot_records = [r for r in caplog.records if r.message.startswith("portfolio_snapshot")]
    assert len(snapshot_records) == 1
    assert re.fullmatch(
        r"portfolio_snapshot cash=\S+ collateral=\S+ mtm=\S+ exposure=\S+ "
        r"realized_pnl=\S+ fees=\S+ open=\d+",
        snapshot_records[0].getMessage(),
    )


async def test_snapshot_loop_survives_transient_503(
    rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    positions_payload = {
        "market_positions": [
            _position(ticker="KXHIGHDEN-26JUN01-T70", market_exposure="0.830000"),
        ],
        "cursor": "",
    }
    call_counter = {"positions": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            call_counter["positions"] += 1
            if call_counter["positions"] == 1:
                return httpx.Response(503, text="boom")
            return httpx.Response(200, json=positions_payload)
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)

    async def _wrapped() -> None:
        try:
            await _portfolio_snapshot_once(app)
        except Exception:
            logging.getLogger("bot.main").exception(
                "loop_iteration_failed name=portfolio_snapshot_loop"
            )

    try:
        caplog.set_level(logging.ERROR, logger="bot.main")
        await _wrapped()
        with app.session_factory() as session:
            assert session.scalars(select(PortfolioSnapshot)).all() == []
        caplog.clear()
        caplog.set_level(logging.INFO, logger="bot.main")
        await _portfolio_snapshot_once(app)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        rows = session.scalars(select(PortfolioSnapshot)).all()
    assert len(rows) == 1


async def test_attribution_uses_placed_at_not_last_status_at(rsa_pem: Path) -> None:
    ticker = "KXHIGHDEN-26JUN01-T70"
    positions_payload = {
        "market_positions": [
            _position(ticker=ticker, market_exposure="0.830000", realized_pnl="1.530000"),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json=positions_payload)
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    base = _now()
    with app.session_factory() as session:
        _seed_demo_order(
            session,
            cid="kw-edge-old",
            ticker=ticker,
            placed_at=base.replace(hour=10),
            last_status_at=base.replace(hour=12),
        )
        _seed_demo_order(
            session,
            cid="kw-edge-new",
            ticker=ticker,
            placed_at=base.replace(hour=11),
            last_status_at=base.replace(hour=11),
        )
        _seed_demo_order(
            session,
            cid="kw-edge-resting",
            ticker=ticker,
            status="resting",
            filled=0,
            placed_at=base.replace(hour=11, minute=30),
        )
    try:
        await _portfolio_snapshot_once(app)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        rows = {r.client_order_id: r for r in session.scalars(select(DemoOrderRow)).all()}
    assert rows["kw-edge-new"].realized_pnl_dollars == Decimal("1.530000")
    assert rows["kw-edge-old"].realized_pnl_dollars is None
    assert rows["kw-edge-resting"].realized_pnl_dollars is None


async def test_attribution_no_write_when_only_resting_row(rsa_pem: Path) -> None:
    ticker = "KXHIGHDEN-26JUN01-T70"
    positions_payload = {
        "market_positions": [
            _position(ticker=ticker, market_exposure="0.830000", realized_pnl="1.530000"),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json=positions_payload)
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    with app.session_factory() as session:
        _seed_demo_order(session, cid="kw-edge-r", ticker=ticker, status="resting", filled=0)
    try:
        await _portfolio_snapshot_once(app)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        row = session.scalars(select(DemoOrderRow)).one()
    assert row.realized_pnl_dollars is None


async def test_kw_backfill_excluded_even_when_newer(rsa_pem: Path) -> None:
    ticker = "KXHIGHDEN-26JUN01-T70"
    positions_payload = {
        "market_positions": [
            _position(ticker=ticker, market_exposure="0.830000", realized_pnl="1.530000"),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json=positions_payload)
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    base = _now()
    with app.session_factory() as session:
        _seed_demo_order(
            session,
            cid="kw-edge-natural",
            ticker=ticker,
            placed_at=base.replace(hour=10),
        )
        _seed_demo_order(
            session,
            cid="kw-backfill-EX-NEW",
            ticker=ticker,
            placed_at=base.replace(hour=12),
        )
    try:
        await _portfolio_snapshot_once(app)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        rows = {r.client_order_id: r for r in session.scalars(select(DemoOrderRow)).all()}
    assert rows["kw-edge-natural"].realized_pnl_dollars == Decimal("1.530000")
    assert rows["kw-backfill-EX-NEW"].realized_pnl_dollars is None


async def test_kw_backfill_only_match_no_write(rsa_pem: Path) -> None:
    ticker = "KXHIGHDEN-26JUN01-T70"
    positions_payload = {
        "market_positions": [
            _position(ticker=ticker, market_exposure="0.830000", realized_pnl="1.530000"),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json=positions_payload)
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    with app.session_factory() as session:
        _seed_demo_order(session, cid="kw-backfill-EX-ONLY", ticker=ticker)
    try:
        await _portfolio_snapshot_once(app)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        row = session.scalars(select(DemoOrderRow)).one()
    assert row.realized_pnl_dollars is None


async def test_snapshot_loop_releases_db_lock_across_http_polls(rsa_pem: Path) -> None:
    positions_payload = {
        "market_positions": [
            _position(ticker="KXHIGHDEN-26JUN01-T70", market_exposure="0.830000"),
        ],
        "cursor": "",
    }
    poll_events = {"balance": asyncio.Event(), "positions": asyncio.Event()}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            poll_events["balance"].set()
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            poll_events["positions"].set()
            return httpx.Response(200, json=positions_payload)
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    holder_release = asyncio.Event()

    async def _hold_lock() -> None:
        async with app.db_lock:
            await holder_release.wait()

    try:
        holder_task = asyncio.create_task(_hold_lock())
        await asyncio.sleep(0)
        snapshot_task = asyncio.create_task(_portfolio_snapshot_once(app))
        await asyncio.wait_for(poll_events["balance"].wait(), timeout=2.0)
        await asyncio.wait_for(poll_events["positions"].wait(), timeout=2.0)
        assert app.db_lock.locked()
        holder_release.set()
        await holder_task
        await asyncio.wait_for(snapshot_task, timeout=2.0)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        rows = session.scalars(select(PortfolioSnapshot)).all()
    assert len(rows) == 1


async def test_mid_cursor_pagination_failure_aborts_tick(rsa_pem: Path) -> None:
    pages = [
        {
            "market_positions": [
                _position(ticker="KXHIGHDEN-26JUN01-T70", market_exposure="0.830000"),
            ],
            "cursor": "next-page",
        },
    ]
    cursors_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        cursors_seen.append(cursor)
        if cursor is None:
            return httpx.Response(200, json=pages[0])
        return httpx.Response(503, text="boom")

    client = await _client_with_handler(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await aggregate_positions(app.kalshi)
    finally:
        await client.aclose()

    assert cursors_seen == [None, "next-page"]
    with app.session_factory() as session:
        rows = session.scalars(select(PortfolioSnapshot)).all()
    assert rows == []
