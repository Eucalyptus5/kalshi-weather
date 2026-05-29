from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from sqlalchemy import select

from bot.config import Settings
from bot.execution.order_reconciler import KalshiPosition, local_position_aggregate
from bot.kalshi_client import KalshiDemoClient
from bot.storage.sqlite import (
    Base,
    DemoOrder as DemoOrderRow,
    make_engine,
    make_session_factory,
)
from scripts.reconcile.backfill_from_kalshi import _backfill_ticker, _find_orphans


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("backfill") / "demo.pem"
    path.write_bytes(pem)
    return path


def _settings(pem_path: Path) -> Settings:
    return Settings(
        mode="demo",
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=pem_path,
    )


async def _make_client(pem_path: Path, handler) -> KalshiDemoClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    )
    client = KalshiDemoClient(_settings(pem_path), http_client=http)
    await client.aopen()
    return client


def test_find_orphans_returns_tickers_with_local_zero() -> None:
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    try:
        positions = [
            KalshiPosition(ticker="KXHIGHNY-26MAY31-T79", position_fp=-4),
            KalshiPosition(ticker="KXHIGHNY-26MAY31-B78.5", position_fp=0),
        ]
        orphans = _find_orphans(sf, positions)
        assert [o.ticker for o in orphans] == ["KXHIGHNY-26MAY31-T79"]
    finally:
        engine.dispose()


async def test_backfill_ticker_materializes_demo_orders_row(rsa_pem: Path) -> None:
    ticker = "KXHIGHNY-26MAY31-T79"
    eid = "EX-RECOVERED"
    cid = "kw-edge-no-existing"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        if path.endswith("/portfolio/orders"):
            return httpx.Response(
                200,
                json={
                    "orders": [
                        {
                            "order_id": eid,
                            "client_order_id": cid,
                            "ticker": ticker,
                            "side": "no",
                            "status": "executed",
                            "initial_count_fp": "4.00",
                            "fill_count_fp": "4.00",
                            "remaining_count_fp": "0.00",
                            "no_price_dollars": "0.30",
                            "taker_fees_dollars": "0.05",
                            "maker_fees_dollars": "0.00",
                            "taker_fill_cost_dollars": "1.20",
                            "maker_fill_cost_dollars": "0.00",
                        }
                    ],
                    "cursor": "",
                },
            )
        if path.endswith("/portfolio/fills"):
            assert params.get("ticker") == ticker
            return httpx.Response(
                200,
                json={
                    "fills": [
                        {
                            "fill_id": "F1",
                            "order_id": eid,
                            "ticker": ticker,
                            "outcome_side": "no",
                            "book_side": "no",
                            "count_fp": "4.00",
                            "yes_price_dollars": "0.70",
                            "no_price_dollars": "0.30",
                            "is_taker": True,
                            "created_time": "2026-05-31T01:00:00Z",
                            "fee_cost": "0.05",
                        }
                    ],
                    "cursor": "",
                },
            )
        return httpx.Response(404)

    client = await _make_client(rsa_pem, handler)
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    try:
        orders_count, inserted = await _backfill_ticker(client, sf, ticker)
        with sf() as session:
            rows = session.scalars(
                select(DemoOrderRow).where(DemoOrderRow.market_ticker == ticker)
            ).all()
            agg = local_position_aggregate(session, ticker)
    finally:
        await client.aclose()
        engine.dispose()

    assert orders_count == 1
    assert inserted == 1
    assert len(rows) == 1
    assert rows[0].side == "no"
    assert rows[0].filled_contracts == 4
    assert rows[0].exchange_order_id == eid
    assert agg == -4
