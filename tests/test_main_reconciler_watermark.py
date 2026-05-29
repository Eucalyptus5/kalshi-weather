from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as _timezone
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

import bot.main as bot_main
from bot.config import Settings
from bot.kalshi_client import KalshiDemoClient
from bot.main import App, _demo_startup_backfill, _load_reconciler_watermark, _reconcile_once
from bot.storage.sqlite import (
    Base,
    ReconcilerState,
    make_engine,
    make_session_factory,
)


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("watermark") / "demo.pem"
    path.write_bytes(pem)
    return path


def _settings(pem_path: Path) -> Settings:
    return Settings(
        mode="demo",
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=pem_path,
    )


async def _make_client(rsa_pem: Path, handler) -> KalshiDemoClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    )
    client = KalshiDemoClient(_settings(rsa_pem), http_client=http)
    await client.aopen()
    return client


def _make_app(rsa_pem: Path, kalshi: KalshiDemoClient) -> App:
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    settings = _settings(rsa_pem)
    return App(
        settings=settings,
        engine=engine,
        session_factory=sf,
        meteo=None,  # type: ignore[arg-type]
        kalshi=kalshi,  # type: ignore[arg-type]
        acis=None,  # type: ignore[arg-type]
        series_list=("KXHIGHDEN",),
    )


def _captured_min_ts(calls: list[dict[str, str]], path_suffix: str) -> int:
    for c in calls:
        if c["_path"].endswith(path_suffix):
            return int(c["min_ts"])
    raise AssertionError(f"no call to {path_suffix} captured")


async def test_first_ever_boot_uses_seven_day_fallback(rsa_pem: Path) -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        params["_path"] = request.url.path
        calls.append(params)
        if request.url.path.endswith("/portfolio/orders"):
            return httpx.Response(200, json={"orders": [], "cursor": ""})
        return httpx.Response(200, json={"fills": [], "cursor": ""})

    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    try:
        watermark = _load_reconciler_watermark(app)
        assert watermark is None
        fallback = datetime.now(tz=_timezone.utc) - timedelta(days=7)
        await _reconcile_once(app, fallback)
    finally:
        await client.aclose()
        app.engine.dispose()

    lower = int((datetime.now(tz=_timezone.utc) - timedelta(days=7, minutes=1)).timestamp())
    upper = int(
        (datetime.now(tz=_timezone.utc) - timedelta(days=6, hours=23, minutes=59)).timestamp()
    )
    orders_ts = _captured_min_ts(calls, "/portfolio/orders")
    fills_ts = _captured_min_ts(calls, "/portfolio/fills")
    assert lower <= orders_ts <= upper
    assert lower <= fills_ts <= upper


async def test_persisted_watermark_used_for_next_poll(rsa_pem: Path) -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        params["_path"] = request.url.path
        calls.append(params)
        if request.url.path.endswith("/portfolio/orders"):
            return httpx.Response(200, json={"orders": [], "cursor": ""})
        return httpx.Response(200, json={"fills": [], "cursor": ""})

    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    persisted = datetime.now(tz=_timezone.utc) - timedelta(hours=3)
    try:
        with app.session_factory() as session:
            session.merge(ReconcilerState(key="last_poll_ts", value=persisted.isoformat()))
            session.commit()

        loaded = _load_reconciler_watermark(app)
        assert loaded is not None
        await _reconcile_once(app, loaded)
    finally:
        await client.aclose()
        app.engine.dispose()

    expected = int(persisted.timestamp())
    assert _captured_min_ts(calls, "/portfolio/orders") == expected
    assert _captured_min_ts(calls, "/portfolio/fills") == expected


async def test_empty_response_still_advances_watermark(rsa_pem: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/orders"):
            return httpx.Response(200, json={"orders": [], "cursor": ""})
        return httpx.Response(200, json={"fills": [], "cursor": ""})

    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    initial = datetime.now(tz=_timezone.utc) - timedelta(hours=2)
    try:
        before = datetime.now(tz=_timezone.utc)
        new_watermark = await _reconcile_once(app, initial)
        after = datetime.now(tz=_timezone.utc)
        persisted = _load_reconciler_watermark(app)
    finally:
        await client.aclose()
        app.engine.dispose()

    assert new_watermark > initial
    lower = before - timedelta(seconds=61)
    upper = after - timedelta(seconds=59)
    assert lower <= new_watermark <= upper
    assert persisted is not None
    assert persisted == new_watermark


async def test_rollback_on_mid_pagination_failure_preserves_watermark(
    rsa_pem: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        params["_path"] = request.url.path
        calls.append(params)
        if request.url.path.endswith("/portfolio/orders"):
            return httpx.Response(200, json={"orders": [], "cursor": ""})
        return httpx.Response(200, json={"fills": [], "cursor": ""})

    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)

    prior_watermark = datetime.now(tz=_timezone.utc) - timedelta(hours=4)

    async def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-pagination failure")

    try:
        with app.session_factory() as session:
            session.merge(ReconcilerState(key="last_poll_ts", value=prior_watermark.isoformat()))
            session.commit()

        monkeypatch.setattr(bot_main, "poll_open_orders", boom)
        loaded = _load_reconciler_watermark(app)
        assert loaded is not None
        with pytest.raises(RuntimeError):
            await _reconcile_once(app, loaded)

        after_failure = _load_reconciler_watermark(app)
        assert after_failure is not None
        assert after_failure == prior_watermark

        monkeypatch.undo()
        calls.clear()

        next_load = _load_reconciler_watermark(app)
        assert next_load == prior_watermark
        await _reconcile_once(app, next_load)
    finally:
        await client.aclose()
        app.engine.dispose()

    assert _captured_min_ts(calls, "/portfolio/orders") == int(prior_watermark.timestamp())


async def test_cold_start_backfill_uses_persisted_watermark_when_older_than_one_hour(
    rsa_pem: Path,
) -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        params["_path"] = request.url.path
        calls.append(params)
        if request.url.path.endswith("/portfolio/orders"):
            return httpx.Response(200, json={"orders": [], "cursor": ""})
        if request.url.path.endswith("/portfolio/fills"):
            return httpx.Response(200, json={"fills": [], "cursor": ""})
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"balance_dollars": "500"})
        return httpx.Response(404)

    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    old_watermark = datetime.now(tz=_timezone.utc) - timedelta(hours=5)
    try:
        with app.session_factory() as session:
            session.merge(ReconcilerState(key="last_poll_ts", value=old_watermark.isoformat()))
            session.commit()

        await _demo_startup_backfill(app)
    finally:
        await client.aclose()
        app.engine.dispose()

    assert _captured_min_ts(calls, "/portfolio/orders") == int(old_watermark.timestamp())
    assert _captured_min_ts(calls, "/portfolio/fills") == int(old_watermark.timestamp())


async def test_cold_start_backfill_falls_back_to_seven_days_when_no_watermark(
    rsa_pem: Path,
) -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        params["_path"] = request.url.path
        calls.append(params)
        if request.url.path.endswith("/portfolio/orders"):
            return httpx.Response(200, json={"orders": [], "cursor": ""})
        if request.url.path.endswith("/portfolio/fills"):
            return httpx.Response(200, json={"fills": [], "cursor": ""})
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"balance_dollars": "500"})
        return httpx.Response(404)

    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    try:
        await _demo_startup_backfill(app)
    finally:
        await client.aclose()
        app.engine.dispose()

    lower = int((datetime.now(tz=_timezone.utc) - timedelta(days=7, minutes=1)).timestamp())
    upper = int(
        (datetime.now(tz=_timezone.utc) - timedelta(days=6, hours=23, minutes=59)).timestamp()
    )
    orders_ts = _captured_min_ts(calls, "/portfolio/orders")
    assert lower <= orders_ts <= upper
