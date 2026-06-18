from __future__ import annotations

import logging
from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from bot.config import Settings
from bot.execution.order_reconciler import (
    KalshiPosition,
    local_position_aggregate,
    poll_positions,
)
from bot.kalshi_client import KalshiDemoClient
from bot.main import App, _assert_boot_position_invariant
from bot.storage.sqlite import (
    Base,
    DemoOrder as DemoOrderRow,
    make_engine,
    make_session_factory,
)


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("boot-invariant") / "demo.pem"
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


def _make_app(pem_path: Path, kalshi: KalshiDemoClient) -> App:
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    return App(
        settings=_settings(pem_path),
        engine=engine,
        session_factory=sf,
        meteo=None,  # type: ignore[arg-type]
        kalshi=kalshi,  # type: ignore[arg-type]
        kalshi_read=kalshi,  # type: ignore[arg-type]
        acis=None,  # type: ignore[arg-type]
        series_list=("KXHIGHNY",),
    )


def _seed_demo_order(
    app: App,
    *,
    cid: str,
    ticker: str,
    side: str,
    filled: int,
    status: str = "executed",
) -> None:
    now = datetime(2026, 5, 31, 12, 0, tzinfo=_timezone.utc)
    with app.session_factory() as session:
        session.add(
            DemoOrderRow(
                client_order_id=cid,
                exchange_order_id=f"EX-{cid}",
                market_ticker=ticker,
                strategy="edge",
                side=side,
                requested_contracts=filled,
                filled_contracts=filled,
                requested_yes_price_dollars=Decimal("0.50"),
                fair_at_entry=Decimal("0.55"),
                q_raw=Decimal("0.55"),
                intended_at=now,
                avg_fill_price=Decimal("0.50"),
                fee_dollars=Decimal("0.01"),
                realized_pnl_dollars=None,
                status=status,
                placed_at=now,
                last_status_at=now,
            )
        )
        session.commit()


def _positions_handler(positions: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json={"market_positions": positions, "cursor": ""})
        return httpx.Response(404, json={"error": "not found"})

    return handler


def test_local_position_aggregate_signed_yes_long(rsa_pem: Path) -> None:
    eng = make_engine(":memory:")
    Base.metadata.create_all(eng)
    sf = make_session_factory(eng)
    now = datetime(2026, 5, 31, 12, 0, tzinfo=_timezone.utc)
    with sf() as session:
        session.add(
            DemoOrderRow(
                client_order_id="kw-a",
                exchange_order_id="EX-A",
                market_ticker="KXHIGHNY-26MAY31-T79",
                strategy="edge",
                side="yes",
                requested_contracts=3,
                filled_contracts=3,
                requested_yes_price_dollars=Decimal("0.50"),
                fair_at_entry=Decimal("0.55"),
                q_raw=Decimal("0.55"),
                intended_at=now,
                avg_fill_price=Decimal("0.50"),
                fee_dollars=Decimal("0.01"),
                realized_pnl_dollars=None,
                status="executed",
                placed_at=now,
                last_status_at=now,
            )
        )
        session.add(
            DemoOrderRow(
                client_order_id="kw-b",
                exchange_order_id="EX-B",
                market_ticker="KXHIGHNY-26MAY31-T79",
                strategy="edge",
                side="no",
                requested_contracts=2,
                filled_contracts=2,
                requested_yes_price_dollars=Decimal("0.50"),
                fair_at_entry=Decimal("0.55"),
                q_raw=Decimal("0.55"),
                intended_at=now,
                avg_fill_price=Decimal("0.50"),
                fee_dollars=Decimal("0.01"),
                realized_pnl_dollars=None,
                status="executed",
                placed_at=now,
                last_status_at=now,
            )
        )
        session.commit()

        agg = local_position_aggregate(session, "KXHIGHNY-26MAY31-T79")
    eng.dispose()
    assert agg == 1


async def test_poll_positions_paginates_via_cursor(rsa_pem: Path) -> None:
    pages = [
        {
            "market_positions": [
                {"ticker": "KXHIGHNY-26MAY31-T79", "position": -4},
            ],
            "cursor": "page2",
        },
        {
            "market_positions": [
                {"ticker": "KXHIGHNY-26MAY31-B78.5", "position": 2},
            ],
            "cursor": "",
        },
    ]
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        page = pages[len(calls) - 1]
        return httpx.Response(200, json=page)

    client = await _make_client(rsa_pem, handler)
    try:
        positions = await poll_positions(client)
    finally:
        await client.aclose()

    assert positions == [
        KalshiPosition(ticker="KXHIGHNY-26MAY31-T79", position_fp=-4),
        KalshiPosition(ticker="KXHIGHNY-26MAY31-B78.5", position_fp=2),
    ]
    assert calls[1].get("cursor") == "page2"


async def test_boot_invariant_violation_exits_and_logs_hint(
    rsa_pem: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KW_RESUME_AFTER_CRASH", raising=False)
    handler = _positions_handler(
        [{"ticker": "KXHIGHNY-26MAY31-T79", "position": -2}],
    )
    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)

    caplog.set_level(logging.WARNING, logger="bot.main")
    try:
        with pytest.raises(SystemExit) as exc:
            await _assert_boot_position_invariant(app, require_parity=True)
    finally:
        await client.aclose()
        app.engine.dispose()

    assert exc.value.code != 0

    violation = [r for r in caplog.records if "boot_position_invariant_violated" in r.getMessage()]
    hints = [r for r in caplog.records if "boot_position_invariant_recovery_hint" in r.getMessage()]
    assert len(violation) == 1
    assert violation[0].levelno == logging.ERROR
    assert "KXHIGHNY-26MAY31-T79" in violation[0].getMessage()
    assert "kalshi=-2" in violation[0].getMessage()
    assert "local=0" in violation[0].getMessage()
    assert len(hints) == 1
    assert "scripts/reconcile/backfill_from_kalshi.py" in hints[0].getMessage()


async def test_boot_invariant_env_var_suppresses_exit_and_downgrades_log(
    rsa_pem: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KW_RESUME_AFTER_CRASH", "1")
    handler = _positions_handler(
        [{"ticker": "KXHIGHNY-26MAY31-T79", "position": -2}],
    )
    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)

    caplog.set_level(logging.WARNING, logger="bot.main")
    try:
        await _assert_boot_position_invariant(app, require_parity=True)
    finally:
        await client.aclose()
        app.engine.dispose()

    violation = [r for r in caplog.records if "boot_position_invariant_violated" in r.getMessage()]
    hints = [r for r in caplog.records if "boot_position_invariant_recovery_hint" in r.getMessage()]
    assert len(violation) == 1
    assert violation[0].levelno == logging.WARNING
    assert hints == []


async def test_boot_invariant_parity_match_passes_silently(
    rsa_pem: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KW_RESUME_AFTER_CRASH", raising=False)
    handler = _positions_handler(
        [{"ticker": "KXHIGHNY-26MAY31-T79", "position": -2}],
    )
    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)
    _seed_demo_order(app, cid="kw-edge-no-x", ticker="KXHIGHNY-26MAY31-T79", side="no", filled=2)

    caplog.set_level(logging.INFO, logger="bot.main")
    try:
        await _assert_boot_position_invariant(app, require_parity=True)
    finally:
        await client.aclose()
        app.engine.dispose()

    violation = [r for r in caplog.records if "boot_position_invariant_violated" in r.getMessage()]
    hints = [r for r in caplog.records if "boot_position_invariant_recovery_hint" in r.getMessage()]
    assert violation == []
    assert hints == []


async def test_boot_invariant_skipped_when_flag_off(
    rsa_pem: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KW_RESUME_AFTER_CRASH", raising=False)
    called: dict[str, int] = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        called["n"] += 1
        return httpx.Response(200, json={"market_positions": [], "cursor": ""})

    client = await _make_client(rsa_pem, handler)
    app = _make_app(rsa_pem, client)

    try:
        await _assert_boot_position_invariant(app, require_parity=False)
    finally:
        await client.aclose()
        app.engine.dispose()

    assert called["n"] == 0


def _drive_main_with_args(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, object]:
    import bot.main as bot_main

    monkeypatch.setattr("sys.argv", argv)

    captured: dict[str, object] = {}

    class _FakeSettings:
        mode = "demo"
        log_level = "INFO"
        kalshi_demo_key_id = "k"
        kalshi_demo_private_key_path = Path("/dev/null")
        kalshi_prod_key_id = "p"
        kalshi_prod_private_key_path = Path("/dev/null")

        def model_dump(self):
            return {"mode": self.mode}

    fake = _FakeSettings()
    monkeypatch.setattr(bot_main, "get_settings", lambda: fake)
    monkeypatch.setattr(bot_main, "make_engine", lambda p: make_engine(":memory:"))
    monkeypatch.setattr(bot_main, "upgrade_schema", lambda p: None)

    class _FakeKalshi:
        async def aopen(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    def _kalshi_factory(_settings_arg):
        return _FakeKalshi()

    monkeypatch.setattr(bot_main, "KalshiDemoClient", _kalshi_factory)
    monkeypatch.setattr(bot_main, "KalshiReadClient", _kalshi_factory)

    async def _noop_backfill(app):
        return None

    async def _noop_run(app, duration):
        return None

    async def _spy(app, *, require_parity: bool) -> None:
        captured["require_parity"] = require_parity

    monkeypatch.setattr(bot_main, "_demo_startup_backfill", _noop_backfill)
    monkeypatch.setattr(bot_main, "_assert_boot_position_invariant", _spy)
    monkeypatch.setattr(bot_main, "run", _noop_run)

    bot_main.main()
    return captured


def test_cli_require_position_parity_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _drive_main_with_args(
        monkeypatch,
        ["bot.main", "--mode=demo", "--series=KXHIGHDEN", "--duration=1s"],
    )
    assert captured.get("require_parity") is True


def test_cli_no_require_position_parity_disables_check(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _drive_main_with_args(
        monkeypatch,
        [
            "bot.main",
            "--mode=demo",
            "--series=KXHIGHDEN",
            "--duration=1s",
            "--no-require-position-parity",
        ],
    )
    assert captured.get("require_parity") is False
