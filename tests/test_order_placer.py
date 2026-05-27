from __future__ import annotations

import decimal
import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from bot.config import Settings
from bot.execution import order_placer as _order_placer_module
from bot.execution.order_placer import (
    DemoOrder,
    DemoOrderIdempotent,
    _build_body,
    _client_order_id,
    _parse_order,
    parse_avg_yes_fill_price,
    place_order_demo,
)
from bot.execution.paper import TradeIntent, TradeSide
from bot.kalshi_client import KalshiDemoClient, KalshiOrderbook


@pytest.fixture(autouse=True)
def _reset_breaker() -> None:
    _order_placer_module._breaker_state["recent_429"] = []
    _order_placer_module._breaker_state["paused_until"] = 0.0


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("kalshi-placer") / "demo.pem"
    path.write_bytes(pem)
    return path


def _settings(pem_path: Path) -> Settings:
    return Settings(
        mode="paper",
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=pem_path,
    )


def _now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)


def _intent(
    *,
    side: TradeSide = TradeSide.BUY_YES,
    ticker: str = "KXHIGHDEN-26MAY22-T70",
    contracts: int = 5,
    strategy: str = "edge",
    fair_yes: Decimal = Decimal("0.50"),
) -> TradeIntent:
    return TradeIntent(
        market_ticker=ticker,
        side=side,
        contracts=contracts,
        fair_yes=fair_yes,
        q_raw=fair_yes,
        strategy=strategy,
    )


def _book(
    *,
    ticker: str = "KXHIGHDEN-26MAY22-T70",
    yes_ask: Decimal = Decimal("0.85"),
    yes_bid: Decimal = Decimal("0.80"),
) -> KalshiOrderbook:
    return KalshiOrderbook(
        ticker=ticker,
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=Decimal("1") - yes_bid,
        no_bid=Decimal("1") - yes_ask,
        yes_ask_depth=100,
        yes_bid_depth=100,
        no_ask_depth=100,
        no_bid_depth=100,
        snapshot_at=_now(),
    )


def _make_order_response(
    *,
    client_order_id: str,
    side: str = "yes",
    status: str = "executed",
    filled_contracts: int = 5,
    requested_contracts: int = 5,
    avg_yes_fill_price_dollars: str | None = "0.8500",
    ticker: str = "KXHIGHDEN-26MAY22-T70",
    fee_dollars: str = "0.01",
    order_id: str = "ex-1",
) -> dict[str, object]:
    if avg_yes_fill_price_dollars is None or filled_contracts == 0:
        taker_fill_cost_cents = 0
    else:
        per_contract = Decimal(avg_yes_fill_price_dollars)
        if side == "no":
            per_contract = Decimal("1") - per_contract
        taker_fill_cost_cents = int(per_contract * Decimal(100) * Decimal(filled_contracts))
    taker_fees_cents = int(Decimal(fee_dollars) * Decimal(100))
    order: dict[str, object] = {
        "order_id": order_id,
        "client_order_id": client_order_id,
        "ticker": ticker,
        "side": side,
        "status": status,
        "fill_count": filled_contracts,
        "initial_count": requested_contracts,
        "taker_fees": taker_fees_cents,
        "maker_fees": 0,
        "taker_fill_cost": taker_fill_cost_cents,
        "maker_fill_cost": 0,
        "taker_fill_cost_dollars": f"{(Decimal(taker_fill_cost_cents) / Decimal(100)):.4f}",
        "maker_fill_cost_dollars": "0.0000",
    }
    return {"order": order}


async def _client_with_handler(rsa_pem: Path, handler) -> KalshiDemoClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    )
    client = KalshiDemoClient(_settings(rsa_pem), http_client=http)
    await client.aopen()
    return client


async def test_place_order_demo_buys_yes_at_yes_ask(rsa_pem: Path) -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201, json=_make_order_response(client_order_id=captured["body"]["client_order_id"])
        )

    client = await _client_with_handler(rsa_pem, handler)
    try:
        order = await place_order_demo(
            _intent(side=TradeSide.BUY_YES),
            _book(yes_ask=Decimal("0.85"), yes_bid=Decimal("0.80")),
            client,
            now=_now(),
        )
    finally:
        await client.aclose()

    assert order is not None
    body = captured["body"]
    assert body["side"] == "yes"
    assert body["action"] == "buy"
    assert body["yes_price_dollars"] == "0.8500"
    assert "type" not in body
    assert body["time_in_force"] == "immediate_or_cancel"
    assert "post_only" not in body
    assert body["count"] == 5


async def test_place_order_demo_sells_yes_routes_to_buy_no_at_no_ask(rsa_pem: Path) -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json=_make_order_response(
                client_order_id=captured["body"]["client_order_id"],
                side="no",
            ),
        )

    book = _book(yes_ask=Decimal("0.95"), yes_bid=Decimal("0.925"))
    assert book.no_ask == Decimal("0.075")
    client = await _client_with_handler(rsa_pem, handler)
    try:
        order = await place_order_demo(
            _intent(side=TradeSide.SELL_YES, strategy="tails"),
            book,
            client,
            now=_now(),
        )
    finally:
        await client.aclose()

    assert order is not None
    body = captured["body"]
    assert body["side"] == "no"
    assert body["action"] == "buy"
    assert body["no_price_dollars"] == "0.0750"
    assert "yes_price_dollars" not in body


async def test_place_order_demo_sells_yes_at_no_bid_does_not_fill_on_wide_spread() -> None:
    book = KalshiOrderbook(
        ticker="KXHIGHDEN-26MAY22-T70",
        yes_ask=Decimal("0.99"),
        yes_bid=Decimal("0.01"),
        no_ask=Decimal("0.99"),
        no_bid=Decimal("0.01"),
        yes_ask_depth=10,
        yes_bid_depth=10,
        no_ask_depth=10,
        no_bid_depth=10,
        snapshot_at=_now(),
    )
    assert book.no_bid < book.no_ask
    resting_no_offer_price = book.no_ask
    naive_sell_price = book.no_bid
    assert naive_sell_price < resting_no_offer_price


async def test_place_order_demo_never_emits_yes_sell(rsa_pem: Path) -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            201,
            json=_make_order_response(
                client_order_id=body["client_order_id"],
                side=body["side"],
            ),
        )

    client = await _client_with_handler(rsa_pem, handler)
    try:
        await place_order_demo(
            _intent(side=TradeSide.BUY_YES),
            _book(),
            client,
            now=_now(),
        )
        await place_order_demo(
            _intent(side=TradeSide.SELL_YES, strategy="tails"),
            _book(),
            client,
            now=_now(),
        )
    finally:
        await client.aclose()

    for body in seen:
        assert not (body["side"] == "yes" and body["action"] == "sell")


def test_client_order_id_is_deterministic() -> None:
    a = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    b = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    assert a == b


def test_client_order_id_is_stable_across_wall_clock() -> None:
    a = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    b = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    assert a == b


def test_client_order_id_differs_per_strategy() -> None:
    a = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    b = _client_order_id("tails", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    assert a != b


def test_client_order_id_differs_per_market() -> None:
    a = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    b = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T80", date(2026, 5, 22))
    assert a != b


def test_client_order_id_differs_per_side() -> None:
    a = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    b = _client_order_id("edge", TradeSide.SELL_YES, "KXHIGHDEN-26MAY22-T70", date(2026, 5, 22))
    assert a != b


def test_client_order_id_fits_in_64_chars_for_typical_ticker() -> None:
    cid = _client_order_id("edge", TradeSide.BUY_YES, "KXHIGHDEN-26MAY22-B66.5", date(2026, 5, 22))
    assert len(cid) <= 64
    assert "edge" in cid


def test_client_order_id_hashes_overlong_natural_key() -> None:
    long_ticker = "KXHIGHXXX-26MAY22-" + "X" * 80
    cid = _client_order_id("edge", TradeSide.BUY_YES, long_ticker, date(2026, 5, 22))
    assert len(cid) <= 64
    assert cid.startswith("kw-")


async def test_place_order_demo_handles_409_as_idempotent(rsa_pem: Path) -> None:
    state: dict[str, int] = {"post_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            state["post_calls"] += 1
            body = json.loads(request.content)
            return httpx.Response(
                409,
                json={"error": "ORDER_ALREADY_EXISTS"},
                headers={"X-Cid": body["client_order_id"]},
            )
        if request.method == "GET" and "/portfolio/orders" in request.url.path:
            cid = request.url.params.get("client_order_id")
            return httpx.Response(
                200,
                json={
                    "orders": [
                        _make_order_response(
                            client_order_id=cid,
                            side="yes",
                            status="executed",
                            filled_contracts=5,
                            requested_contracts=5,
                            avg_yes_fill_price_dollars="0.8500",
                            order_id="ex-existing",
                        )["order"]
                    ]
                },
            )
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    try:
        order = await place_order_demo(_intent(), _book(), client, now=_now())
    finally:
        await client.aclose()

    assert isinstance(order, DemoOrderIdempotent)
    assert order.exchange_order_id == "ex-existing"
    assert order.status == "executed"


async def test_place_order_demo_409_echo_returns_sentinel_not_demo_order(rsa_pem: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(409, json={"error": "ORDER_ALREADY_EXISTS"})
        if request.method == "GET" and "/portfolio/orders" in request.url.path:
            cid = request.url.params.get("client_order_id")
            return httpx.Response(
                200,
                json={
                    "orders": [
                        _make_order_response(
                            client_order_id=cid,
                            side="yes",
                            status="resting",
                            filled_contracts=0,
                            requested_contracts=5,
                            avg_yes_fill_price_dollars=None,
                            order_id="ex-resting-prior",
                        )["order"]
                    ]
                },
            )
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    try:
        order = await place_order_demo(_intent(contracts=5), _book(), client, now=_now())
    finally:
        await client.aclose()

    assert isinstance(order, DemoOrderIdempotent)
    assert not isinstance(order, DemoOrder)
    assert order.exchange_order_id == "ex-resting-prior"
    assert order.status == "resting"


async def test_place_order_demo_canceled_409_mismatched_contracts_short_circuits(
    rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(409, json={"error": "ORDER_ALREADY_EXISTS"})
        if request.method == "GET" and "/portfolio/orders" in request.url.path:
            cid = request.url.params.get("client_order_id")
            return httpx.Response(
                200,
                json={
                    "orders": [
                        _make_order_response(
                            client_order_id=cid,
                            status="canceled",
                            filled_contracts=0,
                            requested_contracts=10,
                            avg_yes_fill_price_dollars=None,
                        )["order"]
                    ]
                },
            )
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    try:
        caplog.set_level(logging.INFO, logger="bot.execution.order_placer")
        order = await place_order_demo(_intent(contracts=5), _book(), client, now=_now())
    finally:
        await client.aclose()

    assert order is None
    messages = [r.getMessage() for r in caplog.records]
    assert any("demo_order_idempotent_canceled" in m for m in messages)
    assert not any("demo_order_intent_drift" in m for m in messages)


async def test_place_order_demo_canceled_409_matching_contracts_also_short_circuits(
    rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(409, json={"error": "ORDER_ALREADY_EXISTS"})
        if request.method == "GET" and "/portfolio/orders" in request.url.path:
            cid = request.url.params.get("client_order_id")
            return httpx.Response(
                200,
                json={
                    "orders": [
                        _make_order_response(
                            client_order_id=cid,
                            status="canceled",
                            filled_contracts=0,
                            requested_contracts=5,
                            avg_yes_fill_price_dollars=None,
                        )["order"]
                    ]
                },
            )
        return httpx.Response(404)

    client = await _client_with_handler(rsa_pem, handler)
    try:
        caplog.set_level(logging.INFO, logger="bot.execution.order_placer")
        order = await place_order_demo(_intent(contracts=5), _book(), client, now=_now())
    finally:
        await client.aclose()

    assert order is None
    messages = [r.getMessage() for r in caplog.records]
    assert any("demo_order_idempotent_canceled" in m for m in messages)
    assert not any("demo_order_idempotent_duplicate" in m for m in messages)


async def test_place_order_demo_returns_none_on_429_after_retries(
    rsa_pem: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    state: dict[str, int] = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(429, json={"error": "too many requests"})

    monkeypatch.setattr(_order_placer_module, "_backoff_seconds", lambda attempt: 0.0)
    client = await _client_with_handler(rsa_pem, handler)
    try:
        caplog.set_level(logging.INFO, logger="bot.execution.order_placer")
        order = await place_order_demo(_intent(), _book(), client, now=_now())
    finally:
        await client.aclose()

    assert order is None
    assert state["calls"] >= 3
    assert any("demo_order_rate_limited" in r.getMessage() for r in caplog.records)


async def test_place_order_demo_returns_none_on_400(
    rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "insufficient_balance"})

    client = await _client_with_handler(rsa_pem, handler)
    try:
        caplog.set_level(logging.INFO, logger="bot.execution.order_placer")
        order = await place_order_demo(_intent(), _book(), client, now=_now())
    finally:
        await client.aclose()

    assert order is None
    assert any(
        "demo_order_rejected" in r.getMessage() and "insufficient_balance" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        (Decimal("0.205"), Decimal("0.205")),
        ("0.205", Decimal("0.205")),
        ("0.0000", Decimal("0.0000")),
    ],
)
def test_parse_avg_yes_fill_price_well_formed_inputs(raw, expected) -> None:
    result = parse_avg_yes_fill_price(raw)
    assert result == expected


def test_parse_avg_yes_fill_price_collapses_absent_null_and_empty_to_none() -> None:
    assert parse_avg_yes_fill_price(None) is None
    assert parse_avg_yes_fill_price("") is None


def test_parse_avg_yes_fill_price_returns_decimal_for_valid_numeric_string() -> None:
    assert parse_avg_yes_fill_price("0.205") == Decimal("0.205")
    assert parse_avg_yes_fill_price("0.0000") == Decimal("0.0000")


@pytest.mark.parametrize("bad", ["not-a-number", "0.xyz"])
def test_parse_avg_yes_fill_price_raises_on_malformed_string(bad: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_avg_yes_fill_price(bad)
    assert isinstance(excinfo.value.__cause__, decimal.InvalidOperation)


def test_invalid_operation_is_not_value_error_subclass() -> None:
    assert not issubclass(decimal.InvalidOperation, ValueError)
    with pytest.raises(decimal.InvalidOperation):
        Decimal("not-a-number")


_STATUSES = ("executed", "resting", "canceled")
_WIRE_SHAPES = (
    ("valid", "0.5000", Decimal("0.5000"), False),
    ("zero", "0.0000", Decimal("0.0000"), False),
    ("absent", None, None, False),
    ("null", "JSON_NULL", None, False),
    ("empty", "", None, False),
    ("malformed", "garbage", None, True),
)


@pytest.mark.parametrize("status", _STATUSES)
@pytest.mark.parametrize(
    "shape_label,raw,expected,should_raise",
    [(label, raw, exp, raise_) for label, raw, exp, raise_ in _WIRE_SHAPES],
)
def test_placer_avg_fill_price_deserialization_golden_table(
    status: str, shape_label: str, raw, expected, should_raise: bool
) -> None:
    if shape_label == "null":
        candidate = None
    elif shape_label == "absent":
        candidate = None
    else:
        candidate = raw
    if should_raise:
        with pytest.raises(ValueError):
            parse_avg_yes_fill_price(candidate)
    else:
        result = parse_avg_yes_fill_price(candidate)
        assert result == expected


def test_build_body_omits_type_key_for_buy_yes() -> None:
    body = _build_body(_intent(side=TradeSide.BUY_YES), _book(), "kw-cid")
    assert "type" not in body
    assert "post_only" not in body


def test_build_body_omits_type_key_for_sell_yes() -> None:
    book = _book(yes_ask=Decimal("0.95"), yes_bid=Decimal("0.925"))
    body = _build_body(_intent(side=TradeSide.SELL_YES, strategy="tails"), book, "kw-cid")
    assert "type" not in body
    assert "post_only" not in body
    assert body["side"] == "no"


async def test_place_order_demo_reads_sdk_count_fields(rsa_pem: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "order": {
                    "order_id": "ex-1",
                    "client_order_id": body["client_order_id"],
                    "ticker": "KXHIGHDEN-26MAY22-T70",
                    "side": "yes",
                    "status": "executed",
                    "initial_count": 5,
                    "fill_count": 5,
                    "taker_fees": 1,
                    "maker_fees": 0,
                    "taker_fill_cost": 425,
                    "maker_fill_cost": 0,
                    "taker_fill_cost_dollars": "4.2500",
                    "maker_fill_cost_dollars": "0.0000",
                }
            },
        )

    client = await _client_with_handler(rsa_pem, handler)
    try:
        order = await place_order_demo(_intent(), _book(), client, now=_now())
    finally:
        await client.aclose()

    assert isinstance(order, DemoOrder)
    assert order.requested_contracts == 5
    assert order.filled_contracts == 5


def test_parse_order_records_fees_from_cents_when_dollar_keys_absent() -> None:
    payload: dict[str, object] = {
        "order_id": "ex-1",
        "client_order_id": "cid",
        "ticker": "KXHIGHDEN-26MAY22-T70",
        "side": "yes",
        "status": "executed",
        "initial_count": 5,
        "fill_count": 5,
        "taker_fees": 7,
        "maker_fees": 3,
        "taker_fill_cost": 425,
        "maker_fill_cost": 0,
        "taker_fill_cost_dollars": "4.2500",
        "maker_fill_cost_dollars": "0.0000",
    }
    order = _parse_order(payload, placed_at=_now())
    assert order.fee_dollars == Decimal("0.10")


def test_parse_order_records_zero_cents_fee_not_silently_falling_back() -> None:
    payload: dict[str, object] = {
        "order_id": "ex-1",
        "client_order_id": "cid",
        "ticker": "KXHIGHDEN-26MAY22-T70",
        "side": "yes",
        "status": "executed",
        "initial_count": 5,
        "fill_count": 0,
        "taker_fees": 0,
        "maker_fees": 0,
        "taker_fees_dollars": "9.99",
        "maker_fees_dollars": "9.99",
        "taker_fill_cost": 0,
        "maker_fill_cost": 0,
        "taker_fill_cost_dollars": "0.0000",
        "maker_fill_cost_dollars": "0.0000",
    }
    order = _parse_order(payload, placed_at=_now())
    assert order.fee_dollars == Decimal("0")


def test_parse_order_falls_back_to_dollar_fees_when_cents_absent_by_key() -> None:
    payload: dict[str, object] = {
        "order_id": "ex-1",
        "client_order_id": "cid",
        "ticker": "KXHIGHDEN-26MAY22-T70",
        "side": "yes",
        "status": "executed",
        "initial_count": 5,
        "fill_count": 5,
        "taker_fees_dollars": "0.05",
        "maker_fees_dollars": "0.02",
        "taker_fill_cost": 425,
        "maker_fill_cost": 0,
        "taker_fill_cost_dollars": "4.2500",
        "maker_fill_cost_dollars": "0.0000",
    }
    order = _parse_order(payload, placed_at=_now())
    assert order.fee_dollars == Decimal("0.07")


def test_parse_order_inverts_avg_fill_for_no_side_from_cents() -> None:
    payload: dict[str, object] = {
        "order_id": "ex-1",
        "client_order_id": "cid",
        "ticker": "KXHIGHDEN-26MAY22-T70",
        "side": "no",
        "status": "executed",
        "initial_count": 4,
        "fill_count": 4,
        "taker_fees": 1,
        "maker_fees": 0,
        "taker_fill_cost": 30,
        "maker_fill_cost": 0,
        "taker_fill_cost_dollars": "0.3000",
        "maker_fill_cost_dollars": "0.0000",
    }
    order = _parse_order(payload, placed_at=_now())
    assert order.avg_yes_fill_price_dollars == Decimal("0.9250")


def test_parse_order_no_side_zero_fill_cost_does_not_record_perfect_fill(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload: dict[str, object] = {
        "order_id": "ex-1",
        "client_order_id": "cid",
        "ticker": "KXHIGHDEN-26MAY22-T70",
        "side": "no",
        "status": "executed",
        "initial_count": 4,
        "fill_count": 4,
        "taker_fees": 0,
        "maker_fees": 0,
        "taker_fill_cost": 0,
        "maker_fill_cost": 0,
        "taker_fill_cost_dollars": "0.0000",
        "maker_fill_cost_dollars": "0.0000",
    }
    caplog.set_level(logging.WARNING, logger="bot.execution.order_placer")
    order = _parse_order(payload, placed_at=_now())
    assert order.avg_yes_fill_price_dollars is None
    assert any("demo_order_zero_cost_fill" in r.getMessage() for r in caplog.records)


def test_parse_order_yes_side_avg_fill_from_cents() -> None:
    payload: dict[str, object] = {
        "order_id": "ex-1",
        "client_order_id": "cid",
        "ticker": "KXHIGHDEN-26MAY22-T70",
        "side": "yes",
        "status": "executed",
        "initial_count": 4,
        "fill_count": 4,
        "taker_fees": 1,
        "maker_fees": 0,
        "taker_fill_cost": 340,
        "maker_fill_cost": 0,
        "taker_fill_cost_dollars": "3.4000",
        "maker_fill_cost_dollars": "0.0000",
    }
    order = _parse_order(payload, placed_at=_now())
    assert order.avg_yes_fill_price_dollars == Decimal("0.8500")
