from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from websockets.exceptions import ConnectionClosed

from bot.config import Settings
from bot.kalshi_ws import (
    BookDelta,
    BookLevel,
    BookSnapshot,
    GapDetected,
    KalshiWSClient,
    TradePrint,
)

_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("ws") / "prod.pem"
    path.write_bytes(pem)
    return path


def _settings_with_pem(pem_path: Path) -> Settings:
    return Settings(
        mode="paper",
        kalshi_prod_key_id="prod-key-id",
        kalshi_prod_private_key_path=pem_path,
    )


def _settings_no_key() -> Settings:
    return Settings(mode="paper", kalshi_prod_key_id=None, kalshi_prod_private_key_path=None)


class FakeWSConn:
    def __init__(self, messages: list[str], end_exception: BaseException | None = None) -> None:
        self._messages: deque[str] = deque(messages)
        self._end_exception = end_exception
        self.sent: list[str] = []
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if self._messages:
            return self._messages.popleft()
        if self._end_exception is not None:
            raise self._end_exception
        await asyncio.sleep(3600)
        raise RuntimeError("recv drained without terminator")

    async def close(self) -> None:
        self.closed = True


class FakeConnectCtx:
    def __init__(self, conn: FakeWSConn) -> None:
        self.conn = conn

    async def __aenter__(self) -> FakeWSConn:
        return self.conn

    async def __aexit__(self, *args: object) -> None:
        await self.conn.close()


def _make_connect_fn(conns: list[FakeWSConn]) -> tuple[object, list[tuple[str, dict[str, str]]]]:
    ctxs = deque(FakeConnectCtx(c) for c in conns)
    calls: list[tuple[str, dict[str, str]]] = []

    def connect(url: str, headers: dict[str, str]) -> FakeConnectCtx:
        calls.append((url, headers))
        return ctxs.popleft()

    return connect, calls


def _snapshot_frame(sid: int, seq: int, ticker: str = "KXHIGHDEN-26MAY08-T96.5") -> str:
    return json.dumps(
        {
            "type": "orderbook_snapshot",
            "sid": sid,
            "seq": seq,
            "msg": {
                "market_ticker": ticker,
                "yes_dollars_fp": [["0.0800", "300.00"], ["0.2200", "333.00"]],
                "no_dollars_fp": [["0.5400", "20.00"], ["0.5600", "146.00"]],
            },
        }
    )


def _delta_frame(sid: int, seq: int, ticker: str = "KXHIGHDEN-26MAY08-T96.5") -> str:
    return json.dumps(
        {
            "type": "orderbook_delta",
            "sid": sid,
            "seq": seq,
            "msg": {
                "market_ticker": ticker,
                "price_dollars": "0.960",
                "delta_fp": "-54.00",
                "side": "yes",
                "ts_ms": 1669149841000,
            },
        }
    )


def _trade_frame(sid: int, ticker: str = "KXHIGHDEN-26MAY08-T96.5") -> str:
    return json.dumps(
        {
            "type": "trade",
            "sid": sid,
            "msg": {
                "trade_id": "abc",
                "market_ticker": ticker,
                "yes_price_dollars": "0.360",
                "no_price_dollars": "0.640",
                "count_fp": "136.00",
                "taker_outcome_side": "no",
                "taker_side": "no",
                "ts": 1669149841,
                "ts_ms": 1669149841000,
            },
        }
    )


class _RecordingAuth:
    calls: list[tuple[str, str]] = []

    def __init__(self, *, key_id: str, private_key_pem: str) -> None:
        self.key_id = key_id
        self.private_key_pem = private_key_pem

    def create_auth_headers(self, method: str, url: str) -> dict[str, str]:
        type(self).calls.append((method, url))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": "1700000000000",
            "KALSHI-ACCESS-SIGNATURE": "sig",
        }


async def test_snapshot_decode(rsa_pem: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    conn = FakeWSConn([_snapshot_frame(sid=2, seq=2)], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn, FakeWSConn([], ConnectionClosed(None, None))])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    snap = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(snap, BookSnapshot)
    assert snap.ticker == "KXHIGHDEN-26MAY08-T96.5"
    assert snap.sid == 2
    assert snap.seq == 2
    assert snap.yes_levels == (
        BookLevel(Decimal("0.0800"), Decimal("300.00")),
        BookLevel(Decimal("0.2200"), Decimal("333.00")),
    )
    assert isinstance(snap.yes_levels[0].price, Decimal)
    assert isinstance(snap.yes_levels[0].size, Decimal)


async def test_snapshot_decode_missing_side(rsa_pem: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    yes_only = json.dumps(
        {
            "type": "orderbook_snapshot",
            "sid": 4,
            "seq": 1,
            "msg": {
                "market_ticker": "KXHIGHDEN-26MAY08-T96.5",
                "yes_dollars_fp": [["0.1000", "10.00"]],
            },
        }
    )
    no_only = json.dumps(
        {
            "type": "orderbook_snapshot",
            "sid": 5,
            "seq": 1,
            "msg": {
                "market_ticker": "KXHIGHDEN-26MAY08-T96.5",
                "no_dollars_fp": [["0.5000", "5.00"]],
            },
        }
    )
    conn = FakeWSConn([yes_only, no_only], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn, FakeWSConn([], ConnectionClosed(None, None))])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    first = await asyncio.wait_for(anext(events), timeout=1.0)
    second = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(first, BookSnapshot)
    assert first.no_levels == ()
    assert first.yes_levels == (BookLevel(Decimal("0.1000"), Decimal("10.00")),)
    assert isinstance(second, BookSnapshot)
    assert second.yes_levels == ()
    assert second.no_levels == (BookLevel(Decimal("0.5000"), Decimal("5.00")),)


async def test_delta_decode(rsa_pem: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    conn = FakeWSConn(
        [_snapshot_frame(sid=1, seq=1), _delta_frame(sid=1, seq=2)],
        end_exception=ConnectionClosed(None, None),
    )
    connect_fn, _ = _make_connect_fn([conn, FakeWSConn([], ConnectionClosed(None, None))])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    await asyncio.wait_for(anext(events), timeout=1.0)
    delta = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(delta, BookDelta)
    assert delta.ticker == "KXHIGHDEN-26MAY08-T96.5"
    assert delta.sid == 1
    assert delta.seq == 2
    assert delta.side == "yes"
    assert delta.price == Decimal("0.960")
    assert delta.delta == Decimal("-54.00")
    assert isinstance(delta.price, Decimal)
    assert isinstance(delta.delta, Decimal)
    assert delta.ts_ms == 1669149841000


async def test_trade_decode_prefers_outcome_side(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    both = json.dumps(
        {
            "type": "trade",
            "sid": 11,
            "msg": {
                "trade_id": "8f5b9f2e-1234-4abc-9def-000000000001",
                "market_ticker": "KXHIGHDEN-26MAY08-T96.5",
                "yes_price_dollars": "0.360",
                "no_price_dollars": "0.640",
                "count_fp": "136.00",
                "taker_side": "yes",
                "taker_outcome_side": "no",
                "ts_ms": 1669149841000,
            },
        }
    )
    conn = FakeWSConn([both], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn, FakeWSConn([], ConnectionClosed(None, None))])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["trade"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    trade = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(trade, TradePrint)
    assert trade.trade_id == "8f5b9f2e-1234-4abc-9def-000000000001"
    assert trade.taker_side == "no"
    assert trade.yes_price == Decimal("0.360")
    assert trade.no_price == Decimal("0.640")
    assert trade.count == Decimal("136.00")
    assert trade.ts_ms == 1669149841000


async def test_trade_decode_falls_back_to_taker_side(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    only_legacy = json.dumps(
        {
            "type": "trade",
            "sid": 11,
            "msg": {
                "trade_id": "abc",
                "market_ticker": "KXHIGHDEN-26MAY08-T96.5",
                "yes_price_dollars": "0.100",
                "no_price_dollars": "0.900",
                "count_fp": "1.00",
                "taker_side": "yes",
                "ts_ms": 1669149841000,
            },
        }
    )
    conn = FakeWSConn([only_legacy], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn, FakeWSConn([], ConnectionClosed(None, None))])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["trade"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    trade = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(trade, TradePrint)
    assert trade.taker_side == "yes"


async def test_seq_gap_detection(rsa_pem: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    frames = [
        _snapshot_frame(sid=1, seq=1),
        _delta_frame(sid=1, seq=2),
        _delta_frame(sid=1, seq=4),
    ]
    conn = FakeWSConn(frames, end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn, FakeWSConn([], ConnectionClosed(None, None))])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    e1 = await asyncio.wait_for(anext(events), timeout=1.0)
    e2 = await asyncio.wait_for(anext(events), timeout=1.0)
    e3 = await asyncio.wait_for(anext(events), timeout=1.0)
    e4 = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(e1, BookSnapshot)
    assert isinstance(e2, BookDelta)
    assert e2.seq == 2
    assert isinstance(e3, GapDetected)
    assert e3.reason == "seq_skip"
    assert e3.sid == 1
    assert e3.last_seq == 2
    assert e3.next_seq == 4
    assert isinstance(e4, BookDelta)
    assert e4.seq == 4


async def test_resubscribe_after_close_does_not_flag_gap(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    conn1 = FakeWSConn(
        [_snapshot_frame(sid=1, seq=1), _delta_frame(sid=1, seq=2)],
        end_exception=ConnectionClosed(None, None),
    )
    conn2 = FakeWSConn([_snapshot_frame(sid=1, seq=1)], end_exception=ConnectionClosed(None, None))
    spare = FakeWSConn([], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn1, conn2, spare])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["A"])

    events = client.events()
    assert isinstance(await asyncio.wait_for(anext(events), timeout=1.0), BookSnapshot)
    assert isinstance(await asyncio.wait_for(anext(events), timeout=1.0), BookDelta)

    await client.aclose()
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["A", "B"])

    events = client.events()
    first = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(first, BookSnapshot)
    assert first.sid == 1
    assert first.seq == 1


async def test_terminal_error_10_triggers_resubscribe(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    error_frame = json.dumps({"type": "error", "sid": 1, "msg": {"code": 10, "msg": "channel"}})
    conn1 = FakeWSConn([_snapshot_frame(sid=1, seq=1), error_frame])
    conn2 = FakeWSConn([_snapshot_frame(sid=2, seq=1)], end_exception=ConnectionClosed(None, None))
    connect_fn, calls = _make_connect_fn([conn1, conn2])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    e1 = await asyncio.wait_for(anext(events), timeout=1.0)
    e2 = await asyncio.wait_for(anext(events), timeout=1.0)
    e3 = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(e1, BookSnapshot)
    assert e1.sid == 1
    assert isinstance(e2, GapDetected)
    assert e2.reason == "terminal_error_10"
    assert isinstance(e3, BookSnapshot)
    assert e3.sid == 2
    assert len(calls) == 2
    assert len(conn2.sent) == 1
    resub = json.loads(conn2.sent[0])
    assert resub["cmd"] == "subscribe"
    assert resub["params"]["market_tickers"] == ["KXHIGHDEN-26MAY08-T96.5"]


async def test_connection_reset_triggers_resubscribe(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    conn1 = FakeWSConn([], end_exception=ConnectionClosed(None, None))
    conn2 = FakeWSConn([_snapshot_frame(sid=9, seq=1)], end_exception=ConnectionClosed(None, None))
    connect_fn, calls = _make_connect_fn([conn1, conn2])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["A"])

    events = client.events()
    e1 = await asyncio.wait_for(anext(events), timeout=1.0)
    e2 = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(e1, GapDetected)
    assert e1.reason == "connection_reset"
    assert isinstance(e2, BookSnapshot)
    assert e2.sid == 9
    assert len(calls) == 2


async def test_subscribe_uses_market_tickers_plural(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    conn = FakeWSConn([], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["A", "B"])
    await client.aclose()

    assert len(conn.sent) == 1
    payload = json.loads(conn.sent[0])
    assert payload == {
        "id": 1,
        "cmd": "subscribe",
        "params": {"channels": ["orderbook_delta"], "market_tickers": ["A", "B"]},
    }


async def test_subscribe_id_increments_on_resubscribe(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    conn1 = FakeWSConn([], end_exception=ConnectionClosed(None, None))
    conn2 = FakeWSConn([_snapshot_frame(sid=1, seq=1)], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn1, conn2])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["A"])

    events = client.events()
    gap = await asyncio.wait_for(anext(events), timeout=1.0)
    snap = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(gap, GapDetected)
    assert isinstance(snap, BookSnapshot)
    assert json.loads(conn1.sent[0])["id"] == 1
    assert json.loads(conn2.sent[0])["id"] == 2


async def test_subscribe_rejects_unknown_channel(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    conn = FakeWSConn([], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    with pytest.raises(ValueError):
        await client.subscribe(["private_orders"], ["A"])
    await client.aclose()


def test_no_order_methods_on_client() -> None:
    forbidden_prefixes = ("place", "cancel", "submit", "create_order", "delete_order")
    for name in dir(KalshiWSClient):
        assert not any(name.startswith(p) for p in forbidden_prefixes), name


async def test_auth_header_shape(rsa_pem: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingAuth.calls.clear()
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    conn = FakeWSConn([], end_exception=ConnectionClosed(None, None))
    connect_fn, calls = _make_connect_fn([conn])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["A"])
    await client.aclose()

    assert _RecordingAuth.calls == [("GET", "/trade-api/ws/v2")]
    assert calls[0][1]["KALSHI-ACCESS-KEY"] == "prod-key-id"


async def test_trade_has_no_gap_detection(rsa_pem: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    frames = [
        _snapshot_frame(sid=1, seq=1),
        _trade_frame(sid=1),
        _delta_frame(sid=1, seq=2),
    ]
    conn = FakeWSConn(frames, end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn, FakeWSConn([], ConnectionClosed(None, None))])
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem), _URL, connect_fn=connect_fn, backoff_seconds=(0.0,)
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta", "trade"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    e1 = await asyncio.wait_for(anext(events), timeout=1.0)
    e2 = await asyncio.wait_for(anext(events), timeout=1.0)
    e3 = await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(e1, BookSnapshot)
    assert isinstance(e2, TradePrint)
    assert isinstance(e3, BookDelta)


async def test_no_key_raises(rsa_pem: Path) -> None:
    client = KalshiWSClient(_settings_no_key(), _URL)
    with pytest.raises(RuntimeError):
        await client.aopen()


async def test_frame_sink_sees_every_frame(rsa_pem: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.kalshi_ws.KalshiAuth", _RecordingAuth)
    valid = _snapshot_frame(sid=1, seq=1)
    error = json.dumps({"type": "error", "sid": 1, "msg": {"code": 99, "msg": "throttled"}})
    unknown = json.dumps({"type": "market_lifecycle", "sid": 1, "msg": {"open": True}})
    garbage = "not json {"
    conn = FakeWSConn([valid, error, unknown, garbage], end_exception=ConnectionClosed(None, None))
    connect_fn, _ = _make_connect_fn([conn])
    seen: list[tuple[str, datetime]] = []
    client = KalshiWSClient(
        _settings_with_pem(rsa_pem),
        _URL,
        connect_fn=connect_fn,
        backoff_seconds=(0.0,),
        frame_sink=lambda raw, received_at: seen.append((raw, received_at)),
    )
    await client.aopen()
    await client.subscribe(["orderbook_delta"], ["KXHIGHDEN-26MAY08-T96.5"])

    events = client.events()
    snap = await asyncio.wait_for(anext(events), timeout=1.0)
    with pytest.raises(json.JSONDecodeError):
        await asyncio.wait_for(anext(events), timeout=1.0)
    await client.aclose()

    assert isinstance(snap, BookSnapshot)
    assert snap.seq == 1
    assert [raw for raw, _ in seen] == [valid, error, unknown, garbage]
    assert all(received_at.tzinfo is not None for _, received_at in seen)
