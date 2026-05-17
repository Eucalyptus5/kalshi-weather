from __future__ import annotations

import ast
import asyncio
import inspect
import time
from datetime import date
from decimal import Decimal
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from bot.execution.paper import PaperTrade, TradeSide
from bot.markets.parser import ParsedTicker, parse_ticker
from bot.validation import reconcile as reconcile_mod
from bot.validation.reconcile import (
    ACISClient,
    Reconciliation,
    reconcile_trade,
    settle_bracket,
    settle_tail,
)


def _resp(value: object) -> dict[str, object]:
    return {
        "meta": {"name": "DENVER INTL AP", "sids": ["KDEN 1"]},
        "data": [["2026-04-28", value]] if value != "__empty__" else [],
    }


def _make_trade(
    side: TradeSide,
    price: Decimal,
    contracts: int,
    fee: Decimal,
) -> PaperTrade:
    from datetime import datetime, timezone

    return PaperTrade(
        intended_at=datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc),
        market_ticker="KXHIGHDEN-26APR28-T70.5-72.5",
        side=side,
        contracts=contracts,
        simulated_price=price,
        fee_dollars=fee,
        fair_at_entry=Decimal("0.50"),
        strategy="t",
    )


async def test_acis_url_construction() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, json=_resp("57"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        await client.fetch_daily_high("KDEN", date(2026, 4, 28))

    req = captured["req"]
    parsed = urlparse(str(req.url))
    assert parsed.netloc == "data.rcc-acis.org"
    assert parsed.path == "/StnData"
    qs = parse_qs(parsed.query)
    assert qs["sid"] == ["KDEN"]
    assert qs["sdate"] == ["2026-04-28"]
    assert qs["edate"] == ["2026-04-28"]
    assert qs["elems"] == ["maxt"]
    assert qs["output"] == ["json"]
    assert req.headers.get("User-Agent")


async def test_acis_happy_path_parse() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp("57"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        result = await client.fetch_daily_high("KDEN", date(2026, 4, 28))

    assert result == Decimal("57")


async def test_acis_missing_data_sentinel() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp("M"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        result = await client.fetch_daily_high("KDEN", date(2026, 4, 28))

    assert result is None


async def test_acis_empty_data_array() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp("__empty__"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        result = await client.fetch_daily_high("KDEN", date(2026, 4, 28))

    assert result is None


async def test_acis_non_numeric_value_other_than_m() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp("trace"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        result = await client.fetch_daily_high("KDEN", date(2026, 4, 28))

    assert result is None


async def test_acis_negative_integer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp("-12"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        result = await client.fetch_daily_high("KMSP", date(2026, 4, 28))

    assert result == Decimal("-12")


async def test_acis_raises_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        with pytest.raises(httpx.HTTPStatusError):
            await client.fetch_daily_high("KDEN", date(2026, 4, 28))


async def test_default_client_is_closed_by_aclose(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp("57"))

    transport = httpx.MockTransport(handler)
    real_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    client = ACISClient()
    result = await client.fetch_daily_high("KDEN", date(2026, 4, 28))
    assert result == Decimal("57")
    await client.aclose()
    assert client._http.is_closed


async def test_caller_owned_client_not_closed_by_aclose() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp("57"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        await client.fetch_daily_high("KDEN", date(2026, 4, 28))
        await client.aclose()
        assert not http.is_closed


async def test_acis_wait_for_fires_on_hung_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(60)
        return httpx.Response(200, json=_resp("57"))

    monkeypatch.setattr(reconcile_mod, "_ACIS_FETCH_TIMEOUT_SECONDS", 0.5)
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = ACISClient(http_client=http)
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await client.fetch_daily_high("KDEN", date(2026, 4, 28))
        elapsed = time.monotonic() - start
    assert 0.4 < elapsed < 2.0


def test_settle_bracket_interior() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5-72.5")
    assert settle_bracket(parsed, Decimal("71")) is True
    assert settle_bracket(parsed, Decimal("72")) is True
    assert settle_bracket(parsed, Decimal("70")) is False
    assert settle_bracket(parsed, Decimal("73")) is False


def test_settle_bracket_boundaries() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5-72.5")
    assert settle_bracket(parsed, Decimal("70.5")) is True
    assert settle_bracket(parsed, Decimal("72.5")) is False


def test_settle_bracket_rejects_tail() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5")
    with pytest.raises(ValueError):
        settle_bracket(parsed, Decimal("71"))


def test_settle_tail_below() -> None:
    parsed = ParsedTicker(
        series="KXHIGHDEN",
        event_date=date(2026, 5, 6),
        is_monthly=False,
        strikes=(Decimal("42.5"),),
        kind="below",
        raw="synthetic-below",
    )
    assert settle_tail(parsed, Decimal("42")) is True
    assert settle_tail(parsed, Decimal("43")) is False
    assert settle_tail(parsed, Decimal("42.5")) is False


def test_settle_tail_above() -> None:
    parsed = parse_ticker("KXHIGHDEN-26MAY06-T96")
    assert settle_tail(parsed, Decimal("97")) is True
    assert settle_tail(parsed, Decimal("95")) is False
    assert settle_tail(parsed, Decimal("96")) is True


def test_settle_tail_rejects_bracket() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5-72.5")
    with pytest.raises(ValueError):
        settle_tail(parsed, Decimal("71"))


def test_reconcile_trade_buy_yes_yes_true() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5-72.5")
    trade = _make_trade(TradeSide.BUY_YES, Decimal("0.40"), 10, Decimal("0.05"))
    out = reconcile_trade(trade, parsed, Decimal("71"))
    assert isinstance(out, Reconciliation)
    assert out.observed_high == Decimal("71")
    assert out.yes_settled is True
    assert out.won is True
    assert out.realized_pnl == Decimal("5.95")


def test_reconcile_trade_buy_yes_yes_false() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5-72.5")
    trade = _make_trade(TradeSide.BUY_YES, Decimal("0.40"), 10, Decimal("0.05"))
    out = reconcile_trade(trade, parsed, Decimal("73"))
    assert out.yes_settled is False
    assert out.won is False
    assert out.realized_pnl == Decimal("-4.05")


def test_reconcile_trade_sell_yes_yes_false() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5-72.5")
    trade = _make_trade(TradeSide.SELL_YES, Decimal("0.85"), 10, Decimal("0.04"))
    out = reconcile_trade(trade, parsed, Decimal("73"))
    assert out.yes_settled is False
    assert out.won is True
    assert out.realized_pnl == Decimal("8.46")


def test_reconcile_trade_sell_yes_yes_true() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5-72.5")
    trade = _make_trade(TradeSide.SELL_YES, Decimal("0.85"), 10, Decimal("0.04"))
    out = reconcile_trade(trade, parsed, Decimal("71"))
    assert out.yes_settled is True
    assert out.won is False
    assert out.realized_pnl == Decimal("-1.54")


def test_reconcile_trade_dispatches_below_tail() -> None:
    parsed = parse_ticker("KXHIGHDEN-26MAY06-B42.5")
    trade = _make_trade(TradeSide.BUY_YES, Decimal("0.40"), 10, Decimal("0.05"))
    out = reconcile_trade(trade, parsed, Decimal("42"))
    assert out.yes_settled is True
    assert out.won is True


def test_reconcile_trade_dispatches_above_tail() -> None:
    parsed = parse_ticker("KXHIGHDEN-26MAY06-T96")
    trade = _make_trade(TradeSide.BUY_YES, Decimal("0.40"), 10, Decimal("0.05"))
    out = reconcile_trade(trade, parsed, Decimal("95"))
    assert out.yes_settled is False
    assert out.won is False


def test_b_form_bracket_semantic_pending_defect_c() -> None:
    from datetime import datetime, timezone

    parsed = parse_ticker("KXHIGHDEN-26MAY19-B49.5")
    trade = PaperTrade(
        intended_at=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
        market_ticker="KXHIGHDEN-26MAY19-B49.5",
        side=TradeSide.SELL_YES,
        contracts=10,
        simulated_price=Decimal("0.85"),
        fee_dollars=Decimal("0.04"),
        fair_at_entry=Decimal("0.50"),
        strategy="t",
    )
    out = reconcile_trade(trade, parsed, Decimal("48"))
    assert out.yes_settled is False


def test_reconcile_trade_reads_only_documented_papertrade_attributes() -> None:
    src = inspect.getsource(reconcile_trade)
    tree = ast.parse(src)
    accessed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "trade":
                accessed.add(node.attr)
    assert accessed == {"side", "simulated_price", "contracts", "fee_dollars"}
