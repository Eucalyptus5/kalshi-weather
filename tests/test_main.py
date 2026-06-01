from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect as py_inspect
import logging
import re as _re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from sqlalchemy import select

import bot.main as bot_main
from bot.execution.gate_cost_basis import cost_per_contract_from_book
from bot.execution.paper import PaperTrade, TradeIntent, TradeSide
from bot.forecast.cdf import EnsembleCDF
from bot.forecast.open_meteo import StationForecast
from bot.kalshi_client import KalshiDemoClient, KalshiMarket, KalshiOrderbook
from bot.main import (
    STATIONS,
    STRATEGY_BLACKLIST,
    App,
    _build_intents,
    _checkpoint_wal,
    _compute_lead_time_hours,
    _forecast_loop,
    _gate_ctx_for,
    _on_demand_reconcile,
    _parse_duration,
    _parse_series_arg,
    _settlement_loop,
    evaluate_strategies,
    main,
    reconcile_settled_trades,
    refresh_forecasts,
    refresh_markets,
    run,
)
from bot.markets.parser import parse_ticker
from bot.storage.sqlite import (
    Base,
    Forecast,
    GateFailure,
    Market,
    OrderbookSnapshot,
    PaperTradeRow,
    SimulatedPnl,
    make_engine,
    make_session_factory,
)
from bot.strategy import edge as edge_strategy
from bot.strategy import tails as tails_strategy
from bot.strategy.sizing import sigma_t_median_for_lead


class _StubMeteo:
    def __init__(self, forecasts: dict[str, StationForecast] | StationForecast) -> None:
        if isinstance(forecasts, StationForecast):
            self._forecasts = {forecasts.station: forecasts}
        else:
            self._forecasts = forecasts
        self.calls: list[tuple[str, float, float, str]] = []

    async def fetch_station(
        self,
        station: str,
        latitude: float,
        longitude: float,
        timezone: str,
        forecast_days: int = 7,
    ) -> StationForecast:
        self.calls.append((station, latitude, longitude, timezone))
        return self._forecasts[station]

    async def aclose(self) -> None:
        return None


class _StubKalshi:
    def __init__(
        self,
        markets: list[KalshiMarket],
        orderbooks: dict[str, KalshiOrderbook],
    ) -> None:
        self._markets = markets
        self._orderbooks = orderbooks

    async def aopen(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def list_open_markets_for_series(self, series_prefix: str) -> list[KalshiMarket]:
        return [m for m in self._markets if m.ticker.startswith(f"{series_prefix}-")]

    async def get_orderbook(self, ticker: str) -> KalshiOrderbook:
        return self._orderbooks[ticker]


class _StubACIS:
    def __init__(self, value: Decimal | None) -> None:
        self.value = value
        self.calls: list[tuple[str, date]] = []

    async def fetch_daily_high(self, station: str, settled_date: date) -> Decimal | None:
        self.calls.append((station, settled_date))
        return self.value

    async def aclose(self) -> None:
        return None


class _MultiStationACIS:
    def __init__(self, values: dict[tuple[str, date], Decimal | None]) -> None:
        self._values = values
        self.calls: list[tuple[str, date]] = []

    async def fetch_daily_high(self, station: str, settled_date: date) -> Decimal | None:
        self.calls.append((station, settled_date))
        return self._values.get((station, settled_date))

    async def aclose(self) -> None:
        return None


def _make_app(
    meteo: _StubMeteo | None = None,
    kalshi: _StubKalshi | None = None,
    acis: object | None = None,
    series_list: tuple[str, ...] = ("KXHIGHDEN",),
) -> App:
    from bot.config import Settings

    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    settings = Settings(mode="paper")
    return App(
        settings=settings,
        engine=engine,
        session_factory=sf,
        meteo=meteo,  # type: ignore[arg-type]
        kalshi=kalshi,  # type: ignore[arg-type]
        acis=acis if acis is not None else _StubACIS(None),  # type: ignore[arg-type]
        series_list=series_list,
    )


def _forecast_with_two_days() -> StationForecast:
    rng = np.random.default_rng(0)
    members_a = rng.normal(72.0, 5.0, size=31)
    members_b = rng.normal(78.0, 5.0, size=31)
    return StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={
            date(2026, 5, 7): members_a,
            date(2026, 5, 8): members_b,
        },
    )


def _book_from(
    ticker: str,
    yes_ask: str,
    yes_bid: str,
    *,
    now: datetime | None = None,
    snapshot_at: datetime | None = None,
    yes_ask_depth: int = 1000,
    yes_bid_depth: int = 1000,
    no_ask_depth: int = 1000,
    no_bid_depth: int = 1000,
) -> KalshiOrderbook:
    if snapshot_at is None:
        anchor = now if now is not None else datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
        snapshot_at = anchor - timedelta(seconds=1)
    return KalshiOrderbook(
        ticker=ticker,
        yes_ask=Decimal(yes_ask),
        yes_bid=Decimal(yes_bid),
        no_ask=Decimal("1") - Decimal(yes_bid),
        no_bid=Decimal("1") - Decimal(yes_ask),
        yes_ask_depth=yes_ask_depth,
        yes_bid_depth=yes_bid_depth,
        no_ask_depth=no_ask_depth,
        no_bid_depth=no_bid_depth,
        snapshot_at=snapshot_at,
    )


def _market_from(
    ticker: str,
    yes_ask: str,
    yes_bid: str,
    close_at: datetime,
    status: str = "active",
) -> KalshiMarket:
    series = ticker.split("-", 1)[0]
    event_ticker = "-".join(ticker.split("-")[:2])
    return KalshiMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        series=series,
        status=status,
        close_time=close_at,
        yes_ask=Decimal(yes_ask),
        yes_bid=Decimal(yes_bid),
    )


def _lift_caps(monkeypatch: pytest.MonkeyPatch, *, lift_market: bool = True) -> None:
    if lift_market:
        monkeypatch.setattr(bot_main, "market_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "event_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "series_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "aggregate_exposure_cap", lambda app=None: Decimal("100000"))


def _lift_per_key_caps_keep_aggregate(
    monkeypatch: pytest.MonkeyPatch, *, aggregate_cap: Decimal
) -> None:
    monkeypatch.setattr(bot_main, "market_position_cap", lambda app=None: Decimal("100"))
    monkeypatch.setattr(bot_main, "event_position_cap", lambda app=None: Decimal("100"))
    monkeypatch.setattr(bot_main, "series_position_cap", lambda app=None: Decimal("100"))
    monkeypatch.setattr(bot_main, "aggregate_exposure_cap", lambda app=None: aggregate_cap)


_AGGREGATE_TEST_PRICES: tuple[tuple[Decimal, Decimal, Decimal], ...] = (
    (Decimal("0.30"), Decimal("0.27"), Decimal("0.37")),
    (Decimal("0.10"), Decimal("0.09"), Decimal("0.19")),
    (Decimal("0.73"), Decimal("0.70"), Decimal("0.63")),
)


def _intents_defaults() -> dict:
    return {
        "sigma_T_median": Decimal("2.0"),
        "event_budget_remaining": Decimal("9999"),
        "market_budget_remaining": Decimal("9999"),
        "buy_yes_depth": 10_000,
        "sell_yes_depth": 10_000,
    }


def test_aggregate_test_prices_route_through_sizer() -> None:
    for yes_ask, yes_bid, fair in _AGGREGATE_TEST_PRICES:
        ctx = edge_strategy.EdgeContext(
            yes_ask=yes_ask,
            yes_bid=yes_bid,
            fair_yes=fair,
            ensemble_spread=Decimal("3.0"),
            bankroll=Decimal("500"),
            is_same_day=False,
            is_blacklisted=False,
            nbm_divergence=None,
            sigma_T_median=Decimal("3.0"),
            event_budget_remaining=Decimal("7.50"),
            market_budget_remaining=Decimal("9999"),
            depth_at_price=10_000,
            price_per_contract=yes_ask if fair > yes_ask else Decimal("1") - yes_bid,
        )
        sig = edge_strategy.evaluate(ctx)
        assert sig.action is not edge_strategy.EdgeAction.SKIP
        assert sig.contracts > 0
        assert sig.notional_dollars <= Decimal("7.50")


def test_parse_duration_hours() -> None:
    assert _parse_duration("24h") == timedelta(hours=24)


def test_parse_duration_minutes() -> None:
    assert _parse_duration("30m") == timedelta(minutes=30)


def test_parse_duration_seconds() -> None:
    assert _parse_duration("60s") == timedelta(seconds=60)


def test_parse_duration_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _parse_duration("forever")
    with pytest.raises(ValueError):
        _parse_duration("3d")
    with pytest.raises(ValueError):
        _parse_duration("")


def test_stations_map_has_all_twenty() -> None:
    assert len(STATIONS) == 20
    assert "KXHIGHINFLATION" not in STATIONS
    assert STATIONS["KXHIGHTPHX"].timezone == "America/Phoenix"
    assert STATIONS["KXHIGHDEN"].station == "KDEN"
    for series, cfg in STATIONS.items():
        assert cfg.series == series


def test_strategy_blacklist_is_lax_and_mia_only() -> None:
    assert STRATEGY_BLACKLIST == frozenset({"KXHIGHLAX", "KXHIGHMIA"})


async def test_refresh_forecasts_persists_rows_and_populates_cache() -> None:
    meteo = _StubMeteo(_forecast_with_two_days())
    app = _make_app(meteo=meteo)

    count = await refresh_forecasts(app)

    assert count == 2
    with app.session_factory() as session:
        rows = session.scalars(select(Forecast)).all()
    assert len(rows) == 2
    assert {r.valid_date for r in rows} == {date(2026, 5, 7), date(2026, 5, 8)}
    assert all(r.station == "KDEN" for r in rows)

    assert ("KDEN", date(2026, 5, 7)) in app.forecast_cdfs
    assert ("KDEN", date(2026, 5, 8)) in app.forecast_cdfs

    cdf = app.forecast_cdfs[("KDEN", date(2026, 5, 7))]
    assert 0.0 < cdf.cdf(72.0) < 1.0

    assert ("KDEN", date(2026, 5, 7)) in app.ensemble_spreads
    assert app.ensemble_spreads[("KDEN", date(2026, 5, 7))] > Decimal("0")


async def test_refresh_forecasts_runs_for_each_series() -> None:
    rng = np.random.default_rng(7)
    den_fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): rng.normal(72.0, 5.0, size=31)},
    )
    nyc_fc = StationForecast(
        station="KNYC",
        latitude=40.7790,
        longitude=-73.9692,
        timezone="America/New_York",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): rng.normal(85.0, 4.0, size=31)},
    )
    meteo = _StubMeteo({"KDEN": den_fc, "KNYC": nyc_fc})
    app = _make_app(meteo=meteo, series_list=("KXHIGHDEN", "KXHIGHNY"))

    count = await refresh_forecasts(app)

    assert count == 2
    assert ("KDEN", date(2026, 5, 8)) in app.forecast_cdfs
    assert ("KNYC", date(2026, 5, 8)) in app.forecast_cdfs
    with app.session_factory() as session:
        stations = sorted({r.station for r in session.scalars(select(Forecast)).all()})
    assert stations == ["KDEN", "KNYC"]
    called_stations = sorted({c[0] for c in meteo.calls})
    assert called_stations == ["KDEN", "KNYC"]


async def test_refresh_markets_persists_markets_and_orderbooks() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market_a = _market_from("KXHIGHDEN-26MAY07-T70-75", "0.45", "0.43", close_at)
    market_b = _market_from("KXHIGHDEN-26MAY07-T75-80", "0.30", "0.28", close_at)
    book_a = _book_from(market_a.ticker, "0.45", "0.43")
    book_b = _book_from(market_b.ticker, "0.30", "0.28")

    kalshi = _StubKalshi(
        markets=[market_a, market_b],
        orderbooks={market_a.ticker: book_a, market_b.ticker: book_b},
    )
    app = _make_app(kalshi=kalshi)

    count = await refresh_markets(app)

    assert count == 2
    with app.session_factory() as session:
        market_rows = session.scalars(select(Market)).all()
        ob_rows = session.scalars(select(OrderbookSnapshot)).all()
    assert {m.ticker for m in market_rows} == {market_a.ticker, market_b.ticker}
    assert {ob.ticker for ob in ob_rows} == {market_a.ticker, market_b.ticker}

    assert app.latest_markets[market_a.ticker] is market_a
    assert app.latest_orderbooks[market_b.ticker] is book_b


async def test_refresh_markets_persists_per_series() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    den = _market_from("KXHIGHDEN-26MAY07-T70-75", "0.45", "0.43", close_at)
    nyc = _market_from("KXHIGHNY-26MAY07-T80-85", "0.30", "0.28", close_at)
    den_book = _book_from(den.ticker, "0.45", "0.43")
    nyc_book = _book_from(nyc.ticker, "0.30", "0.28")

    kalshi = _StubKalshi(
        markets=[den, nyc],
        orderbooks={den.ticker: den_book, nyc.ticker: nyc_book},
    )
    app = _make_app(kalshi=kalshi, series_list=("KXHIGHDEN", "KXHIGHNY"))

    count = await refresh_markets(app)

    assert count == 2
    with app.session_factory() as session:
        rows = session.scalars(select(Market)).all()
    by_series = {r.series for r in rows}
    assert by_series == {"KXHIGHDEN", "KXHIGHNY"}


async def test_refresh_markets_upserts_existing_ticker() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHDEN-26MAY07-T70-75", "0.45", "0.43", close_at)
    book = _book_from(market.ticker, "0.45", "0.43")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(kalshi=kalshi)

    await refresh_markets(app)
    await refresh_markets(app)

    with app.session_factory() as session:
        rows = session.scalars(select(Market)).all()
    assert len(rows) == 1


class _PartialFailKalshi(_StubKalshi):
    def __init__(
        self,
        markets: list[KalshiMarket],
        orderbooks: dict[str, KalshiOrderbook],
        fail_on: str,
        exc: Exception,
    ) -> None:
        super().__init__(markets, orderbooks)
        self._fail_on = fail_on
        self._exc = exc

    async def get_orderbook(self, ticker: str) -> KalshiOrderbook:
        if ticker == self._fail_on:
            raise self._exc
        return self._orderbooks[ticker]


async def test_refresh_markets_isolates_orderbook_http_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    tickers = [
        "KXHIGHDEN-26MAY07-T60-65",
        "KXHIGHDEN-26MAY07-T65-70",
        "KXHIGHDEN-26MAY07-T70-75",
        "KXHIGHDEN-26MAY07-T75-80",
        "KXHIGHDEN-26MAY07-T80-85",
    ]
    markets = [_market_from(t, "0.20", "0.18", close_at) for t in tickers]
    books = {t: _book_from(t, "0.20", "0.18") for t in tickers}

    fail_ticker = tickers[2]
    response = httpx.Response(502, request=httpx.Request("GET", "https://x"))
    exc = httpx.HTTPStatusError("502", request=response.request, response=response)
    kalshi = _PartialFailKalshi(markets, books, fail_on=fail_ticker, exc=exc)

    app = _make_app(kalshi=kalshi)
    caplog.set_level(logging.WARNING, logger="bot.main")

    count = await refresh_markets(app)

    assert count == 4
    with app.session_factory() as session:
        ob_tickers = {ob.ticker for ob in session.scalars(select(OrderbookSnapshot)).all()}
    assert fail_ticker not in ob_tickers
    assert len(ob_tickers) == 4
    assert fail_ticker not in app.latest_markets
    assert fail_ticker not in app.latest_orderbooks
    matches = [
        r
        for r in caplog.records
        if "kalshi_orderbook_fetch_failed" in r.getMessage() and fail_ticker in r.getMessage()
    ]
    assert matches


async def test_refresh_markets_isolates_orderbook_key_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    good = _market_from("KXHIGHDEN-26MAY07-T60-65", "0.20", "0.18", close_at)
    bad = _market_from("KXHIGHDEN-26MAY07-T65-70", "0.30", "0.28", close_at)
    books = {good.ticker: _book_from(good.ticker, "0.20", "0.18")}

    kalshi = _PartialFailKalshi(
        [good, bad], books, fail_on=bad.ticker, exc=KeyError("orderbook_fp")
    )
    app = _make_app(kalshi=kalshi)
    caplog.set_level(logging.WARNING, logger="bot.main")

    count = await refresh_markets(app)

    assert count == 1
    assert bad.ticker not in app.latest_markets
    assert good.ticker in app.latest_markets
    matches = [
        r
        for r in caplog.records
        if "kalshi_orderbook_fetch_failed" in r.getMessage() and bad.ticker in r.getMessage()
    ]
    assert matches


@pytest.fixture(scope="module")
def _rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("kalshi") / "demo.pem"
    path.write_bytes(pem)
    return path


async def test_refresh_markets_isolates_orderbook_json_error(
    _rsa_pem: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from bot.config import Settings as _Settings

    close_at = "2026-05-08T23:00:00Z"
    good_ticker = "KXHIGHDEN-26MAY07-T70-75"
    bad_ticker = "KXHIGHDEN-26MAY07-T75-80"

    good_orderbook = {
        "orderbook_fp": {
            "yes_dollars": [["0.30", "100"]],
            "no_dollars": [["0.55", "50"]],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/markets"):
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": good_ticker,
                            "event_ticker": "KXHIGHDEN-26MAY07",
                            "status": "open",
                            "close_time": close_at,
                            "yes_ask_dollars": "0.45",
                            "yes_bid_dollars": "0.43",
                        },
                        {
                            "ticker": bad_ticker,
                            "event_ticker": "KXHIGHDEN-26MAY07",
                            "status": "open",
                            "close_time": close_at,
                            "yes_ask_dollars": "0.30",
                            "yes_bid_dollars": "0.28",
                        },
                    ]
                },
            )
        if good_ticker in path:
            return httpx.Response(200, json=good_orderbook)
        return httpx.Response(200, content=b"<html>not json</html>")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        settings = _Settings(
            mode="paper",
            kalshi_demo_key_id="demo-key-id",
            kalshi_demo_private_key_path=_rsa_pem,
        )
        client = KalshiDemoClient(settings, http_client=http)
        await client.aopen()

        engine = make_engine(":memory:")
        Base.metadata.create_all(engine)
        sf = make_session_factory(engine)
        app = App(
            settings=settings,
            engine=engine,
            session_factory=sf,
            meteo=None,  # type: ignore[arg-type]
            kalshi=client,
            acis=_StubACIS(None),  # type: ignore[arg-type]
            series_list=("KXHIGHDEN",),
        )

        caplog.set_level(logging.WARNING, logger="bot.main")
        count = await refresh_markets(app)

    assert count == 1
    assert good_ticker in app.latest_markets
    assert bad_ticker not in app.latest_markets
    matches = [
        r
        for r in caplog.records
        if "kalshi_orderbook_fetch_failed" in r.getMessage() and bad_ticker in r.getMessage()
    ]
    assert matches


async def test_refresh_markets_evicts_stale_tickers_per_city() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    persistent = _market_from("KXHIGHDEN-26MAY07-T70-75", "0.45", "0.43", close_at)
    vanishing = _market_from("KXHIGHDEN-26MAY07-T75-80", "0.30", "0.28", close_at)
    books = {
        persistent.ticker: _book_from(persistent.ticker, "0.45", "0.43"),
        vanishing.ticker: _book_from(vanishing.ticker, "0.30", "0.28"),
    }
    kalshi = _StubKalshi(markets=[persistent, vanishing], orderbooks=books)
    app = _make_app(kalshi=kalshi)

    await refresh_markets(app)
    assert vanishing.ticker in app.latest_markets

    kalshi._markets = [persistent]
    await refresh_markets(app)

    assert persistent.ticker in app.latest_markets
    assert vanishing.ticker not in app.latest_markets
    assert vanishing.ticker not in app.latest_orderbooks


class _ListFailKalshi(_StubKalshi):
    def __init__(
        self,
        markets: list[KalshiMarket],
        orderbooks: dict[str, KalshiOrderbook],
        fail_series: str,
    ) -> None:
        super().__init__(markets, orderbooks)
        self._fail_series = fail_series

    async def list_open_markets_for_series(self, series_prefix: str) -> list[KalshiMarket]:
        if series_prefix == self._fail_series:
            raise httpx.ConnectError("connection refused")
        return [m for m in self._markets if m.ticker.startswith(f"{series_prefix}-")]


async def test_refresh_markets_keeps_cache_when_list_fetch_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    den = _market_from("KXHIGHDEN-26MAY07-T70-75", "0.45", "0.43", close_at)
    nyc = _market_from("KXHIGHNY-26MAY07-T80-85", "0.30", "0.28", close_at)
    books = {
        den.ticker: _book_from(den.ticker, "0.45", "0.43"),
        nyc.ticker: _book_from(nyc.ticker, "0.30", "0.28"),
    }
    kalshi = _StubKalshi(markets=[den, nyc], orderbooks=books)
    app = _make_app(kalshi=kalshi, series_list=("KXHIGHDEN", "KXHIGHNY"))

    await refresh_markets(app)
    assert den.ticker in app.latest_markets
    assert nyc.ticker in app.latest_markets

    failing = _ListFailKalshi(markets=[den, nyc], orderbooks=books, fail_series="KXHIGHNY")
    app.kalshi = failing  # type: ignore[assignment]
    caplog.set_level(logging.WARNING, logger="bot.main")
    await refresh_markets(app)

    assert nyc.ticker in app.latest_markets
    matches = [
        r
        for r in caplog.records
        if "kalshi_list_markets_failed" in r.getMessage() and "KXHIGHNY" in r.getMessage()
    ]
    assert matches


async def test_refresh_markets_skips_unparseable_ticker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    good = _market_from("KXHIGHDEN-26MAY07-T70-75", "0.45", "0.43", close_at)
    bad = KalshiMarket(
        ticker="KXHIGHDEN-not-a-real-ticker",
        event_ticker="KXHIGHDEN-26MAY07",
        series="KXHIGHDEN",
        status="open",
        close_time=close_at,
        yes_ask=Decimal("0.30"),
        yes_bid=Decimal("0.28"),
    )
    books = {
        good.ticker: _book_from(good.ticker, "0.45", "0.43"),
        bad.ticker: _book_from(bad.ticker, "0.30", "0.28"),
    }
    kalshi = _StubKalshi(markets=[good, bad], orderbooks=books)
    app = _make_app(kalshi=kalshi)

    caplog.set_level(logging.WARNING, logger="bot.main")
    count = await refresh_markets(app)

    assert count == 1
    with app.session_factory() as session:
        rows = session.scalars(select(Market)).all()
    assert {r.ticker for r in rows} == {good.ticker}
    matches = [r for r in caplog.records if "market_unparseable_ticker" in r.getMessage()]
    assert matches


async def test_evaluate_strategies_runs_edge_buy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)

    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)

    await refresh_forecasts(app)
    await refresh_markets(app)

    _lift_caps(monkeypatch)
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert n_trades >= 1
    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert any(t.strategy == "edge" for t in trades)


async def test_evaluate_strategies_skips_market_without_forecast() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.50",
        "0.48",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.50", "0.48")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(kalshi=kalshi)

    await refresh_markets(app)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert n_trades == 0
    with app.session_factory() as session:
        rows = session.scalars(select(PaperTradeRow)).all()
    assert rows == []


async def test_evaluate_strategies_runs_tails_on_single_strike_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(2).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)

    tail = _market_from(
        "KXHIGHDEN-26MAY08-T100",
        "0.50",
        "0.48",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(tail.ticker, "0.50", "0.48")
    kalshi = _StubKalshi(markets=[tail], orderbooks={tail.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    seen: list[tails_strategy.TailsContext] = []
    real_evaluate = tails_strategy.evaluate

    def recorder(ctx: tails_strategy.TailsContext, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(ctx)
        return real_evaluate(ctx, **kwargs)

    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", recorder)

    _lift_caps(monkeypatch)
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert seen, "tails_strategy.evaluate was not reached for tail ticker"
    assert n_trades >= 1
    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert any(t.strategy == "tails" for t in trades)


async def test_evaluate_strategies_tail_market_no_trade_when_gates_block() -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(2).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)

    tail = _market_from(
        "KXHIGHDEN-26MAY08-T100",
        "0.05",
        "0.03",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(tail.ticker, "0.05", "0.03")
    kalshi = _StubKalshi(markets=[tail], orderbooks={tail.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert n_trades == 0
    with app.session_factory() as session:
        rows = session.scalars(select(PaperTradeRow)).all()
    assert rows == []


async def test_evaluate_strategies_uses_per_ticker_cdf_not_app_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(11)
    den_fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): rng.normal(72.0, 4.0, size=31)},
    )
    nyc_fc = StationForecast(
        station="KNYC",
        latitude=40.7790,
        longitude=-73.9692,
        timezone="America/New_York",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): rng.normal(85.0, 3.0, size=31)},
    )
    meteo = _StubMeteo({"KDEN": den_fc, "KNYC": nyc_fc})

    nyc_market = _market_from(
        "KXHIGHNY-26MAY08-T70-75",
        "0.50",
        "0.48",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    nyc_book = _book_from(nyc_market.ticker, "0.50", "0.48")
    kalshi = _StubKalshi(markets=[nyc_market], orderbooks={nyc_market.ticker: nyc_book})

    app = _make_app(meteo=meteo, kalshi=kalshi, series_list=("KXHIGHDEN", "KXHIGHNY"))

    await refresh_forecasts(app)
    await refresh_markets(app)

    _lift_caps(monkeypatch)
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert n_trades >= 1
    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert all(t.market_ticker == nyc_market.ticker for t in trades)
    assert any(Decimal(t.fair_at_entry) < Decimal("0.001") for t in trades)


async def test_evaluate_strategies_marks_blacklisted_series() -> None:
    rng = np.random.default_rng(13)
    lax_fc = StationForecast(
        station="KLAX",
        latitude=33.9382,
        longitude=-118.3866,
        timezone="America/Los_Angeles",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): rng.normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo({"KLAX": lax_fc})

    market = _market_from(
        "KXHIGHLAX-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi, series_list=("KXHIGHLAX",))

    await refresh_forecasts(app)
    await refresh_markets(app)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert n_trades == 0
    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert trades == []


def test_build_intents_skips_blacklisted_lax_series() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHLAX-26MAY08-T70-75", "0.20", "0.18", close_at)
    book = _book_from(market.ticker, "0.20", "0.18")
    intents = _build_intents(
        app=_make_app(),
        mode="paper",
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.05"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=True,
        is_tail=False,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        **_intents_defaults(),
    )
    assert intents == []


def test_build_intents_skips_blacklisted_mia_series() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHMIA-26MAY08-T70-75", "0.20", "0.18", close_at)
    book = _book_from(market.ticker, "0.20", "0.18")
    intents = _build_intents(
        app=_make_app(),
        mode="paper",
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.05"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=True,
        is_tail=False,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        **_intents_defaults(),
    )
    assert intents == []


def test_build_intents_emits_for_normal_series() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHDEN-26MAY08-T70-75", "0.20", "0.18", close_at)
    book = _book_from(market.ticker, "0.20", "0.18")
    intents = _build_intents(
        app=_make_app(),
        mode="paper",
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.80"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=False,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        **_intents_defaults(),
    )
    assert intents
    assert any(i.strategy == "edge" for i in intents)


async def test_evaluate_strategies_skips_unparseable_ticker(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(14)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): rng.normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)

    good = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    bad = KalshiMarket(
        ticker="KXHIGHDEN-not-a-real-ticker",
        event_ticker="KXHIGHDEN-26MAY08",
        series="KXHIGHDEN",
        status="open",
        close_time=datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
        yes_ask=Decimal("0.30"),
        yes_bid=Decimal("0.28"),
    )
    good_book = _book_from(good.ticker, "0.20", "0.18")
    bad_book = _book_from(bad.ticker, "0.30", "0.28")
    kalshi = _StubKalshi(
        markets=[good, bad], orderbooks={good.ticker: good_book, bad.ticker: bad_book}
    )

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    app.latest_markets[good.ticker] = good
    app.latest_orderbooks[good.ticker] = good_book
    app.latest_markets[bad.ticker] = bad
    app.latest_orderbooks[bad.ticker] = bad_book

    caplog.set_level(logging.WARNING, logger="bot.main")
    _lift_caps(monkeypatch)
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert n_trades >= 1
    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert all(t.market_ticker == good.ticker for t in trades)
    matches = [
        r
        for r in caplog.records
        if "eval_unparseable_ticker" in r.getMessage() and bad.ticker in r.getMessage()
    ]
    assert matches


def test_cli_rejects_unsupported_series(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["bot.main", "--mode=paper", "--series=KXHIGHFAKE,KXHIGHDEN", "--duration=1m"],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "KXHIGHFAKE" in (captured.err + captured.out)


def test_cli_rejects_live_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["bot.main", "--mode=live", "--series=KXHIGHDEN", "--duration=1m"],
    )
    with pytest.raises(SystemExit):
        main()


def test_parse_series_arg_all_returns_full_set() -> None:
    out = _parse_series_arg("all")
    assert set(out) == set(STATIONS.keys())
    assert len(out) == 20


def test_parse_series_arg_comma_list() -> None:
    out = _parse_series_arg("KXHIGHDEN,KXHIGHNY")
    assert out == ("KXHIGHDEN", "KXHIGHNY")


def test_parse_series_arg_rejects_unknown(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse_series_arg("KXHIGHFAKE,KXHIGHDEN")
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "KXHIGHFAKE" in err


def test_parse_series_arg_rejects_empty(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse_series_arg("")
    assert excinfo.value.code == 2


_SAMPLE_TICKERS_FOR_PARSER_TEST: dict[str, str] = {
    "KXHIGHDEN": "KXHIGHDEN-26MAY07-T68",
    "KXHIGHAUS": "KXHIGHAUS-26MAY07-T77",
    "KXHIGHCHI": "KXHIGHCHI-26MAY07-T65",
    "KXHIGHNY": "KXHIGHNY-26MAY07-T71",
    "KXHIGHPHIL": "KXHIGHPHIL-26MAY07-T70",
    "KXHIGHTATL": "KXHIGHTATL-26MAY07-T81",
    "KXHIGHTBOS": "KXHIGHTBOS-26MAY07-T69",
    "KXHIGHTDAL": "KXHIGHTDAL-26MAY07-T79",
    "KXHIGHTDC": "KXHIGHTDC-26MAY07-T68",
    "KXHIGHTHOU": "KXHIGHTHOU-26MAY07-T83",
    "KXHIGHTLV": "KXHIGHTLV-26MAY07-T96",
    "KXHIGHTMIN": "KXHIGHTMIN-26MAY07-T66",
    "KXHIGHTNOLA": "KXHIGHTNOLA-26MAY07-T86",
    "KXHIGHTOKC": "KXHIGHTOKC-26MAY07-T78",
    "KXHIGHTPHX": "KXHIGHTPHX-26MAY07-T99",
    "KXHIGHTSATX": "KXHIGHTSATX-26MAY07-T80",
    "KXHIGHTSEA": "KXHIGHTSEA-26MAY07-T73",
    "KXHIGHTSFO": "KXHIGHTSFO-26MAY07-T69",
    "KXHIGHLAX": "KXHIGHLAX-26MAY07-T75",
    "KXHIGHMIA": "KXHIGHMIA-26MAY07-T94",
}


@pytest.mark.parametrize(
    "series,ticker",
    sorted(_SAMPLE_TICKERS_FOR_PARSER_TEST.items()),
)
def test_parse_real_demo_ticker_per_series(series: str, ticker: str) -> None:
    parsed = parse_ticker(ticker)
    assert parsed.series == series
    assert series in STATIONS


def _insert_paper_trade(
    app: App,
    *,
    market_ticker: str,
    side: str,
    contracts: int,
    simulated_price: Decimal,
    fee_dollars: Decimal,
    fair_at_entry: Decimal,
    strategy: str,
    intended_at: datetime,
) -> int:
    with app.session_factory() as session:
        row = PaperTradeRow(
            intended_at=intended_at,
            market_ticker=market_ticker,
            side=side,
            contracts=contracts,
            simulated_price=simulated_price,
            fee_dollars=fee_dollars,
            fair_at_entry=fair_at_entry,
            q_raw=fair_at_entry,
            strategy=strategy,
        )
        session.add(row)
        session.commit()
        return row.id


async def test_reconcile_settled_trades_writes_simulated_pnl() -> None:
    acis = _StubACIS(Decimal("71"))
    app = _make_app(acis=acis)

    intended = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    bracket_id = _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY05-T70.5-72.5",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=intended,
    )
    tail_id = _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY05-B70.5",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.30"),
        fee_dollars=Decimal("0.04"),
        fair_at_entry=Decimal("0.20"),
        strategy="tails",
        intended_at=intended,
    )

    now = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    n = await reconcile_settled_trades(app, now)

    assert n == 2
    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl).order_by(SimulatedPnl.paper_trade_id)).all()

    assert {r.paper_trade_id for r in rows} == {bracket_id, tail_id}
    by_id = {r.paper_trade_id: r for r in rows}

    assert by_id[bracket_id].outcome == "won"
    assert by_id[bracket_id].realized_pnl == Decimal("5.95")
    assert by_id[bracket_id].settled_at == now

    assert by_id[tail_id].outcome == "lost"
    assert by_id[tail_id].realized_pnl == Decimal("-3.04")
    assert by_id[tail_id].settled_at == now

    assert acis.calls == [("KDEN", date(2026, 5, 5))]


async def test_reconcile_settled_trades_skips_eligibility() -> None:
    acis = _StubACIS(Decimal("71"))
    app = _make_app(acis=acis)

    _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY06-T70.5-72.5",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=datetime(2026, 5, 6, 18, 0, tzinfo=timezone.utc),
    )

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n = await reconcile_settled_trades(app, now)

    assert n == 0
    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl)).all()
    assert rows == []
    assert acis.calls == []


async def test_reconcile_grace_yesterday_is_eligible() -> None:
    acis = _StubACIS(Decimal("71"))
    app = _make_app(acis=acis)

    _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY13-T70-72",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=datetime(2026, 5, 13, 18, 0, tzinfo=timezone.utc),
    )

    now = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    n = await reconcile_settled_trades(app, now)

    assert n == 1
    assert acis.calls == [("KDEN", date(2026, 5, 13))]


async def test_reconcile_grace_today_still_skipped() -> None:
    acis = _StubACIS(Decimal("71"))
    app = _make_app(acis=acis)

    _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY14-T70-72",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=datetime(2026, 5, 14, 18, 0, tzinfo=timezone.utc),
    )

    now = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    n = await reconcile_settled_trades(app, now)

    assert n == 0
    assert acis.calls == []


async def test_reconcile_grace_two_days_ago_still_eligible() -> None:
    acis = _StubACIS(Decimal("71"))
    app = _make_app(acis=acis)

    _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY12-T70-72",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=datetime(2026, 5, 12, 18, 0, tzinfo=timezone.utc),
    )

    now = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    n = await reconcile_settled_trades(app, now)

    assert n == 1
    assert acis.calls == [("KDEN", date(2026, 5, 12))]


async def test_reconcile_settled_trades_idempotent() -> None:
    acis = _StubACIS(Decimal("71"))
    app = _make_app(acis=acis)

    _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY05-T70.5-72.5",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
    )

    now = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    first = await reconcile_settled_trades(app, now)
    second = await reconcile_settled_trades(app, now)

    assert first == 1
    assert second == 0
    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl)).all()
    assert len(rows) == 1


async def test_reconcile_settled_trades_pending_when_acis_returns_none() -> None:
    acis = _StubACIS(None)
    app = _make_app(acis=acis)

    _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY05-T70.5-72.5",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
    )

    now = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    n = await reconcile_settled_trades(app, now)

    assert n == 0
    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl)).all()
    assert rows == []


async def test_reconcile_settled_trades_routes_per_series_station() -> None:
    acis = _MultiStationACIS(
        {
            ("KDEN", date(2026, 5, 5)): Decimal("71"),
            ("KNYC", date(2026, 5, 5)): Decimal("82"),
        }
    )
    app = _make_app(acis=acis, series_list=("KXHIGHDEN", "KXHIGHNY"))

    intended = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    den_id = _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY05-T70-72",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.02"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=intended,
    )
    nyc_id = _insert_paper_trade(
        app,
        market_ticker="KXHIGHNY-26MAY05-T80-83",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.02"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=intended,
    )

    now = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    n = await reconcile_settled_trades(app, now)

    assert n == 2
    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl).order_by(SimulatedPnl.paper_trade_id)).all()
    by_id = {r.paper_trade_id: r for r in rows}
    assert by_id[den_id].outcome == "won"
    assert by_id[nyc_id].outcome == "won"
    called = sorted(acis.calls)
    assert ("KDEN", date(2026, 5, 5)) in called
    assert ("KNYC", date(2026, 5, 5)) in called


class _FlakyACIS:
    def __init__(self, results: list[Decimal | Exception]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, date]] = []

    async def fetch_daily_high(self, station: str, settled_date: date) -> Decimal | None:
        self.calls.append((station, settled_date))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self) -> None:
        return None


async def test_acis_http_error_logs_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    acis = _FlakyACIS([httpx.RequestError("boom"), Decimal("82")])
    app = _make_app(acis=acis, series_list=("KXHIGHDEN", "KXHIGHNY"))

    intended = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    den_id = _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY05-T70-72",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.02"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=intended,
    )
    nyc_id = _insert_paper_trade(
        app,
        market_ticker="KXHIGHNY-26MAY05-T80-83",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.02"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=intended,
    )

    now = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    caplog.set_level(logging.WARNING, logger="bot.main")
    n = await reconcile_settled_trades(app, now)

    assert n == 1
    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl)).all()
    settled_ids = {r.paper_trade_id for r in rows}
    assert den_id not in settled_ids
    assert nyc_id in settled_ids
    assert len(acis.calls) == 2
    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "acis_fetch_failed" in r.getMessage()
    ]
    assert matches, "expected a WARNING log with acis_fetch_failed"


class _FailingACIS:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch_daily_high(self, station: str, settled_date: date) -> Decimal | None:
        self.calls += 1
        raise RuntimeError("simulated transient failure")

    async def aclose(self) -> None:
        return None


async def test_settlement_loop_logs_and_continues_on_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    acis = _FailingACIS()
    app = _make_app(acis=acis)  # type: ignore[arg-type]

    _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-25JAN02-T70.5-72.5",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=datetime(2025, 1, 2, 18, 0, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(bot_main, "SETTLEMENT_INTERVAL_SECONDS", 0.05)

    stop = asyncio.Event()
    caplog.set_level(logging.ERROR, logger="bot.main")

    task = asyncio.create_task(_settlement_loop(app, stop))
    for _ in range(200):
        if acis.calls >= 1:
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert task.exception() is None
    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and "loop_iteration_failed" in r.getMessage()
        and "name=settlement_loop" in r.getMessage()
    ]
    assert matches, "expected an ERROR log with loop_iteration_failed name=settlement_loop"


async def test_db_lock_serializes_concurrent_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    meteo = _StubMeteo(_forecast_with_two_days())

    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHDEN-26MAY07-T70-75", "0.45", "0.43", close_at)
    book = _book_from(market.ticker, "0.45", "0.43")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)

    assert isinstance(app.db_lock, asyncio.Lock)

    in_flight = 0
    overlap_detected = False

    real_persist = bot_main._persist_forecast

    def slow_persist(app_arg: App, forecast: StationForecast) -> int:
        nonlocal in_flight, overlap_detected
        in_flight += 1
        if in_flight > 1:
            overlap_detected = True
        try:
            return real_persist(app_arg, forecast)
        finally:
            in_flight -= 1

    async def slow_get_orderbook(ticker: str) -> KalshiOrderbook:
        await asyncio.sleep(0.02)
        return book

    monkeypatch.setattr(bot_main, "_persist_forecast", slow_persist)
    monkeypatch.setattr(kalshi, "get_orderbook", slow_get_orderbook)

    async def forecast_writer() -> int:
        n = 0
        for _ in range(5):
            n = await refresh_forecasts(app)
        return n

    async def market_writer() -> int:
        n = 0
        for _ in range(5):
            n = await refresh_markets(app)
        return n

    f, m = await asyncio.gather(forecast_writer(), market_writer())
    assert f == 2
    assert m == 1
    assert not overlap_detected

    await asyncio.wait_for(app.db_lock.acquire(), timeout=0.05)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(app.db_lock.acquire(), timeout=0.05)
    app.db_lock.release()


async def test_forecast_loop_retries_quickly_on_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app = _make_app()

    monkeypatch.setattr(bot_main, "FORECAST_RETRY_INTERVAL_SECONDS", 0.05)
    refresh_mock = AsyncMock(side_effect=[RuntimeError("transient"), 7, 7, 7, 7])
    monkeypatch.setattr(bot_main, "refresh_forecasts", refresh_mock)

    stop = asyncio.Event()
    caplog.set_level(logging.ERROR, logger="bot.main")

    task = asyncio.create_task(_forecast_loop(app, stop))
    for _ in range(200):
        if refresh_mock.await_count >= 2:
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert task.exception() is None
    assert refresh_mock.await_count >= 2
    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and "loop_iteration_failed" in r.getMessage()
        and "name=forecast_loop" in r.getMessage()
    ]
    assert matches, "expected an ERROR log with loop_iteration_failed name=forecast_loop"


async def test_evaluate_strategies_bracket_uses_prob_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    seen: list[edge_strategy.EdgeContext] = []
    real_evaluate = edge_strategy.evaluate

    def recorder(ctx: edge_strategy.EdgeContext, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(ctx)
        return real_evaluate(ctx, **kwargs)

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", recorder)

    parsed = parse_ticker(market.ticker)
    assert parsed.kind == "bracket"

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)

    assert seen, "edge_strategy.evaluate was not reached for bracket ticker"
    cdf = app.forecast_cdfs[("KDEN", date(2026, 5, 8))]
    expected = Decimal(str(cdf.prob_range(70.0, 75.0)))
    assert seen[0].fair_yes == expected


def test_fair_yes_per_market_form() -> None:
    members = np.array(
        [72.0 + delta for delta in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)],
        dtype=np.float64,
    )
    cdf = EnsembleCDF.from_members(members, smoothing=1.0)

    bracket_fair = Decimal(str(cdf.prob_range(70.0, 75.0)))
    above_fair = Decimal(str(1.0 - cdf.cdf(75.0)))
    below_fair = Decimal(str(cdf.cdf(70.0)))

    assert bracket_fair > Decimal("0")
    assert above_fair > Decimal("0")
    assert below_fair > Decimal("0")
    total = bracket_fair + above_fair + below_fair
    assert total <= Decimal("1") + Decimal("0.001")


def test_build_intents_routes_tail_to_tails_only(monkeypatch: pytest.MonkeyPatch) -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHDEN-26MAY08-T100", "0.50", "0.48", close_at)
    book = _book_from(market.ticker, "0.50", "0.48")

    edge_calls: list[edge_strategy.EdgeContext] = []
    tails_calls: list[tails_strategy.TailsContext] = []

    def edge_rec(ctx: edge_strategy.EdgeContext, **_):  # type: ignore[no-untyped-def]
        edge_calls.append(ctx)
        return edge_strategy.EdgeSignal(
            action=edge_strategy.EdgeAction.SKIP,
            contracts=0,
            notional_dollars=Decimal("0"),
            reason="stub",
        )

    def tails_rec(ctx: tails_strategy.TailsContext, **_):  # type: ignore[no-untyped-def]
        tails_calls.append(ctx)
        return tails_strategy.TailsSignal(
            action=tails_strategy.TailsAction.SKIP,
            contracts=0,
            notional_dollars=Decimal("0"),
            reason="stub",
        )

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", edge_rec)
    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", tails_rec)

    _build_intents(
        app=_make_app(),
        mode="paper",
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.04"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=True,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        **_intents_defaults(),
    )

    assert edge_calls == []
    assert len(tails_calls) == 1


def test_build_intents_routes_bracket_to_edge_only(monkeypatch: pytest.MonkeyPatch) -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHDEN-26MAY08-T70-75", "0.20", "0.18", close_at)
    book = _book_from(market.ticker, "0.20", "0.18")

    edge_calls: list[edge_strategy.EdgeContext] = []
    tails_calls: list[tails_strategy.TailsContext] = []

    def edge_rec(ctx: edge_strategy.EdgeContext, **_):  # type: ignore[no-untyped-def]
        edge_calls.append(ctx)
        return edge_strategy.EdgeSignal(
            action=edge_strategy.EdgeAction.SKIP,
            contracts=0,
            notional_dollars=Decimal("0"),
            reason="stub",
        )

    def tails_rec(ctx: tails_strategy.TailsContext, **_):  # type: ignore[no-untyped-def]
        tails_calls.append(ctx)
        return tails_strategy.TailsSignal(
            action=tails_strategy.TailsAction.SKIP,
            contracts=0,
            notional_dollars=Decimal("0"),
            reason="stub",
        )

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", edge_rec)
    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", tails_rec)

    _build_intents(
        app=_make_app(),
        mode="paper",
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.50"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=False,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        **_intents_defaults(),
    )

    assert len(edge_calls) == 1
    assert tails_calls == []


def test_build_intents_blacklisted_skips_both(monkeypatch: pytest.MonkeyPatch) -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHLAX-26MAY08-T100", "0.50", "0.48", close_at)
    book = _book_from(market.ticker, "0.50", "0.48")

    edge_calls: list[edge_strategy.EdgeContext] = []
    tails_calls: list[tails_strategy.TailsContext] = []

    def edge_rec(ctx: edge_strategy.EdgeContext, **_):  # type: ignore[no-untyped-def]
        edge_calls.append(ctx)
        return edge_strategy.EdgeSignal(
            action=edge_strategy.EdgeAction.SKIP,
            contracts=0,
            notional_dollars=Decimal("0"),
            reason="stub",
        )

    def tails_rec(ctx: tails_strategy.TailsContext, **_):  # type: ignore[no-untyped-def]
        tails_calls.append(ctx)
        return tails_strategy.TailsSignal(
            action=tails_strategy.TailsAction.SKIP,
            contracts=0,
            notional_dollars=Decimal("0"),
            reason="stub",
        )

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", edge_rec)
    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", tails_rec)

    for is_tail in (True, False):
        edge_calls.clear()
        tails_calls.clear()
        _build_intents(
            app=_make_app(),
            mode="paper",
            ticker=market.ticker,
            market=market,
            book=book,
            fair_yes=Decimal("0.04"),
            spread=Decimal("3.0"),
            is_same_day=False,
            is_blacklisted=True,
            is_tail=is_tail,
            now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
            **_intents_defaults(),
        )
        if is_tail:
            assert tails_calls == []
        else:
            assert len(edge_calls) == 1


def test_build_intents_routes_b_form_bracket_to_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = parse_ticker("KXHIGHTNOLA-26MAY19-B86.5")
    assert parsed.is_bracket is True
    assert parsed.is_tail is False

    close_at = datetime(2026, 5, 19, 4, 0, tzinfo=timezone.utc)
    market = _market_from(parsed.raw, "0.40", "0.38", close_at)
    book = _book_from(market.ticker, "0.40", "0.38")

    edge_calls: list[edge_strategy.EdgeContext] = []
    tails_calls: list[tails_strategy.TailsContext] = []

    def edge_rec(ctx: edge_strategy.EdgeContext, **_):  # type: ignore[no-untyped-def]
        edge_calls.append(ctx)
        return edge_strategy.EdgeSignal(
            action=edge_strategy.EdgeAction.SKIP,
            contracts=0,
            notional_dollars=Decimal("0"),
            reason="stub",
        )

    def tails_rec(ctx: tails_strategy.TailsContext, **_):  # type: ignore[no-untyped-def]
        tails_calls.append(ctx)
        return tails_strategy.TailsSignal(
            action=tails_strategy.TailsAction.SKIP,
            contracts=0,
            notional_dollars=Decimal("0"),
            reason="stub",
        )

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", edge_rec)
    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", tails_rec)

    _build_intents(
        app=_make_app(),
        mode="paper",
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.30"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=parsed.is_tail,
        now=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
        **_intents_defaults(),
    )

    assert len(edge_calls) == 1
    assert tails_calls == []


async def test_evaluate_strategies_b_form_uses_prob_range_not_cdf() -> None:
    class _RecordingCdf:
        def __init__(self) -> None:
            self.prob_range_calls: list[tuple[float, float]] = []
            self.cdf_calls: list[float] = []

        def prob_range(self, lo: float, hi: float) -> float:
            self.prob_range_calls.append((lo, hi))
            return 0.30

        def cdf(self, x: float) -> float:
            self.cdf_calls.append(x)
            return 0.99

    close_at = datetime(2026, 5, 19, 4, 0, tzinfo=timezone.utc)
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHTNOLA-26MAY19-B86.5", "0.40", "0.38", close_at)
    book = _book_from(market.ticker, "0.40", "0.38", now=now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(kalshi=kalshi, series_list=("KXHIGHTNOLA",))

    cdf = _RecordingCdf()
    event_date = date(2026, 5, 19)
    app.forecast_cdfs[("KMSY", event_date)] = cdf  # type: ignore[assignment]
    app.ensemble_spreads[("KMSY", event_date)] = Decimal("3.0")
    app.forecast_run_times[("KMSY", event_date)] = now

    await refresh_markets(app)
    await evaluate_strategies(app, now)

    assert cdf.prob_range_calls == [(86.0, 87.0)]
    assert cdf.cdf_calls == []


async def test_evaluate_strategies_no_index_error_on_single_strike() -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(3).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    tail = _market_from(
        "KXHIGHDEN-26MAY08-T70",
        "0.50",
        "0.48",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(tail.ticker, "0.50", "0.48")
    kalshi = _StubKalshi(markets=[tail], orderbooks={tail.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    parsed = parse_ticker(tail.ticker)
    assert len(parsed.strikes) == 1

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)


async def test_on_demand_reconcile_writes_simulated_pnl(
    caplog: pytest.LogCaptureFixture,
) -> None:
    acis = _StubACIS(Decimal("71"))
    app = _make_app(acis=acis)

    trade_id = _insert_paper_trade(
        app,
        market_ticker="KXHIGHDEN-26MAY05-T70.5-72.5",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
        intended_at=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
    )

    caplog.set_level(logging.INFO, logger="bot.main")
    await _on_demand_reconcile(app)

    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl)).all()
    assert len(rows) == 1
    assert rows[0].paper_trade_id == trade_id
    assert rows[0].outcome == "won"
    assert rows[0].realized_pnl == Decimal("5.95")

    matches = [r for r in caplog.records if "reconcile_on_demand reconciled=1" in r.getMessage()]
    assert matches, "expected log: reconcile_on_demand reconciled=1"


async def test_on_demand_reconcile_logs_zero_when_nothing_eligible(
    caplog: pytest.LogCaptureFixture,
) -> None:
    acis = _StubACIS(Decimal("71"))
    app = _make_app(acis=acis)

    caplog.set_level(logging.INFO, logger="bot.main")
    await _on_demand_reconcile(app)

    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl)).all()
    assert rows == []

    matches = [r for r in caplog.records if "reconcile_on_demand reconciled=0" in r.getMessage()]
    assert matches, "expected log: reconcile_on_demand reconciled=0"


class _LockObservingACIS:
    def __init__(self, app: App, value: Decimal) -> None:
        self._app = app
        self._value = value
        self.active = 0
        self.max_active = 0
        self.lock_held_during_calls: list[bool] = []
        self.calls: list[tuple[str, date]] = []

    async def fetch_daily_high(self, station: str, settled_date: date) -> Decimal | None:
        self.calls.append((station, settled_date))
        self.lock_held_during_calls.append(self._app.reconcile_lock.locked())
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return self._value
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        return None


async def test_on_demand_reconcile_lock_serializes_with_periodic() -> None:
    app = _make_app()
    acis = _LockObservingACIS(app, Decimal("71"))
    app.acis = acis  # type: ignore[assignment]

    for day in (date(2026, 5, 5), date(2026, 5, 6)):
        _insert_paper_trade(
            app,
            market_ticker=f"KXHIGHDEN-{day.strftime('%y%b%d').upper()}-T70.5-72.5",
            side="buy_yes",
            contracts=10,
            simulated_price=Decimal("0.40"),
            fee_dollars=Decimal("0.05"),
            fair_at_entry=Decimal("0.50"),
            strategy="edge",
            intended_at=datetime(day.year, day.month, day.day, 18, 0, tzinfo=timezone.utc),
        )

    await asyncio.gather(_on_demand_reconcile(app), _on_demand_reconcile(app))

    assert acis.calls, "ACIS was never invoked"
    assert all(acis.lock_held_during_calls), "reconcile_lock was not held during ACIS fetch"
    assert acis.max_active == 1, "two reconciles ran concurrently inside the lock"

    with app.session_factory() as session:
        rows = session.scalars(select(SimulatedPnl)).all()
    assert len(rows) == 2


async def test_evaluate_strategies_caps_market_after_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = datetime(
        2026, 5, 6, 11, 30, tzinfo=timezone.utc
    )

    monkeypatch.setattr(bot_main, "market_position_cap", lambda app=None: Decimal("40"))
    monkeypatch.setattr(bot_main, "event_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "series_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "aggregate_exposure_cap", lambda app=None: Decimal("100000"))

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    for _ in range(20):
        await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()

    assert len(trades) >= 1
    market_dollars = sum(
        (
            Decimal(t.contracts) * t.simulated_price
            for t in trades
            if t.market_ticker == market.ticker
        ),
        Decimal("0"),
    )
    assert market_dollars <= Decimal("40")


async def test_evaluate_strategies_caps_event_across_brackets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    m_a = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    m_b = _market_from(
        "KXHIGHDEN-26MAY08-T75-80",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book_a = _book_from(m_a.ticker, "0.20", "0.18")
    book_b = _book_from(m_b.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[m_a, m_b], orderbooks={m_a.ticker: book_a, m_b.ticker: book_b})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = datetime(
        2026, 5, 6, 11, 30, tzinfo=timezone.utc
    )

    monkeypatch.setattr(bot_main, "market_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "event_position_cap", lambda app=None: Decimal("10"))
    monkeypatch.setattr(bot_main, "series_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "aggregate_exposure_cap", lambda app=None: Decimal("100000"))

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    for _ in range(50):
        await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    event_dollars = sum(
        (Decimal(t.contracts) * t.simulated_price for t in trades),
        Decimal("0"),
    )
    assert event_dollars <= Decimal("10")


async def test_evaluate_strategies_within_cycle_event_cap_blocks_second_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    m_a = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    m_b = _market_from(
        "KXHIGHDEN-26MAY08-T75-80",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book_a = _book_from(m_a.ticker, "0.20", "0.18")
    book_b = _book_from(m_b.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[m_a, m_b], orderbooks={m_a.ticker: book_a, m_b.ticker: book_b})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    def stub_intent(
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        now,
        app,
        mode,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=10,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)
    monkeypatch.setattr(bot_main, "event_position_cap", lambda app=None: Decimal("3"))

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
        cap_failures = session.scalars(
            select(GateFailure).where(GateFailure.gate_name == "within_event_cap")
        ).all()
    assert len(trades) == 1
    assert cap_failures


class ReadOnlyOverlay(dict):
    def __setitem__(self, key, value):
        return None

    def __delitem__(self, key):
        return None

    def update(self, *args, **kwargs):
        return None

    def setdefault(self, key, default=None):
        return super().get(key, default)

    def __ior__(self, other):
        return self

    def pop(self, *args, **kwargs):
        return None

    def popitem(self):
        return None

    def clear(self):
        return None


async def test_evaluate_strategies_overlay_strip_regression_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    m_a = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    m_b = _market_from(
        "KXHIGHDEN-26MAY08-T75-80",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book_a = _book_from(m_a.ticker, "0.20", "0.18")
    book_b = _book_from(m_b.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[m_a, m_b], orderbooks={m_a.ticker: book_a, m_b.ticker: book_b})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    def stub_intent(
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        now,
        app,
        mode,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=10,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)
    monkeypatch.setattr(bot_main, "EVENT_POSITION_CAP", Decimal("3"))

    seeds: list[tuple[type, type, type]] = []
    ids: list[tuple[int, int, int]] = []
    real_gate_ctx = bot_main._gate_ctx_for

    def gate_ctx_recorder(**kwargs):
        import inspect

        frame = inspect.currentframe().f_back
        seeds.append(
            (
                type(frame.f_locals["overlay_market"]),
                type(frame.f_locals["overlay_event"]),
                type(frame.f_locals["overlay_series"]),
            )
        )
        ids.append(
            (
                id(frame.f_locals["overlay_market"]),
                id(frame.f_locals["overlay_event"]),
                id(frame.f_locals["overlay_series"]),
            )
        )
        return real_gate_ctx(**kwargs)

    monkeypatch.setattr(bot_main, "_gate_ctx_for", gate_ctx_recorder)

    def fake_open_exposures(session, *, now=None, grace_days=None):
        return ReadOnlyOverlay(), ReadOnlyOverlay(), ReadOnlyOverlay(), Decimal("0")

    monkeypatch.setattr(bot_main, "open_exposures", fake_open_exposures)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)

    assert seeds, "_gate_ctx_for was never reached"
    for seed_market_t, seed_event_t, seed_series_t in seeds:
        assert seed_market_t is ReadOnlyOverlay
        assert seed_event_t is ReadOnlyOverlay
        assert seed_series_t is ReadOnlyOverlay
    first_ids = ids[0]
    for triple in ids[1:]:
        assert triple == first_ids

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(trades) == 2


def _gate_ctx_kwargs(intent: TradeIntent, market: KalshiMarket, book: KalshiOrderbook) -> dict:
    return dict(
        intent=intent,
        market=market,
        fair_yes=Decimal("0.50"),
        spread=Decimal("3.0"),
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        book=book,
        market_existing_dollars=Decimal("0"),
        event_existing_dollars=Decimal("0"),
        series_existing_dollars=Decimal("0"),
        aggregate_existing_dollars=Decimal("0"),
        buy_yes_depth=book.no_bid_depth,
        sell_yes_depth=book.yes_bid_depth,
    )


def test_gate_ctx_for_sell_yes_uses_book_no_ask_on_wide_book() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.50",
        "0.20",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.50", "0.20")
    assert book.no_ask == Decimal("0.80")
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.SELL_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    ctx = _gate_ctx_for(**_gate_ctx_kwargs(intent, market, book))
    assert ctx.order_size_dollars == Decimal("0.80") * Decimal(10)


def test_gate_ctx_for_sell_yes_book_no_ask_diverges_from_market_complement_when_snapshots_lag() -> (
    None
):
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.50",
        "0.25",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.50", "0.20")
    assert cost_per_contract_from_book(TradeSide.SELL_YES, book) == Decimal("0.80")
    assert Decimal("1") - market.yes_bid == Decimal("0.75")
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.SELL_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    ctx = _gate_ctx_for(**_gate_ctx_kwargs(intent, market, book))
    assert ctx.order_size_dollars == Decimal("0.80") * Decimal(10)


def test_gate_ctx_for_buy_yes_uses_book_yes_ask() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.40",
        "0.20",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.40", "0.20")
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    ctx = _gate_ctx_for(**_gate_ctx_kwargs(intent, market, book))
    assert ctx.order_size_dollars == Decimal("0.40") * Decimal(10)


def test_gate_ctx_for_requires_book_kwarg() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.40",
        "0.20",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    with pytest.raises(TypeError):
        _gate_ctx_for(
            intent=intent,
            market=market,
            fair_yes=Decimal("0.50"),
            spread=Decimal("3.0"),
            run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
            now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
            market_existing_dollars=Decimal("0"),
            event_existing_dollars=Decimal("0"),
            series_existing_dollars=Decimal("0"),
            aggregate_existing_dollars=Decimal("0"),
            buy_yes_depth=10,
            sell_yes_depth=10,
        )


def test_gate_ctx_for_caps_scale_with_demo_app_bankroll() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.40",
        "0.20",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.40", "0.20")
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    app = _make_demo_app(bankroll=Decimal("1000"))
    ctx = _gate_ctx_for(**_gate_ctx_kwargs(intent, market, book), app=app)

    assert ctx.market_position_cap == Decimal("1000") * bot_main.MARKET_POSITION_FRAC
    assert ctx.event_position_cap == Decimal("1000") * bot_main.EVENT_POSITION_FRAC
    assert ctx.series_position_cap == Decimal("1000") * bot_main.SERIES_POSITION_FRAC
    assert ctx.aggregate_exposure_cap == Decimal("1000") * bot_main.AGGREGATE_EXPOSURE_FRAC
    assert ctx.account_balance == Decimal("1000")


def test_gate_ctx_for_caps_match_legacy_constants_without_app() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.40",
        "0.20",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.40", "0.20")
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    ctx = _gate_ctx_for(**_gate_ctx_kwargs(intent, market, book))

    base = bot_main.PAPER_BANKROLL
    assert ctx.market_position_cap == base * bot_main.MARKET_POSITION_FRAC
    assert ctx.event_position_cap == base * bot_main.EVENT_POSITION_FRAC
    assert ctx.series_position_cap == base * bot_main.SERIES_POSITION_FRAC
    assert ctx.aggregate_exposure_cap == base * bot_main.AGGREGATE_EXPOSURE_FRAC
    assert ctx.account_balance == base


async def test_evaluate_strategies_partial_fill_on_thin_book(monkeypatch) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "1.00",
        "0.99",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "1.00", "0.99", yes_ask_depth=1, yes_bid_depth=1)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    def stub_intent(
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        now,
        app,
        mode,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.SELL_YES,
                contracts=7194,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)
    _lift_caps(monkeypatch)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(trades) == 1
    assert trades[0].contracts == 1
    assert trades[0].attempted_contracts == 7194


async def test_b_form_evaluate_strategies_records_a_paper_trade_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RecordingCdf:
        def prob_range(self, lo: float, hi: float) -> float:
            return 0.30

        def cdf(self, x: float) -> float:
            return 0.99

    close_at = datetime(2026, 5, 19, 4, 0, tzinfo=timezone.utc)
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHTNOLA-26MAY19-B86.5", "0.40", "0.38", close_at)
    book = _book_from(market.ticker, "0.40", "0.38", now=now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(kalshi=kalshi, series_list=("KXHIGHTNOLA",))

    cdf = _RecordingCdf()
    event_date = date(2026, 5, 19)
    app.forecast_cdfs[("KMSY", event_date)] = cdf  # type: ignore[assignment]
    app.ensemble_spreads[("KMSY", event_date)] = Decimal("3.0")
    app.forecast_run_times[("KMSY", event_date)] = now

    await refresh_markets(app)
    _lift_caps(monkeypatch)
    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        rows = session.scalars(
            select(PaperTradeRow).where(PaperTradeRow.market_ticker == market.ticker)
        ).all()
    assert len(rows) == 1


async def test_evaluate_strategies_partial_fill_overlay_uses_trade_contracts_across_all_three_dicts(
    monkeypatch,
) -> None:
    from tests.fixtures.partial_fill_paper_trade_row import make_partial_fill_intent_and_trade

    intent, _trade, _book = make_partial_fill_intent_and_trade()
    ticker = intent.market_ticker

    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 20): np.random.default_rng(2).normal(50.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        ticker,
        "1.00",
        "0.99",
        datetime(2026, 5, 20, 23, 0, tzinfo=timezone.utc),
    )
    now = datetime(2026, 5, 19, 16, 1, tzinfo=timezone.utc)
    book = _book_from(market.ticker, "1.00", "0.99", now=now, yes_ask_depth=1, yes_bid_depth=1)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    def stub_intent(
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        now,
        app,
        mode,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.SELL_YES,
                contracts=7194,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)

    captured_overlays: dict[str, dict] = {}

    real_gate_ctx = bot_main._gate_ctx_for

    def recorder(**kwargs):
        frame = py_inspect.currentframe().f_back
        captured_overlays["market"] = frame.f_locals["overlay_market"]
        captured_overlays["event"] = frame.f_locals["overlay_event"]
        captured_overlays["series"] = frame.f_locals["overlay_series"]
        return real_gate_ctx(**kwargs)

    monkeypatch.setattr(bot_main, "_gate_ctx_for", recorder)
    _lift_caps(monkeypatch)

    await evaluate_strategies(app, now)

    parsed = parse_ticker(ticker)
    event_key = market.event_ticker
    series_key = parsed.series

    assert captured_overlays["market"].get(ticker) == Decimal("0.01"), (
        f"overlay_market for {ticker} drifted: got {captured_overlays['market'].get(ticker)}"
    )
    assert captured_overlays["event"].get(event_key) == Decimal("0.01"), (
        f"overlay_event for {event_key} drifted: got {captured_overlays['event'].get(event_key)}"
    )
    assert captured_overlays["series"].get(series_key) == Decimal("0.01"), (
        f"overlay_series for {series_key} drifted: got {captured_overlays['series'].get(series_key)}"
    )


async def test_evaluate_strategies_stale_orderbook_skips(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    book = _book_from(market.ticker, "0.20", "0.18", snapshot_at=now - timedelta(seconds=300))
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    def stub_intent(
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        now,
        app,
        mode,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=10,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)
    caplog.set_level(logging.INFO, logger="bot.execution.paper")

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert trades == []
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("paper_trade_stale_orderbook" in m for m in messages)


async def test_evaluate_strategies_stale_snapshot_drives_zero_trades(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    book = _book_from(market.ticker, "0.20", "0.18", snapshot_at=now - timedelta(seconds=300))
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    def stub_intent(
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        now,
        app,
        mode,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=10,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)
    caplog.set_level(logging.INFO, logger="bot.main")

    n_trades = await evaluate_strategies(app, now)

    assert n_trades == 0
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("stale_skips=" in m and "intents=" in m for m in messages)


async def test_evaluate_strategies_stale_skip_ratio_warns(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    markets = [
        _market_from(
            f"KXHIGHDEN-26MAY08-T{60 + i}-{65 + i}",
            "0.20",
            "0.18",
            datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
        )
        for i in range(10)
    ]
    books = {
        m.ticker: _book_from(m.ticker, "0.20", "0.18", snapshot_at=now - timedelta(seconds=300))
        for m in markets
    }
    kalshi = _StubKalshi(markets=markets, orderbooks=books)
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    for m in markets:
        app.latest_markets[m.ticker] = m
        app.latest_orderbooks[m.ticker] = books[m.ticker]

    def stub_intent(
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        now,
        app,
        mode,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=10,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)
    from bot.risk.gates import RiskCheck as _RiskCheck

    monkeypatch.setattr(
        bot_main,
        "evaluate_gates",
        lambda *_a, **_k: _RiskCheck(overall_passed=True, all_results=(), failures=()),
    )
    caplog.set_level(logging.WARNING, logger="bot.execution.paper")

    await evaluate_strategies(app, now)

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("stale_skip_ratio_high" in m for m in messages)


async def test_evaluate_strategies_per_series_stale_skip_warns_on_one_stuck_series(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc_den = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    fc_nyc = StationForecast(
        station="KNYC",
        latitude=40.7790,
        longitude=-73.9692,
        timezone="America/New_York",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(2).normal(85.0, 4.0, size=31)},
    )
    meteo = _StubMeteo({"KDEN": fc_den, "KNYC": fc_nyc})

    fresh_markets = [
        _market_from(
            f"KXHIGHDEN-26MAY08-T{60 + i}-{65 + i}",
            "0.20",
            "0.18",
            datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
        )
        for i in range(20)
    ]
    stale_markets = [
        _market_from(
            f"KXHIGHNY-26MAY08-T{60 + i}-{65 + i}",
            "0.20",
            "0.18",
            datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
        )
        for i in range(5)
    ]
    fresh_books = {m.ticker: _book_from(m.ticker, "0.20", "0.18", now=now) for m in fresh_markets}
    stale_books = {
        m.ticker: _book_from(m.ticker, "0.20", "0.18", snapshot_at=now - timedelta(seconds=300))
        for m in stale_markets
    }
    all_markets = fresh_markets + stale_markets
    all_books = {**fresh_books, **stale_books}
    kalshi = _StubKalshi(markets=all_markets, orderbooks=all_books)

    app = _make_app(meteo=meteo, kalshi=kalshi, series_list=("KXHIGHDEN", "KXHIGHNY"))
    await refresh_forecasts(app)
    for m in all_markets:
        app.latest_markets[m.ticker] = m
        app.latest_orderbooks[m.ticker] = all_books[m.ticker]

    def stub_intent(
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        now,
        app,
        mode,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=10,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)
    from bot.risk.gates import RiskCheck as _RiskCheck

    monkeypatch.setattr(
        bot_main,
        "evaluate_gates",
        lambda *_a, **_k: _RiskCheck(overall_passed=True, all_results=(), failures=()),
    )
    caplog.set_level(logging.WARNING, logger="bot.execution.paper")

    await evaluate_strategies(app, now)

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("stale_skip_ratio_high_series" in m and "KXHIGHNY" in m for m in messages)


async def test_evaluate_strategies_persists_sigma_t(monkeypatch: pytest.MonkeyPatch) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    _lift_caps(monkeypatch)
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    expected_sigma = app.ensemble_spreads[("KDEN", date(2026, 5, 8))]
    n_trades = await evaluate_strategies(app, now)

    assert n_trades >= 1
    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert trades
    assert trades[0].ensemble_spread_sigma_t is not None
    assert abs(trades[0].ensemble_spread_sigma_t - expected_sigma) < Decimal("0.00001")


def test_paper_trade_row_persists_attempted_contracts_from_intent() -> None:
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)

    trade = PaperTrade(
        intended_at=datetime(2026, 5, 5, 18, 30, tzinfo=timezone.utc),
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        side=TradeSide.SELL_YES,
        contracts=1,
        simulated_price=Decimal("0.99"),
        fee_dollars=Decimal("0.01"),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        attempted_contracts=7194,
    )
    row = bot_main._paper_trade_row(trade)
    with sf() as session:
        session.add(row)
        session.commit()
    with sf() as session:
        got = session.scalars(select(PaperTradeRow)).one()
    assert got.attempted_contracts == 7194
    assert got.contracts == 1
    engine.dispose()


async def test_evaluate_strategies_no_market_open_failure_on_active() -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        market_open_failures = session.scalars(
            select(GateFailure).where(GateFailure.gate_name == "market_open")
        ).all()
    assert market_open_failures == []


async def test_evaluate_strategies_handles_market_with_null_close_time() -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(2).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)

    market = KalshiMarket(
        ticker="KXHIGHDEN-26MAY08-T70-75",
        event_ticker="KXHIGHDEN-26MAY08",
        series="KXHIGHDEN",
        status="active",
        close_time=None,
        yes_ask=Decimal("0.20"),
        yes_bid=Decimal("0.18"),
    )
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    book = _book_from(market.ticker, "0.20", "0.18", now=now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    for t in trades:
        assert t.lead_time_hours is None


def test_compute_lead_time_hours_pre_close_returns_positive_decimal() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 6, 13, 0, tzinfo=timezone.utc),
    )
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    out = _compute_lead_time_hours(market, now)
    assert out == Decimal("1.0")


def test_compute_lead_time_hours_null_close_returns_none() -> None:
    market = KalshiMarket(
        ticker="KXHIGHDEN-26MAY08-T70-75",
        event_ticker="KXHIGHDEN-26MAY08",
        series="KXHIGHDEN",
        status="active",
        close_time=None,
        yes_ask=Decimal("0.20"),
        yes_bid=Decimal("0.18"),
    )
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    assert _compute_lead_time_hours(market, now) is None


def test_compute_lead_time_hours_post_close_returns_none() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 6, 11, 59, 59, tzinfo=timezone.utc),
    )
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    assert _compute_lead_time_hours(market, now) is None


def test_compute_lead_time_hours_exact_boundary_returns_zero() -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHDEN-26MAY08-T70-75", "0.20", "0.18", now)
    out = _compute_lead_time_hours(market, now)
    assert out == Decimal("0.0")


async def test_evaluate_strategies_persists_nbm_divergence_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    _lift_caps(monkeypatch)
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert trades
    for t in trades:
        assert t.nbm_divergence is None


async def test_refresh_markets_persists_depth_on_orderbook_snapshot() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHDEN-26MAY07-T70-75", "0.45", "0.43", close_at)
    book = _book_from(
        market.ticker,
        "0.45",
        "0.43",
        yes_ask_depth=17,
        yes_bid_depth=23,
        no_ask_depth=29,
        no_bid_depth=31,
    )
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(kalshi=kalshi)

    await refresh_markets(app)

    with app.session_factory() as session:
        ob = session.scalars(select(OrderbookSnapshot)).one()
    assert ob.yes_ask_depth == 17
    assert ob.yes_bid_depth == 23
    assert ob.no_ask_depth == 29
    assert ob.no_bid_depth == 31


def test_checkpoint_wal_truncates_wal_file(tmp_path) -> None:
    db_file = tmp_path / "wal.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    with sf() as session:
        session.add(
            GateFailure(
                evaluated_at=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
                gate_name="market_open",
                reason="market_status=closed",
                mode="paper",
                market_ticker="KXHIGHDEN-26MAY06-T70-75",
                last_seen_at=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
            )
        )
        session.commit()

    busy, _log_pages, _checkpointed = _checkpoint_wal(engine)
    assert busy == 0
    engine.dispose()


def test_checkpoint_wal_retries_after_busy_via_monkeypatched_pragma(tmp_path, monkeypatch) -> None:
    db_file = tmp_path / "wal.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    with sf() as session:
        session.add(
            GateFailure(
                evaluated_at=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
                gate_name="market_open",
                reason="market_status=closed",
                mode="paper",
                market_ticker="KXHIGHDEN-26MAY06-T70-75",
                last_seen_at=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
            )
        )
        session.commit()

    from sqlalchemy.engine import Connection

    call_count = {"n": 0}
    real = Connection.exec_driver_sql

    class _FakeResult:
        def __init__(self, row):
            self._row = row

        def one(self):
            return self._row

    def patched(self, statement, *args, **kwargs):
        if "wal_checkpoint" in statement.lower():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _FakeResult((1, 5, 0))
            return _FakeResult((0, 0, 5))
        return real(self, statement, *args, **kwargs)

    monkeypatch.setattr(Connection, "exec_driver_sql", patched)

    dispose_calls = {"n": 0}
    real_dispose = engine.dispose

    def dispose_spy(*args, **kwargs):
        dispose_calls["n"] += 1
        return real_dispose(*args, **kwargs)

    monkeypatch.setattr(engine, "dispose", dispose_spy)

    busy, _log_pages, _checkpointed = _checkpoint_wal(engine)
    assert busy == 0
    assert dispose_calls["n"] == 1
    assert call_count["n"] == 2
    real_dispose()


def test_checkpoint_wal_idle_returns_busy_zero_first_call(tmp_path, monkeypatch) -> None:
    db_file = tmp_path / "wal.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)

    dispose_calls = {"n": 0}
    real_dispose = engine.dispose

    def dispose_spy(*args, **kwargs):
        dispose_calls["n"] += 1
        return real_dispose(*args, **kwargs)

    monkeypatch.setattr(engine, "dispose", dispose_spy)

    busy, _log_pages, _checkpointed = _checkpoint_wal(engine)
    assert busy == 0
    assert dispose_calls["n"] == 0
    real_dispose()


async def test_run_checkpoints_wal_on_shutdown(monkeypatch) -> None:
    from bot.config import Settings as _Settings

    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)

    class _NoopMeteo:
        async def aclose(self) -> None:
            return None

    class _NoopKalshi:
        async def aopen(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        async def list_open_markets_for_series(self, series_prefix: str):
            return []

        async def get_orderbook(self, ticker: str):
            raise RuntimeError("no orderbook in noop kalshi")

    class _NoopACIS:
        async def fetch_daily_high(self, station, settled_date):
            return None

        async def aclose(self) -> None:
            return None

    app = App(
        settings=_Settings(mode="paper"),
        engine=engine,
        session_factory=sf,
        meteo=_NoopMeteo(),  # type: ignore[arg-type]
        kalshi=_NoopKalshi(),  # type: ignore[arg-type]
        acis=_NoopACIS(),  # type: ignore[arg-type]
        series_list=("KXHIGHDEN",),
    )

    async def fast_loop(app, stop):
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except asyncio.TimeoutError:
            pass

    monkeypatch.setattr(bot_main, "_forecast_loop", fast_loop)
    monkeypatch.setattr(bot_main, "_market_loop", fast_loop)
    monkeypatch.setattr(bot_main, "_eval_loop", fast_loop)
    monkeypatch.setattr(bot_main, "_settlement_loop", fast_loop)

    calls: list[object] = []
    real_checkpoint = bot_main._checkpoint_wal

    def spy(eng):
        calls.append(eng)
        return real_checkpoint(eng)

    monkeypatch.setattr(bot_main, "_checkpoint_wal", spy)

    await run(app, timedelta(milliseconds=200))

    assert calls
    assert all(c is engine for c in calls)


def _has_app_nbm_divergences_writer(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        else:
            continue
        for tgt in targets:
            if not isinstance(tgt, ast.Subscript):
                continue
            value = tgt.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "nbm_divergences"
                and isinstance(value.value, ast.Name)
                and value.value.id == "app"
            ):
                return True
    return False


def test_app_carries_no_dead_nbm_divergences_field() -> None:
    field_names = {f.name for f in dataclasses.fields(App)}
    if "nbm_divergences" not in field_names:
        return
    src = py_inspect.getsource(bot_main)
    assert _has_app_nbm_divergences_writer(src), (
        "App.nbm_divergences exists but has no subscript-assign writer"
    )


def test_nbm_divergences_writer_detector_accepts_subscript_assign() -> None:
    assert _has_app_nbm_divergences_writer("app.nbm_divergences[k] = v")


def test_nbm_divergences_writer_detector_accepts_aug_assign() -> None:
    assert _has_app_nbm_divergences_writer("app.nbm_divergences[k] += 1")


def test_nbm_divergences_writer_detector_rejects_reader_only() -> None:
    assert not _has_app_nbm_divergences_writer("x = app.nbm_divergences[k]")


def test_nbm_divergences_writer_detector_rejects_dict_method_call() -> None:
    assert not _has_app_nbm_divergences_writer("app.nbm_divergences.get(k)")


def test_nbm_divergences_writer_detector_rejects_unrelated_subscript_assign() -> None:
    assert not _has_app_nbm_divergences_writer("app.something_else[k] = v")


def test_build_intents_passes_literal_none_for_nbm_divergence() -> None:
    src = py_inspect.getsource(_build_intents)
    tree = ast.parse(src)
    keyword_constants: list[bool] = []
    found_any = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword):
            continue
        if node.arg != "nbm_divergence":
            continue
        found_any = True
        assert isinstance(node.value, ast.Constant), (
            f"nbm_divergence kwarg is not a literal Constant: {ast.dump(node.value)}"
        )
        assert node.value.value is None
        keyword_constants.append(True)
    assert found_any, "no nbm_divergence keyword arg found in _build_intents"


class _FixedFairCdf:
    def __init__(self, fair: Decimal) -> None:
        self._fair = float(fair)

    def prob_range(self, lo: float, hi: float) -> float:
        return self._fair

    def cdf(self, x: float) -> float:
        return self._fair


_AGGREGATE_SERIES: tuple[str, ...] = (
    "KXHIGHDEN",
    "KXHIGHAUS",
    "KXHIGHCHI",
    "KXHIGHNY",
    "KXHIGHPHIL",
    "KXHIGHTATL",
    "KXHIGHTBOS",
    "KXHIGHTDAL",
    "KXHIGHTDC",
)


_AGGREGATE_EVENT_DATES: tuple[date, ...] = (
    date(2026, 5, 8),
    date(2026, 5, 9),
    date(2026, 5, 10),
)


_AGGREGATE_RUN_TIME: datetime = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
_AGGREGATE_NOW: datetime = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)


def _ticker_for(series: str, event_date: date, lo: int, hi: int) -> str:
    date_str = event_date.strftime("%y%b%d").upper()
    return f"{series}-{date_str}-T{lo}-{hi}"


def _seed_aggregate_fixture(app: App) -> list[tuple[KalshiMarket, KalshiOrderbook, Decimal]]:
    fixtures: list[tuple[KalshiMarket, KalshiOrderbook, Decimal]] = []
    close_at = datetime(2026, 5, 20, 23, 0, tzinfo=timezone.utc)
    idx = 0
    for series in _AGGREGATE_SERIES:
        for event_date in _AGGREGATE_EVENT_DATES:
            yes_ask, yes_bid, fair = _AGGREGATE_TEST_PRICES[idx % len(_AGGREGATE_TEST_PRICES)]
            ticker = _ticker_for(series, event_date, 70 + idx, 75 + idx)
            market = _market_from(
                ticker, str(yes_ask), str(yes_bid), close_at=close_at, status="active"
            )
            book = _book_from(ticker, str(yes_ask), str(yes_bid), now=_AGGREGATE_NOW)
            station = STATIONS[series].station
            app.forecast_cdfs[(station, event_date)] = _FixedFairCdf(fair)  # type: ignore[assignment]
            app.ensemble_spreads[(station, event_date)] = Decimal("3.0")
            app.forecast_run_times[(station, event_date)] = _AGGREGATE_RUN_TIME
            app.latest_markets[ticker] = market
            app.latest_orderbooks[ticker] = book
            fixtures.append((market, book, fair))
            idx += 1
    return fixtures


def _pin_seven_fifty_per_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    def stub(*, ticker, market, book, fair_yes, **_kwargs):
        if fair_yes > book.yes_ask:
            side = TradeSide.BUY_YES
            price = book.yes_ask
        else:
            side = TradeSide.SELL_YES
            price = Decimal("1") - book.yes_bid
        contracts = int(Decimal("7.50") / price)
        return [
            TradeIntent(
                market_ticker=ticker,
                side=side,
                contracts=contracts,
                fair_yes=fair_yes,
                q_raw=fair_yes,
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub)


async def test_evaluate_strategies_blocks_on_aggregate_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lift_per_key_caps_keep_aggregate(monkeypatch, aggregate_cap=Decimal("200"))
    _pin_seven_fifty_per_intent(monkeypatch)

    app = _make_app(series_list=_AGGREGATE_SERIES)
    fixtures = _seed_aggregate_fixture(app)
    assert len(fixtures) == 27
    events = {m.event_ticker for m, _, _ in fixtures}
    series = {m.series for m, _, _ in fixtures}
    assert len(events) >= 13
    assert len(series) >= 9

    await evaluate_strategies(app, _AGGREGATE_NOW)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
        agg_failures = session.scalars(
            select(GateFailure).where(GateFailure.gate_name == "within_aggregate_cap")
        ).all()
        per_key_failures = session.scalars(
            select(GateFailure).where(
                GateFailure.gate_name.in_(["within_event_cap", "within_series_cap"])
            )
        ).all()

    assert len(trades) == 26, (
        f"expected 26 trades (cumulative $195, trade 27 would push to $202.50 > $200); "
        f"got {len(trades)}"
    )
    assert len(agg_failures) == 1
    assert per_key_failures == []


async def test_evaluate_strategies_aggregate_overlay_updates_within_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lift_per_key_caps_keep_aggregate(monkeypatch, aggregate_cap=Decimal("40"))
    _pin_seven_fifty_per_intent(monkeypatch)

    app = _make_app(series_list=_AGGREGATE_SERIES)
    _seed_aggregate_fixture(app)

    await evaluate_strategies(app, _AGGREGATE_NOW)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
        agg_failures = session.scalars(
            select(GateFailure).where(GateFailure.gate_name == "within_aggregate_cap")
        ).all()

    assert len(trades) == 5, (
        f"expected 5 trades (cumulative $37.50 < $40 < $45.00 on 6th); got {len(trades)}"
    )
    assert agg_failures


async def test_evaluate_strategies_aggregate_kwarg_threads_across_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lift_per_key_caps_keep_aggregate(monkeypatch, aggregate_cap=Decimal("40"))
    _pin_seven_fifty_per_intent(monkeypatch)

    app = _make_app(series_list=_AGGREGATE_SERIES)
    _seed_aggregate_fixture(app)

    observed_aggregates: list[Decimal] = []
    real_gate_ctx = bot_main._gate_ctx_for

    def recorder(**kwargs):  # type: ignore[no-untyped-def]
        observed_aggregates.append(kwargs["aggregate_existing_dollars"])
        return real_gate_ctx(**kwargs)

    monkeypatch.setattr(bot_main, "_gate_ctx_for", recorder)

    await evaluate_strategies(app, _AGGREGATE_NOW)

    assert len(observed_aggregates) >= 6
    expected_prefix = [
        Decimal("0"),
        Decimal("7.50"),
        Decimal("15.00"),
        Decimal("22.50"),
        Decimal("30.00"),
        Decimal("37.50"),
    ]
    assert observed_aggregates[:6] == expected_prefix


async def test_bankroll_accessor_drives_all_downstream_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = Decimal("777.77")
    monkeypatch.setattr(bot_main, "bankroll", lambda app=None: sentinel)

    edge_seen: list[Decimal] = []
    tails_seen: list[Decimal] = []
    real_edge_evaluate = edge_strategy.evaluate
    real_tails_evaluate = tails_strategy.evaluate

    def edge_recorder(ctx, **kwargs):  # type: ignore[no-untyped-def]
        edge_seen.append(ctx.bankroll)
        return real_edge_evaluate(ctx, **kwargs)

    def tails_recorder(ctx, **kwargs):  # type: ignore[no-untyped-def]
        tails_seen.append(ctx.bankroll)
        return real_tails_evaluate(ctx, **kwargs)

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", edge_recorder)
    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", tails_recorder)

    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    bracket = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    bracket_book = _book_from(bracket.ticker, "0.20", "0.18")
    tail = _market_from(
        "KXHIGHDEN-26MAY08-T100",
        "0.50",
        "0.48",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    tail_book = _book_from(tail.ticker, "0.50", "0.48")
    meteo = _StubMeteo(fc)
    kalshi = _StubKalshi(
        markets=[bracket, tail],
        orderbooks={bracket.ticker: bracket_book, tail.ticker: tail_book},
    )
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)

    intent = TradeIntent(
        market_ticker=bracket.ticker,
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    direct_ctx = _gate_ctx_for(
        intent=intent,
        market=bracket,
        fair_yes=Decimal("0.50"),
        spread=Decimal("3.0"),
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        now=now,
        book=bracket_book,
        market_existing_dollars=Decimal("0"),
        event_existing_dollars=Decimal("0"),
        series_existing_dollars=Decimal("0"),
        aggregate_existing_dollars=Decimal("0"),
        buy_yes_depth=bracket_book.no_bid_depth,
        sell_yes_depth=bracket_book.yes_bid_depth,
    )

    assert edge_seen, "edge_strategy.evaluate was not reached"
    assert tails_seen, "tails_strategy.evaluate was not reached"
    assert all(v == sentinel for v in edge_seen)
    assert all(v == sentinel for v in tails_seen)
    assert direct_ctx.account_balance == sentinel
    assert direct_ctx.market_position_cap == sentinel * bot_main.MARKET_POSITION_FRAC
    assert direct_ctx.event_position_cap == sentinel * bot_main.EVENT_POSITION_FRAC
    assert direct_ctx.series_position_cap == sentinel * bot_main.SERIES_POSITION_FRAC
    assert direct_ctx.aggregate_exposure_cap == sentinel * bot_main.AGGREGATE_EXPOSURE_FRAC


def test_caps_observed_at_runtime_follow_bankroll_accessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_cap = getattr(bot_main, "MARKET_POSITION_CAP")
    event_cap = getattr(bot_main, "EVENT_POSITION_CAP")
    series_cap = getattr(bot_main, "SERIES_POSITION_CAP")
    aggregate_cap = getattr(bot_main, "AGGREGATE_EXPOSURE_CAP")
    if bot_main.LIVE_BANKROLL_ENABLED:
        monkeypatch.setattr(bot_main, "bankroll", lambda: Decimal("1000"))
        bk = bot_main.bankroll()
        assert market_cap == bk * bot_main.MARKET_POSITION_FRAC
        assert event_cap == bk * bot_main.EVENT_POSITION_FRAC
        assert series_cap == bk * bot_main.SERIES_POSITION_FRAC
        assert aggregate_cap == bk * bot_main.AGGREGATE_EXPOSURE_FRAC
    else:
        seed = bot_main.PAPER_BANKROLL
        assert market_cap == seed * bot_main.MARKET_POSITION_FRAC
        assert event_cap == seed * bot_main.EVENT_POSITION_FRAC
        assert series_cap == seed * bot_main.SERIES_POSITION_FRAC
        assert aggregate_cap == seed * bot_main.AGGREGATE_EXPOSURE_FRAC


_CAP_NAME_PATTERN = _re.compile(r"^[A-Z_]+_(POSITION|EXPOSURE)_CAP$")


def _find_paper_bankroll_offenders(source: str) -> list[tuple[int, str, str]]:
    tree = ast.parse(source)

    bankroll_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "bankroll":
            for descendant in ast.walk(node):
                if hasattr(descendant, "lineno"):
                    bankroll_lines.add(descendant.lineno)

    paper_bankroll_decl_lines: set[int] = set()
    cap_derivation_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tgt_id = node.target.id
            if tgt_id == "PAPER_BANKROLL":
                paper_bankroll_decl_lines.add(node.lineno)
                continue
            if _CAP_NAME_PATTERN.match(tgt_id) and node.value is not None:
                has_paper_bankroll = False
                has_frac = False
                for descendant in ast.walk(node.value):
                    if isinstance(descendant, ast.Name):
                        if descendant.id == "PAPER_BANKROLL":
                            has_paper_bankroll = True
                        elif descendant.id.endswith("_FRAC"):
                            has_frac = True
                if has_paper_bankroll and has_frac:
                    for descendant in ast.walk(node):
                        if hasattr(descendant, "lineno"):
                            cap_derivation_lines.add(descendant.lineno)

    offenders: list[tuple[int, str, str]] = []
    allowed_lines = bankroll_lines | paper_bankroll_decl_lines | cap_derivation_lines
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "PAPER_BANKROLL":
            if node.lineno not in allowed_lines:
                offenders.append((node.lineno, "Name", node.id))
        elif isinstance(node, ast.Attribute) and node.attr == "PAPER_BANKROLL":
            offenders.append((node.lineno, "Attribute", node.attr))
        elif isinstance(node, ast.Constant) and node.value == "PAPER_BANKROLL":
            offenders.append((node.lineno, "Constant", str(node.value)))
    return offenders


def test_only_bankroll_accessor_reads_paper_bankroll_directly() -> None:
    src = py_inspect.getsource(bot_main)
    offenders = _find_paper_bankroll_offenders(src)
    assert offenders == [], f"PAPER_BANKROLL accessed outside allowed sites: {offenders}"


def test_paper_bankroll_offender_predicate_flags_attribute_access() -> None:
    synthetic = "import bot.main as m\ndef f():\n    return m.PAPER_BANKROLL\n"
    offenders = _find_paper_bankroll_offenders(synthetic)
    assert offenders, "predicate failed to flag attribute access to PAPER_BANKROLL"
    assert any(kind == "Attribute" for _line, kind, _name in offenders)


def test_paper_bankroll_offender_predicate_accepts_real_main() -> None:
    src = py_inspect.getsource(bot_main)
    offenders = _find_paper_bankroll_offenders(src)
    assert offenders == [], f"real bot/main.py flagged: {offenders}"


def _find_cap_import_offenders(source: str, filename: str) -> list[tuple[str, int, str, str]]:
    tree = ast.parse(source)
    bot_main_aliases: set[str] = set()
    offenders: list[tuple[str, int, str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "bot.main":
                    bot_main_aliases.add(alias.asname or "bot")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "bot.main":
                for alias in node.names:
                    if alias.name == "*":
                        offenders.append((filename, node.lineno, "from-star", "*"))
                    elif _CAP_NAME_PATTERN.match(alias.name):
                        kind = "from-alias" if alias.asname else "from-bare"
                        offenders.append((filename, node.lineno, kind, alias.name))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _CAP_NAME_PATTERN.match(node.attr):
            base = node.value
            head: str | None = None
            if isinstance(base, ast.Name):
                head = base.id
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                if base.value.id == "bot" and base.attr == "main":
                    head = "bot"
            if head is not None and head in bot_main_aliases:
                offenders.append((filename, node.lineno, "attribute", node.attr))
    return offenders


def test_no_test_imports_cap_constants_by_name() -> None:
    tests_dir = Path(__file__).parent
    offenders_all: list[tuple[str, int, str, str]] = []
    for test_file in sorted(tests_dir.glob("test_*.py")):
        src = test_file.read_text()
        offenders_all.extend(_find_cap_import_offenders(src, test_file.name))
    assert offenders_all == [], (
        "tests imported cap constants by name (silent-no-op shape: monkeypatch.setattr "
        "on bot.main does not update test-module-local bindings); offenders=" + repr(offenders_all)
    )


def test_cap_import_offender_predicate_flags_bare_from_import() -> None:
    src = "from bot.main import MARKET_POSITION_CAP\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders
    assert any(kind == "from-bare" for _f, _l, kind, _n in offenders)


def test_cap_import_offender_predicate_flags_aliased_from_import() -> None:
    src = "from bot.main import MARKET_POSITION_CAP as cap\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders
    assert any(kind == "from-alias" for _f, _l, kind, _n in offenders)


def test_cap_import_offender_predicate_flags_parenthesized_multi_name() -> None:
    src = "from bot.main import (App, MARKET_POSITION_CAP, evaluate_strategies)\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders
    assert any(name == "MARKET_POSITION_CAP" for _f, _l, _k, name in offenders)


def test_cap_import_offender_predicate_flags_star_import() -> None:
    src = "from bot.main import *\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders
    assert any(kind == "from-star" for _f, _l, kind, _n in offenders)


def test_cap_import_offender_predicate_flags_attribute_via_bare_import() -> None:
    src = "import bot.main\nx = bot.main.MARKET_POSITION_CAP\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders
    assert any(kind == "attribute" for _f, _l, kind, _n in offenders)


def test_cap_import_offender_predicate_flags_attribute_via_aliased_import() -> None:
    src = "import bot.main as m\nx = m.MARKET_POSITION_CAP\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders
    assert any(kind == "attribute" for _f, _l, kind, _n in offenders)


def test_cap_import_offender_predicate_accepts_aliased_import_with_no_attribute() -> None:
    src = "import bot.main as bot_main\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders == []


def test_cap_import_offender_predicate_accepts_app_import() -> None:
    src = "from bot.main import App\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders == []


def test_cap_import_offender_predicate_accepts_evaluate_strategies_import() -> None:
    src = "from bot.main import evaluate_strategies\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders == []


def test_cap_import_offender_predicate_accepts_paper_bankroll_import() -> None:
    src = "from bot.main import PAPER_BANKROLL\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders == []


def test_cap_import_offender_predicate_accepts_required_cushion_import() -> None:
    src = "from bot.main import REQUIRED_CUSHION\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders == []


def test_cap_import_offender_predicate_accepts_unrelated_module_import() -> None:
    src = "from elsewhere import MARKET_POSITION_CAP\n"
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders == []


def test_cap_import_offender_predicate_accepts_monkeypatch_string_literal() -> None:
    src = (
        "import bot.main as bot_main\n"
        "def t(monkeypatch):\n"
        "    monkeypatch.setattr(bot_main, 'MARKET_POSITION_CAP', 1)\n"
    )
    offenders = _find_cap_import_offenders(src, "synthetic.py")
    assert offenders == []


def test_bankroll_defaults_to_paper_bankroll_when_no_app() -> None:
    assert bot_main.bankroll() == bot_main.PAPER_BANKROLL
    assert bot_main.bankroll(None) == bot_main.PAPER_BANKROLL


def test_bankroll_defaults_to_paper_bankroll_when_app_bankroll_is_none() -> None:
    app = _make_app()
    assert app.bankroll is None
    assert bot_main.bankroll(app) == bot_main.PAPER_BANKROLL


def test_bankroll_returns_app_override_when_set() -> None:
    app = _make_app()
    app.bankroll = Decimal("1234.56")
    assert bot_main.bankroll(app) == Decimal("1234.56")


def test_cap_helpers_follow_app_bankroll_override() -> None:
    app = _make_app()
    app.bankroll = Decimal("1000")
    assert bot_main.market_position_cap(app) == Decimal("1000") * bot_main.MARKET_POSITION_FRAC
    assert bot_main.event_position_cap(app) == Decimal("1000") * bot_main.EVENT_POSITION_FRAC
    assert bot_main.series_position_cap(app) == Decimal("1000") * bot_main.SERIES_POSITION_FRAC
    assert (
        bot_main.aggregate_exposure_cap(app) == Decimal("1000") * bot_main.AGGREGATE_EXPOSURE_FRAC
    )


def test_cap_helpers_fall_back_to_paper_bankroll_when_no_app() -> None:
    base = bot_main.PAPER_BANKROLL
    assert bot_main.market_position_cap() == base * bot_main.MARKET_POSITION_FRAC
    assert bot_main.event_position_cap(None) == base * bot_main.EVENT_POSITION_FRAC
    assert bot_main.series_position_cap() == base * bot_main.SERIES_POSITION_FRAC
    assert bot_main.aggregate_exposure_cap(None) == base * bot_main.AGGREGATE_EXPOSURE_FRAC


from bot.config import Settings as _DemoSettings  # noqa: E402
from bot.execution.order_placer import DemoOrder, DemoOrderIdempotent  # noqa: E402
from bot.storage.sqlite import DemoOrder as DemoOrderRow  # noqa: E402


class _DemoKalshi(_StubKalshi):
    def __init__(self, markets, orderbooks) -> None:
        super().__init__(markets, orderbooks)
        self.balance = Decimal("500")

    async def get_balance(self) -> Decimal:
        return self.balance


def _make_demo_app(
    meteo: _StubMeteo | None = None,
    kalshi: _StubKalshi | None = None,
    series_list: tuple[str, ...] = ("KXHIGHDEN",),
    bankroll: Decimal | None = None,
) -> App:
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    settings = _DemoSettings(mode="demo", kalshi_demo_key_id="demo-key-id")
    return App(
        settings=settings,
        engine=engine,
        session_factory=sf,
        meteo=meteo,  # type: ignore[arg-type]
        kalshi=kalshi,  # type: ignore[arg-type]
        acis=_StubACIS(None),  # type: ignore[arg-type]
        series_list=series_list,
        bankroll=bankroll,
    )


def _demo_bracket_setup(now: datetime):
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    return meteo, market, book


def _one_intent_stub(
    side: TradeSide = TradeSide.BUY_YES, contracts: int = 5, strategy: str = "edge"
):
    def stub_intent(
        *,
        app,
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        mode,
        now,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=side,
                contracts=contracts,
                fair_yes=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy=strategy,
            )
        ]

    return stub_intent


def _demo_order(
    *,
    cid: str = "kw-edge-yes-KXHIGHDEN-26MAY08-T70-75-2026-05-08",
    eid: str = "EX-1",
    ticker: str = "KXHIGHDEN-26MAY08-T70-75",
    side_kalshi: str = "yes",
    requested: int = 5,
    filled: int = 5,
    avg: Decimal | None = Decimal("0.20"),
    status: str = "executed",
    now: datetime | None = None,
) -> DemoOrder:
    return DemoOrder(
        client_order_id=cid,
        exchange_order_id=eid,
        ticker=ticker,
        side_kalshi=side_kalshi,
        requested_contracts=requested,
        filled_contracts=filled,
        requested_yes_price_dollars=Decimal("0.20"),
        avg_yes_fill_price_dollars=avg,
        fee_dollars=Decimal("0.01"),
        status=status,
        placed_at=now or datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
    )


async def test_evaluate_strategies_routes_to_simulator_in_paper_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    posted: list[object] = []

    async def boom(*a, **k):
        posted.append(a)
        raise AssertionError("placer must not be called in paper mode")

    monkeypatch.setattr(bot_main, "place_order_demo", boom)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
        demo_orders = session.scalars(select(DemoOrderRow)).all()
    assert len(trades) == 1
    assert demo_orders == []
    assert posted == []


async def test_evaluate_strategies_routes_to_demo_placer_in_demo_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch)

    async def fake_place(intent, book, client, *, now):
        return _demo_order(now=now)

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(demo_orders) == 1
    assert len(trades) == 1
    assert trades[0].demo_order_client_id == demo_orders[0].client_order_id


async def test_evaluate_strategies_persists_demo_order_on_idempotent_409_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch)

    sentinel = DemoOrderIdempotent(
        client_order_id="kw-edge-yes-KXHIGHDEN-26MAY08-T70-75-2026-05-08",
        exchange_order_id="EX-PRIOR",
        status="executed",
        filled_contracts=5,
        requested_yes_price_dollars=Decimal("0.20"),
    )

    async def fake_place(intent, book, client, *, now):
        return sentinel

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(demo_orders) == 1
    row = demo_orders[0]
    assert row.client_order_id == sentinel.client_order_id
    assert row.exchange_order_id == "EX-PRIOR"
    assert row.status == "executed"
    assert row.filled_contracts == 5
    assert row.strategy == "edge"
    assert row.market_ticker == market.ticker
    assert trades == []


async def test_evaluate_strategies_persists_demo_order_on_idempotent_409_with_empty_eid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch)

    cid = "kw-edge-yes-KXHIGHDEN-26MAY08-T70-75-2026-05-08"
    sentinel = DemoOrderIdempotent(
        client_order_id=cid,
        exchange_order_id="",
        status="executed",
        filled_contracts=5,
        requested_yes_price_dollars=Decimal("0.20"),
    )

    async def fake_place(intent, book, client, *, now):
        return sentinel

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
    assert len(demo_orders) == 1
    row = demo_orders[0]
    assert row.client_order_id == cid
    assert row.exchange_order_id is None
    assert row.status == "executed"

    poll_record = DemoOrder(
        client_order_id=cid,
        exchange_order_id="EX-LATE",
        ticker=market.ticker,
        side_kalshi="yes",
        requested_contracts=5,
        filled_contracts=5,
        requested_yes_price_dollars=Decimal("0.20"),
        avg_yes_fill_price_dollars=Decimal("0.20"),
        fee_dollars=Decimal("0.01"),
        status="executed",
        placed_at=now,
    )
    from bot.execution.order_reconciler import upsert_exchange_record

    with app.session_factory() as session:
        upsert_exchange_record(session, poll_record)
        session.commit()
        rows = session.scalars(select(DemoOrderRow)).all()
        assert len(rows) == 1
        stitched = rows[0]
        assert stitched.client_order_id == cid
        assert stitched.exchange_order_id == "EX-LATE"
        assert stitched.strategy == "edge"
        backfill = session.scalars(
            select(DemoOrderRow).where(DemoOrderRow.client_order_id.like("kw-backfill-%"))
        ).all()
        assert backfill == []


async def test_evaluate_strategies_demo_does_not_post_on_failing_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, _market, book = _demo_bracket_setup(now)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
        status="closed",
    )
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    # closed markets are dropped by refresh; seed caches directly.
    app.latest_markets[market.ticker] = market
    app.latest_orderbooks[market.ticker] = book
    _lift_caps(monkeypatch)

    posted: list[object] = []

    async def boom(*a, **k):
        posted.append(a)
        raise AssertionError("must not post on failing gate")

    monkeypatch.setattr(bot_main, "place_order_demo", boom)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
        gate_failures = session.scalars(
            select(GateFailure).where(GateFailure.gate_name == "market_open")
        ).all()
    assert trades == []
    assert posted == []
    assert gate_failures
    assert all(f.mode == "demo" for f in gate_failures)


async def test_evaluate_strategies_demo_mode_asserts_no_cost_per_contract_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    posted: list[object] = []

    async def boom(*a, **k):
        posted.append(a)
        return _demo_order(now=now)

    monkeypatch.setattr(bot_main, "place_order_demo", boom)

    def bad_intents(
        *,
        app,
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        mode,
        now,
        **_kwargs,
    ):
        edge_strategy.evaluate(
            edge_strategy.EdgeContext(
                yes_ask=book.yes_ask,
                yes_bid=book.yes_bid,
                fair_yes=fair_yes,
                ensemble_spread=spread,
                bankroll=Decimal("500"),
                is_same_day=False,
                is_blacklisted=False,
                nbm_divergence=None,
                sigma_T_median=Decimal("2.0"),
                event_budget_remaining=Decimal("9999"),
                market_budget_remaining=Decimal("9999"),
                depth_at_price=10_000,
                price_per_contract=book.yes_ask,
                no_cost_per_contract=None,
            ),
            mode="demo",
        )
        return []

    monkeypatch.setattr(bot_main, "_build_intents", bad_intents)

    with pytest.raises(RuntimeError, match="demo mode requires book-derived cost basis"):
        await evaluate_strategies(app, now)
    assert posted == []


async def test_evaluate_strategies_paper_mode_none_cost_per_contract_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    def ok_intents(
        *,
        app,
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        mode,
        now,
        **_kwargs,
    ):
        edge_strategy.evaluate(
            edge_strategy.EdgeContext(
                yes_ask=book.yes_ask,
                yes_bid=book.yes_bid,
                fair_yes=fair_yes,
                ensemble_spread=spread,
                bankroll=Decimal("500"),
                is_same_day=False,
                is_blacklisted=False,
                nbm_divergence=None,
                sigma_T_median=Decimal("2.0"),
                event_budget_remaining=Decimal("9999"),
                market_budget_remaining=Decimal("9999"),
                depth_at_price=10_000,
                price_per_contract=book.yes_ask,
                no_cost_per_contract=None,
            ),
            mode="paper",
        )
        return []

    monkeypatch.setattr(bot_main, "_build_intents", ok_intents)
    await evaluate_strategies(app, now)


async def test_evaluate_strategies_does_not_hold_db_lock_during_intent_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    observed: list[bool] = []
    real_stub = _one_intent_stub()

    def spy(**kwargs):
        observed.append(app.db_lock.locked())
        return real_stub(**kwargs)

    monkeypatch.setattr(bot_main, "_build_intents", spy)
    await evaluate_strategies(app, now)

    assert observed
    assert all(v is False for v in observed)


async def test_snapshot_markets_atomically_returns_consistent_pair_under_contention() -> None:
    app = _make_app()
    for i in range(50):
        t = f"KXHIGHDEN-26MAY08-T{i}"
        app.latest_markets[t] = _market_from(
            t, "0.20", "0.18", datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
        )
        app.latest_orderbooks[t] = _book_from(t, "0.20", "0.18")

    stop = asyncio.Event()

    async def mutator():
        flip = True
        while not stop.is_set():
            async with app.db_lock:
                if flip:
                    t = "KXHIGHDEN-26MAY08-Tnew"
                    app.latest_markets[t] = _market_from(
                        t, "0.20", "0.18", datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
                    )
                    app.latest_orderbooks[t] = _book_from(t, "0.20", "0.18")
                else:
                    app.latest_markets.pop("KXHIGHDEN-26MAY08-Tnew", None)
                    app.latest_orderbooks.pop("KXHIGHDEN-26MAY08-Tnew", None)
                flip = not flip
            await asyncio.sleep(0)

    task = asyncio.create_task(mutator())
    try:
        for _ in range(200):
            markets, books = await bot_main._snapshot_markets_atomically(app)
            assert {t for t, _m in markets} == set(books.keys())
    finally:
        stop.set()
        await task


async def test_naive_two_line_snapshot_can_disagree() -> None:
    app = _make_app()
    t = "KXHIGHDEN-26MAY08-Tnew"
    app.latest_markets[t] = _market_from(
        t, "0.20", "0.18", datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    )
    app.latest_orderbooks[t] = _book_from(t, "0.20", "0.18")

    async def naive_snapshot():
        markets = tuple(app.latest_markets.items())
        await asyncio.sleep(0)
        books = dict(app.latest_orderbooks)
        return markets, books

    snap_task = asyncio.create_task(naive_snapshot())
    await asyncio.sleep(0)
    app.latest_orderbooks.pop(t, None)
    markets, books = await snap_task
    assert {tk for tk, _m in markets} != set(books.keys())


async def test_snapshot_is_immune_to_post_call_mutation() -> None:
    app = _make_app()
    t = "KXHIGHDEN-26MAY08-T1"
    app.latest_markets[t] = _market_from(
        t, "0.20", "0.18", datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    )
    app.latest_orderbooks[t] = _book_from(t, "0.20", "0.18")

    markets, books = await bot_main._snapshot_markets_atomically(app)
    app.latest_markets.clear()
    app.latest_orderbooks.clear()

    assert {tk for tk, _m in markets} == {t}
    assert set(books.keys()) == {t}


async def test_evaluate_strategies_write_phase_blocks_on_competing_lock_holder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    snapshot_done = asyncio.Event()
    holder_acquired = asyncio.Event()
    release = asyncio.Event()
    real_snapshot = bot_main._snapshot_markets_atomically

    async def snapshot_spy(app):
        pair = await real_snapshot(app)
        snapshot_done.set()
        await holder_acquired.wait()
        return pair

    monkeypatch.setattr(bot_main, "_snapshot_markets_atomically", snapshot_spy)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    async def hold_lock():
        await snapshot_done.wait()
        async with app.db_lock:
            holder_acquired.set()
            await release.wait()

    holder = asyncio.create_task(hold_lock())
    eval_task = asyncio.create_task(evaluate_strategies(app, now))
    await asyncio.wait_for(holder_acquired.wait(), timeout=2.0)
    assert app.db_lock.locked()
    assert not eval_task.done()

    release.set()
    await holder
    await asyncio.wait_for(eval_task, timeout=2.0)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(trades) == 1


async def test_phase1_skips_post_if_ticker_evicted_before_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    posted: list[str] = []

    async def fake_place(intent, book, client, *, now):
        posted.append(intent.market_ticker)
        return _demo_order(now=now)

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    app.latest_markets.pop(market.ticker, None)

    await evaluate_strategies(app, now)

    assert posted == []
    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
    assert demo_orders == []


async def test_phase1_skips_post_if_ticker_evicted_after_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch)

    posted: list[str] = []

    async def fake_place(intent, book, client, *, now):
        posted.append(intent.market_ticker)
        return _demo_order(now=now)

    real_drain = bot_main.place_orders_resilient

    async def evicting_drain(items, placer):
        assert list(items), "intent must reach the drain before eviction"
        app.latest_markets.pop(market.ticker, None)
        return await real_drain(items, placer)

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    monkeypatch.setattr(bot_main, "place_orders_resilient", evicting_drain)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    assert market.ticker in app.latest_markets

    await evaluate_strategies(app, now)

    assert posted == []
    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
    assert demo_orders == []


def _phase2_drain_app() -> App:
    return _make_demo_app()


async def _run_phase2(app: App, *, demo_rows=None, paper_rows=None, gate_failures=None):
    await bot_main._commit_phase2(
        app,
        gate_failures=gate_failures or [],
        paper_rows=paper_rows or [],
        demo_rows=demo_rows or [],
    )


def _demo_values(now: datetime, **overrides) -> dict:
    intent = TradeIntent(
        market_ticker="KXHIGHDEN-26MAY08-T70-75",
        side=TradeSide.BUY_YES,
        contracts=5,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    order = _demo_order(
        now=now,
        **{
            k: overrides.pop(k)
            for k in list(overrides)
            if k in {"cid", "eid", "side_kalshi", "filled", "avg", "status"}
        },
    )
    values = bot_main._demo_order_values(order, intent, now)
    values.update(overrides)
    return values


async def test_phase2_legacy_merge_pattern_raises_integrity_error() -> None:
    from sqlalchemy.exc import IntegrityError

    app = _phase2_drain_app()
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    with app.session_factory() as session:
        session.add(
            DemoOrderRow(
                client_order_id="kw-edge-yes-X",
                exchange_order_id="EXR",
                market_ticker="KXHIGHDEN-26MAY08-T70-75",
                strategy="edge",
                side="yes",
                requested_contracts=5,
                filled_contracts=5,
                status="executed",
                placed_at=now,
                last_status_at=now,
            )
        )
        session.commit()

    with pytest.raises(IntegrityError):
        with app.session_factory() as session:
            session.merge(
                DemoOrderRow(
                    client_order_id="kw-edge-yes-X",
                    exchange_order_id="EX-OTHER",
                    market_ticker="KXHIGHDEN-26MAY08-T70-75",
                    strategy="edge",
                    side="yes",
                    requested_contracts=5,
                    filled_contracts=0,
                    status="resting",
                    placed_at=now,
                    last_status_at=now,
                )
            )
            session.flush()


async def test_phase2_upsert_preserves_reconciler_terminal_state() -> None:
    app = _phase2_drain_app()
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    cid = "kw-edge-yes-X"
    with app.session_factory() as session:
        session.add(
            DemoOrderRow(
                client_order_id=cid,
                exchange_order_id=None,
                market_ticker="KXHIGHDEN-26MAY08-T70-75",
                strategy=None,
                side="yes",
                requested_contracts=5,
                filled_contracts=10,
                avg_fill_price=Decimal("0.205"),
                fee_dollars=Decimal("0.07"),
                status="executed",
                placed_at=now,
                last_status_at=now,
            )
        )
        session.commit()

    values = _demo_values(now, cid=cid, eid="")
    values["status"] = "resting"
    values["filled_contracts"] = 0
    await _run_phase2(app, demo_rows=[values])

    with app.session_factory() as session:
        row = session.scalars(select(DemoOrderRow).where(DemoOrderRow.client_order_id == cid)).one()
    assert row.status == "executed"
    assert row.filled_contracts == 10
    assert row.avg_fill_price == Decimal("0.205")
    assert row.fee_dollars == Decimal("0.07")
    assert row.strategy == "edge"
    assert row.fair_at_entry == Decimal("0.50")
    assert row.intended_at == now


async def test_phase2_upsert_inserts_on_no_collision() -> None:
    app = _phase2_drain_app()
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    values = _demo_values(now, cid="kw-edge-yes-FRESH", eid="")
    await _run_phase2(app, demo_rows=[values])

    with app.session_factory() as session:
        rows = session.scalars(select(DemoOrderRow)).all()
    assert len(rows) == 1
    assert rows[0].client_order_id == "kw-edge-yes-FRESH"
    assert rows[0].strategy == "edge"
    assert rows[0].status == "executed"


async def test_phase2_upsert_commits_gate_failures_despite_demo_row_savepoint_rollback() -> None:
    app = _phase2_drain_app()
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    # Seed a holder row with a different exchange id so stitch raises case-4 ValueError.
    with app.session_factory() as session:
        session.add(
            DemoOrderRow(
                client_order_id="kw-edge-yes-COLLIDE",
                exchange_order_id="EX-OTHER",
                market_ticker="KXHIGHDEN-26MAY08-T70-75",
                strategy="edge",
                side="yes",
                requested_contracts=5,
                filled_contracts=5,
                status="executed",
                placed_at=now,
                last_status_at=now,
            )
        )
        session.add(
            DemoOrderRow(
                client_order_id="kw-backfill-EX-2",
                exchange_order_id="EX-2",
                market_ticker="KXHIGHDEN-26MAY08-T70-75",
                strategy=None,
                side="yes",
                requested_contracts=5,
                filled_contracts=0,
                status="resting",
                placed_at=now,
                last_status_at=now,
            )
        )
        session.commit()

    values = _demo_values(now, cid="kw-edge-yes-COLLIDE", eid="EX-2")
    gate_failure = GateFailure(
        evaluated_at=now,
        gate_name="edge_threshold",
        reason="x",
        mode="demo",
        market_ticker="KXHIGHDEN-26MAY08-T70-75",
        last_seen_at=now,
    )
    await _run_phase2(app, demo_rows=[values], gate_failures=[gate_failure])

    with app.session_factory() as session:
        failures = session.scalars(select(GateFailure)).all()
    assert len(failures) == 1


async def test_phase2_paper_rows_upsert_survives_duplicate_demo_order_client_id() -> None:
    app = _phase2_drain_app()
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    cid = "kw-edge-yes-DUP"
    with app.session_factory() as session:
        session.add(
            PaperTradeRow(
                intended_at=now,
                market_ticker="KXHIGHDEN-26MAY08-T70-75",
                side="buy_yes",
                contracts=5,
                simulated_price=Decimal("0.20"),
                fee_dollars=Decimal("0.01"),
                fair_at_entry=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
                demo_order_client_id=cid,
            )
        )
        session.commit()

    dup_row = PaperTradeRow(
        intended_at=now,
        market_ticker="KXHIGHDEN-26MAY08-T70-75",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.20"),
        fee_dollars=Decimal("0.01"),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        demo_order_client_id=cid,
    )
    legacy_row = PaperTradeRow(
        intended_at=now,
        market_ticker="KXHIGHDEN-26MAY08-T70-75",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.20"),
        fee_dollars=Decimal("0.01"),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        demo_order_client_id=None,
    )
    gate_failure = GateFailure(
        evaluated_at=now,
        gate_name="edge_threshold",
        reason="x",
        mode="demo",
        market_ticker="T",
        last_seen_at=now,
    )
    await _run_phase2(app, paper_rows=[dup_row, legacy_row], gate_failures=[gate_failure])

    with app.session_factory() as session:
        for_cid = session.scalars(
            select(PaperTradeRow).where(PaperTradeRow.demo_order_client_id == cid)
        ).all()
        legacy = session.scalars(
            select(PaperTradeRow).where(PaperTradeRow.demo_order_client_id.is_(None))
        ).all()
        failures = session.scalars(select(GateFailure)).all()
    assert len(for_cid) == 1
    assert len(legacy) == 1
    assert len(failures) == 1


async def test_phase2_gate_failure_upsert_dedupes_identical_tuple() -> None:
    app = _phase2_drain_app()
    base = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    reason = "fair_value_sane fair_yes=9.649e-09 outside [0.01, 0.99]"
    last: datetime | None = None
    for i in range(5):
        ts = base + timedelta(minutes=i)
        last = ts
        await _run_phase2(
            app,
            gate_failures=[
                GateFailure(
                    evaluated_at=ts,
                    gate_name="fair_value_sane",
                    reason=reason,
                    mode="paper",
                    market_ticker="KXHIGHTPHX-26JUN01-B106.5",
                    last_seen_at=ts,
                )
            ],
        )

    with app.session_factory() as session:
        rows = session.scalars(select(GateFailure)).all()
    assert len(rows) == 1
    assert rows[0].count == 5
    assert rows[0].last_seen_at == last


async def test_phase2_gate_failure_upsert_keeps_distinct_tuples_separate() -> None:
    app = _phase2_drain_app()
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    failures = [
        GateFailure(
            evaluated_at=now,
            gate_name="fair_value_sane",
            reason="fair_yes=9.649e-09 outside [0.01, 0.99]",
            mode="paper",
            market_ticker="KXHIGHTPHX-26JUN01-B106.5",
            last_seen_at=now,
        ),
        GateFailure(
            evaluated_at=now,
            gate_name="fair_value_sane",
            reason="fair_yes=9.649e-09 outside [0.01, 0.99]",
            mode="paper",
            market_ticker="KXHIGHTNY-26JUN01-B72.5",
            last_seen_at=now,
        ),
        GateFailure(
            evaluated_at=now,
            gate_name="model_fresh",
            reason="model_age_hours=7.5 > 6",
            mode="paper",
            market_ticker="KXHIGHTPHX-26JUN01-B106.5",
            last_seen_at=now,
        ),
    ]
    await _run_phase2(app, gate_failures=failures)
    await _run_phase2(app, gate_failures=failures)

    with app.session_factory() as session:
        rows = session.scalars(select(GateFailure)).all()
    assert len(rows) == 3
    assert all(r.count == 2 for r in rows)


async def test_phase2_resting_order_skips_paper_row_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch)

    async def fake_place(intent, book, client, *, now):
        return _demo_order(now=now, status="resting", filled=0, avg=None)

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(demo_orders) == 1
    assert demo_orders[0].status == "resting"
    assert trades == []


async def test_phase2_late_fill_via_reconciler_inserts_paper_row() -> None:
    app = _make_demo_app()
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    cid = "kw-edge-yes-LATE"
    with app.session_factory() as session:
        session.add(
            DemoOrderRow(
                client_order_id=cid,
                exchange_order_id="EX-LATE",
                market_ticker="KXHIGHDEN-26MAY08-T70-75",
                strategy="edge",
                side="yes",
                requested_contracts=10,
                filled_contracts=0,
                requested_yes_price_dollars=Decimal("0.20"),
                fair_at_entry=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                intended_at=now,
                status="resting",
                placed_at=now,
                last_status_at=now,
            )
        )
        session.commit()

    from bot.execution.order_reconciler import DemoFill, reconcile_fills_into_demo_orders

    fill = DemoFill(
        fill_id="F1",
        order_id="EX-LATE",
        ticker="KXHIGHDEN-26MAY08-T70-75",
        outcome_side="yes",
        book_side="yes",
        count=10,
        yes_price_dollars=Decimal("0.795"),
        no_price_dollars=Decimal("0.205"),
        is_taker=True,
        created_time="",
        fee_cost=Decimal("0.02"),
    )
    order = _demo_order(cid=cid, eid="EX-LATE", filled=10, avg=Decimal("0.795"), now=now)
    async with app.db_lock:
        with app.session_factory() as session:
            reconcile_fills_into_demo_orders(session, [fill], [order])
            session.commit()

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(trades) == 1
    assert trades[0].simulated_price == Decimal("0.795")
    assert trades[0].strategy == "edge"


async def test_demo_intended_at_uses_pre_post_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch)

    async def slow_drain(items, placer):
        out = []
        for item in items:
            await asyncio.sleep(0.05)
            r = await placer(item)
            if r is not None:
                out.append(r)
        return out

    async def fake_place(intent, book, client, *, now):
        return _demo_order(now=now + timedelta(milliseconds=100))

    monkeypatch.setattr(bot_main, "place_orders_resilient", slow_drain)
    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        row = session.scalars(select(DemoOrderRow)).one()
    assert row.intended_at == now


async def test_paper_intended_at_uses_simulate_taker_fill_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        row = session.scalars(select(PaperTradeRow)).one()
    assert row.intended_at == now


def _wide_tails_market(now: datetime):
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(7).normal(40.0, 5.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T100",
        "0.98",
        "0.10",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.98", "0.10")
    return meteo, market, book


async def _capture_tails_call(app, meteo, market, book, now, monkeypatch, *, lift_market=True):
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_cdfs[("KDEN", date(2026, 5, 8))] = _StrongTailCdf()
    app.ensemble_spreads[("KDEN", date(2026, 5, 8))] = Decimal("5.0")
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch, lift_market=lift_market)

    captured: list[dict] = []
    real = tails_strategy.evaluate

    def rec(ctx, **kwargs):
        captured.append(kwargs)
        return real(ctx, **kwargs)

    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", rec)
    await evaluate_strategies(app, now)
    return captured


class _StrongTailCdf:
    def prob_range(self, lo: float, hi: float) -> float:
        return 0.01

    def cdf(self, x: float) -> float:
        return 0.99


async def test_tails_call_site_paper_mode_baseline_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _wide_tails_market(now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    captured = await _capture_tails_call(app, meteo, market, book, now, monkeypatch)

    assert captured
    for kwargs in captured:
        assert "position_cap" not in kwargs
        assert "contracts_cap" not in kwargs
        assert kwargs.get("mode") == "paper"


async def test_tails_call_site_demo_mode_market_budget_clamps_via_sizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _wide_tails_market(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi, bankroll=Decimal("500"))

    async def fake_place(intent, book, client, *, now):
        return _demo_order(now=now, ticker=intent.market_ticker, side_kalshi="no")

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    captured_ctx: list[tails_strategy.TailsContext] = []
    real = tails_strategy.evaluate

    def rec(ctx, **kwargs):
        captured_ctx.append(ctx)
        return real(ctx, **kwargs)

    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", rec)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_cdfs[("KDEN", date(2026, 5, 8))] = _StrongTailCdf()
    app.ensemble_spreads[("KDEN", date(2026, 5, 8))] = Decimal("5.0")
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch, lift_market=False)
    await evaluate_strategies(app, now)

    assert captured_ctx
    ctx = captured_ctx[0]
    assert ctx.market_budget_remaining == Decimal("7.50")
    assert int(ctx.market_budget_remaining / ctx.price_per_contract) == 8


async def test_tails_market_budget_scales_with_app_bankroll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _wide_tails_market(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi, bankroll=Decimal("1000"))

    async def fake_place(intent, book, client, *, now):
        return _demo_order(now=now, ticker=intent.market_ticker, side_kalshi="no")

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    captured_ctx: list[tails_strategy.TailsContext] = []
    real = tails_strategy.evaluate

    def rec(ctx, **kwargs):
        captured_ctx.append(ctx)
        return real(ctx, **kwargs)

    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", rec)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_cdfs[("KDEN", date(2026, 5, 8))] = _StrongTailCdf()
    app.ensemble_spreads[("KDEN", date(2026, 5, 8))] = Decimal("5.0")
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    _lift_caps(monkeypatch, lift_market=False)
    await evaluate_strategies(app, now)

    assert captured_ctx
    ctx = captured_ctx[0]
    assert ctx.market_budget_remaining == Decimal("15.00")


async def test_tails_call_site_demo_mode_zero_wired_cost_omits_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(7).normal(40.0, 5.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T100",
        "1.00",
        "1.00",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "1.00", "1.00")
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi, bankroll=Decimal("500"))
    captured = await _capture_tails_call(app, meteo, market, book, now, monkeypatch)

    assert captured
    for kwargs in captured:
        assert "position_cap" not in kwargs
        assert "contracts_cap" not in kwargs


async def test_tails_call_site_paper_mode_baseline_uses_sizer_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _wide_tails_market(now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    captured = await _capture_tails_call(app, meteo, market, book, now, monkeypatch)
    assert captured
    for kwargs in captured:
        assert "position_cap" not in kwargs
        assert "contracts_cap" not in kwargs
        assert "kelly_fraction" not in kwargs


async def test_startup_backfill_runs_before_eval_loop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    _rsa_pem: Path,
) -> None:
    order = _demo_order(now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc))

    from bot.execution.order_reconciler import DemoFill

    fill = DemoFill(
        fill_id="F1",
        order_id=order.exchange_order_id,
        ticker=order.ticker,
        outcome_side="yes",
        book_side="yes",
        count=5,
        yes_price_dollars=Decimal("0.20"),
        no_price_dollars=Decimal("0.80"),
        is_taker=True,
        created_time="",
        fee_cost=Decimal("0.01"),
    )

    order_calls: list[int] = []

    async def fake_poll_orders(client, watermark):
        order_calls.append(1)
        return [order]

    async def fake_poll_fills(client, watermark):
        return [fill]

    sequence: list[str] = []

    async def fake_run(app, duration):
        sequence.append("run")

    monkeypatch.setattr(bot_main, "poll_open_orders", fake_poll_orders)
    monkeypatch.setattr(bot_main, "poll_fills", fake_poll_fills)
    monkeypatch.setattr(bot_main, "run", fake_run)

    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    settings = _DemoSettings(
        mode="demo", kalshi_demo_key_id="demo-key-id", kalshi_demo_private_key_path=_rsa_pem
    )

    class _BalanceKalshi(_DemoKalshi):
        async def aopen(self) -> None:
            sequence.append("aopen")

        async def get_balance(self) -> Decimal:
            sequence.append("balance")
            return Decimal("500")

    kalshi = _BalanceKalshi(markets=[], orderbooks={})
    app = App(
        settings=settings,
        engine=engine,
        session_factory=sf,
        meteo=_StubMeteo({}),  # type: ignore[arg-type]
        kalshi=kalshi,
        acis=_StubACIS(None),  # type: ignore[arg-type]
        series_list=("KXHIGHDEN",),
    )

    async def _go() -> None:
        await app.kalshi.aopen()
        if app.settings.mode == "demo":
            await bot_main._demo_startup_backfill(app)
        await bot_main.run(app, timedelta(seconds=1))

    caplog.set_level(logging.INFO, logger="bot.main")
    await _go()

    assert order_calls
    assert sequence.index("balance") < sequence.index("run")
    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
    assert len(demo_orders) == 1
    assert any("demo_startup_backfill" in r.getMessage() for r in caplog.records)
    engine.dispose()


def test_cli_accepts_demo_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["bot.main", "--mode=demo", "--series=KXHIGHDEN", "--duration=1s"],
    )
    captured: dict[str, object] = {}

    def fake_run(coro):
        coro.close()
        captured["ran"] = True

    monkeypatch.setattr(
        bot_main, "get_settings", lambda: _DemoSettings(mode="demo", kalshi_demo_key_id="k")
    )
    monkeypatch.setattr(bot_main.asyncio, "run", fake_run)
    monkeypatch.setattr(bot_main, "make_engine", lambda p: make_engine(":memory:"))
    main()
    assert captured.get("ran") is True


def test_cli_mode_demo_overrides_env_paper(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["bot.main", "--mode=demo", "--series=KXHIGHDEN", "--duration=1s"],
    )
    seen: dict[str, object] = {}

    def fake_run(coro):
        coro.close()

    monkeypatch.setattr(
        bot_main, "get_settings", lambda: _DemoSettings(mode="paper", kalshi_demo_key_id="k")
    )
    monkeypatch.setattr(bot_main.asyncio, "run", fake_run)
    monkeypatch.setattr(bot_main, "make_engine", lambda p: make_engine(":memory:"))

    real_app = bot_main.App

    def app_spy(*args, **kwargs):
        seen["mode"] = kwargs["settings"].mode
        return real_app(*args, **kwargs)

    monkeypatch.setattr(bot_main, "App", app_spy)
    main()
    assert seen["mode"] == "demo"


def test_cli_mode_demo_without_key_raises_validation_error(monkeypatch) -> None:
    from pydantic import ValidationError

    monkeypatch.setattr(
        "sys.argv",
        ["bot.main", "--mode=demo", "--series=KXHIGHDEN", "--duration=1s"],
    )
    monkeypatch.setattr(
        bot_main, "get_settings", lambda: _DemoSettings(mode="paper", kalshi_demo_key_id=None)
    )
    constructed: list[int] = []
    real_app = bot_main.App

    def app_spy(*args, **kwargs):
        constructed.append(1)
        return real_app(*args, **kwargs)

    monkeypatch.setattr(bot_main, "App", app_spy)
    monkeypatch.setattr(bot_main, "make_engine", lambda p: make_engine(":memory:"))
    with pytest.raises(ValidationError):
        main()
    assert constructed == []


async def test_demo_integration_one_cycle_via_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    _rsa_pem: Path,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)

    cid = "kw-edge-yes-KXHIGHDEN-26MAY08-T70-75-2026-05-08"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/orders") and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "order": {
                        "client_order_id": cid,
                        "order_id": "EX-INT-1",
                        "ticker": market.ticker,
                        "side": "yes",
                        "status": "executed",
                        "initial_count_fp": "5.00",
                        "fill_count_fp": "5.00",
                        "remaining_count_fp": "0.00",
                        "yes_price_dollars": "0.2000",
                        "taker_fees_dollars": "0.010000",
                        "maker_fees_dollars": "0.000000",
                        "taker_fill_cost_dollars": "1.000000",
                        "maker_fill_cost_dollars": "0.000000",
                    }
                },
            )
        return httpx.Response(200, json={"orders": [], "fills": []})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    ) as http:
        settings = _DemoSettings(
            mode="demo", kalshi_demo_key_id="demo-key-id", kalshi_demo_private_key_path=_rsa_pem
        )
        client = KalshiDemoClient(settings, http_client=http)
        await client.aopen()

        engine = make_engine(":memory:")
        Base.metadata.create_all(engine)
        sf = make_session_factory(engine)
        app = App(
            settings=settings,
            engine=engine,
            session_factory=sf,
            meteo=meteo,  # type: ignore[arg-type]
            kalshi=client,
            acis=_StubACIS(None),  # type: ignore[arg-type]
            series_list=("KXHIGHDEN",),
        )
        await refresh_forecasts(app)
        app.latest_markets[market.ticker] = market
        app.latest_orderbooks[market.ticker] = book
        app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
        _lift_caps(monkeypatch)
        monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

        await evaluate_strategies(app, now)
        await client.aclose()

    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(demo_orders) == 1
    assert demo_orders[0].status == "executed"
    assert len(trades) == 1
    assert trades[0].demo_order_client_id == demo_orders[0].client_order_id
    engine.dispose()


async def test_demo_integration_paper_mode_regression_demo_orders_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _demo_bracket_setup(now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)
    monkeypatch.setattr(bot_main, "_build_intents", _one_intent_stub())

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        demo_orders = session.scalars(select(DemoOrderRow)).all()
        trades = session.scalars(select(PaperTradeRow)).all()
    assert demo_orders == []
    assert len(trades) == 1


async def test_intent_with_thin_edge_on_low_priced_contract_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T60-65",
        "0.07",
        "0.05",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.07", "0.05", now=now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    def stub_intent(
        *,
        app,
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        mode,
        now,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=1,
                fair_yes=fair_yes,
                q_raw=fair_yes,
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
        failures = session.scalars(
            select(GateFailure).where(GateFailure.gate_name == "edge_after_friction")
        ).all()
    assert trades == []
    assert failures


async def test_intent_with_thick_edge_still_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.07",
        "0.05",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.07", "0.05", now=now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    def stub_intent(
        *,
        app,
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        mode,
        now,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=1,
                fair_yes=fair_yes,
                q_raw=fair_yes,
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
    assert len(trades) == 1


async def test_paper_mode_edge_after_friction_failure_blocks_papertrade_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T60-65",
        "0.07",
        "0.05",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.07", "0.05", now=now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    def stub_intent(
        *,
        app,
        ticker,
        market,
        book,
        fair_yes,
        spread,
        is_same_day,
        is_blacklisted,
        is_tail,
        mode,
        now,
        **_kwargs,
    ):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=1,
                fair_yes=fair_yes,
                q_raw=fair_yes,
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub_intent)

    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        trades = session.scalars(select(PaperTradeRow)).all()
        failures = session.scalars(
            select(GateFailure).where(GateFailure.gate_name == "edge_after_friction")
        ).all()
    assert trades == []
    assert failures


def test_gate_ctx_for_buy_yes_uses_side_specific_edge() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.40",
        "0.30",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.40", "0.30")
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    ctx = _gate_ctx_for(
        intent=intent,
        market=market,
        fair_yes=Decimal("0.50"),
        spread=Decimal("3.0"),
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        book=book,
        market_existing_dollars=Decimal("0"),
        event_existing_dollars=Decimal("0"),
        series_existing_dollars=Decimal("0"),
        aggregate_existing_dollars=Decimal("0"),
        buy_yes_depth=book.no_bid_depth,
        sell_yes_depth=book.yes_bid_depth,
    )
    assert ctx.edge == Decimal("0.10")
    mid = Decimal("0.35")
    assert ctx.edge != Decimal("0.50") - mid


def test_gate_ctx_for_sell_yes_uses_side_specific_edge() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.40",
        "0.30",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.40", "0.30")
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.SELL_YES,
        contracts=10,
        fair_yes=Decimal("0.20"),
        q_raw=Decimal("0.20"),
        strategy="edge",
    )
    ctx = _gate_ctx_for(
        intent=intent,
        market=market,
        fair_yes=Decimal("0.20"),
        spread=Decimal("3.0"),
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        book=book,
        market_existing_dollars=Decimal("0"),
        event_existing_dollars=Decimal("0"),
        series_existing_dollars=Decimal("0"),
        aggregate_existing_dollars=Decimal("0"),
        buy_yes_depth=book.no_bid_depth,
        sell_yes_depth=book.yes_bid_depth,
    )
    assert ctx.edge == book.yes_bid - Decimal("0.20")
    assert ctx.edge == Decimal("0.10")
    mid = Decimal("0.35")
    assert ctx.edge != mid - Decimal("0.20")


def test_gate_ctx_for_sources_edge_from_book_not_market() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.40",
        "0.30",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.41", "0.30")
    assert market.yes_ask == Decimal("0.40")
    assert book.yes_ask == Decimal("0.41")
    intent = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    ctx = _gate_ctx_for(
        intent=intent,
        market=market,
        fair_yes=Decimal("0.50"),
        spread=Decimal("3.0"),
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        book=book,
        market_existing_dollars=Decimal("0"),
        event_existing_dollars=Decimal("0"),
        series_existing_dollars=Decimal("0"),
        aggregate_existing_dollars=Decimal("0"),
        buy_yes_depth=book.no_bid_depth,
        sell_yes_depth=book.yes_bid_depth,
    )
    assert ctx.edge == Decimal("0.09")
    assert ctx.edge != Decimal("0.50") - market.yes_ask


def test_gate_ctx_for_passes_orderbook_depth_through() -> None:
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.40",
        "0.30",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(
        market.ticker,
        "0.40",
        "0.30",
        yes_bid_depth=11,
        no_bid_depth=7,
    )
    buy = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
    )
    sell = TradeIntent(
        market_ticker=market.ticker,
        side=TradeSide.SELL_YES,
        contracts=10,
        fair_yes=Decimal("0.20"),
        q_raw=Decimal("0.20"),
        strategy="edge",
    )
    base = dict(
        market=market,
        spread=Decimal("3.0"),
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        book=book,
        market_existing_dollars=Decimal("0"),
        event_existing_dollars=Decimal("0"),
        series_existing_dollars=Decimal("0"),
        aggregate_existing_dollars=Decimal("0"),
        buy_yes_depth=book.no_bid_depth,
        sell_yes_depth=book.yes_bid_depth,
    )
    ctx_buy = _gate_ctx_for(intent=buy, fair_yes=Decimal("0.50"), **base)
    ctx_sell = _gate_ctx_for(intent=sell, fair_yes=Decimal("0.20"), **base)
    assert ctx_buy.depth_at_price == 7
    assert ctx_sell.depth_at_price == 11


def test_no_strategy_caller_uses_hardcoded_contracts() -> None:
    edge_src = Path("bot/strategy/edge.py").read_text()
    tails_src = Path("bot/strategy/tails.py").read_text()
    assert "compute_stake_contracts" in edge_src
    assert "compute_stake_contracts" in tails_src


async def test_build_intents_passes_event_budget_to_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18", now=now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    seen: list[Decimal] = []
    real = edge_strategy.evaluate

    def rec(ctx, **kwargs):
        seen.append(ctx.event_budget_remaining)
        return real(ctx, **kwargs)

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", rec)
    await evaluate_strategies(app, now)
    assert seen
    assert all(v >= Decimal("0") for v in seen)


async def test_build_intents_uses_lead_time_sigma_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    near_market = _market_from(
        "KXHIGHDEN-26MAY07-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 7, 18, 0, tzinfo=timezone.utc),
    )
    far_market = _market_from(
        "KXHIGHDEN-26MAY09-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )

    seen: list[tuple[str, Decimal]] = []
    real = edge_strategy.evaluate

    def rec(ctx, **kwargs):
        seen.append(("ctx", ctx.sigma_T_median))
        return real(ctx, **kwargs)

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", rec)

    book = _book_from(near_market.ticker, "0.20", "0.18", now=now)
    _build_intents(
        app=_make_app(),
        ticker=near_market.ticker,
        market=near_market,
        book=book,
        fair_yes=Decimal("0.80"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=False,
        mode="paper",
        now=now,
        sigma_T_median=sigma_t_median_for_lead(
            int((near_market.close_time - now).total_seconds() / 3600)
        ),
        event_budget_remaining=Decimal("9999"),
        market_budget_remaining=Decimal("9999"),
        buy_yes_depth=10_000,
        sell_yes_depth=10_000,
    )
    book2 = _book_from(far_market.ticker, "0.20", "0.18", now=now)
    _build_intents(
        app=_make_app(),
        ticker=far_market.ticker,
        market=far_market,
        book=book2,
        fair_yes=Decimal("0.80"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=False,
        mode="paper",
        now=now,
        sigma_T_median=sigma_t_median_for_lead(
            int((far_market.close_time - now).total_seconds() / 3600)
        ),
        event_budget_remaining=Decimal("9999"),
        market_budget_remaining=Decimal("9999"),
        buy_yes_depth=10_000,
        sell_yes_depth=10_000,
    )
    assert len(seen) == 2
    near_sigma = seen[0][1]
    far_sigma = seen[1][1]
    assert far_sigma > near_sigma


@pytest.mark.parametrize(
    "fair_yes, yes_ask, yes_bid, expected_depth, expected_action",
    [
        (
            Decimal("0.45"),
            "0.20",
            "0.18",
            11,
            edge_strategy.EdgeAction.BUY_YES,
        ),
        (
            Decimal("0.05"),
            "0.85",
            "0.80",
            7,
            edge_strategy.EdgeAction.SELL_YES,
        ),
    ],
)
async def test_gate_ctx_and_build_intents_share_depth_ints(
    monkeypatch: pytest.MonkeyPatch,
    fair_yes: Decimal,
    yes_ask: str,
    yes_bid: str,
    expected_depth: int,
    expected_action: edge_strategy.EdgeAction,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        yes_ask,
        yes_bid,
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(
        market.ticker,
        yes_ask,
        yes_bid,
        now=now,
        yes_bid_depth=7,
        no_bid_depth=11,
    )
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    app.forecast_cdfs[("KDEN", date(2026, 5, 8))] = _FixedFairCdf(fair_yes)
    _lift_caps(monkeypatch)

    edge_depths: list[int] = []
    edge_actions: list[edge_strategy.EdgeAction] = []
    real_edge = edge_strategy.evaluate

    def rec_edge(ctx, **kwargs):
        edge_depths.append(ctx.depth_at_price)
        sig = real_edge(ctx, **kwargs)
        edge_actions.append(sig.action)
        return sig

    gate_depths: list[int] = []
    real_gate = bot_main._gate_ctx_for

    def rec_gate(**kwargs):
        ctx = real_gate(**kwargs)
        gate_depths.append(ctx.depth_at_price)
        return ctx

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", rec_edge)
    monkeypatch.setattr(bot_main, "_gate_ctx_for", rec_gate)
    await evaluate_strategies(app, now)

    assert edge_actions == [expected_action]
    assert edge_depths == [expected_depth]
    assert gate_depths == [expected_depth]


async def test_event_budget_recomputed_per_iteration_not_per_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    m_a = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    m_b = _market_from(
        "KXHIGHDEN-26MAY08-T75-80",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book_a = _book_from(m_a.ticker, "0.20", "0.18", now=now)
    book_b = _book_from(m_b.ticker, "0.20", "0.18", now=now)
    kalshi = _StubKalshi(markets=[m_a, m_b], orderbooks={m_a.ticker: book_a, m_b.ticker: book_b})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    contracts = 5

    def stub(*, ticker, market, book, fair_yes, **_kwargs):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=contracts,
                fair_yes=fair_yes,
                q_raw=fair_yes,
                strategy="edge",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub)

    seen_event_budget: list[Decimal] = []
    real_gate = bot_main._gate_ctx_for

    def rec_gate(**kwargs):
        ctx = real_gate(**kwargs)
        seen_event_budget.append(ctx.event_position_cap - kwargs["event_existing_dollars"])
        return ctx

    monkeypatch.setattr(bot_main, "_gate_ctx_for", rec_gate)
    await evaluate_strategies(app, now)

    debit = book_a.yes_ask * Decimal(contracts)
    assert len(seen_event_budget) == 2
    assert seen_event_budget[1] == seen_event_budget[0] - debit
    assert seen_event_budget[1] < seen_event_budget[0]


async def test_build_intents_threads_mode_and_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.30",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.30", "0.18", now=now)

    seen_paper: list[Decimal] = []
    real = edge_strategy.evaluate

    def rec(ctx, **kwargs):
        seen_paper.append(ctx.price_per_contract)
        return real(ctx, **kwargs)

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", rec)
    _build_intents(
        app=_make_app(),
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.05"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=False,
        mode="paper",
        now=now,
        sigma_T_median=Decimal("2.0"),
        event_budget_remaining=Decimal("9999"),
        market_budget_remaining=Decimal("9999"),
        buy_yes_depth=10_000,
        sell_yes_depth=10_000,
    )
    assert seen_paper == [Decimal("1") - book.yes_bid]

    seen_demo: list[Decimal] = []

    def rec_demo(ctx, **kwargs):
        seen_demo.append(ctx.price_per_contract)
        return real(ctx, **kwargs)

    monkeypatch.setattr(bot_main.edge_strategy, "evaluate", rec_demo)
    _build_intents(
        app=_make_app(),
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.05"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=False,
        mode="demo",
        now=now,
        sigma_T_median=Decimal("2.0"),
        event_budget_remaining=Decimal("9999"),
        market_budget_remaining=Decimal("9999"),
        buy_yes_depth=10_000,
        sell_yes_depth=10_000,
    )
    assert seen_demo == [book.no_ask]


async def test_demo_path_no_longer_passes_contracts_cap_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    meteo, market, book = _wide_tails_market(now)
    kalshi = _DemoKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi, bankroll=Decimal("500"))

    async def fake_place(intent, book, client, *, now):
        return _demo_order(now=now, ticker=intent.market_ticker, side_kalshi="no")

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)
    captured = await _capture_tails_call(
        app, meteo, market, book, now, monkeypatch, lift_market=False
    )
    assert captured
    for kwargs in captured:
        assert "contracts_cap" not in kwargs
        assert "position_cap" not in kwargs


def _two_market_event_books(
    now: datetime,
) -> tuple[KalshiMarket, KalshiMarket, KalshiOrderbook, KalshiOrderbook]:
    m_a = _market_from(
        "KXHIGHDEN-26MAY08-T70-75",
        "0.85",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    m_b = _market_from(
        "KXHIGHDEN-26MAY08-T75-80",
        "0.85",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )

    def book(ticker: str) -> KalshiOrderbook:
        return KalshiOrderbook(
            ticker=ticker,
            yes_ask=Decimal("0.85"),
            yes_bid=Decimal("0.18"),
            no_ask=Decimal("0.85"),
            no_bid=Decimal("0.15"),
            yes_ask_depth=10_000,
            yes_bid_depth=10_000,
            no_ask_depth=10_000,
            no_bid_depth=10_000,
            snapshot_at=now - timedelta(seconds=1),
        )

    return m_a, m_b, book(m_a.ticker), book(m_b.ticker)


async def test_overlay_debit_uses_paper_basis_in_paper_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(7).normal(85.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    m_a, m_b, book_a, book_b = _two_market_event_books(now)
    kalshi = _StubKalshi(markets=[m_a, m_b], orderbooks={m_a.ticker: book_a, m_b.ticker: book_b})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    contracts = 5

    def stub(*, ticker, market, book, fair_yes, **_kwargs):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.SELL_YES,
                contracts=contracts,
                fair_yes=fair_yes,
                q_raw=fair_yes,
                strategy="tails",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub)

    event_existing_seen: list[Decimal] = []
    real_gate = bot_main._gate_ctx_for

    def rec_gate(**kwargs):
        event_existing_seen.append(kwargs["event_existing_dollars"])
        return real_gate(**kwargs)

    monkeypatch.setattr(bot_main, "_gate_ctx_for", rec_gate)

    await evaluate_strategies(app, now)

    paper_basis = Decimal("1") - book_a.yes_bid
    demo_basis = book_a.no_ask
    assert paper_basis != demo_basis
    assert len(event_existing_seen) == 2
    assert event_existing_seen[0] == Decimal("0")
    assert event_existing_seen[1] == paper_basis * Decimal(contracts)
    assert event_existing_seen[1] != demo_basis * Decimal(contracts)


async def test_overlay_debit_uses_no_ask_in_demo_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(7).normal(85.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    m_a, m_b, book_a, book_b = _two_market_event_books(now)
    kalshi = _DemoKalshi(markets=[m_a, m_b], orderbooks={m_a.ticker: book_a, m_b.ticker: book_b})
    app = _make_demo_app(meteo=meteo, kalshi=kalshi, bankroll=Decimal("500"))
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    async def fake_place(intent, book, client, *, now):
        return _demo_order(now=now, ticker=intent.market_ticker, side_kalshi="no")

    monkeypatch.setattr(bot_main, "place_order_demo", fake_place)

    contracts = 5

    def stub(*, ticker, market, book, fair_yes, **_kwargs):
        return [
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.SELL_YES,
                contracts=contracts,
                fair_yes=fair_yes,
                q_raw=fair_yes,
                strategy="tails",
            )
        ]

    monkeypatch.setattr(bot_main, "_build_intents", stub)

    event_existing_seen: list[Decimal] = []
    real_gate = bot_main._gate_ctx_for

    def rec_gate(**kwargs):
        event_existing_seen.append(kwargs["event_existing_dollars"])
        return real_gate(**kwargs)

    monkeypatch.setattr(bot_main, "_gate_ctx_for", rec_gate)

    await evaluate_strategies(app, now)

    paper_basis = Decimal("1") - book_a.yes_bid
    demo_basis = book_a.no_ask
    assert paper_basis != demo_basis
    assert len(event_existing_seen) == 2
    assert event_existing_seen[0] == Decimal("0")
    assert event_existing_seen[1] == demo_basis * Decimal(contracts)
    assert event_existing_seen[1] != paper_basis * Decimal(contracts)


async def test_market_budget_first_iteration_lands_at_seven_fifty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(7).normal(85.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T100",
        "0.20",
        "0.07",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.07", now=now)
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    seen: list[Decimal] = []
    real = tails_strategy.evaluate

    def rec(ctx, **kwargs):
        seen.append(ctx.market_budget_remaining)
        return real(ctx, **kwargs)

    monkeypatch.setattr(bot_main.tails_strategy, "evaluate", rec)
    app.forecast_cdfs[("KDEN", date(2026, 5, 8))] = _FixedFairCdf(Decimal("0.995"))
    app.ensemble_spreads[("KDEN", date(2026, 5, 8))] = Decimal("3.0")
    app.forecast_run_times[("KDEN", date(2026, 5, 8))] = now
    monkeypatch.setattr(bot_main, "event_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "series_position_cap", lambda app=None: Decimal("10000"))
    monkeypatch.setattr(bot_main, "aggregate_exposure_cap", lambda app=None: Decimal("100000"))

    await evaluate_strategies(app, now)

    assert seen
    assert seen[0] == Decimal("500") * bot_main.MARKET_POSITION_FRAC


async def test_evaluate_strategies_skips_market_with_none_close_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=now - timedelta(hours=1),
        daily_highs={date(2026, 5, 8): np.random.default_rng(1).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    a_ticker = "KXHIGHDEN-26MAY08-T60-65"
    b_ticker = "KXHIGHDEN-26MAY08-T70-75"
    market_a = KalshiMarket(
        ticker=a_ticker,
        event_ticker="KXHIGHDEN-26MAY08",
        series="KXHIGHDEN",
        status="active",
        close_time=None,
        yes_ask=Decimal("0.20"),
        yes_bid=Decimal("0.18"),
    )
    market_b = _market_from(
        b_ticker,
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book_a = _book_from(a_ticker, "0.20", "0.18", now=now)
    book_b = _book_from(b_ticker, "0.20", "0.18", now=now)
    kalshi = _StubKalshi(
        markets=[market_a, market_b],
        orderbooks={a_ticker: book_a, b_ticker: book_b},
    )
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    app.latest_markets[a_ticker] = market_a
    app.latest_markets[b_ticker] = market_b
    app.latest_orderbooks[a_ticker] = book_a
    app.latest_orderbooks[b_ticker] = book_b
    _lift_caps(monkeypatch)

    seen_tickers: list[str] = []
    real = bot_main._build_intents

    def rec(*, ticker, **kwargs):
        seen_tickers.append(ticker)
        return real(ticker=ticker, **kwargs)

    monkeypatch.setattr(bot_main, "_build_intents", rec)
    await evaluate_strategies(app, now)
    assert a_ticker not in seen_tickers
    assert b_ticker in seen_tickers


async def test_build_intents_skips_blacklisted_tail_series() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHLAX-26MAY08-T100", "0.20", "0.18", close_at)
    book = _book_from(market.ticker, "0.20", "0.18")
    intents = _build_intents(
        app=_make_app(),
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.05"),
        spread=Decimal("3.0"),
        is_same_day=False,
        is_blacklisted=True,
        is_tail=True,
        mode="paper",
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        **_intents_defaults(),
    )
    assert all(i.strategy != "tails" for i in intents)


def test_calibration_cold_start_log_emits_at_init(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="bot.main")
    _make_app()
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "calibration_maps_loaded n_buckets_fit=0 n_buckets_skipped=24 reason=cold_start" in m
        for m in messages
    )


def test_gate_failure_reason_helper_formats_tokens() -> None:
    from bot.validation.calibration import format_gate_failure_reason

    out = format_gate_failure_reason(
        "fair_value_sane fair_yes=0.005 outside [0.01, 0.99]",
        q_raw=Decimal("0.005"),
        fair_yes=Decimal("0.005"),
        bucket_key=("tails", 0, 1),
    )
    assert "q_raw=" in out
    assert "q_corrected=" in out
    assert "bucket=" in out
    assert len(out) < 512


async def test_gate_failure_reason_includes_raw_corrected_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(0).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)
    market = _market_from(
        "KXHIGHDEN-26MAY08-T100",
        "0.20",
        "0.18",
        datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
    )
    book = _book_from(market.ticker, "0.20", "0.18")
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)
    _lift_caps(monkeypatch)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    await evaluate_strategies(app, now)

    with app.session_factory() as session:
        failures = session.scalars(select(GateFailure)).all()
    assert failures, "expected at least one gate failure for this TAILS market"
    for f in failures:
        assert "q_raw=" in f.reason
        assert "q_corrected=" in f.reason
        assert "bucket=" in f.reason


def test_seconds_until_next_refit_jitters_within_bound() -> None:
    from bot.main import (
        CALIBRATION_REFIT_JITTER_SECONDS,
        CALIBRATION_REFIT_TARGET_HOUR_UTC,
        _seconds_until_next_refit,
    )

    now = datetime(2026, 5, 6, 3, 0, tzinfo=timezone.utc)
    base = (
        now.replace(hour=CALIBRATION_REFIT_TARGET_HOUR_UTC, minute=0, second=0, microsecond=0) - now
    ).total_seconds()
    seen = {round(_seconds_until_next_refit(now), 3) for _ in range(50)}
    assert len(seen) >= 5
    for v in seen:
        assert abs(v - base) <= CALIBRATION_REFIT_JITTER_SECONDS + 1


def test_seconds_until_next_refit_handles_after_04_utc() -> None:
    from bot.main import (
        CALIBRATION_REFIT_JITTER_SECONDS,
        _seconds_until_next_refit,
    )

    now = datetime(2026, 5, 6, 4, 30, tzinfo=timezone.utc)
    base = (24 - 0.5) * 3600
    for _ in range(20):
        v = _seconds_until_next_refit(now)
        assert abs(v - base) <= CALIBRATION_REFIT_JITTER_SECONDS + 1


async def test_calibration_refit_loop_continues_after_refit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bot.main import _calibration_refit_loop

    app = _make_app()
    call_count = {"n": 0}

    def fake_refit(session, prev_maps=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("synthetic refit failure")
        return prev_maps or bot_main._empty_calibration_maps()

    monkeypatch.setattr(bot_main, "refit_all", fake_refit)
    monkeypatch.setattr(bot_main, "_seconds_until_next_refit", lambda now: 0.001)

    stop = asyncio.Event()

    async def stop_after_two_iters() -> None:
        while call_count["n"] < 2:
            await asyncio.sleep(0.005)
        stop.set()

    await asyncio.gather(
        _calibration_refit_loop(app, stop),
        stop_after_two_iters(),
    )
    assert call_count["n"] >= 2


async def test_calibration_refit_loop_logs_bss_aggregate_tokens(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from bot.main import _calibration_refit_loop
    from bot.validation.calibration import BSS_AGGREGATE_NA, CalibrationMaps

    app = _make_app()
    fake_maps = CalibrationMaps(
        maps={},
        fitted_at=datetime(2026, 5, 6, 4, 0, tzinfo=timezone.utc),
        n_samples_per_bucket={},
        holdout_bs_new={},
        holdout_bs_prev={},
        holdout_n_per_bucket={},
        climatological_rate_per_bucket={},
        bss_aggregate_per_stratum={"tails": Decimal("0.123456"), "edge": Decimal("0.234567")},
    )
    call_count = {"n": 0}

    def fake_refit(session, prev_maps=None):
        call_count["n"] += 1
        return fake_maps

    monkeypatch.setattr(bot_main, "refit_all", fake_refit)
    monkeypatch.setattr(bot_main, "_seconds_until_next_refit", lambda now: 0.001)

    stop = asyncio.Event()

    async def stop_after_one_iter() -> None:
        while call_count["n"] < 1:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.01)
        stop.set()

    with caplog.at_level(logging.INFO, logger="bot.main"):
        await asyncio.gather(
            _calibration_refit_loop(app, stop),
            stop_after_one_iter(),
        )
    complete_lines = [
        rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith("calibration_refit_complete ")
    ]
    assert complete_lines, "expected a calibration_refit_complete log line"
    line = complete_lines[0]
    assert "tails_bss_aggregate=0.123456" in line
    assert "edge_bss_aggregate=0.234567" in line
    assert "tails_holdout_bs_new=" in line
    assert "edge_holdout_bs_new=" in line
    assert BSS_AGGREGATE_NA not in line.split("tails_bss_aggregate=")[1].split(" ")[0]


def _balance_json(balance_dollars: str = "788.3901", balance: int = 78839) -> dict:
    return {
        "balance": balance,
        "balance_breakdown": [{"balance": balance_dollars, "exchange_index": 0}],
        "balance_dollars": balance_dollars,
        "portfolio_value": 13434,
        "updated_ts": 1780428732,
    }


def _position_dict(
    *,
    ticker: str,
    market_exposure: str = "0.830000",
    realized_pnl: str = "0.000000",
    fees_paid: str = "0.000000",
    position_fp: str = "-1.00",
    drop: tuple[str, ...] = (),
    nulls: tuple[str, ...] = (),
) -> dict:
    base = {
        "fees_paid_dollars": fees_paid,
        "last_updated_ts": "2026-06-02T11:00:00Z",
        "market_exposure_dollars": market_exposure,
        "position_fp": position_fp,
        "realized_pnl_dollars": realized_pnl,
        "resting_orders_count": 0,
        "ticker": ticker,
        "total_traded_dollars": market_exposure,
    }
    for k in drop:
        base.pop(k, None)
    for k in nulls:
        base[k] = None
    return base


def _make_snapshot_demo_app(_rsa_pem: Path, kalshi) -> App:
    from bot.config import Settings as _Settings

    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    settings = _Settings(
        mode="demo",
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=_rsa_pem,
    )
    return App(
        settings=settings,
        engine=engine,
        session_factory=sf,
        meteo=None,  # type: ignore[arg-type]
        kalshi=kalshi,
        acis=None,  # type: ignore[arg-type]
        series_list=("KXHIGHDEN",),
    )


async def _demo_client(_rsa_pem: Path, handler) -> KalshiDemoClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport, base_url="https://demo-api.kalshi.co/trade-api/v2"
    )
    from bot.config import Settings as _Settings

    settings = _Settings(
        mode="demo",
        kalshi_demo_key_id="demo-key-id",
        kalshi_demo_private_key_path=_rsa_pem,
    )
    client = KalshiDemoClient(settings, http_client=http)
    await client.aopen()
    return client


def test_paper_mode_skips_portfolio_snapshot_loop_task() -> None:
    src = Path(bot_main.__file__).read_text()
    tree = ast.parse(src)

    enclosing_func: list[str] = []

    class _RunVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found_in_demo_branch = False
            self.found_unconditionally = False

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node.name == "run":
                enclosing_func.append(node.name)
                self.generic_visit(node)
                enclosing_func.pop()
                return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if not enclosing_func:
                self.generic_visit(node)
                return
            func = node.func
            is_create_task = (
                isinstance(func, ast.Attribute)
                and func.attr == "create_task"
                and isinstance(func.value, ast.Name)
                and func.value.id == "asyncio"
            )
            if is_create_task and node.args:
                arg = node.args[0]
                if (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Name)
                    and arg.func.id == "_portfolio_snapshot_loop"
                ):
                    parent = _find_parent_if(node, tree)
                    if parent is not None and _if_tests_demo_mode(parent):
                        self.found_in_demo_branch = True
                    else:
                        self.found_unconditionally = True
            self.generic_visit(node)

    def _find_parent_if(target: ast.AST, root: ast.AST) -> ast.If | None:
        for node in ast.walk(root):
            if isinstance(node, ast.If):
                for sub in ast.walk(node):
                    if sub is target:
                        return node
        return None

    def _if_tests_demo_mode(node: ast.If) -> bool:
        test = node.test
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and isinstance(test.left.value, ast.Attribute)
            and test.left.value.attr == "settings"
            and test.left.attr == "mode"
            and any(isinstance(c, ast.Constant) and c.value == "demo" for c in test.comparators)
        )

    v = _RunVisitor()
    v.visit(tree)
    assert v.found_in_demo_branch is True
    assert v.found_unconditionally is False


def test_app_bankroll_assignment_site_is_demo_startup_backfill_only() -> None:
    src = Path(bot_main.__file__).read_text()
    tree = ast.parse(src)
    enclosing_func_names: list[str] = []

    sites: list[str] = []

    class _V(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            enclosing_func_names.append(node.name)
            self.generic_visit(node)
            enclosing_func_names.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            enclosing_func_names.append(node.name)
            self.generic_visit(node)
            enclosing_func_names.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and tgt.attr == "bankroll"
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "app"
                ):
                    sites.append(enclosing_func_names[-1])
            self.generic_visit(node)

    _V().visit(tree)
    assert set(sites) == {"_demo_startup_backfill"}
    assert len(sites) == 1


@pytest.mark.parametrize(
    "missing_field",
    ["market_exposure_dollars", "realized_pnl_dollars", "fees_paid_dollars"],
)
async def test_aggregator_raises_key_error_on_missing_required_field(
    _rsa_pem: Path, missing_field: str
) -> None:
    from bot.execution.portfolio_snapshot import aggregate_positions

    payload = {
        "market_positions": [
            _position_dict(ticker="KXHIGHDEN-26JUN01-T70", drop=(missing_field,)),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = await _demo_client(_rsa_pem, handler)
    app = _make_snapshot_demo_app(_rsa_pem, client)
    try:
        with pytest.raises(KeyError):
            await aggregate_positions(client)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        from bot.storage.sqlite import PortfolioSnapshot as _PS

        assert session.scalars(select(_PS)).all() == []


async def test_aggregator_raises_key_error_on_null_realized_pnl(_rsa_pem: Path) -> None:
    from bot.execution.portfolio_snapshot import aggregate_positions

    payload = {
        "market_positions": [
            _position_dict(ticker="KXHIGHDEN-26JUN01-T70", nulls=("realized_pnl_dollars",)),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = await _demo_client(_rsa_pem, handler)
    try:
        with pytest.raises(KeyError):
            await aggregate_positions(client)
    finally:
        await client.aclose()


def _seed_demo_order_row(
    session,
    *,
    cid: str,
    ticker: str = "KXHIGHDEN-26JUN01-T70",
    status: str = "executed",
    filled: int = 10,
    placed_at: datetime | None = None,
    realized: Decimal | None = None,
):
    from bot.storage.sqlite import DemoOrder as _DemoOrderRow

    placed = (
        placed_at if placed_at is not None else datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    )
    row = _DemoOrderRow(
        client_order_id=cid,
        exchange_order_id="EX-" + cid,
        market_ticker=ticker,
        strategy=None if cid.startswith("kw-backfill-") else "edge",
        side="no",
        requested_contracts=filled if filled else 10,
        filled_contracts=filled,
        requested_yes_price_dollars=Decimal("0.58"),
        fair_at_entry=Decimal("0.62"),
        intended_at=placed,
        avg_fill_price=Decimal("0.205"),
        fee_dollars=Decimal("0.07"),
        realized_pnl_dollars=realized,
        status=status,
        placed_at=placed,
        last_status_at=placed,
    )
    session.add(row)
    session.commit()
    return row


async def test_zero_row_update_emits_snapshot_pnl_unattributed_for_backfill_only(
    _rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from bot.execution.portfolio_snapshot import update_demo_realized_pnl

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = await _demo_client(_rsa_pem, handler)
    app = _make_snapshot_demo_app(_rsa_pem, client)
    snapshot_at = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    with app.session_factory() as session:
        _seed_demo_order_row(session, cid="kw-backfill-EX1")
        caplog.set_level(logging.INFO, logger="bot.execution.portfolio_snapshot")
        rc = update_demo_realized_pnl(
            session, "KXHIGHDEN-26JUN01-T70", Decimal("1.530000"), snapshot_at
        )
        session.commit()
    await client.aclose()

    assert rc == 0
    matching = [r for r in caplog.records if "snapshot_pnl_unattributed" in r.getMessage()]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "ticker=KXHIGHDEN-26JUN01-T70" in msg
    assert "pnl=1.530000" in msg
    assert "reason=no_eligible_row" in msg


async def test_zero_row_update_emits_log_when_natural_row_placed_after_snapshot(
    _rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from bot.execution.portfolio_snapshot import update_demo_realized_pnl

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = await _demo_client(_rsa_pem, handler)
    app = _make_snapshot_demo_app(_rsa_pem, client)
    snapshot_at = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    with app.session_factory() as session:
        _seed_demo_order_row(
            session, cid="kw-edge-late", placed_at=snapshot_at + timedelta(minutes=5)
        )
        caplog.set_level(logging.INFO, logger="bot.execution.portfolio_snapshot")
        rc = update_demo_realized_pnl(
            session, "KXHIGHDEN-26JUN01-T70", Decimal("1.530000"), snapshot_at
        )
        session.commit()
    await client.aclose()
    assert rc == 0
    matching = [r for r in caplog.records if "snapshot_pnl_unattributed" in r.getMessage()]
    assert len(matching) == 1


async def test_zero_row_update_no_log_when_match_lands(
    _rsa_pem: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from bot.execution.portfolio_snapshot import update_demo_realized_pnl

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = await _demo_client(_rsa_pem, handler)
    app = _make_snapshot_demo_app(_rsa_pem, client)
    snapshot_at = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    with app.session_factory() as session:
        _seed_demo_order_row(session, cid="kw-edge-natural", placed_at=snapshot_at)
        caplog.set_level(logging.INFO, logger="bot.execution.portfolio_snapshot")
        rc = update_demo_realized_pnl(
            session, "KXHIGHDEN-26JUN01-T70", Decimal("1.530000"), snapshot_at
        )
        session.commit()
    await client.aclose()
    assert rc == 1
    matching = [r for r in caplog.records if "snapshot_pnl_unattributed" in r.getMessage()]
    assert matching == []


async def test_snapshot_loop_skips_ticks_until_auth_present(_rsa_pem: Path) -> None:
    from bot.kalshi_client import BalancePayload as _BP
    from bot.main import _portfolio_snapshot_once

    polled = {"balance": 0, "positions": 0}

    class _StubClient:
        def __init__(self) -> None:
            self._auth = None

        async def get_balance_full(self):
            polled["balance"] += 1
            return _BP.model_validate(_balance_json())

    async def fake_aggregate(client):
        polled["positions"] += 1
        from bot.execution.portfolio_snapshot import PositionAggregate

        return PositionAggregate(
            per_ticker_realized_pnl={},
            total_exposure_dollars=Decimal("0"),
            realized_pnl_dollars=Decimal("0"),
            fees_paid_dollars=Decimal("0"),
            open_positions_count=0,
        )

    stub = _StubClient()
    app = _make_snapshot_demo_app(_rsa_pem, stub)

    import bot.main as _main

    saved = _main.aggregate_positions
    _main.aggregate_positions = fake_aggregate  # type: ignore[assignment]
    try:
        await _portfolio_snapshot_once(app)
        await _portfolio_snapshot_once(app)
        assert polled == {"balance": 0, "positions": 0}
        with app.session_factory() as session:
            from bot.storage.sqlite import PortfolioSnapshot as _PS

            assert session.scalars(select(_PS)).all() == []

        stub._auth = object()
        await _portfolio_snapshot_once(app)
        assert polled["balance"] == 1
        assert polled["positions"] == 1
        with app.session_factory() as session:
            from bot.storage.sqlite import PortfolioSnapshot as _PS

            rows = session.scalars(select(_PS)).all()
        assert len(rows) == 1
    finally:
        _main.aggregate_positions = saved  # type: ignore[assignment]


async def test_per_ticker_update_binds_decimal_without_coercion(_rsa_pem: Path) -> None:
    from bot.main import _portfolio_snapshot_once
    from bot.storage.sqlite import DemoOrder as _DemoOrderRow

    snapshot_at = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    ticker = "KXHIGHDEN-26JUN01-T70"
    payload = {
        "market_positions": [
            _position_dict(
                ticker=ticker,
                realized_pnl="1.53",
                market_exposure="0.83",
                fees_paid="0.01",
                position_fp="-1.00",
            ),
        ],
        "cursor": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/portfolio/balance"):
            return httpx.Response(200, json=_balance_json())
        if request.url.path.endswith("/portfolio/positions"):
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    client = await _demo_client(_rsa_pem, handler)
    app = _make_snapshot_demo_app(_rsa_pem, client)
    with app.session_factory() as session:
        _seed_demo_order_row(
            session,
            cid="kw-edge-natural",
            ticker=ticker,
            placed_at=snapshot_at - timedelta(minutes=5),
        )

    try:
        await _portfolio_snapshot_once(app)
    finally:
        await client.aclose()

    with app.session_factory() as session:
        row = session.scalars(
            select(_DemoOrderRow).where(_DemoOrderRow.client_order_id == "kw-edge-natural")
        ).one()
    assert row.realized_pnl_dollars == Decimal("1.530000")
    assert isinstance(row.realized_pnl_dollars, Decimal)


async def test_predicate_tiebreaker_on_id_is_stable(_rsa_pem: Path) -> None:
    from bot.execution.portfolio_snapshot import update_demo_realized_pnl
    from bot.storage.sqlite import DemoOrder as _DemoOrderRow

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = await _demo_client(_rsa_pem, handler)
    app = _make_snapshot_demo_app(_rsa_pem, client)
    ticker = "KXHIGHDEN-26JUN01-T70"
    snapshot_at = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    placed = snapshot_at - timedelta(minutes=5)

    with app.session_factory() as session:
        _seed_demo_order_row(session, cid="kw-edge-a", ticker=ticker, placed_at=placed)
        _seed_demo_order_row(session, cid="kw-edge-b", ticker=ticker, placed_at=placed)
        rows = session.scalars(
            select(_DemoOrderRow).where(_DemoOrderRow.market_ticker == ticker)
        ).all()
        max_id = max(r.id for r in rows)

        for _ in range(10):
            rc = update_demo_realized_pnl(session, ticker, Decimal("1.530000"), snapshot_at)
            assert rc == 1
            session.commit()
            rows_post = session.scalars(
                select(_DemoOrderRow).where(_DemoOrderRow.market_ticker == ticker)
            ).all()
            with_pnl = [r for r in rows_post if r.realized_pnl_dollars is not None]
            assert len(with_pnl) == 1
            assert with_pnl[0].id == max_id
            assert with_pnl[0].realized_pnl_dollars == Decimal("1.530000")
            for r in rows_post:
                r.realized_pnl_dollars = None
            session.commit()
    await client.aclose()


async def test_expire_all_makes_same_session_read_see_updated_pnl(_rsa_pem: Path) -> None:
    from bot.execution.portfolio_snapshot import update_demo_realized_pnl
    from bot.storage.sqlite import DemoOrder as _DemoOrderRow

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = await _demo_client(_rsa_pem, handler)
    app = _make_snapshot_demo_app(_rsa_pem, client)
    ticker = "KXHIGHDEN-26JUN01-T70"
    snapshot_at = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)

    with app.session_factory() as session:
        _seed_demo_order_row(session, cid="kw-edge-a", ticker=ticker, placed_at=snapshot_at)
        loaded = session.scalars(
            select(_DemoOrderRow).where(_DemoOrderRow.client_order_id == "kw-edge-a")
        ).one()
        assert loaded.realized_pnl_dollars is None

        rc = update_demo_realized_pnl(session, ticker, Decimal("1.530000"), snapshot_at)
        assert rc == 1
        post = session.scalars(
            select(_DemoOrderRow).where(_DemoOrderRow.client_order_id == "kw-edge-a")
        ).one()
        assert post.realized_pnl_dollars == Decimal("1.530000")

    with app.session_factory() as session:
        _seed_demo_order_row(
            session, cid="kw-edge-b", ticker="KXHIGHCHI-26JUN01-T70", placed_at=snapshot_at
        )
        loaded = session.scalars(
            select(_DemoOrderRow).where(_DemoOrderRow.client_order_id == "kw-edge-b")
        ).one()
        assert loaded.realized_pnl_dollars is None

        from sqlalchemy import Numeric, bindparam, text as _text
        from bot.storage.sqlite import UtcDateTime as _UDT

        stmt = _text(
            """
            UPDATE demo_orders SET realized_pnl_dollars = :pnl
             WHERE id = (SELECT id FROM demo_orders
                          WHERE market_ticker = :ticker
                            AND status = 'executed'
                            AND filled_contracts > 0
                            AND client_order_id NOT LIKE 'kw-backfill-%'
                            AND placed_at <= :snapshot_at
                          ORDER BY placed_at DESC, id DESC LIMIT 1)
            """
        ).bindparams(
            bindparam("pnl", type_=Numeric(10, 6)),
            bindparam("snapshot_at", type_=_UDT()),
        )
        session.execute(
            stmt,
            {
                "pnl": Decimal("2.250000"),
                "ticker": "KXHIGHCHI-26JUN01-T70",
                "snapshot_at": snapshot_at,
            },
        )
        stale = session.scalars(
            select(_DemoOrderRow).where(_DemoOrderRow.client_order_id == "kw-edge-b")
        ).one()
        assert stale.realized_pnl_dollars is None

    await client.aclose()
