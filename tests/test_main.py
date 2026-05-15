from __future__ import annotations

import asyncio
import logging
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
from bot.forecast.cdf import EnsembleCDF
from bot.forecast.open_meteo import StationForecast
from bot.kalshi_client import KalshiDemoClient, KalshiMarket, KalshiOrderbook
from bot.main import (
    STATIONS,
    STRATEGY_BLACKLIST,
    App,
    _build_intents,
    _forecast_loop,
    _parse_duration,
    _parse_series_arg,
    _settlement_loop,
    evaluate_strategies,
    main,
    reconcile_settled_trades,
    refresh_forecasts,
    refresh_markets,
)
from bot.markets.parser import parse_ticker
from bot.storage.sqlite import (
    Base,
    Forecast,
    Market,
    OrderbookSnapshot,
    PaperTradeRow,
    SimulatedPnl,
    make_engine,
    make_session_factory,
)
from bot.strategy import edge as edge_strategy
from bot.strategy import tails as tails_strategy


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
    settings = Settings(paper_mode=True)
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


def _book_from(ticker: str, yes_ask: str, yes_bid: str) -> KalshiOrderbook:
    return KalshiOrderbook(
        ticker=ticker,
        yes_ask=Decimal(yes_ask),
        yes_bid=Decimal(yes_bid),
        no_ask=Decimal("1") - Decimal(yes_bid),
        no_bid=Decimal("1") - Decimal(yes_ask),
        snapshot_at=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
    )


def _market_from(ticker: str, yes_ask: str, yes_bid: str, close_at: datetime) -> KalshiMarket:
    series = ticker.split("-", 1)[0]
    event_ticker = "-".join(ticker.split("-")[:2])
    return KalshiMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        series=series,
        status="open",
        close_time=close_at,
        yes_ask=Decimal(yes_ask),
        yes_bid=Decimal(yes_bid),
    )


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
            paper_mode=True,
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


async def test_evaluate_strategies_runs_edge_buy_path() -> None:
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


async def test_evaluate_strategies_uses_per_ticker_cdf_not_app_series() -> None:
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
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.05"),
        spread=Decimal("3.0"),
        mid=Decimal("0.19"),
        is_same_day=False,
        is_blacklisted=True,
        is_tail=False,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
    )
    assert intents == []


def test_build_intents_skips_blacklisted_mia_series() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHMIA-26MAY08-T70-75", "0.20", "0.18", close_at)
    book = _book_from(market.ticker, "0.20", "0.18")
    intents = _build_intents(
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.05"),
        spread=Decimal("3.0"),
        mid=Decimal("0.19"),
        is_same_day=False,
        is_blacklisted=True,
        is_tail=False,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
    )
    assert intents == []


def test_build_intents_emits_for_normal_series() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = _market_from("KXHIGHDEN-26MAY08-T70-75", "0.20", "0.18", close_at)
    book = _book_from(market.ticker, "0.20", "0.18")
    intents = _build_intents(
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.80"),
        spread=Decimal("3.0"),
        mid=Decimal("0.19"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=False,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
    )
    assert intents
    assert any(i.strategy == "edge" for i in intents)


async def test_evaluate_strategies_skips_unparseable_ticker(
    caplog: pytest.LogCaptureFixture,
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


def test_cli_rejects_non_paper_mode(monkeypatch) -> None:
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
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.04"),
        spread=Decimal("3.0"),
        mid=Decimal("0.49"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=True,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
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
        ticker=market.ticker,
        market=market,
        book=book,
        fair_yes=Decimal("0.50"),
        spread=Decimal("3.0"),
        mid=Decimal("0.19"),
        is_same_day=False,
        is_blacklisted=False,
        is_tail=False,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
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
            ticker=market.ticker,
            market=market,
            book=book,
            fair_yes=Decimal("0.04"),
            spread=Decimal("3.0"),
            mid=Decimal("0.49"),
            is_same_day=False,
            is_blacklisted=True,
            is_tail=is_tail,
            now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        )
        if is_tail:
            assert tails_calls == []
        else:
            assert len(edge_calls) == 1


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
        "KXHIGHDEN-26MAY08-B70",
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
