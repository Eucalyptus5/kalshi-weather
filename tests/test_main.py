from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import select

from bot.forecast.open_meteo import StationForecast
from bot.kalshi_client import KalshiMarket, KalshiOrderbook
from bot.main import (
    STATIONS,
    App,
    _parse_duration,
    evaluate_strategies,
    main,
    refresh_forecasts,
    refresh_markets,
)
from bot.storage.sqlite import (
    Base,
    Forecast,
    Market,
    OrderbookSnapshot,
    PaperTradeRow,
    make_engine,
    make_session_factory,
)


class _StubMeteo:
    def __init__(self, forecast: StationForecast) -> None:
        self._forecast = forecast
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
        return self._forecast

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
        return [m for m in self._markets if m.ticker.startswith(series_prefix)]

    async def get_orderbook(self, ticker: str) -> KalshiOrderbook:
        return self._orderbooks[ticker]


def _make_app(meteo: _StubMeteo | None = None, kalshi: _StubKalshi | None = None) -> App:
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
        series="KXHIGHDEN",
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


async def test_refresh_markets_persists_markets_and_orderbooks() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market_a = KalshiMarket(
        ticker="KXHIGHDEN-26MAY07-T70-75",
        event_ticker="KXHIGHDEN-26MAY07",
        series="KXHIGHDEN",
        status="open",
        close_time=close_at,
        yes_ask=Decimal("0.45"),
        yes_bid=Decimal("0.43"),
    )
    market_b = KalshiMarket(
        ticker="KXHIGHDEN-26MAY07-T75-80",
        event_ticker="KXHIGHDEN-26MAY07",
        series="KXHIGHDEN",
        status="open",
        close_time=close_at,
        yes_ask=Decimal("0.30"),
        yes_bid=Decimal("0.28"),
    )
    book_a = KalshiOrderbook(
        ticker=market_a.ticker,
        yes_ask=Decimal("0.45"),
        yes_bid=Decimal("0.43"),
        no_ask=Decimal("0.57"),
        no_bid=Decimal("0.55"),
        snapshot_at=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
    )
    book_b = KalshiOrderbook(
        ticker=market_b.ticker,
        yes_ask=Decimal("0.30"),
        yes_bid=Decimal("0.28"),
        no_ask=Decimal("0.72"),
        no_bid=Decimal("0.70"),
        snapshot_at=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
    )

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


async def test_refresh_markets_upserts_existing_ticker() -> None:
    close_at = datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc)
    market = KalshiMarket(
        ticker="KXHIGHDEN-26MAY07-T70-75",
        event_ticker="KXHIGHDEN-26MAY07",
        series="KXHIGHDEN",
        status="open",
        close_time=close_at,
        yes_ask=Decimal("0.45"),
        yes_bid=Decimal("0.43"),
    )
    book = KalshiOrderbook(
        ticker=market.ticker,
        yes_ask=Decimal("0.45"),
        yes_bid=Decimal("0.43"),
        no_ask=Decimal("0.57"),
        no_bid=Decimal("0.55"),
        snapshot_at=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
    )
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(kalshi=kalshi)

    await refresh_markets(app)
    await refresh_markets(app)

    with app.session_factory() as session:
        rows = session.scalars(select(Market)).all()
    assert len(rows) == 1


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

    market = KalshiMarket(
        ticker="KXHIGHDEN-26MAY08-T70-75",
        event_ticker="KXHIGHDEN-26MAY08",
        series="KXHIGHDEN",
        status="open",
        close_time=datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
        yes_ask=Decimal("0.20"),
        yes_bid=Decimal("0.18"),
    )
    book = KalshiOrderbook(
        ticker=market.ticker,
        yes_ask=Decimal("0.20"),
        yes_bid=Decimal("0.18"),
        no_ask=Decimal("0.82"),
        no_bid=Decimal("0.80"),
        snapshot_at=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
    )
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
    market = KalshiMarket(
        ticker="KXHIGHDEN-26MAY08-T70-75",
        event_ticker="KXHIGHDEN-26MAY08",
        series="KXHIGHDEN",
        status="open",
        close_time=datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
        yes_ask=Decimal("0.50"),
        yes_bid=Decimal("0.48"),
    )
    book = KalshiOrderbook(
        ticker=market.ticker,
        yes_ask=Decimal("0.50"),
        yes_bid=Decimal("0.48"),
        no_ask=Decimal("0.52"),
        no_bid=Decimal("0.50"),
        snapshot_at=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
    )
    kalshi = _StubKalshi(markets=[market], orderbooks={market.ticker: book})
    app = _make_app(kalshi=kalshi)

    await refresh_markets(app)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert n_trades == 0
    with app.session_factory() as session:
        rows = session.scalars(select(PaperTradeRow)).all()
    assert rows == []


async def test_evaluate_strategies_skips_tail_markets() -> None:
    fc = StationForecast(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
        run_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        daily_highs={date(2026, 5, 8): np.random.default_rng(2).normal(73.0, 4.0, size=31)},
    )
    meteo = _StubMeteo(fc)

    tail = KalshiMarket(
        ticker="KXHIGHDEN-26MAY08-T100",
        event_ticker="KXHIGHDEN-26MAY08",
        series="KXHIGHDEN",
        status="open",
        close_time=datetime(2026, 5, 8, 23, 0, tzinfo=timezone.utc),
        yes_ask=Decimal("0.05"),
        yes_bid=Decimal("0.03"),
    )
    book = KalshiOrderbook(
        ticker=tail.ticker,
        yes_ask=Decimal("0.05"),
        yes_bid=Decimal("0.03"),
        no_ask=Decimal("0.97"),
        no_bid=Decimal("0.95"),
        snapshot_at=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
    )
    kalshi = _StubKalshi(markets=[tail], orderbooks={tail.ticker: book})

    app = _make_app(meteo=meteo, kalshi=kalshi)
    await refresh_forecasts(app)
    await refresh_markets(app)

    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    n_trades = await evaluate_strategies(app, now)

    assert n_trades == 0


def test_cli_rejects_unsupported_series(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["bot.main", "--mode=paper", "--series=KXHIGHAUS", "--duration=1m"],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "KXHIGHAUS" in (captured.err + captured.out)


def test_cli_rejects_non_paper_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["bot.main", "--mode=live", "--series=KXHIGHDEN", "--duration=1m"],
    )
    with pytest.raises(SystemExit):
        main()


def test_stations_map_has_kxhighden() -> None:
    cfg = STATIONS["KXHIGHDEN"]
    assert cfg.station == "KDEN"
    assert cfg.timezone == "America/Denver"
