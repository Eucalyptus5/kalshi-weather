from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import signal
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from bot.config import Settings, get_settings
from bot.execution.demo_row_translator import (
    _demo_order_values,
    paper_trade_row_from_demo_order,
    paper_trade_row_values,
)
from bot.execution.gate_cost_basis import (
    cost_per_contract_from_book,
    paper_collateral_per_contract,
)
from bot.execution.order_loop import place_orders_resilient
from bot.execution.order_placer import DemoOrder, DemoOrderIdempotent, place_order_demo
from bot.execution.order_reconciler import (
    idempotent_insert_demo_order_row,
    local_position_aggregate,
    poll_fills,
    poll_open_orders,
    poll_positions,
    reconcile_fills_into_demo_orders,
    stitch_natural_key_order,
    upsert_exchange_record,
)
from bot.execution.portfolio_snapshot import aggregate_positions, update_demo_realized_pnl
from bot.execution.paper import (
    Orderbook,
    PaperTrade,
    TradeIntent,
    TradeSide,
    log_stale_skip_ratio,
    simulate_taker_fill,
)
from bot.forecast.cdf import EnsembleCDF
from bot.forecast.open_meteo import OpenMeteoClient, StationForecast
from bot.kalshi_client import KalshiDemoClient, KalshiMarket, KalshiOrderbook
from bot.markets.observation_window import observation_window
from bot.markets.parser import ParsedTicker, parse_ticker
from bot.risk.gates import CAP_GATE_NAMES, GateContext, GateMode, evaluate as evaluate_gates
from bot.storage.positions import open_exposures
from bot.storage.sqlite import (
    DemoOrder as DemoOrderRow,
    Forecast,
    GateFailure,
    Market,
    OrderbookSnapshot,
    PaperTradeRow,
    PortfolioSnapshot,
    ReconcilerState,
    SimulatedPnl,
    make_engine,
    make_session_factory,
    upgrade_schema,
)
from bot.strategy import edge as edge_strategy
from bot.strategy import tails as tails_strategy
from bot.strategy.sizing import sigma_t_median_for_lead
from bot.validation.calibration import (
    BSS_AGGREGATE_NA,
    CalibrationMaps,
    bucket_for,
    format_gate_failure_reason,
    hours_until,
    refit_all,
)
from bot.validation.reconcile import ACISClient, reconcile_trade

logger = logging.getLogger(__name__)


PAPER_BANKROLL: Decimal = Decimal("500")
LIVE_BANKROLL_ENABLED: bool = False


def bankroll(app: "App | None" = None) -> Decimal:
    if app is not None and app.bankroll is not None:
        return app.bankroll
    return PAPER_BANKROLL


REQUIRED_CUSHION: Decimal = Decimal("100")

MARKET_POSITION_FRAC: Decimal = Decimal("0.015")
EVENT_POSITION_FRAC: Decimal = Decimal("0.03")
SERIES_POSITION_FRAC: Decimal = Decimal("0.05")
AGGREGATE_EXPOSURE_FRAC: Decimal = Decimal("0.40")

MARKET_POSITION_CAP: Decimal = PAPER_BANKROLL * MARKET_POSITION_FRAC
EVENT_POSITION_CAP: Decimal = PAPER_BANKROLL * EVENT_POSITION_FRAC
SERIES_POSITION_CAP: Decimal = PAPER_BANKROLL * SERIES_POSITION_FRAC
AGGREGATE_EXPOSURE_CAP: Decimal = PAPER_BANKROLL * AGGREGATE_EXPOSURE_FRAC


def market_position_cap(app: "App | None" = None) -> Decimal:
    return bankroll(app) * MARKET_POSITION_FRAC


def event_position_cap(app: "App | None" = None) -> Decimal:
    return bankroll(app) * EVENT_POSITION_FRAC


def series_position_cap(app: "App | None" = None) -> Decimal:
    return bankroll(app) * SERIES_POSITION_FRAC


def aggregate_exposure_cap(app: "App | None" = None) -> Decimal:
    return bankroll(app) * AGGREGATE_EXPOSURE_FRAC


MARKET_REFRESH_INTERVAL = 60.0
EVAL_INTERVAL = 60.0
RECONCILE_INTERVAL_SECONDS: float = 30.0
PORTFOLIO_SNAPSHOT_INTERVAL_SECONDS: float = 60.0
SETTLEMENT_INTERVAL_SECONDS: float = 6 * 3600
_SETTLEMENT_GRACE_DAYS: int = 1
GFS_CYCLES_HOURS: tuple[int, ...] = (0, 6, 12, 18)
GFS_CYCLE_OFFSET_MINUTES = 30
FORECAST_RETRY_INTERVAL_SECONDS: float = 60.0
CALIBRATION_REFIT_JITTER_SECONDS: int = 300
CALIBRATION_REFIT_TARGET_HOUR_UTC: int = 4
CALIBRATION_REFIT_TOTAL_BUCKETS: int = 24


@dataclass(frozen=True, slots=True)
class StationConfig:
    series: str
    station: str
    latitude: float
    longitude: float
    timezone: str


STATIONS: dict[str, StationConfig] = {
    "KXHIGHDEN": StationConfig(
        series="KXHIGHDEN",
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
    ),
    "KXHIGHAUS": StationConfig(
        series="KXHIGHAUS",
        station="KAUS",
        latitude=30.1831,
        longitude=-97.6806,
        timezone="America/Chicago",
    ),
    "KXHIGHCHI": StationConfig(
        series="KXHIGHCHI",
        station="KMDW",
        latitude=41.7841,
        longitude=-87.7552,
        timezone="America/Chicago",
    ),
    "KXHIGHNY": StationConfig(
        series="KXHIGHNY",
        station="KNYC",
        latitude=40.7790,
        longitude=-73.9692,
        timezone="America/New_York",
    ),
    "KXHIGHPHIL": StationConfig(
        series="KXHIGHPHIL",
        station="KPHL",
        latitude=39.8733,
        longitude=-75.2268,
        timezone="America/New_York",
    ),
    "KXHIGHTATL": StationConfig(
        series="KXHIGHTATL",
        station="KATL",
        latitude=33.6297,
        longitude=-84.4422,
        timezone="America/New_York",
    ),
    "KXHIGHTBOS": StationConfig(
        series="KXHIGHTBOS",
        station="KBOS",
        latitude=42.3606,
        longitude=-71.0097,
        timezone="America/New_York",
    ),
    "KXHIGHTDAL": StationConfig(
        series="KXHIGHTDAL",
        station="KDFW",
        latitude=32.8974,
        longitude=-97.0220,
        timezone="America/Chicago",
    ),
    "KXHIGHTDC": StationConfig(
        series="KXHIGHTDC",
        station="KDCA",
        latitude=38.8472,
        longitude=-77.0345,
        timezone="America/New_York",
    ),
    "KXHIGHTHOU": StationConfig(
        series="KXHIGHTHOU",
        station="KIAH",
        latitude=29.9844,
        longitude=-95.3607,
        timezone="America/Chicago",
    ),
    "KXHIGHTLV": StationConfig(
        series="KXHIGHTLV",
        station="KLAS",
        latitude=36.0719,
        longitude=-115.1634,
        timezone="America/Los_Angeles",
    ),
    "KXHIGHTMIN": StationConfig(
        series="KXHIGHTMIN",
        station="KMSP",
        latitude=44.8852,
        longitude=-93.2313,
        timezone="America/Chicago",
    ),
    "KXHIGHTNOLA": StationConfig(
        series="KXHIGHTNOLA",
        station="KMSY",
        latitude=29.9974,
        longitude=-90.2777,
        timezone="America/Chicago",
    ),
    "KXHIGHTOKC": StationConfig(
        series="KXHIGHTOKC",
        station="KOKC",
        latitude=35.3843,
        longitude=-97.6003,
        timezone="America/Chicago",
    ),
    "KXHIGHTPHX": StationConfig(
        series="KXHIGHTPHX",
        station="KPHX",
        latitude=33.4278,
        longitude=-112.0037,
        timezone="America/Phoenix",
    ),
    "KXHIGHTSATX": StationConfig(
        series="KXHIGHTSATX",
        station="KSAT",
        latitude=29.5443,
        longitude=-98.4839,
        timezone="America/Chicago",
    ),
    "KXHIGHTSEA": StationConfig(
        series="KXHIGHTSEA",
        station="KSEA",
        latitude=47.4447,
        longitude=-122.3144,
        timezone="America/Los_Angeles",
    ),
    "KXHIGHTSFO": StationConfig(
        series="KXHIGHTSFO",
        station="KSFO",
        latitude=37.6196,
        longitude=-122.3656,
        timezone="America/Los_Angeles",
    ),
    "KXHIGHLAX": StationConfig(
        series="KXHIGHLAX",
        station="KLAX",
        latitude=33.9382,
        longitude=-118.3866,
        timezone="America/Los_Angeles",
    ),
    "KXHIGHMIA": StationConfig(
        series="KXHIGHMIA",
        station="KMIA",
        latitude=25.7881,
        longitude=-80.3169,
        timezone="America/New_York",
    ),
}

assert len(STATIONS) == 20

STRATEGY_BLACKLIST: frozenset[str] = frozenset({"KXHIGHLAX", "KXHIGHMIA"})


def _empty_calibration_maps() -> CalibrationMaps:
    return CalibrationMaps(
        maps={},
        fitted_at=datetime.now(tz=_timezone.utc),
        n_samples_per_bucket={},
        holdout_bs_new={},
        holdout_bs_prev={},
        holdout_n_per_bucket={},
        climatological_rate_per_bucket={},
        bss_aggregate_per_stratum={},
    )


@dataclass
class App:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    meteo: OpenMeteoClient
    kalshi: KalshiDemoClient
    acis: ACISClient
    series_list: tuple[str, ...]
    bankroll: Decimal | None = None
    db_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    reconcile_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    forecast_cdfs: dict[tuple[str, date], EnsembleCDF] = field(default_factory=dict)
    ensemble_spreads: dict[tuple[str, date], Decimal] = field(default_factory=dict)
    forecast_run_times: dict[tuple[str, date], datetime] = field(default_factory=dict)
    latest_markets: dict[str, KalshiMarket] = field(default_factory=dict)
    latest_orderbooks: dict[str, KalshiOrderbook] = field(default_factory=dict)
    calibration_maps: CalibrationMaps = field(default_factory=_empty_calibration_maps)

    def __post_init__(self) -> None:
        logger.info(
            "calibration_maps_loaded n_buckets_fit=%d n_buckets_skipped=%d reason=cold_start",
            0,
            CALIBRATION_REFIT_TOTAL_BUCKETS,
        )

    async def aclose(self) -> None:
        await self.meteo.aclose()
        await self.kalshi.aclose()
        await self.acis.aclose()
        _checkpoint_wal(self.engine)
        self.engine.dispose()


_DURATION_RE = re.compile(r"^(\d+)([hms])$")


def _parse_duration(raw: str) -> timedelta:
    match = _DURATION_RE.match(raw)
    if not match:
        raise ValueError(f"duration must match Nh/Nm/Ns: got {raw!r}")
    n = int(match.group(1))
    unit = match.group(2)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "m":
        return timedelta(minutes=n)
    return timedelta(seconds=n)


async def refresh_forecasts(app: App) -> int:
    total = 0
    for series in app.series_list:
        cfg = STATIONS[series]
        forecast = await app.meteo.fetch_station(
            station=cfg.station,
            latitude=cfg.latitude,
            longitude=cfg.longitude,
            timezone=cfg.timezone,
        )
        async with app.db_lock:
            total += _persist_forecast(app, forecast)
    return total


def _persist_forecast(app: App, forecast: StationForecast) -> int:
    n = 0
    with app.session_factory() as session:
        for valid_date, members in forecast.daily_highs.items():
            members_arr = np.asarray(members, dtype=np.float64)
            existing = session.scalars(
                select(Forecast).where(
                    Forecast.station == forecast.station,
                    Forecast.run_time == forecast.run_time,
                    Forecast.valid_date == valid_date,
                )
            ).one_or_none()
            if existing is None:
                session.add(
                    Forecast(
                        station=forecast.station,
                        run_time=forecast.run_time,
                        valid_date=valid_date,
                        members_json=json.dumps(members_arr.tolist()),
                    )
                )
            key = (forecast.station, valid_date)
            app.forecast_cdfs[key] = EnsembleCDF.from_members(members_arr, smoothing=1.0)
            app.ensemble_spreads[key] = Decimal(str(float(members_arr.std())))
            app.forecast_run_times[key] = forecast.run_time
            n += 1
        session.commit()
    logger.info(
        "refresh_forecasts station=%s valid_dates=%d run_time=%s",
        forecast.station,
        n,
        forecast.run_time.isoformat(),
    )
    return n


async def refresh_markets(app: App) -> int:
    now = datetime.now(tz=_timezone.utc)
    total = 0
    for series in app.series_list:
        try:
            markets = await app.kalshi.list_open_markets_for_series(series)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as err:
            logger.warning("kalshi_list_markets_failed series=%s err=%s", series, err)
            continue

        pairs: list[tuple[KalshiMarket, KalshiOrderbook]] = []
        seen_tickers: set[str] = set()
        for m in markets:
            try:
                book = await app.kalshi.get_orderbook(m.ticker)
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as err:
                logger.warning("kalshi_orderbook_fetch_failed ticker=%s err=%s", m.ticker, err)
                continue
            pairs.append((m, book))
            seen_tickers.add(m.ticker)

        async with app.db_lock:
            with app.session_factory() as session:
                for m, book in pairs:
                    try:
                        parsed = parse_ticker(m.ticker)
                    except ValueError as err:
                        logger.warning("market_unparseable_ticker ticker=%s err=%s", m.ticker, err)
                        continue
                    existing = session.scalars(
                        select(Market).where(Market.ticker == m.ticker)
                    ).one_or_none()
                    if existing is None:
                        session.add(
                            Market(
                                ticker=m.ticker,
                                series=parsed.series,
                                event_date=parsed.event_date,
                                is_monthly=parsed.is_monthly,
                                is_tail=parsed.is_tail,
                                strike_low=parsed.strikes[0],
                                strike_high=parsed.strikes[1] if parsed.is_bracket else None,
                                close_time=m.close_time,
                                status=m.status,
                                last_seen_at=now,
                            )
                        )
                    else:
                        existing.status = m.status
                        existing.close_time = m.close_time
                        existing.last_seen_at = now
                    session.add(
                        OrderbookSnapshot(
                            ticker=m.ticker,
                            snapshot_at=book.snapshot_at,
                            yes_ask=book.yes_ask,
                            yes_bid=book.yes_bid,
                            no_ask=book.no_ask,
                            no_bid=book.no_bid,
                            yes_ask_depth=book.yes_ask_depth,
                            yes_bid_depth=book.yes_bid_depth,
                            no_ask_depth=book.no_ask_depth,
                            no_bid_depth=book.no_bid_depth,
                        )
                    )
                    app.latest_markets[m.ticker] = m
                    app.latest_orderbooks[m.ticker] = book
                    total += 1
                session.commit()

        for ticker in list(app.latest_markets.keys()):
            if ticker.startswith(f"{series}-") and ticker not in seen_tickers:
                app.latest_markets.pop(ticker, None)
                app.latest_orderbooks.pop(ticker, None)

        logger.info("refresh_markets series=%s pairs=%d", series, len(pairs))
    return total


def _gate_mode_for_settings(settings: Settings) -> GateMode:
    if settings.mode == "demo":
        return GateMode.DEMO
    if settings.mode == "paper":
        return GateMode.PAPER
    return GateMode.PAPER


async def _snapshot_markets_atomically(
    app: App,
) -> tuple[tuple[tuple[str, KalshiMarket], ...], dict[str, KalshiOrderbook]]:
    async with app.db_lock:
        return tuple(app.latest_markets.items()), dict(app.latest_orderbooks)


async def evaluate_strategies(app: App, now: datetime) -> int:
    n_trades = 0
    stale_skips_by_series: dict[str, int] = defaultdict(int)
    intents_seen_by_series: dict[str, int] = defaultdict(int)

    markets_snapshot, orderbooks_snapshot = await _snapshot_markets_atomically(app)

    with app.session_factory() as session:
        by_market, by_event, by_series, cycle_aggregate_exposure = open_exposures(session, now=now)
    # type(x)(x) preserves any dict subclass passed in (canary in tests/test_main.py); dict(x)/copy/{**x} would coerce to plain dict.
    overlay_market = type(by_market)(by_market)
    overlay_event = type(by_event)(by_event)
    overlay_series = type(by_series)(by_series)

    gate_mode = _gate_mode_for_settings(app.settings)
    gate_failures: list[GateFailure] = []
    paper_rows: list[PaperTradeRow] = []
    demo_rows: list[dict[str, object]] = []
    pending_intents: list[tuple[TradeIntent, KalshiOrderbook, datetime]] = []

    for ticker, market in markets_snapshot:
        if market.close_time is None:
            continue
        book = orderbooks_snapshot.get(ticker)
        if book is None:
            continue
        try:
            parsed = parse_ticker(ticker)
        except ValueError as err:
            logger.warning("eval_unparseable_ticker ticker=%s err=%s", ticker, err)
            continue

        cfg = STATIONS.get(parsed.series)
        if cfg is None:
            logger.warning("eval_unknown_series ticker=%s series=%s", ticker, parsed.series)
            continue

        cdf_key = (cfg.station, parsed.event_date)
        cdf = app.forecast_cdfs.get(cdf_key)
        if cdf is None:
            continue
        spread = app.ensemble_spreads[cdf_key]
        run_time = app.forecast_run_times[cdf_key]

        if parsed.kind == "bracket":
            lo = float(parsed.strikes[0])
            hi = float(parsed.strikes[1])
            fair_yes = Decimal(str(cdf.prob_range(lo, hi)))
        elif parsed.kind == "above":
            fair_yes = Decimal(str(1.0 - cdf.cdf(float(parsed.strikes[0]))))
        else:
            fair_yes = Decimal(str(cdf.cdf(float(parsed.strikes[0]))))

        start_utc, end_utc = observation_window(cfg.timezone, parsed.event_date)
        is_same_day = start_utc <= now < end_utc

        is_blacklisted = parsed.series in STRATEGY_BLACKLIST

        event_key = market.event_ticker
        series_key = parsed.series

        sigma_T_median = sigma_t_median_for_lead(
            int((market.close_time - now).total_seconds() / 3600)
        )
        event_budget_remaining = max(
            Decimal("0"),
            event_position_cap(app) - overlay_event.get(event_key, Decimal("0")),
        )
        market_budget_remaining = max(
            Decimal("0"),
            market_position_cap(app) - overlay_market.get(ticker, Decimal("0")),
        )
        buy_yes_depth = book.no_bid_depth
        sell_yes_depth = book.yes_bid_depth

        for intent in _build_intents(
            app=app,
            ticker=ticker,
            market=market,
            book=book,
            fair_yes=fair_yes,
            spread=spread,
            is_same_day=is_same_day,
            is_blacklisted=is_blacklisted,
            is_tail=parsed.is_tail,
            mode=app.settings.mode,
            now=now,
            sigma_T_median=sigma_T_median,
            event_budget_remaining=event_budget_remaining,
            market_budget_remaining=market_budget_remaining,
            buy_yes_depth=buy_yes_depth,
            sell_yes_depth=sell_yes_depth,
        ):
            intents_seen_by_series[series_key] += 1
            if app.settings.mode == "paper":
                debit_basis = paper_collateral_per_contract(intent.side, book)
            else:
                debit_basis = cost_per_contract_from_book(intent.side, book)

            gate_ctx = _gate_ctx_for(
                intent=intent,
                market=market,
                fair_yes=fair_yes,
                spread=spread,
                run_time=run_time,
                now=now,
                book=book,
                market_existing_dollars=overlay_market.get(ticker, Decimal("0")),
                event_existing_dollars=overlay_event.get(event_key, Decimal("0")),
                series_existing_dollars=overlay_series.get(series_key, Decimal("0")),
                aggregate_existing_dollars=cycle_aggregate_exposure,
                buy_yes_depth=buy_yes_depth,
                sell_yes_depth=sell_yes_depth,
                app=app,
            )
            check = evaluate_gates(gate_ctx, gate_mode)
            bucket_key = bucket_for(
                intent.strategy, intent.q_raw, hours_until(market.close_time, now)
            )
            for failure in check.failures:
                reason = format_gate_failure_reason(
                    failure.reason or "",
                    q_raw=intent.q_raw,
                    fair_yes=intent.fair_yes,
                    bucket_key=bucket_key,
                )
                gate_failures.append(
                    GateFailure(
                        evaluated_at=now,
                        gate_name=failure.name,
                        reason=reason,
                        mode=gate_mode.value,
                        market_ticker=ticker,
                        last_seen_at=now,
                    )
                )
            if any(f.name in CAP_GATE_NAMES for f in check.failures):
                continue
            if not check.overall_passed:
                continue

            if app.settings.mode == "paper":
                trade = simulate_taker_fill(
                    intent,
                    Orderbook(
                        yes_ask=book.yes_ask,
                        yes_bid=book.yes_bid,
                        yes_ask_depth=book.yes_ask_depth,
                        yes_bid_depth=book.yes_bid_depth,
                        snapshot_at=book.snapshot_at,
                    ),
                    now,
                )
                if trade is None:
                    stale_skips_by_series[series_key] += 1
                    continue
                paper_rows.append(_paper_trade_row(trade))
                delta = debit_basis * Decimal(trade.contracts)
            else:
                if ticker not in app.latest_markets:
                    continue
                now_pre_post: datetime = now
                pending_intents.append((intent, book, now_pre_post))
                delta = debit_basis * Decimal(intent.contracts)

            overlay_market[ticker] = overlay_market.get(ticker, Decimal("0")) + delta
            overlay_event[event_key] = overlay_event.get(event_key, Decimal("0")) + delta
            overlay_series[series_key] = overlay_series.get(series_key, Decimal("0")) + delta
            cycle_aggregate_exposure = cycle_aggregate_exposure + delta
            n_trades += 1

    async def _place(
        triple: tuple[TradeIntent, KalshiOrderbook, datetime],
    ) -> tuple[TradeIntent, DemoOrder, datetime] | None:
        intent, book, now_pre_post = triple
        if intent.market_ticker not in app.latest_markets:
            return None
        order = await place_order_demo(intent, book, app.kalshi, now=now_pre_post)
        if order is None:
            return None
        if isinstance(order, DemoOrderIdempotent):
            side_kalshi = "yes" if intent.side is TradeSide.BUY_YES else "no"
            async with app.db_lock:
                with app.session_factory() as session:
                    idempotent_insert_demo_order_row(
                        session,
                        client_order_id=order.client_order_id,
                        exchange_order_id=order.exchange_order_id or None,
                        market_ticker=intent.market_ticker,
                        strategy=intent.strategy,
                        side=side_kalshi,
                        requested_contracts=intent.contracts,
                        filled_contracts=order.filled_contracts,
                        requested_yes_price_dollars=order.requested_yes_price_dollars,
                        fair_at_entry=intent.fair_yes,
                        q_raw=intent.q_raw,
                        intended_at=now_pre_post,
                        status=order.status,
                        placed_at=now_pre_post,
                    )
                    session.commit()
            return None
        return (intent, order, now_pre_post)

    for intent, order, now_pre_post in await place_orders_resilient(pending_intents, _place):
        demo_rows.append(_demo_order_values(order, intent, now_pre_post))
        row = paper_trade_row_from_demo_order(order, intent, intended_at=now_pre_post)
        if row is not None:
            paper_rows.append(row)

    await _commit_phase2(
        app,
        gate_failures=gate_failures,
        paper_rows=paper_rows,
        demo_rows=demo_rows,
    )

    log_stale_skip_ratio(stale_skips_by_series, intents_seen_by_series)
    logger.info(
        "evaluate_strategies trades=%d markets=%d stale_skips=%d intents=%d",
        n_trades,
        len(markets_snapshot),
        sum(stale_skips_by_series.values()),
        sum(intents_seen_by_series.values()),
    )
    return n_trades


async def _commit_phase2(
    app: App,
    *,
    gate_failures: list[GateFailure],
    paper_rows: list[PaperTradeRow],
    demo_rows: list[dict[str, object]],
) -> None:
    async with app.db_lock:
        with app.session_factory() as session:
            for failure_row in gate_failures:
                stmt = sqlite_insert(GateFailure).values(
                    evaluated_at=failure_row.evaluated_at,
                    gate_name=failure_row.gate_name,
                    reason=failure_row.reason,
                    mode=failure_row.mode,
                    market_ticker=failure_row.market_ticker,
                    count=1,
                    last_seen_at=failure_row.last_seen_at,
                )
                session.execute(
                    stmt.on_conflict_do_update(
                        index_elements=["gate_name", "market_ticker", "reason"],
                        set_={
                            "count": GateFailure.count + 1,
                            "last_seen_at": stmt.excluded.last_seen_at,
                        },
                    )
                )
            for paper_row in paper_rows:
                if paper_row.demo_order_client_id is None:
                    session.add(paper_row)
                else:
                    session.execute(
                        sqlite_insert(PaperTradeRow)
                        .values(**paper_trade_row_values(paper_row))
                        .on_conflict_do_nothing(index_elements=["demo_order_client_id"])
                    )
            for values in demo_rows:
                try:
                    with session.begin_nested():
                        eid = values.get("exchange_order_id")
                        stitched = False
                        if eid is not None:
                            existing = session.execute(
                                select(DemoOrderRow).where(DemoOrderRow.exchange_order_id == eid)
                            ).scalar_one_or_none()
                            if existing is not None:
                                stitch_natural_key_order(
                                    session,
                                    exchange_order_id=eid,  # type: ignore[arg-type]
                                    client_order_id=values["client_order_id"],  # type: ignore[arg-type]
                                    strategy=values["strategy"],  # type: ignore[arg-type]
                                    side=values["side"],  # type: ignore[arg-type]
                                    fair_at_entry=values["fair_at_entry"],  # type: ignore[arg-type]
                                    q_raw=values["q_raw"],  # type: ignore[arg-type]
                                    intended_at=values["intended_at"],  # type: ignore[arg-type]
                                    requested_yes_price_dollars=values[  # type: ignore[arg-type]
                                        "requested_yes_price_dollars"
                                    ],
                                )
                                stitched = True
                        if not stitched:
                            session.execute(
                                sqlite_insert(DemoOrderRow)
                                .values(**values)
                                .on_conflict_do_update(
                                    index_elements=["client_order_id"],
                                    set_={
                                        "strategy": values["strategy"],
                                        "side": values["side"],
                                        "fair_at_entry": values["fair_at_entry"],
                                        "intended_at": values["intended_at"],
                                        "requested_yes_price_dollars": values[
                                            "requested_yes_price_dollars"
                                        ],
                                    },
                                )
                            )
                except (ValueError, SQLAlchemyError) as err:
                    logger.error("demo_row_stitch_corrupted err=%s values=%r", err, values)
                    continue
            session.commit()


def _compute_lead_time_hours(market: KalshiMarket, now: datetime) -> Decimal | None:
    if market.close_time is None:
        return None
    seconds = (market.close_time - now).total_seconds()
    if seconds < 0:
        return None
    return Decimal(str(seconds / 3600))


def _build_intents(
    *,
    app: App,
    ticker: str,
    market: KalshiMarket,
    book: KalshiOrderbook,
    fair_yes: Decimal,
    spread: Decimal,
    is_same_day: bool,
    is_blacklisted: bool,
    is_tail: bool,
    mode: str,
    now: datetime,
    sigma_T_median: Decimal,
    event_budget_remaining: Decimal,
    market_budget_remaining: Decimal,
    buy_yes_depth: int,
    sell_yes_depth: int,
) -> list[TradeIntent]:
    intents: list[TradeIntent] = []
    lead_time_hours = _compute_lead_time_hours(market, now)
    is_demo = mode == "demo"
    ctx_bankroll = bankroll(app) if is_demo else bankroll()
    no_cost_per_contract = book.no_ask if is_demo else None

    if not is_tail:
        if fair_yes > book.yes_ask + edge_strategy.DIRECTION_CUSHION:
            edge_depth = buy_yes_depth
            edge_price_per_contract = book.yes_ask
        elif fair_yes < book.yes_bid - edge_strategy.DIRECTION_CUSHION:
            edge_depth = sell_yes_depth
            edge_price_per_contract = book.no_ask if is_demo else Decimal("1") - book.yes_bid
        else:
            edge_depth = buy_yes_depth
            edge_price_per_contract = book.yes_ask
        edge_ctx = edge_strategy.EdgeContext(
            yes_ask=book.yes_ask,
            yes_bid=book.yes_bid,
            fair_yes=fair_yes,
            ensemble_spread=spread,
            bankroll=ctx_bankroll,
            is_same_day=is_same_day,
            is_blacklisted=is_blacklisted,
            nbm_divergence=None,
            sigma_T_median=sigma_T_median,
            event_budget_remaining=event_budget_remaining,
            market_budget_remaining=market_budget_remaining,
            depth_at_price=edge_depth,
            price_per_contract=edge_price_per_contract,
            no_cost_per_contract=no_cost_per_contract,
        )
        edge_sig = edge_strategy.evaluate(edge_ctx, mode=mode)
        if edge_sig.action is edge_strategy.EdgeAction.BUY_YES:
            intents.append(
                TradeIntent(
                    market_ticker=ticker,
                    side=TradeSide.BUY_YES,
                    contracts=edge_sig.contracts,
                    fair_yes=fair_yes,
                    q_raw=fair_yes,
                    strategy="edge",
                    ensemble_spread_sigma_t=spread,
                    lead_time_hours=lead_time_hours,
                    nbm_divergence=None,
                )
            )
        elif edge_sig.action is edge_strategy.EdgeAction.SELL_YES:
            intents.append(
                TradeIntent(
                    market_ticker=ticker,
                    side=TradeSide.SELL_YES,
                    contracts=edge_sig.contracts,
                    fair_yes=fair_yes,
                    q_raw=fair_yes,
                    strategy="edge",
                    ensemble_spread_sigma_t=spread,
                    lead_time_hours=lead_time_hours,
                    nbm_divergence=None,
                )
            )

    if is_tail and not is_blacklisted:
        tails_price_per_contract = book.no_ask if is_demo else Decimal("1") - book.yes_bid
        tails_ctx = tails_strategy.TailsContext(
            yes_ask=book.yes_ask,
            yes_bid=book.yes_bid,
            no_bid=book.no_bid,
            fair_yes=fair_yes,
            close_time=market.close_time,
            now=now,
            bankroll=ctx_bankroll,
            is_same_day=is_same_day,
            ensemble_spread=spread,
            sigma_T_median=sigma_T_median,
            event_budget_remaining=event_budget_remaining,
            market_budget_remaining=market_budget_remaining,
            depth_at_price=sell_yes_depth,
            price_per_contract=tails_price_per_contract,
            no_cost_per_contract=no_cost_per_contract,
        )
        tails_sig = tails_strategy.evaluate(tails_ctx, mode=mode)
        if tails_sig.action is tails_strategy.TailsAction.SELL_YES:
            intents.append(
                TradeIntent(
                    market_ticker=ticker,
                    side=TradeSide.SELL_YES,
                    contracts=tails_sig.contracts,
                    fair_yes=fair_yes,
                    q_raw=fair_yes,
                    strategy="tails",
                    ensemble_spread_sigma_t=spread,
                    lead_time_hours=lead_time_hours,
                    nbm_divergence=None,
                )
            )

    return intents


def _gate_ctx_for(
    *,
    intent: TradeIntent,
    market: KalshiMarket,
    fair_yes: Decimal,
    spread: Decimal,
    run_time: datetime,
    now: datetime,
    book: KalshiOrderbook,
    market_existing_dollars: Decimal,
    event_existing_dollars: Decimal,
    series_existing_dollars: Decimal,
    aggregate_existing_dollars: Decimal,
    buy_yes_depth: int,
    sell_yes_depth: int,
    app: "App | None" = None,
) -> GateContext:
    if intent.side is TradeSide.BUY_YES:
        edge_dollars = fair_yes - book.yes_ask
        price = book.yes_ask
        depth_at_price = buy_yes_depth
    else:
        edge_dollars = book.yes_bid - fair_yes
        price = Decimal("1") - book.yes_bid
        depth_at_price = sell_yes_depth
    order_dollars = price * Decimal(intent.contracts)
    minutes_to_close = 99999
    if market.close_time is not None:
        delta = (market.close_time - now).total_seconds()
        minutes_to_close = max(0, int(delta // 60))

    model_age_hours = Decimal(str((now - run_time).total_seconds() / 3600))
    return GateContext(
        fair_yes=fair_yes,
        model_age_hours=model_age_hours,
        ensemble_spread=spread,
        edge=edge_dollars,
        price=price,
        depth_at_price=depth_at_price,
        contracts=intent.contracts,
        order_size_dollars=order_dollars,
        market_existing_dollars=market_existing_dollars,
        market_position_cap=market_position_cap(app),
        event_existing_dollars=event_existing_dollars,
        event_position_cap=event_position_cap(app),
        series_existing_dollars=series_existing_dollars,
        series_position_cap=series_position_cap(app),
        aggregate_existing_dollars=aggregate_existing_dollars,
        aggregate_exposure_cap=aggregate_exposure_cap(app),
        account_balance=bankroll(app),
        required_cushion=REQUIRED_CUSHION,
        market_status=market.status,
        minutes_to_close=minutes_to_close,
        circuit_breakers_armed=True,
    )


def _paper_trade_row(trade: PaperTrade) -> PaperTradeRow:
    return PaperTradeRow(
        intended_at=trade.intended_at,
        market_ticker=trade.market_ticker,
        side=trade.side.value,
        contracts=trade.contracts,
        simulated_price=trade.simulated_price,
        fee_dollars=trade.fee_dollars,
        fair_at_entry=trade.fair_at_entry,
        q_raw=trade.q_raw,
        strategy=trade.strategy,
        attempted_contracts=trade.attempted_contracts,
        ensemble_spread_sigma_t=trade.ensemble_spread_sigma_t,
        lead_time_hours=trade.lead_time_hours,
        nbm_divergence=trade.nbm_divergence,
    )


def _next_gfs_cycle(now: datetime) -> datetime:
    candidates: list[datetime] = []
    for offset_days in (0, 1):
        base = (now + timedelta(days=offset_days)).replace(
            minute=GFS_CYCLE_OFFSET_MINUTES, second=0, microsecond=0
        )
        for hour in GFS_CYCLES_HOURS:
            candidate = base.replace(hour=hour)
            if candidate > now:
                candidates.append(candidate)
    return min(candidates)


async def _market_loop(app: App, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await refresh_markets(app)
        except Exception:
            logger.exception("loop_iteration_failed name=market_loop")
        try:
            await asyncio.wait_for(stop.wait(), timeout=MARKET_REFRESH_INTERVAL)
        except asyncio.TimeoutError:
            pass


async def _eval_loop(app: App, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            now = datetime.now(tz=_timezone.utc)
            await evaluate_strategies(app, now)
        except Exception:
            logger.exception("loop_iteration_failed name=eval_loop")
        try:
            await asyncio.wait_for(stop.wait(), timeout=EVAL_INTERVAL)
        except asyncio.TimeoutError:
            pass


_WATERMARK_OVERLAP_SECONDS: int = 60
_COLD_START_GAP = timedelta(hours=1)
_COLD_START_FALLBACK = timedelta(days=7)


def _load_reconciler_watermark(app: App) -> datetime | None:
    with app.session_factory() as session:
        row = session.get(ReconcilerState, "last_poll_ts")
        if row is None:
            return None
        return datetime.fromisoformat(row.value)


async def _reconcile_once(app: App, watermark: datetime) -> datetime:
    poll_started_at = datetime.now(tz=_timezone.utc) - timedelta(seconds=_WATERMARK_OVERLAP_SECONDS)
    fills = await poll_fills(app.kalshi, watermark)
    open_orders = await poll_open_orders(app.kalshi, watermark)
    async with app.db_lock:
        with app.session_factory() as session:
            for record in open_orders:
                upsert_exchange_record(session, record)
            reconcile_fills_into_demo_orders(session, fills, open_orders)
            session.merge(ReconcilerState(key="last_poll_ts", value=poll_started_at.isoformat()))
            session.commit()
    return poll_started_at


async def _order_reconcile_loop(app: App, stop: asyncio.Event) -> None:
    persisted = _load_reconciler_watermark(app)
    watermark = (
        persisted
        if persisted is not None
        else datetime.now(tz=_timezone.utc) - _COLD_START_FALLBACK
    )
    while not stop.is_set():
        if app.settings.mode == "demo":
            try:
                watermark = await _reconcile_once(app, watermark)
            except Exception:
                logger.exception("loop_iteration_failed name=order_reconcile_loop")
        try:
            await asyncio.wait_for(stop.wait(), timeout=RECONCILE_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


async def _portfolio_snapshot_once(app: App) -> None:
    if app.kalshi._auth is None:
        return
    balance = await app.kalshi.get_balance_full()
    aggregate = await aggregate_positions(app.kalshi)
    snapshot_at = datetime.now(tz=_timezone.utc)
    cash_dollars = balance.balance_dollars
    portfolio_value_dollars = cash_dollars + aggregate.total_exposure_dollars
    async with app.db_lock:
        with app.session_factory() as session:
            for ticker, realized_pnl in aggregate.per_ticker_realized_pnl.items():
                update_demo_realized_pnl(session, ticker, realized_pnl, snapshot_at)
            session.add(
                PortfolioSnapshot(
                    snapshot_at=snapshot_at,
                    cash_dollars=cash_dollars,
                    portfolio_value_dollars=portfolio_value_dollars,
                    total_exposure_dollars=aggregate.total_exposure_dollars,
                    realized_pnl_dollars=aggregate.realized_pnl_dollars,
                    fees_paid_dollars=aggregate.fees_paid_dollars,
                    open_positions_count=aggregate.open_positions_count,
                )
            )
            session.commit()
    logger.info(
        "portfolio_snapshot cash=%s value=%s exposure=%s realized_pnl=%s fees=%s open=%d",
        cash_dollars,
        portfolio_value_dollars,
        aggregate.total_exposure_dollars,
        aggregate.realized_pnl_dollars,
        aggregate.fees_paid_dollars,
        aggregate.open_positions_count,
    )


async def _portfolio_snapshot_loop(app: App, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await _portfolio_snapshot_once(app)
        except Exception:
            logger.exception("loop_iteration_failed name=portfolio_snapshot_loop")
        try:
            await asyncio.wait_for(stop.wait(), timeout=PORTFOLIO_SNAPSHOT_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


async def reconcile_settled_trades(app: App, now: datetime) -> int:
    cutoff = (now - timedelta(days=_SETTLEMENT_GRACE_DAYS)).date()

    reconciled = 0
    pending = 0
    skipped = 0

    with app.session_factory() as session:
        unreconciled_q = (
            select(PaperTradeRow)
            .where(~PaperTradeRow.id.in_(select(SimulatedPnl.paper_trade_id)))
            .order_by(PaperTradeRow.id.asc())
        )
        rows = list(session.scalars(unreconciled_q).all())
        for r in rows:
            session.expunge(r)

    eligible: list[tuple[PaperTradeRow, ParsedTicker]] = []
    for row in rows:
        try:
            parsed = parse_ticker(row.market_ticker)
        except ValueError as err:
            logger.warning(
                "settlement_unparseable_ticker ticker=%s err=%s",
                row.market_ticker,
                err,
            )
            skipped += 1
            continue
        if parsed.event_date > cutoff:
            continue
        if parsed.series not in STATIONS:
            logger.warning(
                "settlement_unknown_series ticker=%s series=%s",
                row.market_ticker,
                parsed.series,
            )
            skipped += 1
            continue
        eligible.append((row, parsed))

    observed_by_station_date: dict[tuple[str, date], Decimal | None] = {}
    unique_keys: set[tuple[str, date]] = {
        (STATIONS[parsed.series].station, parsed.event_date) for _row, parsed in eligible
    }
    for station, event_date in sorted(unique_keys):
        try:
            observed = await app.acis.fetch_daily_high(station, event_date)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, asyncio.TimeoutError) as err:
            logger.warning(
                "acis_fetch_failed station=%s date=%s err=%s",
                station,
                event_date,
                err,
            )
            continue
        observed_by_station_date[(station, event_date)] = observed
        if observed is None:
            logger.info(
                "settlement_pending station=%s date=%s",
                station,
                event_date.isoformat(),
            )

    async with app.db_lock:
        with app.session_factory() as session:
            for row, parsed in eligible:
                key = (STATIONS[parsed.series].station, parsed.event_date)
                observed = observed_by_station_date.get(key)
                if observed is None:
                    pending += 1
                    continue
                paper_trade_value = PaperTrade(
                    intended_at=row.intended_at,
                    market_ticker=row.market_ticker,
                    side=TradeSide(row.side),
                    contracts=row.contracts,
                    simulated_price=row.simulated_price,
                    fee_dollars=row.fee_dollars,
                    fair_at_entry=row.fair_at_entry,
                    q_raw=row.q_raw,
                    strategy=row.strategy,
                )
                recon = reconcile_trade(paper_trade_value, parsed, observed)
                # paper-mode notional even when the trade was mirrored to a demo
                # order via paper_trades.demo_order_client_id: the simulator scores
                # (sell-YES premium - $1 payoff if YES wins - fees) while demo
                # execution buys NO at no_ask and gains ($1 if NO wins - cost - fees).
                # Demo-cash realized P&L lives on demo_orders.realized_pnl_dollars,
                # populated by the portfolio snapshot loop.
                session.add(
                    SimulatedPnl(
                        paper_trade_id=row.id,
                        settled_at=now,
                        outcome="won" if recon.won else "lost",
                        realized_pnl=recon.realized_pnl,
                    )
                )
                reconciled += 1

            session.commit()

    logger.info(
        "reconcile_settled_trades reconciled=%d pending=%d skipped=%d",
        reconciled,
        pending,
        skipped,
    )
    return reconciled


async def _settlement_loop(app: App, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            now = datetime.now(tz=_timezone.utc)
            async with app.reconcile_lock:
                await reconcile_settled_trades(app, now)
        except Exception:
            logger.exception("loop_iteration_failed name=settlement_loop")
        try:
            await asyncio.wait_for(stop.wait(), timeout=SETTLEMENT_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


async def _on_demand_reconcile(app: App) -> None:
    async with app.reconcile_lock:
        now = datetime.now(tz=_timezone.utc)
        try:
            reconciled = await reconcile_settled_trades(app, now)
            logger.info("reconcile_on_demand reconciled=%d", reconciled)
        except Exception:
            logger.exception("reconcile_on_demand_failed")


def _seconds_until_next_refit(now: datetime) -> float:
    target = now.replace(hour=CALIBRATION_REFIT_TARGET_HOUR_UTC, minute=0, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    base = (target - now).total_seconds()
    jitter = random.uniform(-CALIBRATION_REFIT_JITTER_SECONDS, CALIBRATION_REFIT_JITTER_SECONDS)
    return max(1.0, base + jitter)


async def _calibration_refit_loop(app: App, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=_seconds_until_next_refit(datetime.now(tz=_timezone.utc))
            )
        except asyncio.TimeoutError:
            pass
        if stop.is_set():
            return
        try:
            async with app.db_lock:
                with app.session_factory() as session:
                    new_maps = refit_all(session, prev_maps=app.calibration_maps)
            app.calibration_maps = new_maps
            n_fit = len(new_maps.maps)
            n_skipped = CALIBRATION_REFIT_TOTAL_BUCKETS - n_fit
            tails_keys = [k for k in new_maps.maps if k[0] == "tails"]
            edge_keys = [k for k in new_maps.maps if k[0] == "edge"]
            tails_bs_new = (
                sum(new_maps.holdout_bs_new[k] for k in tails_keys) / Decimal(len(tails_keys))
                if tails_keys
                else Decimal("0")
            )
            tails_bs_prev = (
                sum(new_maps.holdout_bs_prev[k] for k in tails_keys) / Decimal(len(tails_keys))
                if tails_keys
                else Decimal("0")
            )
            edge_bs_new = (
                sum(new_maps.holdout_bs_new[k] for k in edge_keys) / Decimal(len(edge_keys))
                if edge_keys
                else Decimal("0")
            )
            edge_bs_prev = (
                sum(new_maps.holdout_bs_prev[k] for k in edge_keys) / Decimal(len(edge_keys))
                if edge_keys
                else Decimal("0")
            )
            tails_bss_aggregate = new_maps.bss_aggregate_per_stratum.get("tails", BSS_AGGREGATE_NA)
            edge_bss_aggregate = new_maps.bss_aggregate_per_stratum.get("edge", BSS_AGGREGATE_NA)
            logger.info(
                "calibration_refit_complete n_buckets_fit=%d n_buckets_skipped=%d "
                "tails_bss_aggregate=%s edge_bss_aggregate=%s "
                "tails_holdout_bs_new=%s tails_holdout_bs_prev=%s "
                "edge_holdout_bs_new=%s edge_holdout_bs_prev=%s",
                n_fit,
                n_skipped,
                tails_bss_aggregate,
                edge_bss_aggregate,
                tails_bs_new,
                tails_bs_prev,
                edge_bs_new,
                edge_bs_prev,
            )
        except Exception:
            logger.exception("loop_iteration_failed name=calibration_loop")


async def _forecast_loop(app: App, stop: asyncio.Event) -> None:
    last_succeeded = False
    while not stop.is_set():
        try:
            await refresh_forecasts(app)
            last_succeeded = True
        except Exception:
            last_succeeded = False
            logger.exception("loop_iteration_failed name=forecast_loop")
        if stop.is_set():
            return
        if last_succeeded:
            next_cycle = _next_gfs_cycle(datetime.now(tz=_timezone.utc))
            delay = (next_cycle - datetime.now(tz=_timezone.utc)).total_seconds()
        else:
            delay = FORECAST_RETRY_INTERVAL_SECONDS
        if delay > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass


async def run(app: App, duration: timedelta) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass

    sig_usr1 = getattr(signal, "SIGUSR1", None)
    if sig_usr1 is not None:
        try:
            loop.add_signal_handler(
                sig_usr1, lambda: asyncio.create_task(_on_demand_reconcile(app))
            )
        except NotImplementedError:
            pass

    tasks = [
        asyncio.create_task(_forecast_loop(app, stop), name="forecast_loop"),
        asyncio.create_task(_market_loop(app, stop), name="market_loop"),
        asyncio.create_task(_eval_loop(app, stop), name="eval_loop"),
        asyncio.create_task(_settlement_loop(app, stop), name="settlement_loop"),
        asyncio.create_task(_order_reconcile_loop(app, stop), name="order_reconcile_loop"),
        asyncio.create_task(_calibration_refit_loop(app, stop), name="calibration_loop"),
    ]
    if app.settings.mode == "demo":
        tasks.append(
            asyncio.create_task(_portfolio_snapshot_loop(app, stop), name="portfolio_snapshot_loop")
        )

    try:
        await asyncio.wait_for(stop.wait(), timeout=duration.total_seconds())
    except asyncio.TimeoutError:
        stop.set()
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _checkpoint_wal(app.engine)


def _checkpoint_wal(engine: Engine) -> tuple[int, int, int]:
    with engine.connect() as connection:
        row = connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE);").one()
    busy, log_pages, checkpointed_pages = int(row[0]), int(row[1]), int(row[2])
    if busy == 0:
        return busy, log_pages, checkpointed_pages
    # await asyncio.gather cancels _eval_loop mid-session_factory(); the SQLite
    # connection returns to the pool with an implicit read snapshot, so the first
    # wal_checkpoint(TRUNCATE) sees busy=1. Dispose drops the pool and lets the
    # retry succeed against a fresh connection.
    engine.dispose()
    with engine.connect() as connection:
        row = connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE);").one()
    busy, log_pages, checkpointed_pages = int(row[0]), int(row[1]), int(row[2])
    if busy == 1:
        logger.warning(
            "wal_checkpoint_still_busy log_pages=%d checkpointed_pages=%d",
            log_pages,
            checkpointed_pages,
        )
    return busy, log_pages, checkpointed_pages


def _parse_series_arg(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return tuple(STATIONS.keys())
    requested = tuple(s.strip() for s in raw.split(",") if s.strip())
    unknown = [s for s in requested if s not in STATIONS]
    if unknown:
        sys.stderr.write(f"unsupported series: {unknown}; supported: {sorted(STATIONS)}\n")
        sys.exit(2)
    if not requested:
        sys.stderr.write("--series is empty; pass 'all' or a comma-separated list\n")
        sys.exit(2)
    return requested


async def _assert_boot_position_invariant(app: App, *, require_parity: bool) -> None:
    if not require_parity:
        return

    # KW_RESUME_AFTER_CRASH=1 downgrades the mismatch from a hard exit to a
    # WARNING log; the operator has accepted the drift and the recovery script
    # would be a no-op.
    suppressed = os.environ.get("KW_RESUME_AFTER_CRASH") == "1"
    positions = await poll_positions(app.kalshi)
    nonzero = [p for p in positions if p.position_fp != 0]

    violations: list[tuple[str, int, int]] = []
    with app.session_factory() as session:
        for pos in nonzero:
            local = local_position_aggregate(session, pos.ticker)
            if local != pos.position_fp:
                violations.append((pos.ticker, pos.position_fp, local))

    if not violations:
        logger.info("boot_position_invariant_ok positions=%d", len(nonzero))
        return

    level = logging.WARNING if suppressed else logging.ERROR
    for ticker, kalshi_fp, local in violations:
        logger.log(
            level,
            "boot_position_invariant_violated ticker=%s kalshi=%d local=%d",
            ticker,
            kalshi_fp,
            local,
        )
    if suppressed:
        return
    logger.error(
        "boot_position_invariant_recovery_hint cmd=scripts/reconcile/backfill_from_kalshi.py"
    )
    sys.exit(1)


async def _demo_startup_backfill(app: App) -> None:
    balance = await app.kalshi.get_balance()
    logger.info("demo_account_balance balance_dollars=%s", balance)
    if balance <= Decimal("0"):
        raise RuntimeError("demo account has no balance; fund via demo UI before MODE=demo")
    app.bankroll = balance
    now = datetime.now(tz=_timezone.utc)
    persisted = _load_reconciler_watermark(app)
    if persisted is None:
        watermark = now - _COLD_START_FALLBACK
        reason = "no_persisted_watermark"
    elif now - persisted > _COLD_START_GAP:
        watermark = persisted
        reason = "stale_persisted_watermark"
    else:
        watermark = persisted
        reason = "recent_persisted_watermark"
    logger.info(
        "demo_startup_backfill watermark=%s reason=%s",
        watermark.isoformat(),
        reason,
    )
    await _reconcile_once(app, watermark)


def main() -> None:
    parser = argparse.ArgumentParser(prog="bot.main")
    parser.add_argument("--mode", choices=["paper", "demo"], default=None)
    parser.add_argument("--series", default="all")
    parser.add_argument("--duration", default="24h")
    parser.add_argument(
        "--require-position-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "assert every Kalshi position has a matching local demo_orders row at boot; "
            "set KW_RESUME_AFTER_CRASH=1 to log mismatches and continue instead of exiting"
        ),
    )
    args = parser.parse_args()

    series_list = _parse_series_arg(args.series)
    duration = _parse_duration(args.duration)

    settings = get_settings()
    if args.mode is not None and args.mode != settings.mode:
        settings = Settings.model_validate({**settings.model_dump(), "mode": args.mode})
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    upgrade_schema("data/state.db")
    engine = make_engine("data/state.db")
    session_factory = make_session_factory(engine)
    meteo = OpenMeteoClient()
    kalshi = KalshiDemoClient(settings)
    acis = ACISClient()

    app = App(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        meteo=meteo,
        kalshi=kalshi,
        acis=acis,
        series_list=series_list,
    )

    async def _go() -> None:
        await app.kalshi.aopen()
        if app.settings.mode == "demo":
            await _demo_startup_backfill(app)
            await _assert_boot_position_invariant(app, require_parity=args.require_position_parity)
        try:
            await run(app, duration)
        finally:
            await app.aclose()

    asyncio.run(_go())


if __name__ == "__main__":
    main()
