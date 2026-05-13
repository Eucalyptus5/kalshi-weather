from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import signal
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal

import numpy as np
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from bot.config import Settings, get_settings
from bot.execution.paper import Orderbook, PaperTrade, TradeIntent, TradeSide, simulate_taker_fill
from bot.forecast.cdf import EnsembleCDF
from bot.forecast.open_meteo import OpenMeteoClient, StationForecast
from bot.kalshi_client import KalshiDemoClient, KalshiMarket, KalshiOrderbook
from bot.markets.observation_window import observation_window
from bot.markets.parser import ParsedTicker, parse_ticker
from bot.risk.gates import GateContext, GateMode, evaluate as evaluate_gates
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
from bot.validation.reconcile import ACISClient, reconcile_trade

logger = logging.getLogger(__name__)


PAPER_BANKROLL: Decimal = Decimal("500")
REQUIRED_CUSHION: Decimal = Decimal("100")
MARKET_POSITION_CAP: Decimal = Decimal("250")
SERIES_POSITION_CAP: Decimal = Decimal("400")

MARKET_REFRESH_INTERVAL = 60.0
EVAL_INTERVAL = 60.0
SETTLEMENT_INTERVAL_SECONDS: float = 6 * 3600
_SETTLEMENT_GRACE_DAYS: int = 1
GFS_CYCLES_HOURS: tuple[int, ...] = (0, 6, 12, 18)
GFS_CYCLE_OFFSET_MINUTES = 30
FORECAST_RETRY_INTERVAL_SECONDS: float = 60.0


@dataclass(frozen=True, slots=True)
class _StationConfig:
    station: str
    latitude: float
    longitude: float
    timezone: str


STATIONS: dict[str, _StationConfig] = {
    "KXHIGHDEN": _StationConfig(
        station="KDEN",
        latitude=39.8466,
        longitude=-104.6562,
        timezone="America/Denver",
    ),
}


@dataclass
class App:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    meteo: OpenMeteoClient
    kalshi: KalshiDemoClient
    acis: ACISClient
    series: str
    db_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    forecast_cdfs: dict[tuple[str, date], EnsembleCDF] = field(default_factory=dict)
    ensemble_spreads: dict[tuple[str, date], Decimal] = field(default_factory=dict)
    forecast_run_times: dict[tuple[str, date], datetime] = field(default_factory=dict)
    latest_markets: dict[str, KalshiMarket] = field(default_factory=dict)
    latest_orderbooks: dict[str, KalshiOrderbook] = field(default_factory=dict)

    async def aclose(self) -> None:
        await self.meteo.aclose()
        await self.kalshi.aclose()
        await self.acis.aclose()
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
    cfg = STATIONS[app.series]
    forecast = await app.meteo.fetch_station(
        station=cfg.station,
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        timezone=cfg.timezone,
    )
    async with app.db_lock:
        return _persist_forecast(app, forecast)


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
    markets = await app.kalshi.list_open_markets_for_series(app.series)
    now = datetime.now(tz=_timezone.utc)
    pairs: list[tuple[KalshiMarket, KalshiOrderbook]] = []
    for m in markets:
        book = await app.kalshi.get_orderbook(m.ticker)
        pairs.append((m, book))
    n = 0
    async with app.db_lock:
        with app.session_factory() as session:
            for m, book in pairs:
                parsed = parse_ticker(m.ticker)
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
                    )
                )
                app.latest_markets[m.ticker] = m
                app.latest_orderbooks[m.ticker] = book
                n += 1
            session.commit()
    logger.info("refresh_markets series=%s pairs=%d", app.series, n)
    return n


async def evaluate_strategies(app: App, now: datetime) -> int:
    cfg = STATIONS[app.series]
    n_trades = 0
    async with app.db_lock:
        with app.session_factory() as session:
            for ticker, market in app.latest_markets.items():
                book = app.latest_orderbooks.get(ticker)
                if book is None:
                    continue
                parsed = parse_ticker(ticker)
                if parsed.is_tail:
                    logger.debug("skip_tail ticker=%s", ticker)
                    continue

                cdf_key = (cfg.station, parsed.event_date)
                cdf = app.forecast_cdfs.get(cdf_key)
                if cdf is None:
                    continue
                spread = app.ensemble_spreads[cdf_key]
                run_time = app.forecast_run_times[cdf_key]

                lo = float(parsed.strikes[0])
                hi = float(parsed.strikes[1])
                fair_yes = Decimal(str(cdf.prob_range(lo, hi)))

                start_utc, end_utc = observation_window(cfg.timezone, parsed.event_date)
                is_same_day = start_utc <= now < end_utc

                mid = (book.yes_ask + book.yes_bid) / Decimal("2")

                for intent in _build_intents(
                    ticker=ticker,
                    market=market,
                    book=book,
                    fair_yes=fair_yes,
                    spread=spread,
                    mid=mid,
                    is_same_day=is_same_day,
                    now=now,
                ):
                    gate_ctx = _gate_ctx_for(
                        intent=intent,
                        market=market,
                        fair_yes=fair_yes,
                        spread=spread,
                        mid=mid,
                        run_time=run_time,
                        now=now,
                    )
                    check = evaluate_gates(gate_ctx, GateMode.PAPER)
                    for failure in check.failures:
                        session.add(
                            GateFailure(
                                evaluated_at=now,
                                gate_name=failure.name,
                                reason=failure.reason or "",
                                mode="paper",
                                market_ticker=ticker,
                            )
                        )
                    if not check.overall_passed:
                        continue

                    trade = simulate_taker_fill(intent, Orderbook(book.yes_ask, book.yes_bid), now)
                    session.add(_paper_trade_row(trade))
                    n_trades += 1
            session.commit()
    logger.info("evaluate_strategies trades=%d markets=%d", n_trades, len(app.latest_markets))
    return n_trades


def _build_intents(
    *,
    ticker: str,
    market: KalshiMarket,
    book: KalshiOrderbook,
    fair_yes: Decimal,
    spread: Decimal,
    mid: Decimal,
    is_same_day: bool,
    now: datetime,
) -> list[TradeIntent]:
    intents: list[TradeIntent] = []

    edge_ctx = edge_strategy.EdgeContext(
        yes_ask=book.yes_ask,
        yes_bid=book.yes_bid,
        fair_yes=fair_yes,
        ensemble_spread=spread,
        bankroll=PAPER_BANKROLL,
        is_same_day=is_same_day,
        is_blacklisted=False,
        nbm_divergence=None,
    )
    edge_sig = edge_strategy.evaluate(edge_ctx)
    if edge_sig.action is edge_strategy.EdgeAction.BUY_YES:
        intents.append(
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.BUY_YES,
                contracts=edge_sig.contracts,
                fair_yes=fair_yes,
                strategy="edge",
            )
        )
    elif edge_sig.action is edge_strategy.EdgeAction.SELL_YES:
        intents.append(
            TradeIntent(
                market_ticker=ticker,
                side=TradeSide.SELL_YES,
                contracts=edge_sig.contracts,
                fair_yes=fair_yes,
                strategy="edge",
            )
        )

    if market.close_time is not None:
        tails_ctx = tails_strategy.TailsContext(
            yes_ask=book.yes_ask,
            yes_bid=book.yes_bid,
            no_bid=book.no_bid,
            fair_yes=fair_yes,
            close_time=market.close_time,
            now=now,
            bankroll=PAPER_BANKROLL,
            is_same_day=is_same_day,
        )
        tails_sig = tails_strategy.evaluate(tails_ctx)
        if tails_sig.action is tails_strategy.TailsAction.SELL_YES:
            intents.append(
                TradeIntent(
                    market_ticker=ticker,
                    side=TradeSide.SELL_YES,
                    contracts=tails_sig.contracts,
                    fair_yes=fair_yes,
                    strategy="tails",
                )
            )

    return intents


def _gate_ctx_for(
    *,
    intent: TradeIntent,
    market: KalshiMarket,
    fair_yes: Decimal,
    spread: Decimal,
    mid: Decimal,
    run_time: datetime,
    now: datetime,
) -> GateContext:
    if intent.side is TradeSide.BUY_YES:
        edge_dollars = fair_yes - mid
    else:
        edge_dollars = mid - fair_yes
    cost_per_contract = (
        market.yes_ask if intent.side is TradeSide.BUY_YES else (Decimal("1") - market.yes_bid)
    )
    order_dollars = cost_per_contract * Decimal(intent.contracts)
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
        order_size_dollars=order_dollars,
        market_position_cap=MARKET_POSITION_CAP,
        series_existing_dollars=Decimal("0"),
        series_position_cap=SERIES_POSITION_CAP,
        account_balance=PAPER_BANKROLL,
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
        strategy=trade.strategy,
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


async def reconcile_settled_trades(app: App, now: datetime) -> int:
    cutoff = (now - timedelta(days=_SETTLEMENT_GRACE_DAYS)).date()
    cfg = STATIONS[app.series]

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
        if parsed.event_date >= cutoff:
            continue
        eligible.append((row, parsed))

    observed_by_date: dict[date, Decimal | None] = {}
    unique_dates = sorted({parsed.event_date for _row, parsed in eligible})
    for event_date in unique_dates:
        observed = await app.acis.fetch_daily_high(cfg.station, event_date)
        observed_by_date[event_date] = observed
        if observed is None:
            logger.info(
                "settlement_pending station=%s date=%s",
                cfg.station,
                event_date.isoformat(),
            )

    async with app.db_lock:
        with app.session_factory() as session:
            for row, parsed in eligible:
                observed = observed_by_date[parsed.event_date]
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
                    strategy=row.strategy,
                )
                recon = reconcile_trade(paper_trade_value, parsed, observed)
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
            await reconcile_settled_trades(app, now)
        except Exception:
            logger.exception("loop_iteration_failed name=settlement_loop")
        try:
            await asyncio.wait_for(stop.wait(), timeout=SETTLEMENT_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


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

    tasks = [
        asyncio.create_task(_forecast_loop(app, stop), name="forecast_loop"),
        asyncio.create_task(_market_loop(app, stop), name="market_loop"),
        asyncio.create_task(_eval_loop(app, stop), name="eval_loop"),
        asyncio.create_task(_settlement_loop(app, stop), name="settlement_loop"),
    ]

    try:
        await asyncio.wait_for(stop.wait(), timeout=duration.total_seconds())
    except asyncio.TimeoutError:
        stop.set()
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="bot.main")
    parser.add_argument("--mode", choices=["paper"], default="paper")
    parser.add_argument("--series", default="KXHIGHDEN")
    parser.add_argument("--duration", default="24h")
    args = parser.parse_args()

    if args.series not in STATIONS:
        sys.stderr.write(f"unsupported series {args.series!r}; supported: {sorted(STATIONS)}\n")
        sys.exit(2)

    duration = _parse_duration(args.duration)

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    engine = make_engine("data/state.db")
    Base.metadata.create_all(engine)
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
        series=args.series,
    )

    async def _go() -> None:
        await app.kalshi.aopen()
        try:
            await run(app, duration)
        finally:
            await app.aclose()

    asyncio.run(_go())


if __name__ == "__main__":
    main()
