from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

import bot.backtest.engine as engine_mod
import bot.main as bot_main
from bot.backtest.engine import BacktestConfig, ReplayLog, ReplaySnapshot, run_replay
from bot.backtest.forecast_replay import StationSpec
from bot.backtest.normalize import CanonicalSnapshot
from bot.config import Settings
from bot.forecast.cdf import EnsembleCDF
from bot.kalshi_client import KalshiMarket, KalshiOrderbook
from bot.main import App, evaluate_strategies
from bot.markets.parser import event_id, parse_ticker, series_id
from bot.risk.gates import CAP_GATE_NAMES
from bot.storage.sqlite import (
    Base,
    GateFailure,
    PaperTradeRow,
    make_engine,
    make_session_factory,
)
from sqlalchemy import select

_LEAD = timedelta(hours=24)
_BANKROLL = Decimal("500")
_STATION = "KDEN"
_SERIES = "KXHIGHDEN"
_MEMBERS = 75.0 + 3.0 * np.array([-2, -1, -0.5, 0, 0.5, 1, 2], dtype=np.float64)
_YES_ASK = Decimal("0.10")
_YES_BID = Decimal("0.08")
_DEPTH = 100
_FILL_DEPTH = 1000
_SERIES_CAP = Decimal("12")
_OPEN_CAP = Decimal("500")
_AGGREGATE_CAP = Decimal("500")


@dataclass(frozen=True, slots=True)
class CycleMarket:
    ticker: str
    event_date: date
    close_time: datetime
    as_of: datetime
    run_time: datetime


@dataclass(frozen=True, slots=True)
class ParityTuple:
    market_ticker: str
    side: str | None
    contracts: int | None
    gate_outcome: str
    cap_failure_name: str | None


class _FixedReplay:
    def __init__(self, cdf: EnsembleCDF) -> None:
        self._cdf = cdf

    async def replay(self, station: StationSpec, valid_date: date, as_of: datetime) -> EnsembleCDF:
        return self._cdf


def make_three_cycle_overlay_bind_fixture() -> list[CycleMarket]:
    markets: list[CycleMarket] = []
    for i in range(3):
        event_date = date(2026, 6, 20 + i)
        close_time = datetime(
            event_date.year, event_date.month, event_date.day, 11, 0, tzinfo=timezone.utc
        )
        as_of = close_time - _LEAD
        markets.append(
            CycleMarket(
                ticker=f"{_SERIES}-26JUN{20 + i}-T70-80",
                event_date=event_date,
                close_time=close_time,
                as_of=as_of,
                run_time=as_of - timedelta(hours=5),
            )
        )
    return markets


def _fixture_cdf() -> EnsembleCDF:
    return EnsembleCDF.from_members(_MEMBERS, smoothing=1.0)


def _patch_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bot_main, "market_position_cap", lambda app=None: _OPEN_CAP)
    monkeypatch.setattr(bot_main, "event_position_cap", lambda app=None: _OPEN_CAP)
    monkeypatch.setattr(bot_main, "series_position_cap", lambda app=None: _SERIES_CAP)
    monkeypatch.setattr(bot_main, "aggregate_exposure_cap", lambda app=None: _AGGREGATE_CAP)
    monkeypatch.setattr(engine_mod, "MARKET_POSITION_FRAC", _OPEN_CAP / _BANKROLL)
    monkeypatch.setattr(engine_mod, "EVENT_POSITION_FRAC", _OPEN_CAP / _BANKROLL)
    monkeypatch.setattr(engine_mod, "SERIES_POSITION_FRAC", _SERIES_CAP / _BANKROLL)
    monkeypatch.setattr(engine_mod, "AGGREGATE_EXPOSURE_FRAC", _AGGREGATE_CAP / _BANKROLL)


def _live_book(ticker: str, snapshot_at: datetime) -> KalshiOrderbook:
    return KalshiOrderbook(
        ticker=ticker,
        yes_ask=_YES_ASK,
        yes_bid=_YES_BID,
        no_ask=Decimal("1") - _YES_BID,
        no_bid=Decimal("1") - _YES_ASK,
        yes_ask_depth=_FILL_DEPTH,
        yes_bid_depth=_FILL_DEPTH,
        no_ask_depth=_DEPTH,
        no_bid_depth=_DEPTH,
        snapshot_at=snapshot_at,
    )


def _live_market(cm: CycleMarket) -> KalshiMarket:
    return KalshiMarket(
        ticker=cm.ticker,
        event_ticker=event_id(cm.ticker),
        series=series_id(cm.ticker),
        status="active",
        close_time=cm.close_time,
        yes_ask=_YES_ASK,
        yes_bid=_YES_BID,
    )


async def _live_tuples(
    markets: list[CycleMarket], monkeypatch: pytest.MonkeyPatch
) -> list[ParityTuple]:
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    app = App(
        settings=Settings(mode="paper"),
        engine=engine,
        session_factory=make_session_factory(engine),
        meteo=None,  # type: ignore[arg-type]
        kalshi=None,  # type: ignore[arg-type]
        acis=None,  # type: ignore[arg-type]
        series_list=(_SERIES,),
        bankroll=_BANKROLL,
    )
    cdf = _fixture_cdf()
    spread = Decimal(str(float(_MEMBERS.std())))
    _patch_caps(monkeypatch)

    for cm in markets:
        key = (_STATION, cm.event_date)
        app.forecast_cdfs[key] = cdf
        app.ensemble_spreads[key] = spread
        app.forecast_run_times[key] = cm.run_time
        app.latest_markets = {cm.ticker: _live_market(cm)}
        app.latest_orderbooks = {cm.ticker: _live_book(cm.ticker, cm.as_of - timedelta(seconds=1))}
        await evaluate_strategies(app, cm.as_of)

    with app.session_factory() as session:
        trades = {t.market_ticker: t for t in session.scalars(select(PaperTradeRow)).all()}
        cap_fail_by_ticker: dict[str, str] = {}
        for f in session.scalars(select(GateFailure)).all():
            if f.gate_name in CAP_GATE_NAMES and f.market_ticker is not None:
                cap_fail_by_ticker[f.market_ticker] = f.gate_name
    app.engine.dispose()

    out: list[ParityTuple] = []
    for cm in markets:
        trade = trades.get(cm.ticker)
        if trade is not None:
            out.append(ParityTuple(cm.ticker, trade.side, trade.contracts, "filled", None))
        else:
            out.append(
                ParityTuple(cm.ticker, None, None, "skipped", cap_fail_by_ticker.get(cm.ticker))
            )
    return out


def _canonical_snapshot(cm: CycleMarket) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        ticker=cm.ticker,
        event_ticker=event_id(cm.ticker),
        series_ticker=series_id(cm.ticker),
        status="active",
        result="",
        yes_ask=_YES_ASK,
        yes_bid=_YES_BID,
        no_ask=Decimal("1") - _YES_BID,
        no_bid=Decimal("1") - _YES_ASK,
        last_price=_YES_BID,
        volume=Decimal(0),
        volume_24h=Decimal(0),
        open_interest=Decimal(0),
        close_time=cm.close_time,
        yes_bid_size=Decimal(_DEPTH),
        no_bid_size=Decimal(_DEPTH),
    )


async def _backtest_tuples(
    markets: list[CycleMarket], monkeypatch: pytest.MonkeyPatch
) -> list[ParityTuple]:
    _patch_caps(monkeypatch)
    snapshots = [
        ReplaySnapshot(
            snapshot_at=cm.close_time - _LEAD - timedelta(minutes=1), snap=_canonical_snapshot(cm)
        )
        for cm in markets
    ]
    config = BacktestConfig(lead=_LEAD, bankroll=_BANKROLL)
    log = ReplayLog()
    orders = await run_replay(snapshots, _FixedReplay(_fixture_cdf()), "edge", config, log=log)

    filled = {o.market_ticker: o for o in orders}
    cap_fail_by_ticker: dict[str, str] = {}
    for f in log.failures:
        if f.name in CAP_GATE_NAMES:
            cap_fail_by_ticker[f.market_ticker] = f.name

    out: list[ParityTuple] = []
    for cm in markets:
        order = filled.get(cm.ticker)
        if order is not None:
            out.append(ParityTuple(cm.ticker, order.action.value, order.contracts, "filled", None))
        else:
            out.append(
                ParityTuple(cm.ticker, None, None, "skipped", cap_fail_by_ticker.get(cm.ticker))
            )
    return out


def test_seed_overlays_match_open_exposures_for_a_buy_yes_fill() -> None:
    parsed = parse_ticker(f"{_SERIES}-26JUN20-T70-80")
    contracts = 50
    open_exposures_max_loss = _YES_ASK * Decimal(contracts)
    ledger_delta = bot_main.paper_collateral_per_contract(
        bot_main.TradeSide.BUY_YES,
        KalshiOrderbook(
            ticker=parsed.raw,
            yes_ask=_YES_ASK,
            yes_bid=_YES_BID,
            no_ask=Decimal("1") - _YES_BID,
            no_bid=Decimal("1") - _YES_ASK,
            yes_ask_depth=_FILL_DEPTH,
            yes_bid_depth=_FILL_DEPTH,
            no_ask_depth=_DEPTH,
            no_bid_depth=_DEPTH,
            snapshot_at=datetime(2026, 6, 19, 11, 0, tzinfo=timezone.utc),
        ),
    ) * Decimal(contracts)
    assert open_exposures_max_loss == ledger_delta


async def test_main_loop_and_backtest_engine_agree_on_decision_tuples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markets = make_three_cycle_overlay_bind_fixture()

    live = await _live_tuples(markets, monkeypatch)
    backtest = await _backtest_tuples(markets, monkeypatch)

    assert live == backtest

    cap_failures = [t for t in live if t.cap_failure_name is not None]
    assert cap_failures, "fixture did not bind a cap gate on any cycle"
    assert all(t.cap_failure_name in CAP_GATE_NAMES for t in cap_failures)
    assert cap_failures[0].cap_failure_name == "within_series_cap"
    assert cap_failures[0].market_ticker == markets[2].ticker

    fills = [t for t in live if t.gate_outcome == "filled"]
    assert [t.market_ticker for t in fills] == [markets[0].ticker, markets[1].ticker]
    assert all(t.side == bot_main.TradeSide.BUY_YES.value for t in fills)
