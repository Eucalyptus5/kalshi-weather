from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from bot.backtest.context import BacktestBudgets, build_edge_context, build_tails_context
from bot.backtest.normalize import CanonicalSnapshot
from bot.forecast.cdf import EnsembleCDF
from bot.strategy import edge as edge_strategy
from bot.strategy import tails as tails_strategy


_AS_OF = datetime(2026, 1, 14, 22, 0, tzinfo=timezone.utc)
_CLOSE_TIME = datetime(2026, 1, 15, 2, 0, tzinfo=timezone.utc)
_STATION_TZ = "America/Denver"
_BANKROLL = Decimal("20000")
_BUDGETS = BacktestBudgets(
    event_budget_remaining=Decimal("600"),
    market_budget_remaining=Decimal("300"),
)
_SPREAD = Decimal("2.5")


def make_canonical_snapshot(
    ticker: str,
    *,
    yes_ask: Decimal,
    yes_bid: Decimal,
    no_bid: Decimal,
    no_ask: Decimal | None = None,
) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0],
        series_ticker=ticker.split("-", 1)[0],
        status="active",
        result="",
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=no_ask if no_ask is not None else Decimal("1") - yes_bid,
        no_bid=no_bid,
        last_price=yes_bid,
        volume=Decimal(0),
        volume_24h=Decimal(0),
        open_interest=Decimal(0),
        close_time=_CLOSE_TIME,
    )


def make_above_cdf(target_above: float) -> EnsembleCDF:
    if target_above < 0.05:
        members = np.array([72.5, 73.0, 73.5])
    else:
        members = np.array([73.5, 73.7, 73.8])
    return EnsembleCDF.from_members(members, smoothing=1.0)


def make_bracket_cdf_centered_on(center: float, smoothing: float = 0.65) -> EnsembleCDF:
    return EnsembleCDF.from_members(np.array([center]), smoothing=smoothing)


def test_build_tails_context_drives_sell_yes() -> None:
    snap = make_canonical_snapshot(
        "KXHIGHDEN-26JAN15-T75",
        yes_ask=Decimal("0.15"),
        yes_bid=Decimal("0.12"),
        no_bid=Decimal("0.85"),
    )
    cdf = make_above_cdf(target_above=0.04)

    ctx = build_tails_context(
        snap,
        cdf,
        _AS_OF,
        _BANKROLL,
        _BUDGETS,
        ensemble_spread=_SPREAD,
        depth_at_price=100,
        station_tz=_STATION_TZ,
    )

    assert ctx.yes_ask == Decimal("0.15")
    assert ctx.yes_bid == Decimal("0.12")
    assert ctx.no_bid == Decimal("0.85")
    assert ctx.fair_yes < Decimal("0.07")
    assert ctx.close_time == _CLOSE_TIME
    assert ctx.now == _AS_OF
    assert ctx.is_same_day is False
    assert ctx.ensemble_spread == _SPREAD
    assert ctx.depth_at_price == 100
    assert ctx.price_per_contract == Decimal("0.88")

    sig = tails_strategy.evaluate(ctx)
    assert sig.action is tails_strategy.TailsAction.SELL_YES
    assert sig.reason == "trade"
    assert sig.contracts >= 1


def test_build_tails_context_high_fair_skips() -> None:
    snap = make_canonical_snapshot(
        "KXHIGHDEN-26JAN15-T75",
        yes_ask=Decimal("0.15"),
        yes_bid=Decimal("0.12"),
        no_bid=Decimal("0.85"),
    )
    cdf = make_above_cdf(target_above=0.09)

    ctx = build_tails_context(
        snap,
        cdf,
        _AS_OF,
        _BANKROLL,
        _BUDGETS,
        ensemble_spread=_SPREAD,
        depth_at_price=100,
        station_tz=_STATION_TZ,
    )

    assert ctx.fair_yes >= Decimal("0.07")
    sig = tails_strategy.evaluate(ctx)
    assert sig.action is tails_strategy.TailsAction.SKIP
    assert sig.reason == "fair_too_high"


def test_build_edge_context_drives_buy_yes() -> None:
    snap = make_canonical_snapshot(
        "KXHIGHDEN-26JAN15-B61.5",
        yes_ask=Decimal("0.40"),
        yes_bid=Decimal("0.38"),
        no_bid=Decimal("0.60"),
    )
    cdf = make_bracket_cdf_centered_on(61.5, smoothing=0.65)

    ctx = build_edge_context(
        snap,
        cdf,
        _AS_OF,
        _BANKROLL,
        _BUDGETS,
        ensemble_spread=_SPREAD,
        buy_yes_depth=100,
        sell_yes_depth=80,
        station_tz=_STATION_TZ,
    )

    mid = (ctx.yes_ask + ctx.yes_bid) / Decimal("2")
    assert abs(ctx.fair_yes - mid) > Decimal("0.08")
    assert ctx.fair_yes > ctx.yes_ask + Decimal("0.04")
    assert ctx.is_blacklisted is False
    assert ctx.depth_at_price == 100
    assert ctx.price_per_contract == Decimal("0.40")

    sig = edge_strategy.evaluate(ctx)
    assert sig.action is edge_strategy.EdgeAction.BUY_YES
    assert sig.reason == "trade_buy"
    assert sig.contracts >= 1


def test_build_edge_context_sell_yes_direction_picks_sell_depth() -> None:
    snap = make_canonical_snapshot(
        "KXHIGHDEN-26JAN15-B61.5",
        yes_ask=Decimal("0.60"),
        yes_bid=Decimal("0.58"),
        no_bid=Decimal("0.40"),
    )
    cdf = EnsembleCDF.from_members(np.array([45.0]), smoothing=1.0)

    ctx = build_edge_context(
        snap,
        cdf,
        _AS_OF,
        _BANKROLL,
        _BUDGETS,
        ensemble_spread=_SPREAD,
        buy_yes_depth=100,
        sell_yes_depth=80,
        station_tz=_STATION_TZ,
    )

    assert ctx.fair_yes < ctx.yes_bid - Decimal("0.04")
    assert ctx.depth_at_price == 80
    assert ctx.price_per_contract == Decimal("1") - Decimal("0.58")


def test_build_edge_context_propagates_blacklist() -> None:
    snap = make_canonical_snapshot(
        "KXHIGHLAX-26JAN15-B61.5",
        yes_ask=Decimal("0.40"),
        yes_bid=Decimal("0.38"),
        no_bid=Decimal("0.60"),
    )
    cdf = make_bracket_cdf_centered_on(61.5, smoothing=0.65)

    ctx = build_edge_context(
        snap,
        cdf,
        _AS_OF,
        _BANKROLL,
        _BUDGETS,
        ensemble_spread=_SPREAD,
        buy_yes_depth=100,
        sell_yes_depth=80,
        station_tz="America/Los_Angeles",
        is_blacklisted=True,
    )

    assert ctx.is_blacklisted is True
    sig = edge_strategy.evaluate(ctx)
    assert sig.action is edge_strategy.EdgeAction.SKIP
    assert sig.reason == "blacklisted"


def test_build_tails_context_sigma_t_median_keys_off_lead() -> None:
    snap = make_canonical_snapshot(
        "KXHIGHDEN-26JAN15-T75",
        yes_ask=Decimal("0.15"),
        yes_bid=Decimal("0.12"),
        no_bid=Decimal("0.85"),
    )
    cdf = make_above_cdf(target_above=0.04)

    ctx_short = build_tails_context(
        snap,
        cdf,
        _AS_OF,
        _BANKROLL,
        _BUDGETS,
        ensemble_spread=_SPREAD,
        depth_at_price=100,
        station_tz=_STATION_TZ,
    )
    assert ctx_short.sigma_T_median == Decimal("1.60")

    snap_far = CanonicalSnapshot(**{**snap.model_dump(), "close_time": _AS_OF.replace(day=21)})
    ctx_far = build_tails_context(
        snap_far,
        cdf,
        _AS_OF,
        _BANKROLL,
        _BUDGETS,
        ensemble_spread=_SPREAD,
        depth_at_price=100,
        station_tz=_STATION_TZ,
    )
    assert ctx_far.sigma_T_median == Decimal("4.47")


def test_build_tails_context_requires_close_time() -> None:
    snap = CanonicalSnapshot(
        ticker="KXHIGHDEN-26JAN15-T75",
        event_ticker="KXHIGHDEN-26JAN15",
        series_ticker="KXHIGHDEN",
        status="active",
        result="",
        yes_ask=Decimal("0.15"),
        yes_bid=Decimal("0.12"),
        no_ask=Decimal("0.16"),
        no_bid=Decimal("0.85"),
        last_price=Decimal("0.13"),
        volume=Decimal(0),
        volume_24h=Decimal(0),
        open_interest=Decimal(0),
        close_time=None,
    )
    cdf = make_above_cdf(target_above=0.04)

    with pytest.raises(ValueError, match="close_time"):
        build_tails_context(
            snap,
            cdf,
            _AS_OF,
            _BANKROLL,
            _BUDGETS,
            ensemble_spread=_SPREAD,
            depth_at_price=100,
            station_tz=_STATION_TZ,
        )
