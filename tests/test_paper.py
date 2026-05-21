from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import bot.execution.paper
import bot.main
import tests.test_paper as _self_module
from bot.execution.fees import taker_fee
from bot.execution.paper import (
    MARKET_REFRESH_INTERVAL_SECONDS,
    STALE_ORDERBOOK_THRESHOLD_SECONDS,
    Orderbook,
    PaperTrade,
    TradeIntent,
    TradeSide,
    log_stale_skip_ratio,
    simulate_taker_fill,
    trade_side_for_demo,
)


def _intent(**overrides: object) -> TradeIntent:
    base: dict[str, object] = {
        "market_ticker": "KXHIGHDEN-26MAY05-T80",
        "side": TradeSide.BUY_YES,
        "contracts": 10,
        "fair_yes": Decimal("0.50"),
        "strategy": "edge",
        "ensemble_spread_sigma_t": None,
        "lead_time_hours": None,
        "nbm_divergence": None,
    }
    base.update(overrides)
    return TradeIntent(**base)  # type: ignore[arg-type]


def _book(**overrides: object) -> Orderbook:
    base: dict[str, object] = {
        "yes_ask": Decimal("0.40"),
        "yes_bid": Decimal("0.38"),
        "yes_ask_depth": 1000,
        "yes_bid_depth": 1000,
        "snapshot_at": _now() - timedelta(seconds=1),
    }
    base.update(overrides)
    return Orderbook(**base)  # type: ignore[arg-type]


def _now() -> datetime:
    return datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)


def test_buy_yes_fills_at_yes_ask() -> None:
    trade = simulate_taker_fill(_intent(), _book(), _now())
    assert trade.simulated_price == Decimal("0.40")


def test_buy_yes_fee_uses_simulated_price() -> None:
    trade = simulate_taker_fill(_intent(contracts=10), _book(yes_ask=Decimal("0.40")), _now())
    assert trade.fee_dollars == taker_fee(10, Decimal("0.40"))


def test_buy_yes_preserves_intent_fields() -> None:
    now = _now()
    intent = _intent(
        market_ticker="KXHIGHDEN-26MAY05-T80",
        contracts=10,
        fair_yes=Decimal("0.50"),
        strategy="edge",
    )
    trade = simulate_taker_fill(intent, _book(), now)
    assert trade.market_ticker == "KXHIGHDEN-26MAY05-T80"
    assert trade.side is TradeSide.BUY_YES
    assert trade.contracts == 10
    assert trade.fair_at_entry == Decimal("0.50")
    assert trade.strategy == "edge"
    assert trade.intended_at == now


def test_sell_yes_fills_at_yes_bid() -> None:
    trade = simulate_taker_fill(
        _intent(side=TradeSide.SELL_YES, contracts=20, fair_yes=Decimal("0.20")),
        _book(yes_ask=Decimal("0.32"), yes_bid=Decimal("0.30")),
        _now(),
    )
    assert trade.simulated_price == Decimal("0.30")


def test_sell_yes_fee_uses_bid() -> None:
    trade = simulate_taker_fill(
        _intent(side=TradeSide.SELL_YES, contracts=20, fair_yes=Decimal("0.20")),
        _book(yes_ask=Decimal("0.32"), yes_bid=Decimal("0.30")),
        _now(),
    )
    assert trade.fee_dollars == taker_fee(20, Decimal("0.30"))


def test_decimal_hygiene() -> None:
    trade = simulate_taker_fill(_intent(), _book(), _now())
    assert isinstance(trade.simulated_price, Decimal)
    assert isinstance(trade.fee_dollars, Decimal)
    assert isinstance(trade.fair_at_entry, Decimal)
    assert isinstance(trade.intended_at, datetime)
    assert trade.intended_at.tzinfo is not None


def test_zero_contracts_raises() -> None:
    with pytest.raises(ValueError):
        simulate_taker_fill(_intent(contracts=0), _book(), _now())


def test_negative_contracts_raises() -> None:
    with pytest.raises(ValueError):
        simulate_taker_fill(_intent(contracts=-5), _book(), _now())


def test_naive_datetime_raises() -> None:
    naive_now = datetime(2026, 5, 5, 12, 0)
    with pytest.raises(ValueError):
        simulate_taker_fill(_intent(), _book(), naive_now)


def test_tail_price_buy() -> None:
    trade = simulate_taker_fill(
        _intent(contracts=10, fair_yes=Decimal("0.03")),
        _book(yes_ask=Decimal("0.05"), yes_bid=Decimal("0.03")),
        _now(),
    )
    assert trade.simulated_price == Decimal("0.05")
    assert trade.fee_dollars == taker_fee(10, Decimal("0.05"))


def test_paper_trade_is_frozen() -> None:
    trade = simulate_taker_fill(_intent(), _book(), _now())
    assert isinstance(trade, PaperTrade)
    with pytest.raises(Exception):
        trade.contracts = 99  # type: ignore[misc]


# bot.main.evaluate_strategies feeds one orderbook-anchored cost_per_contract scalar to the cap gate and the overlay update.
# bot.storage.positions.open_exposures reconstructs cross-cycle max-loss from paper_trades.simulated_price.
# If a future slippage haircut alters simulate_taker_fill, the equality pins below break; updating them silently
# decouples the in-cycle cap overlay from the cross-cycle exposure reconstruction.
@pytest.mark.parametrize(
    "side,yes_ask,yes_bid",
    [
        (TradeSide.BUY_YES, Decimal("0.40"), Decimal("0.38")),
        (TradeSide.BUY_YES, Decimal("0.05"), Decimal("0.03")),
        (TradeSide.BUY_YES, Decimal("0.987654"), Decimal("0.987650")),
        (TradeSide.SELL_YES, Decimal("0.40"), Decimal("0.38")),
        (TradeSide.SELL_YES, Decimal("0.05"), Decimal("0.03")),
        (TradeSide.SELL_YES, Decimal("0.987654"), Decimal("0.987650")),
    ],
)
def test_simulate_taker_fill_orderbook_verbatim_for_cap_overlay_coupling(
    side: TradeSide, yes_ask: Decimal, yes_bid: Decimal
) -> None:
    book = _book(yes_ask=yes_ask, yes_bid=yes_bid)
    trade = simulate_taker_fill(_intent(side=side), book, _now())
    expected = yes_ask if side is TradeSide.BUY_YES else yes_bid
    assert trade.simulated_price == expected
    assert trade.simulated_price.compare_total(expected) == Decimal("0")


def test_fill_within_depth_returns_full_size() -> None:
    trade = simulate_taker_fill(_intent(contracts=5), _book(yes_ask_depth=10), _now())
    assert trade is not None
    assert trade.contracts == 5
    assert trade.attempted_contracts == 5


def test_fill_exceeding_depth_returns_partial() -> None:
    trade = simulate_taker_fill(
        _intent(side=TradeSide.SELL_YES, contracts=7194),
        _book(yes_bid=Decimal("0.99"), yes_bid_depth=1),
        _now(),
    )
    assert trade is not None
    assert trade.contracts == 1
    assert trade.attempted_contracts == 7194


def test_fill_zero_depth_returns_none() -> None:
    trade = simulate_taker_fill(_intent(contracts=5), _book(yes_ask_depth=0), _now())
    assert trade is None


def test_simulate_taker_fill_returns_none_on_zero_depth() -> None:
    trade = simulate_taker_fill(_intent(contracts=10), _book(yes_ask_depth=0), _now())
    assert trade is None


def test_stale_orderbook_returns_none() -> None:
    trade = simulate_taker_fill(
        _intent(),
        _book(snapshot_at=_now() - timedelta(seconds=241)),
        _now(),
    )
    assert trade is None


def test_fresh_orderbook_within_threshold_fills() -> None:
    trade = simulate_taker_fill(
        _intent(),
        _book(snapshot_at=_now() - timedelta(seconds=239)),
        _now(),
    )
    assert trade is not None


def test_stale_threshold_clears_observed_live_gap() -> None:
    trade = simulate_taker_fill(
        _intent(),
        _book(snapshot_at=_now() - timedelta(seconds=200)),
        _now(),
    )
    assert trade is not None


def test_stale_threshold_clears_5min_session_p99_9() -> None:
    # 5-min intra-session refresh-gap filter yields p99.9 = 185.4s.
    assert STALE_ORDERBOOK_THRESHOLD_SECONDS >= 186.0


def test_stale_threshold_is_exactly_four_times_interval() -> None:
    assert STALE_ORDERBOOK_THRESHOLD_SECONDS == MARKET_REFRESH_INTERVAL_SECONDS * 4


def test_refresh_interval_matches_main_loop_cadence() -> None:
    assert bot.execution.paper.MARKET_REFRESH_INTERVAL_SECONDS == bot.main.MARKET_REFRESH_INTERVAL


def test_eval_interval_matches_refresh_interval() -> None:
    assert bot.main.EVAL_INTERVAL == bot.execution.paper.MARKET_REFRESH_INTERVAL_SECONDS


def test_naive_snapshot_at_raises_value_error() -> None:
    naive_snap = datetime(2026, 5, 5, 11, 59, 59)
    book = Orderbook(
        yes_ask=Decimal("0.40"),
        yes_bid=Decimal("0.38"),
        yes_ask_depth=1000,
        yes_bid_depth=1000,
        snapshot_at=naive_snap,
    )
    with pytest.raises(ValueError):
        simulate_taker_fill(_intent(), book, _now())


def test_legacy_default_book_is_fresh_enough_to_fill() -> None:
    trade = simulate_taker_fill(_intent(), _book(), _now())
    assert trade is not None


def test_no_fresh_book_helper_exists() -> None:
    assert getattr(_self_module, "_fresh_book", None) is None


def test_buy_yes_uses_yes_ask_depth() -> None:
    trade = simulate_taker_fill(
        _intent(side=TradeSide.BUY_YES, contracts=5),
        _book(yes_ask_depth=2, yes_bid_depth=99),
        _now(),
    )
    assert trade is not None
    assert trade.contracts == 2


def test_sell_yes_uses_yes_bid_depth() -> None:
    trade = simulate_taker_fill(
        _intent(side=TradeSide.SELL_YES, contracts=5),
        _book(yes_ask_depth=99, yes_bid_depth=3),
        _now(),
    )
    assert trade is not None
    assert trade.contracts == 3


def test_paper_trade_carries_feature_fields() -> None:
    intent = _intent(
        ensemble_spread_sigma_t=Decimal("2.5"),
        lead_time_hours=Decimal("36.5"),
        nbm_divergence=Decimal("1.25"),
    )
    trade = simulate_taker_fill(intent, _book(), _now())
    assert trade is not None
    assert trade.ensemble_spread_sigma_t == Decimal("2.5")
    assert trade.lead_time_hours == Decimal("36.5")
    assert trade.nbm_divergence == Decimal("1.25")


def test_papertrade_constructs_with_legacy_call_site_args() -> None:
    trade = PaperTrade(
        intended_at=_now(),
        market_ticker="KXHIGHDEN-26MAY05-T80",
        side=TradeSide.BUY_YES,
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        strategy="edge",
    )
    assert trade.attempted_contracts == 0
    assert trade.ensemble_spread_sigma_t is None
    assert trade.lead_time_hours is None
    assert trade.nbm_divergence is None


def test_log_stale_skip_ratio_aggregate_above_threshold_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="bot.execution.paper")
    log_stale_skip_ratio({"K1": 6}, {"K1": 10})
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("stale_skip_ratio_high" in m for m in messages)


def test_log_stale_skip_ratio_aggregate_below_threshold_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="bot.execution.paper")
    log_stale_skip_ratio({"K1": 3}, {"K1": 10})
    messages = [rec.getMessage() for rec in caplog.records]
    assert not any(m.startswith("stale_skip_ratio_high ") for m in messages)


def test_log_stale_skip_ratio_zero_total_silent(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="bot.execution.paper")
    log_stale_skip_ratio({}, {})
    messages = [rec.getMessage() for rec in caplog.records]
    assert not any("stale_skip_ratio_high" in m for m in messages)


def test_log_stale_skip_ratio_per_series_warns_when_aggregate_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="bot.execution.paper")
    log_stale_skip_ratio({"K20": 10}, {"K1": 190, "K20": 10})
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("stale_skip_ratio_high_series" in m and "K20" in m for m in messages)
    assert not any(m.startswith("stale_skip_ratio_high ") for m in messages)


def test_log_stale_skip_ratio_per_series_below_min_intents_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="bot.execution.paper")
    log_stale_skip_ratio({"FRESH_SERIES": 3}, {"FRESH_SERIES": 3})
    messages = [rec.getMessage() for rec in caplog.records]
    assert not any("stale_skip_ratio_high_series" in m for m in messages)


def test_module_has_logger() -> None:
    assert isinstance(bot.execution.paper.logger, logging.Logger)
    assert bot.execution.paper.logger.name == "bot.execution.paper"


def test_module_logger_emits(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="bot.execution.paper")
    bot.execution.paper.logger.warning("test_module_logger_emits sentinel=42")
    messages = [(rec.name, rec.getMessage()) for rec in caplog.records]
    assert any(name == "bot.execution.paper" and "sentinel=42" in msg for name, msg in messages)


def test_trade_side_for_demo_buy_yes_maps_to_buy_yes() -> None:
    assert trade_side_for_demo("yes") is TradeSide.BUY_YES


def test_trade_side_for_demo_buy_no_maps_to_sell_yes() -> None:
    assert trade_side_for_demo("no") is TradeSide.SELL_YES


@pytest.mark.parametrize("side", ["yes", "no"])
def test_trade_side_for_demo_round_trip_through_trade_side_value(side: str) -> None:
    mapped = trade_side_for_demo(side)
    assert TradeSide(mapped.value) is mapped


def test_trade_side_for_demo_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="maybe"):
        trade_side_for_demo("maybe")


def test_bare_trade_side_constructor_rejects_kalshi_side() -> None:
    with pytest.raises(ValueError):
        TradeSide("yes")
    with pytest.raises(ValueError):
        TradeSide("no")
