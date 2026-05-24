from __future__ import annotations

from decimal import Decimal

import pytest

from bot.execution.paper import TradeSide
from bot.strategy.edge import (
    DEFAULT_MIN_SPREAD,
    EdgeAction,
    EdgeContext,
    evaluate,
)
from bot.strategy.sizing import EDGE_KELLY_FRAC, compute_stake_contracts


def _ctx(**overrides: object) -> EdgeContext:
    base: dict[str, object] = {
        "yes_ask": Decimal("0.40"),
        "yes_bid": Decimal("0.38"),
        "fair_yes": Decimal("0.50"),
        "ensemble_spread": Decimal("2.0"),
        "bankroll": Decimal("1000"),
        "is_same_day": False,
        "is_blacklisted": False,
        "nbm_divergence": None,
        "sigma_T_median": Decimal("2.0"),
        "event_budget_remaining": Decimal("9999"),
        "market_budget_remaining": Decimal("9999"),
        "depth_at_price": 10_000,
        "price_per_contract": Decimal("0.40"),
    }
    base.update(overrides)
    return EdgeContext(**base)  # type: ignore[arg-type]


def test_buy_yes_happy_path() -> None:
    sig = evaluate(_ctx())
    assert sig.action is EdgeAction.BUY_YES
    assert sig.contracts >= 1
    assert sig.reason == "trade_buy"


def test_sell_yes_happy_path() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.32"),
            yes_bid=Decimal("0.30"),
            fair_yes=Decimal("0.20"),
            price_per_contract=Decimal("0.70"),
            no_cost_per_contract=None,
        )
    )
    assert sig.action is EdgeAction.SELL_YES
    assert sig.contracts > 0
    assert sig.reason == "trade_sell"


def test_edge_exactly_at_threshold() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.40"),
            yes_bid=Decimal("0.20"),
            fair_yes=Decimal("0.38"),
        )
    )
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "edge_too_small"


def test_edge_just_above_threshold_passes_outer_gate() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.40"),
            yes_bid=Decimal("0.20"),
            fair_yes=Decimal("0.385"),
        )
    )
    assert sig.reason != "edge_too_small"


def test_spread_exactly_at_min_spread() -> None:
    sig = evaluate(_ctx(ensemble_spread=Decimal("1.0")))
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "spread_too_tight"


def test_spread_just_above_min_spread() -> None:
    sig = evaluate(_ctx(ensemble_spread=Decimal("1.01")))
    assert sig.reason != "spread_too_tight"


def test_custom_min_spread() -> None:
    sig = evaluate(
        _ctx(ensemble_spread=Decimal("1.5")),
        min_spread=Decimal("2.0"),
    )
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "spread_too_tight"


def test_default_min_spread_is_one() -> None:
    assert DEFAULT_MIN_SPREAD == Decimal("1.0")


def test_blacklisted_skips() -> None:
    sig = evaluate(_ctx(is_blacklisted=True))
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "blacklisted"


def test_same_day_skips() -> None:
    sig = evaluate(_ctx(is_same_day=True))
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "same_day"


def test_nbm_divergence_above_limit_skips() -> None:
    sig = evaluate(_ctx(nbm_divergence=Decimal("5.5")))
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "nbm_diverged"


def test_nbm_divergence_exactly_at_limit_passes() -> None:
    sig = evaluate(_ctx(nbm_divergence=Decimal("5")))
    assert sig.reason != "nbm_diverged"


def test_nbm_divergence_none_passes() -> None:
    sig = evaluate(_ctx(nbm_divergence=None))
    assert sig.reason != "nbm_diverged"


def test_no_direction_inside_wide_spread() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.50"),
            yes_bid=Decimal("0.20"),
            fair_yes=Decimal("0.46"),
        )
    )
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "no_direction"


def test_buy_threshold_strict() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.40"),
            yes_bid=Decimal("0.20"),
            fair_yes=Decimal("0.44"),
        )
    )
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "no_direction"


def test_sell_threshold_strict() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.80"),
            yes_bid=Decimal("0.60"),
            fair_yes=Decimal("0.56"),
        )
    )
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "no_direction"


def test_below_min_size_buy() -> None:
    sig = evaluate(_ctx(event_budget_remaining=Decimal("0")))
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "below_min_size"
    assert sig.contracts == 0
    assert sig.notional_dollars == Decimal("0")


def test_decimal_hygiene() -> None:
    sig = evaluate(_ctx())
    assert isinstance(sig.notional_dollars, Decimal)


def test_first_skip_wins_edge_before_spread() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.40"),
            yes_bid=Decimal("0.20"),
            fair_yes=Decimal("0.38"),
            ensemble_spread=Decimal("0.5"),
        )
    )
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "edge_too_small"


def test_signal_is_frozen_dataclass() -> None:
    sig = evaluate(_ctx())
    with pytest.raises(Exception):
        sig.contracts = 99  # type: ignore[misc]


def test_edge_demo_mode_without_cost_basis_raises() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        evaluate(_ctx(no_cost_per_contract=None), mode="demo")
    assert str(excinfo.value) == (
        "demo mode requires book-derived cost basis; _build_intents failed to thread book.no_ask"
    )


def test_edge_paper_mode_without_cost_basis_accepted() -> None:
    sig = evaluate(_ctx(no_cost_per_contract=None), mode="paper")
    assert sig.action is EdgeAction.BUY_YES


def test_edge_depth_zero_emits_distinct_skip_reason() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.30"),
            yes_bid=Decimal("0.28"),
            fair_yes=Decimal("0.50"),
            depth_at_price=0,
        )
    )
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "depth_zero_clamp"


def test_edge_depth_zero_sell_yes_emits_distinct_skip_reason() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.32"),
            yes_bid=Decimal("0.30"),
            fair_yes=Decimal("0.20"),
            no_cost_per_contract=None,
            price_per_contract=Decimal("0.70"),
            depth_at_price=0,
        )
    )
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "depth_zero_clamp"


def test_evaluate_calls_sizer_and_emits_buy_yes() -> None:
    ctx = _ctx(
        yes_ask=Decimal("0.40"),
        yes_bid=Decimal("0.38"),
        fair_yes=Decimal("0.55"),
        price_per_contract=Decimal("0.40"),
        depth_at_price=50,
        event_budget_remaining=Decimal("15"),
        market_budget_remaining=Decimal("9999"),
        bankroll=Decimal("500"),
        sigma_T_median=Decimal("2.0"),
        ensemble_spread=Decimal("2.0"),
    )
    sig = evaluate(ctx)
    expected = compute_stake_contracts(
        side=TradeSide.BUY_YES,
        q=Decimal("0.55"),
        p=Decimal("0.40"),
        sigma_T=Decimal("2.0"),
        sigma_T_median=Decimal("2.0"),
        kelly_frac=EDGE_KELLY_FRAC,
        bankroll=Decimal("500"),
        event_budget_remaining=Decimal("15"),
        market_budget_remaining=Decimal("9999"),
        depth_at_price=50,
        price_per_contract=Decimal("0.40"),
    )
    assert sig.action is EdgeAction.BUY_YES
    assert sig.contracts == expected


def test_evaluate_skips_below_min_size_when_sizer_returns_zero() -> None:
    sig = evaluate(_ctx(event_budget_remaining=Decimal("0.01"), depth_at_price=50))
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "below_min_size"
