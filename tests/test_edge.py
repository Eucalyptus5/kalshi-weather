from __future__ import annotations

from decimal import Decimal

import pytest

from bot.strategy.edge import (
    DEFAULT_MIN_SPREAD,
    KELLY_MULTIPLIER,
    EdgeAction,
    EdgeContext,
    evaluate,
)


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
    }
    base.update(overrides)
    return EdgeContext(**base)  # type: ignore[arg-type]


def test_buy_yes_happy_path() -> None:
    sig = evaluate(_ctx())
    assert sig.action is EdgeAction.BUY_YES
    assert sig.contracts == 62
    assert sig.notional_dollars == Decimal("0.40") * Decimal(62)
    assert sig.notional_dollars == Decimal("24.80")
    assert sig.reason == "trade_buy"


def test_sell_yes_happy_path() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.32"),
            yes_bid=Decimal("0.30"),
            fair_yes=Decimal("0.20"),
            no_cost_per_contract=None,
        )
    )
    assert sig.action is EdgeAction.SELL_YES
    assert sig.contracts == 71
    assert sig.notional_dollars == Decimal("0.70") * Decimal(71)
    assert sig.notional_dollars == Decimal("49.70")
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
    sig = evaluate(_ctx(bankroll=Decimal("0.10")))
    assert sig.action is EdgeAction.SKIP
    assert sig.reason == "below_min_size"
    assert sig.contracts == 0
    assert sig.notional_dollars == Decimal("0")


def test_custom_kelly_multiplier_reduces_notional() -> None:
    default_sig = evaluate(_ctx(bankroll=Decimal("1000")))
    smaller_sig = evaluate(
        _ctx(bankroll=Decimal("1000")),
        kelly_multiplier=Decimal("0.05"),
    )
    assert default_sig.action is EdgeAction.BUY_YES
    assert smaller_sig.action is EdgeAction.BUY_YES
    assert smaller_sig.notional_dollars < default_sig.notional_dollars


def test_kelly_multiplier_default() -> None:
    assert KELLY_MULTIPLIER == Decimal("0.15")


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


def test_edge_sell_yes_paper_mode_no_cost_per_contract_absent_matches_legacy_denominator() -> None:
    ctx = _ctx(
        yes_ask=Decimal("0.22"),
        yes_bid=Decimal("0.20"),
        fair_yes=Decimal("0.05"),
        no_cost_per_contract=None,
    )
    sig = evaluate(ctx)
    assert sig.action is EdgeAction.SELL_YES
    legacy_cost = Decimal("1") - Decimal("0.20")
    assert sig.contracts == 140
    assert sig.notional_dollars == legacy_cost * Decimal(sig.contracts)
    assert sig.notional_dollars == Decimal("112.00")


def test_edge_sell_yes_demo_mode_no_cost_per_contract_present_uses_no_ask() -> None:
    paper = evaluate(
        _ctx(
            yes_ask=Decimal("0.22"),
            yes_bid=Decimal("0.20"),
            fair_yes=Decimal("0.05"),
            no_cost_per_contract=None,
        )
    )
    demo = evaluate(
        _ctx(
            yes_ask=Decimal("0.22"),
            yes_bid=Decimal("0.20"),
            fair_yes=Decimal("0.05"),
            no_cost_per_contract=Decimal("0.40"),
        )
    )
    assert demo.action is EdgeAction.SELL_YES
    assert demo.notional_dollars == Decimal("0.40") * Decimal(demo.contracts)
    assert demo.contracts == 281
    assert demo.contracts == 2 * paper.contracts + 1


def test_edge_buy_yes_unaffected_by_no_cost_per_contract_field() -> None:
    without = evaluate(_ctx(no_cost_per_contract=None))
    with_field = evaluate(_ctx(no_cost_per_contract=Decimal("0.40")))
    assert without.action is EdgeAction.BUY_YES
    assert with_field.action is EdgeAction.BUY_YES
    assert with_field.contracts == without.contracts
    assert with_field.notional_dollars == without.notional_dollars


def test_edge_demo_mode_without_cost_basis_raises() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        evaluate(_ctx(no_cost_per_contract=None), mode="demo")
    assert str(excinfo.value) == (
        "demo mode requires book-derived cost basis; _build_intents failed to thread book.no_ask"
    )


def test_edge_paper_mode_without_cost_basis_accepted() -> None:
    sig = evaluate(_ctx(no_cost_per_contract=None), mode="paper")
    assert sig.action is EdgeAction.BUY_YES
