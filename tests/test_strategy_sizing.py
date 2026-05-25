from __future__ import annotations

from decimal import Decimal

import pytest

from bot.execution.paper import TradeSide
from bot.strategy.sizing import (
    SIGMA_T_MEDIAN_BY_LEAD_H,
    TAILS_ACTIVE_KELLY_FRAC,
    TAILS_KELLY_FRAC_PRE_CALIBRATION,
    compute_stake_contracts,
    sigma_t_median_for_lead,
)


@pytest.mark.parametrize(
    "test_id, side, q, p, sigma_T, sigma_T_median, kelly_frac, bankroll, "
    "event_budget_remaining, market_budget_remaining, depth, price_per_contract, expected",
    [
        (
            "tails_normal_no_shrinkage",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.90"),
            16,
        ),
        (
            "tails_with_shrinkage",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("3.0"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.90"),
            8,
        ),
        (
            "edge_buy_yes_normal",
            TradeSide.BUY_YES,
            Decimal("0.55"),
            Decimal("0.40"),
            Decimal("2.0"),
            Decimal("2.0"),
            Decimal("0.25"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.40"),
            25,
        ),
        (
            "edge_sell_yes_normal",
            TradeSide.SELL_YES,
            Decimal("0.25"),
            Decimal("0.45"),
            Decimal("2.0"),
            Decimal("2.0"),
            Decimal("0.25"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.55"),
            25,
        ),
        (
            "budget_binding",
            TradeSide.BUY_YES,
            Decimal("0.95"),
            Decimal("0.39"),
            Decimal("2.0"),
            Decimal("2.0"),
            Decimal("0.25"),
            Decimal("10000"),
            Decimal("1.0"),
            Decimal("9999"),
            10_000,
            Decimal("0.39"),
            2,
        ),
        (
            "depth_binding",
            TradeSide.BUY_YES,
            Decimal("0.95"),
            Decimal("0.10"),
            Decimal("2.0"),
            Decimal("2.0"),
            Decimal("0.25"),
            Decimal("10000"),
            Decimal("9999"),
            Decimal("9999"),
            3,
            Decimal("0.10"),
            1,
        ),
        (
            "zero_contracts_below_floor",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.05"),
            Decimal("2.0"),
            Decimal("2.0"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("0.50"),
            Decimal("9999"),
            50,
            Decimal("0.95"),
            0,
        ),
        (
            "sigma_T_floor_no_crash",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("0"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.90"),
            16,
        ),
        (
            "event_budget_negative_clamps",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("-0.0001"),
            Decimal("9999"),
            50,
            Decimal("0.90"),
            0,
        ),
        (
            "market_budget_negative_clamps",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("9999"),
            Decimal("-0.0001"),
            50,
            Decimal("0.90"),
            0,
        ),
        (
            "market_budget_binds_tighter_than_event",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("50"),
            Decimal("1.80"),
            50,
            Decimal("0.90"),
            2,
        ),
        (
            "event_budget_binds_tighter_than_market",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("1.80"),
            Decimal("50"),
            50,
            Decimal("0.90"),
            2,
        ),
        (
            "depth_zero_returns_zero",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            0,
            Decimal("0.90"),
            0,
        ),
        (
            "kelly_frac_zero_returns_zero",
            TradeSide.SELL_YES,
            Decimal("0.04"),
            Decimal("0.10"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.90"),
            0,
        ),
        (
            "boundary_price_buy_yes_zero",
            TradeSide.BUY_YES,
            Decimal("0.05"),
            Decimal("0"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.25"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.40"),
            0,
        ),
        (
            "boundary_price_buy_yes_one",
            TradeSide.BUY_YES,
            Decimal("0.95"),
            Decimal("1"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.25"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.40"),
            0,
        ),
        (
            "boundary_price_sell_yes_zero",
            TradeSide.SELL_YES,
            Decimal("0.05"),
            Decimal("0"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.90"),
            0,
        ),
        (
            "boundary_price_sell_yes_one",
            TradeSide.SELL_YES,
            Decimal("0.95"),
            Decimal("1"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("15"),
            Decimal("9999"),
            50,
            Decimal("0.90"),
            0,
        ),
        (
            "boundary_price_per_contract_zero_demo_sell_yes_deep_itm",
            TradeSide.SELL_YES,
            Decimal("0.5"),
            Decimal("0.99"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("15"),
            Decimal("15"),
            10,
            Decimal("0"),
            0,
        ),
        (
            "boundary_price_per_contract_zero_depth_zero",
            TradeSide.SELL_YES,
            Decimal("0.5"),
            Decimal("0.99"),
            Decimal("1.5"),
            Decimal("1.5"),
            Decimal("0.05"),
            Decimal("500"),
            Decimal("15"),
            Decimal("15"),
            0,
            Decimal("0"),
            0,
        ),
    ],
)
def test_compute_stake_contracts_table(
    test_id: str,
    side: TradeSide,
    q: Decimal,
    p: Decimal,
    sigma_T: Decimal,
    sigma_T_median: Decimal,
    kelly_frac: Decimal,
    bankroll: Decimal,
    event_budget_remaining: Decimal,
    market_budget_remaining: Decimal,
    depth: int,
    price_per_contract: Decimal,
    expected: int,
) -> None:
    result = compute_stake_contracts(
        side=side,
        q=q,
        p=p,
        sigma_T=sigma_T,
        sigma_T_median=sigma_T_median,
        kelly_frac=kelly_frac,
        bankroll=bankroll,
        event_budget_remaining=event_budget_remaining,
        market_budget_remaining=market_budget_remaining,
        depth_at_price=depth,
        price_per_contract=price_per_contract,
    )
    assert result == expected, f"{test_id}: expected {expected}, got {result}"


def test_smoke_normal_intent_returns_positive_contracts() -> None:
    result = compute_stake_contracts(
        side=TradeSide.SELL_YES,
        q=Decimal("0.05"),
        p=Decimal("0.10"),
        sigma_T=Decimal("1.5"),
        sigma_T_median=Decimal("1.5"),
        kelly_frac=Decimal("0.05"),
        bankroll=Decimal("500"),
        event_budget_remaining=Decimal("3"),
        market_budget_remaining=Decimal("3"),
        depth_at_price=20,
        price_per_contract=Decimal("0.90"),
    )
    assert result >= 1


def test_tails_normal_no_shrinkage_lands_at_16_contracts() -> None:
    result = compute_stake_contracts(
        side=TradeSide.SELL_YES,
        q=Decimal("0.04"),
        p=Decimal("0.10"),
        sigma_T=Decimal("1.5"),
        sigma_T_median=Decimal("1.5"),
        kelly_frac=Decimal("0.05"),
        bankroll=Decimal("500"),
        event_budget_remaining=Decimal("15"),
        market_budget_remaining=Decimal("9999"),
        depth_at_price=50,
        price_per_contract=Decimal("0.90"),
    )
    assert result == 16, (
        "production sees the tighter event-budget regime; this vector deliberately "
        "relaxes the budget to isolate the sizer's algebra"
    )


def test_tails_active_kelly_frac_equals_pre_calibration() -> None:
    assert TAILS_ACTIVE_KELLY_FRAC == TAILS_KELLY_FRAC_PRE_CALIBRATION


def test_sell_yes_paper_uses_one_minus_yes_bid() -> None:
    result = compute_stake_contracts(
        side=TradeSide.SELL_YES,
        q=Decimal("0.04"),
        p=Decimal("0.18"),
        sigma_T=Decimal("1.5"),
        sigma_T_median=Decimal("1.5"),
        kelly_frac=Decimal("0.05"),
        bankroll=Decimal("100000"),
        event_budget_remaining=Decimal("50"),
        market_budget_remaining=Decimal("9999"),
        depth_at_price=10_000,
        price_per_contract=Decimal("0.82"),
    )
    assert result == 60


def test_sell_yes_demo_uses_no_ask() -> None:
    result = compute_stake_contracts(
        side=TradeSide.SELL_YES,
        q=Decimal("0.04"),
        p=Decimal("0.18"),
        sigma_T=Decimal("1.5"),
        sigma_T_median=Decimal("1.5"),
        kelly_frac=Decimal("0.05"),
        bankroll=Decimal("100000"),
        event_budget_remaining=Decimal("50"),
        market_budget_remaining=Decimal("9999"),
        depth_at_price=10_000,
        price_per_contract=Decimal("0.85"),
    )
    assert result == 58


def test_market_budget_clamp_replaces_demo_contracts_cap() -> None:
    result = compute_stake_contracts(
        side=TradeSide.SELL_YES,
        q=Decimal("0.04"),
        p=Decimal("0.10"),
        sigma_T=Decimal("1.5"),
        sigma_T_median=Decimal("1.5"),
        kelly_frac=Decimal("0.05"),
        bankroll=Decimal("500"),
        event_budget_remaining=Decimal("9999"),
        market_budget_remaining=Decimal("1.80"),
        depth_at_price=50,
        price_per_contract=Decimal("0.90"),
    )
    assert result == 2


def test_sigma_t_median_table_has_24h_buckets_through_168() -> None:
    expected_keys = {0, 24, 48, 72, 96, 120, 144, 168}
    assert set(SIGMA_T_MEDIAN_BY_LEAD_H.keys()) == expected_keys


def test_sigma_t_median_for_lead_floor_lookup_at_tabulated_keys() -> None:
    for k, v in SIGMA_T_MEDIAN_BY_LEAD_H.items():
        assert sigma_t_median_for_lead(k) == v


def test_sigma_t_median_for_lead_floors_between_buckets() -> None:
    assert sigma_t_median_for_lead(30) == SIGMA_T_MEDIAN_BY_LEAD_H[24]
    assert sigma_t_median_for_lead(60) == SIGMA_T_MEDIAN_BY_LEAD_H[48]
    assert sigma_t_median_for_lead(95) == SIGMA_T_MEDIAN_BY_LEAD_H[72]


def test_sigma_t_median_for_lead_clamps_above_max() -> None:
    assert sigma_t_median_for_lead(240) == SIGMA_T_MEDIAN_BY_LEAD_H[168]
    assert sigma_t_median_for_lead(10_000) == SIGMA_T_MEDIAN_BY_LEAD_H[168]


def test_sigma_t_median_for_lead_clamps_below_zero() -> None:
    assert sigma_t_median_for_lead(-2) == SIGMA_T_MEDIAN_BY_LEAD_H[0]
    assert sigma_t_median_for_lead(0) == SIGMA_T_MEDIAN_BY_LEAD_H[0]


def test_key_168_mirrors_144() -> None:
    assert SIGMA_T_MEDIAN_BY_LEAD_H[168] == SIGMA_T_MEDIAN_BY_LEAD_H[144]
