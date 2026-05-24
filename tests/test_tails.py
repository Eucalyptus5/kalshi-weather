from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as _timezone
from decimal import ROUND_CEILING, Decimal

import pytest

from bot.execution.fees import taker_fee
from bot.execution.paper import TradeSide
from bot.strategy.sizing import (
    TAILS_ACTIVE_KELLY_FRAC,
    TAILS_KELLY_FRAC,
    TAILS_KELLY_FRAC_PRE_CALIBRATION,
    compute_stake_contracts,
)
from bot.strategy.tails import (
    FAIR_THRESHOLD,
    YES_BID_FLOOR,
    TailsAction,
    TailsContext,
    evaluate,
)


_NOW = datetime(2026, 5, 5, 12, 0, tzinfo=_timezone.utc)


def _ctx(**overrides: object) -> TailsContext:
    base: dict[str, object] = {
        "yes_ask": Decimal("0.20"),
        "yes_bid": Decimal("0.18"),
        "no_bid": Decimal("0.80"),
        "fair_yes": Decimal("0.05"),
        "close_time": _NOW + timedelta(hours=4),
        "now": _NOW,
        "bankroll": Decimal("1000"),
        "is_same_day": False,
        "ensemble_spread": Decimal("1.5"),
        "sigma_T_median": Decimal("1.5"),
        "event_budget_remaining": Decimal("9999"),
        "market_budget_remaining": Decimal("9999"),
        "depth_at_price": 10_000,
        "price_per_contract": Decimal("0.82"),
    }
    base.update(overrides)
    return TailsContext(**base)  # type: ignore[arg-type]


def test_happy_path_trade_fires() -> None:
    sig = evaluate(_ctx())
    assert sig.action is TailsAction.SELL_YES
    assert sig.contracts >= 1
    assert sig.notional_dollars > Decimal("0")
    assert sig.reason == "trade"


def test_sell_yes_uses_collateral_basis() -> None:
    sig = evaluate(
        _ctx(
            bankroll=Decimal("100000"),
            event_budget_remaining=Decimal("50"),
            price_per_contract=Decimal("0.82"),
        )
    )
    assert sig.action is TailsAction.SELL_YES
    assert sig.contracts == 60


def test_gate_ask_exactly_at_threshold() -> None:
    sig = evaluate(_ctx(yes_ask=Decimal("0.10")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "ask_too_low"
    assert sig.contracts == 0
    assert sig.notional_dollars == Decimal("0")


def test_gate_ask_just_above_threshold_trades() -> None:
    sig = evaluate(_ctx(yes_ask=Decimal("0.11")))
    assert sig.action is TailsAction.SELL_YES
    assert sig.reason == "trade"


def test_gate_fair_exactly_at_threshold() -> None:
    sig = evaluate(_ctx(fair_yes=Decimal("0.07")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "fair_too_high"


def test_gate_fair_just_below_threshold_trades() -> None:
    sig = evaluate(_ctx(fair_yes=Decimal("0.069")))
    assert sig.action is TailsAction.SELL_YES
    assert sig.reason == "trade"


def test_gate_same_day() -> None:
    sig = evaluate(_ctx(is_same_day=True))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "same_day"


def test_gate_close_exactly_60_minutes_away() -> None:
    sig = evaluate(_ctx(close_time=_NOW + timedelta(minutes=60)))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "too_close_to_settle"


def test_gate_close_61_minutes_away_trades() -> None:
    sig = evaluate(_ctx(close_time=_NOW + timedelta(minutes=61)))
    assert sig.action is TailsAction.SELL_YES
    assert sig.reason == "trade"


def test_gate_close_already_passed() -> None:
    sig = evaluate(_ctx(close_time=_NOW - timedelta(minutes=5)))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "too_close_to_settle"


def test_negative_edge() -> None:
    sig = evaluate(_ctx(no_bid=Decimal("0.96")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "negative_edge"


def test_zero_edge_boundary() -> None:
    sig = evaluate(_ctx(no_bid=Decimal("0.95")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "negative_edge"


def test_below_min_size() -> None:
    sig = evaluate(_ctx(event_budget_remaining=Decimal("0")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "below_min_size"
    assert sig.contracts == 0
    assert sig.notional_dollars == Decimal("0")


def test_decimal_hygiene() -> None:
    sig = evaluate(_ctx(bankroll=Decimal("100000")))
    assert isinstance(sig.notional_dollars, Decimal)


def test_skip_reason_priority_order() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.05"),
            fair_yes=Decimal("0.20"),
            is_same_day=True,
        )
    )
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "ask_too_low"


def test_no_bid_zero_skips_without_division_error() -> None:
    sig = evaluate(_ctx(no_bid=Decimal("0")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "no_no_bid"
    assert sig.contracts == 0
    assert sig.notional_dollars == Decimal("0")


def test_no_bid_negative_also_skipped() -> None:
    sig = evaluate(_ctx(no_bid=Decimal("-0.01")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "no_no_bid"


def test_no_bid_positive_still_trades() -> None:
    sig = evaluate(_ctx(no_bid=Decimal("0.93"), fair_yes=Decimal("0.05")))
    assert sig.action is TailsAction.SELL_YES
    assert sig.contracts >= 1


def test_no_no_bid_fires_before_fair_too_high() -> None:
    sig = evaluate(
        _ctx(
            no_bid=Decimal("0"),
            yes_ask=Decimal("0.50"),
            fair_yes=Decimal("0.05"),
        )
    )
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "no_no_bid"


def test_ask_too_low_wins_over_no_no_bid() -> None:
    sig = evaluate(_ctx(no_bid=Decimal("0"), yes_ask=Decimal("0.05")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "ask_too_low"


def test_signal_is_frozen_dataclass() -> None:
    sig = evaluate(_ctx())
    with pytest.raises(Exception):
        sig.contracts = 99  # type: ignore[misc]


def test_yes_bid_zero_skips() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.85"),
            yes_bid=Decimal("0"),
            no_bid=Decimal("0.10"),
            fair_yes=Decimal("0.02"),
        )
    )
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "no_yes_bid"
    assert sig.contracts == 0
    assert sig.notional_dollars == Decimal("0")


def test_yes_bid_below_floor_skips() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.85"),
            yes_bid=Decimal("0.06"),
            no_bid=Decimal("0.10"),
            fair_yes=Decimal("0.02"),
        )
    )
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "no_yes_bid"
    assert sig.contracts == 0


def test_yes_bid_at_floor_passes() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.85"),
            yes_bid=Decimal("0.095"),
            no_bid=Decimal("0.10"),
            fair_yes=Decimal("0.02"),
            price_per_contract=Decimal("0.905"),
        )
    )
    assert sig.action is TailsAction.SELL_YES


def test_yes_bid_floor_covers_fee_cushion_with_safety() -> None:
    assert YES_BID_FLOOR == FAIR_THRESHOLD + Decimal("0.025")
    assert YES_BID_FLOOR == Decimal("0.095")


def test_yes_bid_floor_admit_is_net_ev_positive() -> None:
    ctx = _ctx(
        yes_ask=Decimal("0.85"),
        yes_bid=YES_BID_FLOOR,
        no_bid=Decimal("0.10"),
        fair_yes=FAIR_THRESHOLD - Decimal("0.00001"),
        price_per_contract=Decimal("0.905"),
    )
    sig = evaluate(ctx)
    assert sig.action is TailsAction.SELL_YES
    gross = ctx.yes_bid - ctx.fair_yes
    sell_price = Decimal("1") - ctx.yes_bid
    fee = taker_fee(1, sell_price)
    half_tick = Decimal("0.0125")
    assert gross - fee - half_tick > Decimal("0")


def test_net_ev_anchor_fires_when_taker_rate_breaks_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("bot.execution.fees.TAKER_RATE", Decimal("0.30"))
    ctx = _ctx(
        yes_ask=Decimal("0.85"),
        yes_bid=YES_BID_FLOOR,
        no_bid=Decimal("0.10"),
        fair_yes=FAIR_THRESHOLD - Decimal("0.00001"),
    )
    gross = ctx.yes_bid - ctx.fair_yes
    fee = taker_fee(1, ctx.yes_bid)
    assert gross - fee <= Decimal("0")


@pytest.mark.parametrize(
    "yes_bid",
    [
        Decimal("0.05"),
        Decimal("0.06"),
        Decimal("0.069"),
        Decimal("0.07"),
        Decimal("0.074"),
        Decimal("0.085"),
        Decimal("0.089"),
        Decimal("0.090"),
        Decimal("0.094"),
    ],
)
def test_yes_bid_in_contaminated_band_is_skipped(yes_bid: Decimal) -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.85"),
            yes_bid=yes_bid,
            no_bid=Decimal("0.10"),
            fair_yes=Decimal("0.02"),
        )
    )
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "no_yes_bid"


@pytest.mark.parametrize(
    "overrides, expected_reason",
    [
        ({"yes_bid": Decimal("0.06"), "no_bid": Decimal("0")}, "no_no_bid"),
        ({"yes_bid": Decimal("0.06"), "yes_ask": Decimal("0.05")}, "ask_too_low"),
    ],
)
def test_yes_bid_floor_parametrize(overrides: dict[str, Decimal], expected_reason: str) -> None:
    sig = evaluate(_ctx(**overrides))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == expected_reason


def test_yes_bid_negative_also_skipped() -> None:
    sig = evaluate(_ctx(yes_bid=Decimal("-0.01")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "no_yes_bid"


@pytest.mark.parametrize(
    "overrides, expected_reason",
    [
        ({"yes_bid": Decimal("0"), "no_bid": Decimal("0")}, "no_no_bid"),
        ({"yes_bid": Decimal("0"), "yes_ask": Decimal("0.05")}, "ask_too_low"),
        ({"yes_bid": Decimal("0"), "fair_yes": Decimal("0.10")}, "no_yes_bid"),
    ],
)
def test_yes_bid_gate_ordering(overrides: dict[str, Decimal], expected_reason: str) -> None:
    sig = evaluate(_ctx(**overrides))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == expected_reason


def test_cost_per_contract_at_one_dollar_skips_cost_basis_unusable() -> None:
    sig = evaluate(
        _ctx(
            yes_ask=Decimal("0.98"),
            yes_bid=Decimal("0.10"),
            no_bid=Decimal("0.01"),
            fair_yes=Decimal("0.05"),
            price_per_contract=Decimal("1"),
        )
    )
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "cost_basis_unusable"
    assert sig.contracts == 0
    assert sig.notional_dollars == Decimal("0")


def test_demo_mode_without_cost_basis_raises() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        evaluate(_ctx(no_cost_per_contract=None), mode="demo")
    assert str(excinfo.value) == (
        "demo mode requires book-derived cost basis; _build_intents failed to thread book.no_ask"
    )


def test_paper_mode_without_cost_basis_accepted() -> None:
    sig = evaluate(_ctx(no_cost_per_contract=None), mode="paper")
    assert sig.action is TailsAction.SELL_YES


def test_fee_cushion_constant_removed() -> None:
    from bot.strategy import tails

    assert not hasattr(tails, "FEE_CUSHION")


def test_yes_bid_floor_clears_real_cent_ceiled_fee_with_positive_margin() -> None:
    yes_bid = YES_BID_FLOOR
    fair_yes = Decimal("0.07")
    sell_price = Decimal("1") - yes_bid
    real_fee = (Decimal("0.07") * sell_price * (Decimal("1") - sell_price)).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    assert (yes_bid - fair_yes) - real_fee - Decimal("0.0125") > Decimal("0")


def test_yes_bid_at_prior_target_0090_yields_non_positive_real_margin() -> None:
    yes_bid = Decimal("0.090")
    fair_yes = Decimal("0.07")
    sell_price = Decimal("1") - yes_bid
    real_fee = (Decimal("0.07") * sell_price * (Decimal("1") - sell_price)).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    assert (yes_bid - fair_yes) - real_fee - Decimal("0.0125") <= Decimal("0")


@pytest.mark.parametrize(
    "yes_bid_depth,event_budget_remaining,price_per_contract,expected_contracts",
    [
        (10, Decimal("9999"), Decimal("0.910"), 5),
        (4, Decimal("9999"), Decimal("0.905"), 2),
        (3, Decimal("9999"), Decimal("0.700"), 1),
    ],
)
def test_yes_bid_depth_caps_contracts_via_sizer(
    yes_bid_depth: int,
    event_budget_remaining: Decimal,
    price_per_contract: Decimal,
    expected_contracts: int,
) -> None:
    ctx = TailsContext(
        yes_ask=Decimal("0.85"),
        yes_bid=Decimal("0.095"),
        no_bid=Decimal("0.10"),
        fair_yes=Decimal("0.069"),
        close_time=_NOW + timedelta(hours=4),
        now=_NOW,
        bankroll=Decimal("10000"),
        is_same_day=False,
        ensemble_spread=Decimal("1.5"),
        sigma_T_median=Decimal("1.5"),
        event_budget_remaining=event_budget_remaining,
        market_budget_remaining=Decimal("9999"),
        depth_at_price=yes_bid_depth,
        price_per_contract=price_per_contract,
    )
    sig = evaluate(ctx)
    assert sig.action is TailsAction.SELL_YES
    assert sig.contracts == expected_contracts


def test_market_budget_cap_applies_after_yes_bid_depth() -> None:
    common = dict(
        yes_ask=Decimal("0.85"),
        yes_bid=Decimal("0.095"),
        no_bid=Decimal("0.10"),
        fair_yes=Decimal("0.069"),
        bankroll=Decimal("100000"),
        price_per_contract=Decimal("0.905"),
        event_budget_remaining=Decimal("9999"),
    )
    # depth_dollars at depth=4 ppc=0.905 is 0.5*4*0.905 = 1.81 -> 2 contracts; market_budget_remaining=1.811 also caps at 2 contracts.
    sig_a = evaluate(_ctx(**common, depth_at_price=4, market_budget_remaining=Decimal("1.811")))
    assert sig_a.contracts == 2
    # depth=11 -> depth_dollars=4.9775 -> 5 contracts; market_budget=1.811 -> 2 contracts, market wins.
    sig_b = evaluate(_ctx(**common, depth_at_price=11, market_budget_remaining=Decimal("1.811")))
    assert sig_b.contracts == 2
    # depth=4 -> 2 contracts; market_budget large -> depth wins at 2.
    sig_c = evaluate(_ctx(**common, depth_at_price=4, market_budget_remaining=Decimal("9999")))
    assert sig_c.contracts == 2


def test_yes_bid_depth_zero_emits_distinct_skip_reason_over_below_min_size() -> None:
    ctx = _ctx(
        yes_ask=Decimal("0.85"),
        yes_bid=Decimal("0.095"),
        no_bid=Decimal("0.10"),
        fair_yes=Decimal("0.02"),
        bankroll=Decimal("5"),
        depth_at_price=0,
        price_per_contract=Decimal("0.905"),
    )
    sig = evaluate(ctx)
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "yes_bid_depth_zero"


def test_evaluate_calls_sizer_and_emits_sell_yes() -> None:
    ctx = _ctx(
        yes_ask=Decimal("0.85"),
        yes_bid=Decimal("0.18"),
        no_bid=Decimal("0.80"),
        fair_yes=Decimal("0.04"),
        bankroll=Decimal("500"),
        event_budget_remaining=Decimal("15"),
        market_budget_remaining=Decimal("9999"),
        ensemble_spread=Decimal("1.5"),
        sigma_T_median=Decimal("1.5"),
        depth_at_price=50,
        price_per_contract=Decimal("0.82"),
    )
    sig = evaluate(ctx)
    expected = compute_stake_contracts(
        side=TradeSide.SELL_YES,
        q=Decimal("0.04"),
        p=Decimal("0.18"),
        sigma_T=Decimal("1.5"),
        sigma_T_median=Decimal("1.5"),
        kelly_frac=TAILS_ACTIVE_KELLY_FRAC,
        bankroll=Decimal("500"),
        event_budget_remaining=Decimal("15"),
        market_budget_remaining=Decimal("9999"),
        depth_at_price=50,
        price_per_contract=Decimal("0.82"),
    )
    assert sig.action is TailsAction.SELL_YES
    assert sig.contracts == expected


def test_evaluate_uses_pre_calibration_kelly_frac() -> None:
    assert TAILS_ACTIVE_KELLY_FRAC == TAILS_KELLY_FRAC_PRE_CALIBRATION
    assert TAILS_ACTIVE_KELLY_FRAC < TAILS_KELLY_FRAC
