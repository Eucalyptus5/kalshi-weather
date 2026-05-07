from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal

import pytest

from bot.strategy.tails import (
    DEFAULT_POSITION_CAP,
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
    }
    base.update(overrides)
    return TailsContext(**base)  # type: ignore[arg-type]


def test_happy_path_trade_fires() -> None:
    sig = evaluate(_ctx())
    assert sig.action is TailsAction.SELL_YES
    assert sig.contracts >= 1
    assert sig.notional_dollars > Decimal("0")
    assert sig.notional_dollars <= DEFAULT_POSITION_CAP
    assert sig.reason == "trade"


def test_position_cap_clamps_notional() -> None:
    sig = evaluate(_ctx(bankroll=Decimal("100000")))
    assert sig.action is TailsAction.SELL_YES
    assert sig.notional_dollars <= Decimal("50")
    assert sig.contracts == int(Decimal("50") / Decimal("0.80"))
    assert sig.contracts == 62
    assert sig.notional_dollars == Decimal("0.80") * Decimal(62)


def test_custom_kelly_fraction_reduces_notional() -> None:
    default_sig = evaluate(_ctx(bankroll=Decimal("100")))
    smaller_sig = evaluate(_ctx(bankroll=Decimal("100")), kelly_fraction=Decimal("0.05"))
    assert default_sig.action is TailsAction.SELL_YES
    assert smaller_sig.action is TailsAction.SELL_YES
    assert smaller_sig.notional_dollars < default_sig.notional_dollars


def test_custom_position_cap() -> None:
    sig = evaluate(
        _ctx(bankroll=Decimal("100000")),
        position_cap=Decimal("10"),
    )
    assert sig.action is TailsAction.SELL_YES
    assert sig.notional_dollars <= Decimal("10")


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
    sig = evaluate(_ctx(bankroll=Decimal("0.10")))
    assert sig.action is TailsAction.SKIP
    assert sig.reason == "below_min_size"
    assert sig.contracts == 0
    assert sig.notional_dollars == Decimal("0")


def test_decimal_hygiene() -> None:
    sig = evaluate(_ctx(bankroll=Decimal("100000")))
    assert isinstance(sig.notional_dollars, Decimal)
    assert sig.notional_dollars == Decimal("49.60")


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


def test_kelly_formula_correctness() -> None:
    sig = evaluate(_ctx(bankroll=Decimal("100")))
    assert sig.action is TailsAction.SELL_YES
    assert sig.contracts == 14
    assert sig.notional_dollars == Decimal("11.20")
    assert sig.reason == "trade"


def test_signal_is_frozen_dataclass() -> None:
    sig = evaluate(_ctx())
    with pytest.raises(Exception):
        sig.contracts = 99  # type: ignore[misc]
