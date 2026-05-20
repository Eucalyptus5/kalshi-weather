from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from bot.risk.gates import (
    CAP_GATE_NAMES,
    GATE_NAMES,
    GateContext,
    GateMode,
    GateParams,
    GateResult,
    RiskCheck,
    evaluate,
)


def _ctx(**overrides: object) -> GateContext:
    base: dict[str, object] = {
        "fair_yes": Decimal("0.50"),
        "model_age_hours": Decimal("1"),
        "ensemble_spread": Decimal("2.0"),
        "edge": Decimal("0.06"),
        "order_size_dollars": Decimal("20"),
        "market_existing_dollars": Decimal("0"),
        "market_position_cap": Decimal("50"),
        "event_existing_dollars": Decimal("0"),
        "event_position_cap": Decimal("300"),
        "series_existing_dollars": Decimal("10"),
        "series_position_cap": Decimal("100"),
        "aggregate_existing_dollars": Decimal("0"),
        "aggregate_exposure_cap": Decimal("10000"),
        "account_balance": Decimal("500"),
        "required_cushion": Decimal("100"),
        "market_status": "active",
        "minutes_to_close": 60,
        "circuit_breakers_armed": True,
    }
    base.update(overrides)
    return GateContext(**base)  # type: ignore[arg-type]


def _names(check: RiskCheck) -> list[str]:
    return [r.name for r in check.all_results]


def test_all_pass_paper() -> None:
    check = evaluate(_ctx(), GateMode.PAPER)
    assert check.overall_passed is True
    assert len(check.failures) == 0
    assert len(check.all_results) == len(GATE_NAMES)
    assert all(r.passed for r in check.all_results)
    assert all(r.reason is None for r in check.all_results)
    assert tuple(_names(check)) == GATE_NAMES


def test_all_pass_live() -> None:
    check = evaluate(_ctx(), GateMode.LIVE)
    assert check.overall_passed is True
    assert len(check.failures) == 0
    assert len(check.all_results) == len(GATE_NAMES)
    assert tuple(_names(check)) == GATE_NAMES


def test_fair_value_none_paper_continues() -> None:
    check = evaluate(_ctx(fair_yes=None), GateMode.PAPER)
    assert check.overall_passed is True
    assert len(check.failures) == 1
    assert check.failures[0].name == "fair_value_sane"
    assert check.failures[0].passed is False
    assert check.failures[0].reason is not None


def test_fair_value_none_live_blocks() -> None:
    check = evaluate(_ctx(fair_yes=None), GateMode.LIVE)
    assert check.overall_passed is False
    assert len(check.failures) == 1
    assert check.failures[0].name == "fair_value_sane"


def test_fair_value_below_range() -> None:
    check = evaluate(_ctx(fair_yes=Decimal("0.005")), GateMode.LIVE)
    assert check.overall_passed is False
    failed = [r for r in check.failures if r.name == "fair_value_sane"]
    assert len(failed) == 1


def test_fair_value_above_range() -> None:
    check = evaluate(_ctx(fair_yes=Decimal("0.995")), GateMode.LIVE)
    assert check.overall_passed is False
    failed = [r for r in check.failures if r.name == "fair_value_sane"]
    assert len(failed) == 1


def test_fair_value_lower_boundary_passes() -> None:
    check = evaluate(_ctx(fair_yes=Decimal("0.01")), GateMode.LIVE)
    assert check.overall_passed is True
    assert len(check.failures) == 0


def test_fair_value_upper_boundary_passes() -> None:
    check = evaluate(_ctx(fair_yes=Decimal("0.99")), GateMode.LIVE)
    assert check.overall_passed is True
    assert len(check.failures) == 0


def test_model_stale_fails() -> None:
    check = evaluate(_ctx(model_age_hours=Decimal("7")), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"model_fresh"}


def test_model_age_at_boundary_passes() -> None:
    check = evaluate(_ctx(model_age_hours=Decimal("6")), GateMode.LIVE)
    assert check.overall_passed is True


def test_ensemble_spread_too_low_fails() -> None:
    check = evaluate(_ctx(ensemble_spread=Decimal("0.5")), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"ensemble_spread_ok"}


def test_ensemble_spread_at_boundary_passes() -> None:
    check = evaluate(_ctx(ensemble_spread=Decimal("1.0")), GateMode.LIVE)
    assert check.overall_passed is True


def test_edge_below_threshold_fails() -> None:
    check = evaluate(_ctx(edge=Decimal("0.04")), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"edge_threshold"}


def test_edge_at_boundary_passes() -> None:
    check = evaluate(_ctx(edge=Decimal("0.05")), GateMode.LIVE)
    assert check.overall_passed is True


def test_order_size_above_market_cap_fails() -> None:
    check = evaluate(
        _ctx(
            order_size_dollars=Decimal("60"),
            market_existing_dollars=Decimal("0"),
            market_position_cap=Decimal("50"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"within_market_cap"}


def test_within_market_cap_blocks_when_existing_plus_new_exceeds_cap() -> None:
    check = evaluate(
        _ctx(
            market_existing_dollars=Decimal("240"),
            order_size_dollars=Decimal("20"),
            market_position_cap=Decimal("250"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"within_market_cap"}


def test_within_market_cap_passes_when_below_cap() -> None:
    check = evaluate(
        _ctx(
            market_existing_dollars=Decimal("100"),
            order_size_dollars=Decimal("50"),
            market_position_cap=Decimal("250"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is True
    assert len(check.failures) == 0


def test_within_event_cap_blocks_when_existing_plus_new_exceeds_cap() -> None:
    check = evaluate(
        _ctx(
            event_existing_dollars=Decimal("290"),
            order_size_dollars=Decimal("20"),
            event_position_cap=Decimal("300"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"within_event_cap"}


def test_within_event_cap_passes_when_below_cap() -> None:
    check = evaluate(
        _ctx(
            event_existing_dollars=Decimal("290"),
            order_size_dollars=Decimal("20"),
            event_position_cap=Decimal("310"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is True
    assert len(check.failures) == 0


def test_cap_gate_names_invariant() -> None:
    assert CAP_GATE_NAMES == frozenset(
        {"within_market_cap", "within_event_cap", "within_series_cap", "within_aggregate_cap"}
    )


def test_cap_gate_names_includes_aggregate() -> None:
    assert "within_aggregate_cap" in CAP_GATE_NAMES
    assert CAP_GATE_NAMES == frozenset(
        {"within_market_cap", "within_event_cap", "within_series_cap", "within_aggregate_cap"}
    )


def test_within_aggregate_cap_blocks_when_existing_plus_new_exceeds_cap() -> None:
    check = evaluate(
        _ctx(
            aggregate_existing_dollars=Decimal("195"),
            order_size_dollars=Decimal("10"),
            aggregate_exposure_cap=Decimal("200"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"within_aggregate_cap"}


def test_within_aggregate_cap_passes_when_below_cap() -> None:
    check = evaluate(
        _ctx(
            aggregate_existing_dollars=Decimal("100"),
            order_size_dollars=Decimal("50"),
            aggregate_exposure_cap=Decimal("200"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is True
    assert len(check.failures) == 0


def test_within_aggregate_cap_independent_of_market_event_series() -> None:
    check = evaluate(
        _ctx(
            market_existing_dollars=Decimal("0"),
            event_existing_dollars=Decimal("0"),
            series_existing_dollars=Decimal("0"),
            aggregate_existing_dollars=Decimal("199"),
            order_size_dollars=Decimal("5"),
            market_position_cap=Decimal("250"),
            event_position_cap=Decimal("300"),
            series_position_cap=Decimal("400"),
            aggregate_exposure_cap=Decimal("200"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"within_aggregate_cap"}


def test_within_series_cap_now_correctly_aggregates() -> None:
    check = evaluate(
        _ctx(
            series_existing_dollars=Decimal("380"),
            order_size_dollars=Decimal("50"),
            series_position_cap=Decimal("400"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"within_series_cap"}


def test_paper_mode_cap_failure_still_surfaces_in_failures() -> None:
    check = evaluate(
        _ctx(
            market_existing_dollars=Decimal("240"),
            order_size_dollars=Decimal("20"),
            market_position_cap=Decimal("250"),
        ),
        GateMode.PAPER,
    )
    assert check.overall_passed is True
    assert "within_market_cap" in [f.name for f in check.failures]


def test_series_cap_breached_fails() -> None:
    check = evaluate(
        _ctx(
            series_existing_dollars=Decimal("100"),
            order_size_dollars=Decimal("20"),
            series_position_cap=Decimal("110"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"within_series_cap"}


def test_account_cushion_violation_fails() -> None:
    check = evaluate(
        _ctx(
            account_balance=Decimal("50"),
            required_cushion=Decimal("100"),
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"account_cushion"}


def test_market_closed_fails() -> None:
    check = evaluate(_ctx(market_status="closed"), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"market_open"}


def test_market_open_accepts_active() -> None:
    check = evaluate(_ctx(market_status="active"), GateMode.LIVE)
    assert check.overall_passed is True
    assert "market_open" not in {r.name for r in check.failures}


def test_market_open_rejects_closed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="bot.risk.gates")
    check = evaluate(_ctx(market_status="closed"), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"market_open"}
    messages = [rec.getMessage() for rec in caplog.records]
    assert not any("market_open_unknown_status" in m for m in messages)


def test_market_open_rejects_settled(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="bot.risk.gates")
    check = evaluate(_ctx(market_status="settled"), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"market_open"}
    messages = [rec.getMessage() for rec in caplog.records]
    assert not any("market_open_unknown_status" in m for m in messages)


def test_market_open_unknown_status_fails_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="bot.risk.gates")
    check = evaluate(_ctx(market_status="frobnicated"), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"market_open"}
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("market_open_unknown_status" in m and "frobnicated" in m for m in messages)


def test_market_open_rejects_open_string() -> None:
    check = evaluate(_ctx(market_status="open"), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"market_open"}


def test_too_close_to_close_fails() -> None:
    check = evaluate(_ctx(minutes_to_close=9), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"time_to_close"}


def test_minutes_to_close_at_boundary_passes() -> None:
    check = evaluate(_ctx(minutes_to_close=10), GateMode.LIVE)
    assert check.overall_passed is True


def test_circuit_breakers_unarmed_fails() -> None:
    check = evaluate(_ctx(circuit_breakers_armed=False), GateMode.LIVE)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"circuit_breakers_armed"}


def test_multiple_failures_paper_continues() -> None:
    check = evaluate(
        _ctx(
            fair_yes=None,
            model_age_hours=Decimal("8"),
            circuit_breakers_armed=False,
        ),
        GateMode.PAPER,
    )
    assert check.overall_passed is True
    assert len(check.failures) == 3
    assert {r.name for r in check.failures} == {
        "fair_value_sane",
        "model_fresh",
        "circuit_breakers_armed",
    }


def test_multiple_failures_live_blocks() -> None:
    check = evaluate(
        _ctx(
            fair_yes=None,
            model_age_hours=Decimal("8"),
            circuit_breakers_armed=False,
        ),
        GateMode.LIVE,
    )
    assert check.overall_passed is False
    assert len(check.failures) == 3


def test_no_early_exit_all_results_in_spec_order() -> None:
    check = evaluate(_ctx(fair_yes=None), GateMode.PAPER)
    assert len(check.all_results) == len(GATE_NAMES)
    assert tuple(_names(check)) == GATE_NAMES


def test_custom_params_override_defaults() -> None:
    params = GateParams(min_edge=Decimal("0.08"))
    check = evaluate(_ctx(edge=Decimal("0.06")), GateMode.LIVE, params)
    assert check.overall_passed is False
    assert {r.name for r in check.failures} == {"edge_threshold"}


def test_paper_mode_logs_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="bot.risk.gates")
    evaluate(_ctx(fair_yes=None), GateMode.PAPER)
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("risk gate failed" in m and "fair_value_sane" in m for m in messages)


def test_live_mode_logs_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="bot.risk.gates")
    evaluate(_ctx(fair_yes=None), GateMode.LIVE)
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("risk gate failed" in m and "fair_value_sane" in m for m in messages)


def test_decimal_hygiene_and_reason_strings() -> None:
    check = evaluate(
        _ctx(model_age_hours=Decimal("7.5"), fair_yes=Decimal("0.005")),
        GateMode.LIVE,
    )
    for failure in check.failures:
        assert isinstance(failure, GateResult)
        assert isinstance(failure.reason, str)
        assert len(failure.reason) <= 80


def test_result_is_frozen() -> None:
    check = evaluate(_ctx(), GateMode.PAPER)
    with pytest.raises(Exception):
        check.all_results[0].passed = False  # type: ignore[misc]


def test_gate_ctx_for_overlay_kwargs_are_keyword_only() -> None:
    import inspect

    import bot.main as bot_main

    sig = inspect.signature(bot_main._gate_ctx_for)
    for name in (
        "market_existing_dollars",
        "event_existing_dollars",
        "series_existing_dollars",
        "aggregate_existing_dollars",
    ):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only, got {param.kind}"
        )


def test_caps_derived_from_bankroll_fractions() -> None:
    import bot.main as bot_main

    seed = bot_main.PAPER_BANKROLL
    assert getattr(bot_main, "MARKET_POSITION_CAP") == seed * Decimal("0.015")
    assert getattr(bot_main, "EVENT_POSITION_CAP") == seed * Decimal("0.03")
    assert getattr(bot_main, "SERIES_POSITION_CAP") == seed * Decimal("0.05")
    assert getattr(bot_main, "AGGREGATE_EXPOSURE_CAP") == seed * Decimal("0.40")


@pytest.mark.parametrize(
    "bankroll,expected_market,expected_event,expected_series,expected_aggregate",
    [
        (
            Decimal("500"),
            Decimal("7.500"),
            Decimal("15.00"),
            Decimal("25.00"),
            Decimal("200.00"),
        ),
        (
            Decimal("1000"),
            Decimal("15.000"),
            Decimal("30.00"),
            Decimal("50.00"),
            Decimal("400.00"),
        ),
        (
            Decimal("5000"),
            Decimal("75.000"),
            Decimal("150.00"),
            Decimal("250.00"),
            Decimal("2000.00"),
        ),
        (
            Decimal("10000"),
            Decimal("150.000"),
            Decimal("300.00"),
            Decimal("500.00"),
            Decimal("4000.00"),
        ),
    ],
)
def test_caps_scale_linearly_with_bankroll(
    bankroll: Decimal,
    expected_market: Decimal,
    expected_event: Decimal,
    expected_series: Decimal,
    expected_aggregate: Decimal,
) -> None:
    import bot.main as bot_main

    assert bankroll * bot_main.MARKET_POSITION_FRAC == expected_market
    assert bankroll * bot_main.EVENT_POSITION_FRAC == expected_event
    assert bankroll * bot_main.SERIES_POSITION_FRAC == expected_series
    assert bankroll * bot_main.AGGREGATE_EXPOSURE_FRAC == expected_aggregate
