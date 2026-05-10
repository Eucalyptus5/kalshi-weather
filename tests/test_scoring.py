from __future__ import annotations

from decimal import Decimal

import pytest

from bot.execution.paper import TradeSide
from bot.validation.scoring import (
    BrierReport,
    ReliabilityBin,
    SettledTrade,
    brier_report,
    brier_score,
    cumulative_pnl,
    gate_failure_rates,
    realized_pnl_for_trade,
    reliability_diagram,
)


def _d(s: str) -> Decimal:
    return Decimal(s)


def test_brier_perfect_prediction_is_zero() -> None:
    score = brier_score(
        [_d("1.0"), _d("0.0"), _d("1.0")],
        [1, 0, 1],
    )
    assert score == Decimal("0.000000")


def test_brier_perfect_anti_prediction_is_one() -> None:
    score = brier_score(
        [_d("0.0"), _d("1.0"), _d("0.0")],
        [1, 0, 1],
    )
    assert score == Decimal("1.000000")


def test_brier_coin_flip_is_quarter() -> None:
    score = brier_score(
        [_d("0.5"), _d("0.5"), _d("0.5"), _d("0.5")],
        [1, 0, 1, 0],
    )
    assert score == Decimal("0.250000")


def test_brier_returns_decimal() -> None:
    score = brier_score([_d("0.5")], [1])
    assert isinstance(score, Decimal)


def test_brier_empty_raises() -> None:
    with pytest.raises(ValueError):
        brier_score([], [])


def test_brier_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        brier_score([_d("0.5")], [1, 0])


@pytest.mark.parametrize("bad", [2, -1, 3])
def test_brier_non_binary_outcome_raises(bad: int) -> None:
    with pytest.raises(ValueError):
        brier_score([_d("0.5"), _d("0.5")], [1, bad])


def test_brier_report_model_beats_baseline() -> None:
    report = brier_report(
        model_predictions=[_d("0.9"), _d("0.1"), _d("0.9")],
        market_midpoints=[_d("0.5"), _d("0.5"), _d("0.5")],
        outcomes=[1, 0, 1],
    )
    assert isinstance(report, BrierReport)
    assert report.n_trades == 3
    assert report.model_brier < report.market_brier
    assert report.delta > Decimal("0")
    assert report.delta == report.market_brier - report.model_brier


def test_brier_report_market_beats_model() -> None:
    report = brier_report(
        model_predictions=[_d("0.1"), _d("0.9"), _d("0.1")],
        market_midpoints=[_d("0.5"), _d("0.5"), _d("0.5")],
        outcomes=[1, 0, 1],
    )
    assert report.delta < Decimal("0")


def test_brier_report_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        brier_report(
            model_predictions=[_d("0.5"), _d("0.5")],
            market_midpoints=[_d("0.5")],
            outcomes=[1, 0],
        )


def test_brier_report_decimal_types() -> None:
    report = brier_report(
        model_predictions=[_d("0.5")],
        market_midpoints=[_d("0.5")],
        outcomes=[1],
    )
    assert isinstance(report.model_brier, Decimal)
    assert isinstance(report.market_brier, Decimal)
    assert isinstance(report.delta, Decimal)


def test_reliability_default_bin_partition() -> None:
    bins = reliability_diagram([_d("0.5")], [1])
    assert len(bins) == 10
    expected_edges = [
        (Decimal("0.0"), Decimal("0.1")),
        (Decimal("0.1"), Decimal("0.2")),
        (Decimal("0.2"), Decimal("0.3")),
        (Decimal("0.3"), Decimal("0.4")),
        (Decimal("0.4"), Decimal("0.5")),
        (Decimal("0.5"), Decimal("0.6")),
        (Decimal("0.6"), Decimal("0.7")),
        (Decimal("0.7"), Decimal("0.8")),
        (Decimal("0.8"), Decimal("0.9")),
        (Decimal("0.9"), Decimal("1.0")),
    ]
    for b, (lo, hi) in zip(bins, expected_edges):
        assert b.lo == lo
        assert b.hi == hi


def test_reliability_one_per_bin() -> None:
    preds = [
        _d(s)
        for s in ("0.05", "0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75", "0.85", "0.95")
    ]
    outs = [1] * 10
    bins = reliability_diagram(preds, outs)
    for b, p in zip(bins, preds):
        assert b.n == 1
        assert b.avg_predicted == p.quantize(Decimal("0.000001"))
        assert b.avg_observed == Decimal("1.000000")


def test_reliability_boundary_assignment() -> None:
    preds = [_d("0.1")]
    outs = [1]
    bins = reliability_diagram(preds, outs)
    assert bins[0].n == 0
    assert bins[1].n == 1
    assert bins[1].avg_predicted == Decimal("0.100000")


def test_reliability_final_bin_includes_one() -> None:
    preds = [_d("1.0")]
    outs = [1]
    bins = reliability_diagram(preds, outs)
    assert bins[9].n == 1
    assert bins[9].avg_predicted == Decimal("1.000000")


def test_reliability_empty_bins() -> None:
    preds = [_d("0.42"), _d("0.43"), _d("0.49")]
    outs = [1, 0, 1]
    bins = reliability_diagram(preds, outs)
    for i, b in enumerate(bins):
        if i == 4:
            assert b.n == 3
        else:
            assert b.n == 0
            assert b.avg_predicted == Decimal("0")
            assert b.avg_observed == Decimal("0")


def test_reliability_custom_bins() -> None:
    bins = reliability_diagram([_d("0.5")], [1], n_bins=4)
    assert len(bins) == 4
    expected = [
        (Decimal("0.00"), Decimal("0.25")),
        (Decimal("0.25"), Decimal("0.50")),
        (Decimal("0.50"), Decimal("0.75")),
        (Decimal("0.75"), Decimal("1.00")),
    ]
    for b, (lo, hi) in zip(bins, expected):
        assert b.lo == lo
        assert b.hi == hi


def test_reliability_one_bin_raises() -> None:
    with pytest.raises(ValueError):
        reliability_diagram([_d("0.5")], [1], n_bins=1)


def test_reliability_empty_raises() -> None:
    with pytest.raises(ValueError):
        reliability_diagram([], [])


def test_reliability_returns_decimal_fields() -> None:
    bins = reliability_diagram([_d("0.5")], [1])
    for b in bins:
        assert isinstance(b.lo, Decimal)
        assert isinstance(b.hi, Decimal)
        assert isinstance(b.avg_predicted, Decimal)
        assert isinstance(b.avg_observed, Decimal)


@pytest.mark.parametrize(
    "side, price, contracts, fee, won, expected",
    [
        (TradeSide.BUY_YES, Decimal("0.40"), 10, Decimal("0.05"), True, Decimal("5.95")),
        (TradeSide.BUY_YES, Decimal("0.40"), 10, Decimal("0.05"), False, Decimal("-4.05")),
        (TradeSide.SELL_YES, Decimal("0.30"), 10, Decimal("0.05"), True, Decimal("2.95")),
        (TradeSide.SELL_YES, Decimal("0.30"), 10, Decimal("0.05"), False, Decimal("-7.05")),
    ],
)
def test_realized_pnl_golden(
    side: TradeSide,
    price: Decimal,
    contracts: int,
    fee: Decimal,
    won: bool,
    expected: Decimal,
) -> None:
    assert realized_pnl_for_trade(side, price, contracts, fee, won) == expected


def test_realized_pnl_returns_decimal() -> None:
    pnl = realized_pnl_for_trade(TradeSide.BUY_YES, Decimal("0.5"), 1, Decimal("0.01"), True)
    assert isinstance(pnl, Decimal)


def test_realized_pnl_zero_contracts_raises() -> None:
    with pytest.raises(ValueError):
        realized_pnl_for_trade(TradeSide.BUY_YES, Decimal("0.5"), 0, Decimal("0"), True)


@pytest.mark.parametrize("price", [Decimal("-0.01"), Decimal("1.01")])
def test_realized_pnl_bad_price_raises(price: Decimal) -> None:
    with pytest.raises(ValueError):
        realized_pnl_for_trade(TradeSide.BUY_YES, price, 1, Decimal("0"), True)


def _trade(side: TradeSide, price: str, contracts: int, fee: str, won: bool) -> SettledTrade:
    return SettledTrade(
        side=side,
        simulated_price=Decimal(price),
        contracts=contracts,
        fee_dollars=Decimal(fee),
        won=won,
    )


def test_cumulative_pnl_sum() -> None:
    trades = [
        _trade(TradeSide.BUY_YES, "0.40", 10, "0.05", True),
        _trade(TradeSide.BUY_YES, "0.40", 10, "0.05", False),
        _trade(TradeSide.SELL_YES, "0.30", 10, "0.05", True),
    ]
    expected = Decimal("5.95") + Decimal("-4.05") + Decimal("2.95")
    assert cumulative_pnl(trades) == expected


def test_cumulative_pnl_empty() -> None:
    assert cumulative_pnl([]) == Decimal("0")


def test_cumulative_pnl_returns_decimal() -> None:
    assert isinstance(cumulative_pnl([]), Decimal)


def test_cumulative_pnl_without_fees() -> None:
    trades = [
        _trade(TradeSide.BUY_YES, "0.40", 10, "0.05", True),
        _trade(TradeSide.SELL_YES, "0.30", 10, "0.05", False),
    ]
    with_fees = cumulative_pnl(trades, include_fees=True)
    without_fees = cumulative_pnl(trades, include_fees=False)
    total_fees = sum((t.fee_dollars for t in trades), Decimal("0"))
    assert without_fees == with_fees + total_fees


def test_gate_failure_rates_basic() -> None:
    rates = gate_failure_rates({"fair_value_sane": 5, "model_fresh": 10}, 100)
    assert rates == {
        "fair_value_sane": Decimal("0.050000"),
        "model_fresh": Decimal("0.100000"),
    }


def test_gate_failure_rates_zero_evals() -> None:
    assert gate_failure_rates({"fair_value_sane": 5}, 0) == {}


def test_gate_failure_rates_empty_failures() -> None:
    assert gate_failure_rates({}, 100) == {}


def test_gate_failure_rates_negative_evals_raises() -> None:
    with pytest.raises(ValueError):
        gate_failure_rates({}, -1)


def test_gate_failure_rates_returns_decimal_values() -> None:
    rates = gate_failure_rates({"x": 1}, 4)
    for v in rates.values():
        assert isinstance(v, Decimal)


def test_reliability_bin_dataclass_fields() -> None:
    b = ReliabilityBin(
        lo=Decimal("0"),
        hi=Decimal("0.1"),
        n=0,
        avg_predicted=Decimal("0"),
        avg_observed=Decimal("0"),
    )
    assert b.n == 0
