from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

import numpy as np
import properscoring

from bot.execution.paper import TradeSide


BRIER_QUANTUM: Decimal = Decimal("0.000001")
RATE_QUANTUM: Decimal = Decimal("0.000001")


@dataclass(frozen=True, slots=True)
class BrierReport:
    n_trades: int
    model_brier: Decimal
    market_brier: Decimal
    delta: Decimal


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lo: Decimal
    hi: Decimal
    n: int
    avg_predicted: Decimal
    avg_observed: Decimal


@dataclass(frozen=True, slots=True)
class SettledTrade:
    side: TradeSide
    simulated_price: Decimal
    contracts: int
    fee_dollars: Decimal
    won: bool


def _validate_outcomes(outcomes: Sequence[int]) -> None:
    for o in outcomes:
        if o != 0 and o != 1:
            raise ValueError(f"outcome must be 0 or 1, got {o}")


def _to_float_array(values: Sequence[Decimal]) -> np.ndarray:
    return np.array([float(v) for v in values], dtype=np.float64)


def brier_score(predictions: Sequence[Decimal], outcomes: Sequence[int]) -> Decimal:
    if len(predictions) == 0:
        raise ValueError("predictions must be non-empty")
    if len(predictions) != len(outcomes):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} outcomes={len(outcomes)}"
        )
    _validate_outcomes(outcomes)

    forecasts = _to_float_array(predictions)
    observations = np.array(outcomes, dtype=np.int64)
    scores = properscoring.brier_score(observations, forecasts)
    mean_score = float(scores.mean())
    return Decimal(str(mean_score)).quantize(BRIER_QUANTUM)


def brier_report(
    model_predictions: Sequence[Decimal],
    market_midpoints: Sequence[Decimal],
    outcomes: Sequence[int],
) -> BrierReport:
    n = len(model_predictions)
    if n == 0:
        raise ValueError("inputs must be non-empty")
    if len(market_midpoints) != n or len(outcomes) != n:
        raise ValueError(
            f"length mismatch: model={n} market={len(market_midpoints)} outcomes={len(outcomes)}"
        )

    model_b = brier_score(model_predictions, outcomes)
    market_b = brier_score(market_midpoints, outcomes)
    return BrierReport(
        n_trades=n,
        model_brier=model_b,
        market_brier=market_b,
        delta=market_b - model_b,
    )


def reliability_diagram(
    predictions: Sequence[Decimal],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if len(predictions) == 0:
        raise ValueError("predictions must be non-empty")
    if len(predictions) != len(outcomes):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} outcomes={len(outcomes)}"
        )
    _validate_outcomes(outcomes)

    width = Decimal(1) / Decimal(n_bins)
    decimal_edges = [width * Decimal(i) for i in range(n_bins + 1)]
    float_edges = np.array([float(e) for e in decimal_edges], dtype=np.float64)

    preds = _to_float_array(predictions)
    obs = np.array(outcomes, dtype=np.float64)

    raw_idx = np.digitize(preds, float_edges[1:-1])
    bin_idx = np.clip(raw_idx, 0, n_bins - 1)

    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo = decimal_edges[i]
        hi = decimal_edges[i + 1]
        mask = bin_idx == i
        n = int(mask.sum())
        if n == 0:
            avg_p = Decimal("0")
            avg_o = Decimal("0")
        else:
            avg_p = Decimal(str(float(preds[mask].mean()))).quantize(BRIER_QUANTUM)
            avg_o = Decimal(str(float(obs[mask].mean()))).quantize(BRIER_QUANTUM)
        out.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                n=n,
                avg_predicted=avg_p,
                avg_observed=avg_o,
            )
        )
    return out


def realized_pnl_for_trade(
    side: TradeSide,
    simulated_price: Decimal,
    contracts: int,
    fee_dollars: Decimal,
    won: bool,
) -> Decimal:
    if contracts <= 0:
        raise ValueError(f"contracts must be > 0, got {contracts}")
    if simulated_price < Decimal("0") or simulated_price > Decimal("1"):
        raise ValueError(f"simulated_price must be in [0, 1], got {simulated_price}")

    n = Decimal(contracts)
    if side is TradeSide.BUY_YES:
        gross = (Decimal("1") - simulated_price) * n if won else -simulated_price * n
    else:
        gross = simulated_price * n if won else -(Decimal("1") - simulated_price) * n

    return gross - fee_dollars


def cumulative_pnl(
    trades: Sequence[SettledTrade],
    include_fees: bool = True,
) -> Decimal:
    total = Decimal("0")
    for t in trades:
        pnl = realized_pnl_for_trade(t.side, t.simulated_price, t.contracts, t.fee_dollars, t.won)
        if not include_fees:
            pnl = pnl + t.fee_dollars
        total = total + pnl
    return total


def gate_failure_rates(
    failures_by_gate: dict[str, int],
    total_evaluations: int,
) -> dict[str, Decimal]:
    if total_evaluations < 0:
        raise ValueError(f"total_evaluations must be >= 0, got {total_evaluations}")
    if total_evaluations == 0:
        return {}

    denom = Decimal(total_evaluations)
    return {
        name: (Decimal(count) / denom).quantize(RATE_QUANTUM)
        for name, count in failures_by_gate.items()
    }
