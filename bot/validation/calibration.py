from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal
from typing import Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import select
from sqlalchemy.orm import Session

from bot.risk.gates import GateParams
from bot.storage.sqlite import Market, PaperTradeRow, SimulatedPnl
from bot.validation.scoring import brier_score


logger = logging.getLogger(__name__)


MIN_FIT_SAMPLES: int = 300
SPARSE_BUCKET_MIN_SAMPLES: int = 1000
MIN_POSITIVES_PER_BUCKET: int = 30
REFIT_WINDOW_DAYS: int = 90
HOLDOUT_DAYS: int = 7
SETTLEMENT_LAG_BUFFER_HOURS: int = 1
CORRECTION_QUANTUM: Decimal = Decimal("0.000001")
LEAD_TIME_NONE_SENTINEL_HOURS: int = 9999
LEAD_TIME_NONE_BUCKET_IDX: int = -1
TAILS_PRICE_EDGES: tuple[Decimal, ...] = (Decimal("0.02"),)
EDGE_PRICE_EDGES: tuple[Decimal, ...] = (
    Decimal("0.05"),
    Decimal("0.10"),
    Decimal("0.15"),
    Decimal("0.20"),
    Decimal("0.30"),
)
LEAD_TIME_EDGES_HOURS: tuple[int, ...] = (12, 24, 48)


BSS_AGGREGATE_NA: str = "NA"


@dataclass(frozen=True, slots=True)
class CalibrationMaps:
    maps: Mapping[tuple[str, int, int], IsotonicRegression]
    fitted_at: datetime
    n_samples_per_bucket: Mapping[tuple[str, int, int], int]
    holdout_bs_new: Mapping[tuple[str, int, int], Decimal]
    holdout_bs_prev: Mapping[tuple[str, int, int], Decimal]
    holdout_n_per_bucket: Mapping[tuple[str, int, int], int]
    climatological_rate_per_bucket: Mapping[tuple[str, int, int], Decimal]
    bss_aggregate_per_stratum: Mapping[str, Decimal | str]


def _price_edges_for(strategy: str) -> tuple[Decimal, ...]:
    if strategy == "tails":
        return TAILS_PRICE_EDGES
    if strategy == "edge":
        return EDGE_PRICE_EDGES
    raise ValueError(f"unknown strategy {strategy!r}")


def _price_bucket_idx(strategy: str, q_raw: Decimal) -> int:
    edges = _price_edges_for(strategy)
    for idx, upper in enumerate(edges):
        if q_raw <= upper:
            return idx
    return len(edges)


def _lead_bucket_idx(lead_time_hours: int) -> int:
    for idx, upper in enumerate(LEAD_TIME_EDGES_HOURS):
        if lead_time_hours <= upper:
            return idx
    return len(LEAD_TIME_EDGES_HOURS) - 1


def bucket_for(strategy: str, q_raw: Decimal, lead_time_hours: int) -> tuple[str, int, int]:
    if lead_time_hours == LEAD_TIME_NONE_SENTINEL_HOURS:
        return (strategy, _price_bucket_idx(strategy, q_raw), LEAD_TIME_NONE_BUCKET_IDX)
    return (strategy, _price_bucket_idx(strategy, q_raw), _lead_bucket_idx(lead_time_hours))


def hours_until(close_time: datetime | None, now: datetime) -> int:
    if close_time is None:
        return LEAD_TIME_NONE_SENTINEL_HOURS
    delta = (close_time - now).total_seconds() / 3600.0
    if delta < 0:
        return 0
    return int(delta)


def adaptive_min_samples(strategy: str, price_bucket_idx: int) -> int:
    edges = _price_edges_for(strategy)
    if price_bucket_idx < len(edges):
        upper = edges[price_bucket_idx]
    else:
        upper = Decimal("1")
    expected_positives = upper * Decimal(MIN_FIT_SAMPLES)
    if expected_positives < Decimal("4"):
        return SPARSE_BUCKET_MIN_SAMPLES
    return MIN_FIT_SAMPLES


def _min_nonzero_prediction(bucket_climatological_rate: Decimal) -> Decimal:
    return min(GateParams().fair_min, bucket_climatological_rate / Decimal("2"))


def brier_skill_score(
    model_predictions: Sequence[Decimal],
    outcomes: Sequence[int],
    climatological_rate: Decimal,
) -> Decimal:
    bs_model = brier_score(model_predictions, outcomes)
    bs_clim = climatological_rate * (Decimal("1") - climatological_rate)
    if bs_clim == Decimal("0"):
        raise ValueError("climatological_rate must be in (0, 1)")
    return (Decimal("1") - bs_model / bs_clim).quantize(CORRECTION_QUANTUM)


def fit_isotonic(q_raws: Sequence[Decimal], outcomes: Sequence[int]) -> IsotonicRegression:
    estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    x = np.array([float(q) for q in q_raws], dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    estimator.fit(x, y)
    return estimator


def corrected_probability(
    raw_q: Decimal,
    strategy: str,
    lead_time_hours: int,
    maps: CalibrationMaps,
) -> Decimal:
    key = bucket_for(strategy, raw_q, lead_time_hours)
    estimator = maps.maps.get(key)
    if estimator is None:
        return raw_q
    raw_f = float(raw_q)
    if raw_f < estimator.X_min_ or raw_f > estimator.X_max_:
        return raw_q
    predicted = float(estimator.predict([raw_f])[0])
    floor = _min_nonzero_prediction(maps.climatological_rate_per_bucket[key])
    if Decimal(str(predicted)) < floor:
        return raw_q
    return Decimal(str(predicted)).quantize(CORRECTION_QUANTUM)


def format_gate_failure_reason(
    base: str,
    *,
    q_raw: Decimal,
    fair_yes: Decimal,
    bucket_key: tuple[str, int, int],
) -> str:
    q_raw_q = q_raw.quantize(CORRECTION_QUANTUM)
    q_corr_q = fair_yes.quantize(CORRECTION_QUANTUM)
    return f"{base} q_raw={q_raw_q} q_corrected={q_corr_q} bucket={bucket_key}"


def _quantize_rate(rate: float) -> Decimal:
    return Decimal(str(rate)).quantize(CORRECTION_QUANTUM)


def _apply_bucket(
    estimator: IsotonicRegression, q_raws: np.ndarray, climatological_rate: Decimal
) -> tuple[np.ndarray, int]:
    floor = float(_min_nonzero_prediction(climatological_rate))
    out = np.empty_like(q_raws, dtype=np.float64)
    identity = 0
    for i, raw_f in enumerate(q_raws):
        if raw_f < estimator.X_min_ or raw_f > estimator.X_max_:
            out[i] = raw_f
            identity += 1
            continue
        pred = float(estimator.predict([raw_f])[0])
        if pred < floor:
            out[i] = raw_f
            identity += 1
        else:
            out[i] = pred
    return out, identity


def _mean_brier(predictions: np.ndarray, outcomes: np.ndarray) -> Decimal:
    diffs = predictions - outcomes
    return Decimal(str(float(np.mean(diffs * diffs)))).quantize(CORRECTION_QUANTUM)


def refit_all(session: Session, prev_maps: CalibrationMaps | None = None) -> CalibrationMaps:
    now = datetime.now(tz=_timezone.utc)
    window_start = now - timedelta(days=REFIT_WINDOW_DAYS)
    holdout_start = now - timedelta(days=HOLDOUT_DAYS)
    settlement_cutoff = now - timedelta(hours=SETTLEMENT_LAG_BUFFER_HOURS)

    rows = session.execute(
        select(
            PaperTradeRow.q_raw,
            SimulatedPnl.outcome,
            PaperTradeRow.intended_at,
            Market.close_time,
            PaperTradeRow.strategy,
        )
        .join(SimulatedPnl, SimulatedPnl.paper_trade_id == PaperTradeRow.id)
        .join(Market, Market.ticker == PaperTradeRow.market_ticker, isouter=True)
        .where(PaperTradeRow.intended_at >= window_start)
        .where(SimulatedPnl.settled_at <= settlement_cutoff)
    ).all()

    fit_by_bucket: dict[tuple[str, int, int], list[tuple[Decimal, int]]] = {}
    holdout_by_bucket: dict[tuple[str, int, int], list[tuple[Decimal, int]]] = {}
    for q_raw, outcome, intended_at, close_time, strategy in rows:
        if q_raw is None:
            continue
        lead = hours_until(close_time, intended_at)
        key = bucket_for(strategy, q_raw, lead)
        if key[2] == LEAD_TIME_NONE_BUCKET_IDX:
            continue
        outcome_int = 1 if outcome == "won" else 0
        if intended_at < holdout_start:
            fit_by_bucket.setdefault(key, []).append((q_raw, outcome_int))
        else:
            holdout_by_bucket.setdefault(key, []).append((q_raw, outcome_int))

    maps: dict[tuple[str, int, int], IsotonicRegression] = {}
    n_samples: dict[tuple[str, int, int], int] = {}
    holdout_n: dict[tuple[str, int, int], int] = {}
    holdout_bs_new: dict[tuple[str, int, int], Decimal] = {}
    holdout_bs_prev: dict[tuple[str, int, int], Decimal] = {}
    climatological_rate_per_bucket: dict[tuple[str, int, int], Decimal] = {}
    stratum_fit_preds: dict[str, list[float]] = {"tails": [], "edge": []}
    stratum_fit_outcomes: dict[str, list[int]] = {"tails": [], "edge": []}

    for key, samples in fit_by_bucket.items():
        strategy, price_idx, lead_idx = key
        min_n = adaptive_min_samples(strategy, price_idx)
        if len(samples) < min_n:
            continue
        outcomes_arr = np.array([s[1] for s in samples], dtype=np.int64)
        positives = int(outcomes_arr.sum())
        if positives < MIN_POSITIVES_PER_BUCKET:
            continue
        q_arr = np.array([float(s[0]) for s in samples], dtype=np.float64)
        estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        estimator.fit(q_arr, outcomes_arr.astype(np.float64))
        if estimator.y_thresholds_[0] == 0.0:
            logger.warning(
                "calibration_first_knot_zero strategy=%s price_idx=%d lead_idx=%d",
                strategy,
                price_idx,
                lead_idx,
            )
        maps[key] = estimator
        n_samples[key] = len(samples)
        clim = _quantize_rate(float(outcomes_arr.mean()))
        climatological_rate_per_bucket[key] = clim

        fit_preds, fit_identity = _apply_bucket(estimator, q_arr, clim)
        bss_fit = (
            brier_skill_score(
                [Decimal(str(p)) for p in fit_preds.tolist()],
                outcomes_arr.tolist(),
                clim,
            )
            if clim > Decimal("0") and clim < Decimal("1")
            else Decimal("0")
        )
        stratum_fit_preds[strategy].extend(fit_preds.tolist())
        stratum_fit_outcomes[strategy].extend(outcomes_arr.tolist())
        fit_identity_pct = (
            Decimal(fit_identity) / Decimal(len(samples)) * Decimal("100")
        ).quantize(CORRECTION_QUANTUM)

        holdout_samples = holdout_by_bucket.get(key, [])
        holdout_n[key] = len(holdout_samples)
        if holdout_samples:
            h_q = np.array([float(s[0]) for s in holdout_samples], dtype=np.float64)
            h_y = np.array([s[1] for s in holdout_samples], dtype=np.float64)
            new_preds, holdout_identity = _apply_bucket(estimator, h_q, clim)
            holdout_bs_new[key] = _mean_brier(new_preds, h_y)
            prev_estimator = prev_maps.maps.get(key) if prev_maps is not None else None
            if prev_estimator is not None:
                prev_preds, _ = _apply_bucket(
                    prev_estimator,
                    h_q,
                    prev_maps.climatological_rate_per_bucket[key],
                )
                holdout_bs_prev[key] = _mean_brier(prev_preds, h_y)
            else:
                holdout_bs_prev[key] = _mean_brier(h_q.copy(), h_y)
            holdout_identity_pct = (
                Decimal(holdout_identity) / Decimal(len(holdout_samples)) * Decimal("100")
            ).quantize(CORRECTION_QUANTUM)
        else:
            holdout_bs_new[key] = Decimal("0")
            holdout_bs_prev[key] = Decimal("0")
            holdout_identity_pct = Decimal("0")

        logger.info(
            "calibration_bucket strategy=%s price_idx=%d lead_idx=%d n_fit=%d "
            "n_holdout=%d bss_fit=%s bs_new=%s bs_prev=%s fit_identity_pct=%s "
            "holdout_identity_pct=%s",
            strategy,
            price_idx,
            lead_idx,
            len(samples),
            holdout_n[key],
            bss_fit,
            holdout_bs_new[key],
            holdout_bs_prev[key],
            fit_identity_pct,
            holdout_identity_pct,
        )

    bss_aggregate_per_stratum: dict[str, Decimal | str] = {}
    for stratum in ("tails", "edge"):
        preds = stratum_fit_preds[stratum]
        outcomes = stratum_fit_outcomes[stratum]
        if not preds:
            bss_aggregate_per_stratum[stratum] = BSS_AGGREGATE_NA
            continue
        clim_rate = Decimal(str(sum(outcomes) / len(outcomes))).quantize(CORRECTION_QUANTUM)
        if clim_rate <= Decimal("0") or clim_rate >= Decimal("1"):
            bss_aggregate_per_stratum[stratum] = BSS_AGGREGATE_NA
            continue
        bss_aggregate_per_stratum[stratum] = brier_skill_score(
            [Decimal(str(p)) for p in preds],
            outcomes,
            clim_rate,
        )

    return CalibrationMaps(
        maps=maps,
        fitted_at=now,
        n_samples_per_bucket=n_samples,
        holdout_bs_new=holdout_bs_new,
        holdout_bs_prev=holdout_bs_prev,
        holdout_n_per_bucket=holdout_n,
        climatological_rate_per_bucket=climatological_rate_per_bucket,
        bss_aggregate_per_stratum=bss_aggregate_per_stratum,
    )
