from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from statistics import median
from typing import Literal

import numpy as np


Direction = Literal["greater", "less"]

DIRECTIONS: tuple[Direction, ...] = ("greater", "less")

# A mid-latitude synoptic system lives about three days, so days inside that span share weather
# state and are not independent draws. Fixed from meteorology, never from a statistic on this tape.
BLOCK_DAYS: int = 3

# 0.05 Bonferroni-corrected at four questions.
ALPHA: float = 0.0125
# Holdout replication is one test of a direction the discovery already fixed, so it carries no
# family correction of its own.
HOLDOUT_ALPHA: float = 0.05


@dataclass(frozen=True, slots=True)
class ClusterAggregate:
    cluster: str
    total: Decimal
    weight: Decimal


@dataclass(frozen=True, slots=True)
class ValueCluster:
    cluster: str
    values: tuple[Decimal, ...]


@dataclass(frozen=True, slots=True)
class CorridorDayAggregate:
    corridor: str
    day: date
    total: Decimal
    weight: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapResult:
    estimate: Decimal
    null_value: Decimal
    direction: Direction
    p_value: float
    ci_level: float
    ci_low: float
    ci_high: float
    replicate_spread: float
    degenerate: bool
    n_clusters: int
    resamples: int
    seed: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WildBootstrapResult:
    estimate: Decimal
    null_value: Decimal
    direction: Direction
    standard_error: float
    t_statistic: float
    p_value: float
    ci_level: float
    ci_low: float
    ci_high: float
    n_observations: int
    n_blocks: int
    block_days: int
    degenerate_resamples: int
    resamples: int
    seed: int


def cluster_bootstrap(
    clusters: Sequence[ClusterAggregate],
    *,
    null_value: Decimal,
    direction: Direction,
    resamples: int,
    seed: int,
    ci_level: float,
) -> BootstrapResult:
    if not clusters:
        raise ValueError("a cluster bootstrap needs at least one cluster")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    unweighted = [item.cluster for item in clusters if item.weight <= 0]
    if unweighted:
        raise ValueError("clusters carry a non-positive weight: " + ", ".join(unweighted))

    estimate = sum((item.total for item in clusters), Decimal(0)) / sum(
        (item.weight for item in clusters), Decimal(0)
    )

    totals = np.array([float(item.total) for item in clusters])
    weights = np.array([float(item.weight) for item in clusters])
    rng = np.random.default_rng(seed)
    drawn = rng.integers(len(clusters), size=(resamples, len(clusters)))
    replicates = totals[drawn].sum(axis=1) / weights[drawn].sum(axis=1)

    theta = float(estimate)
    pivot = replicates - theta
    observed = theta - float(null_value)
    extreme = pivot >= observed if direction == "greater" else pivot <= observed

    # The basic interval reflects the replicates through the estimate, so it inverts the same
    # pivot the p-value tests and its upper quantile carries the lower bound.
    low_q, high_q = np.percentile(replicates, [50 * (1 - ci_level), 50 * (1 + ci_level)])
    replicate_spread, degenerate = _replicate_spread(replicates)

    return BootstrapResult(
        estimate=estimate,
        null_value=null_value,
        direction=direction,
        p_value=(1 + int(np.count_nonzero(extreme))) / (resamples + 1),
        ci_level=ci_level,
        ci_low=2 * theta - float(high_q),
        ci_high=2 * theta - float(low_q),
        replicate_spread=replicate_spread,
        degenerate=degenerate,
        n_clusters=len(clusters),
        resamples=resamples,
        seed=seed,
    )


def cluster_median_bootstrap(
    clusters: Sequence[ValueCluster],
    *,
    null_value: Decimal,
    direction: Direction,
    resamples: int,
    seed: int,
    ci_level: float,
) -> BootstrapResult:
    if not clusters:
        raise ValueError("a cluster median bootstrap needs at least one cluster")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    silent = [item.cluster for item in clusters if not item.values]
    if silent:
        raise ValueError("clusters carry no values: " + ", ".join(silent))

    estimate = median(value for item in clusters for value in item.values)

    pooled = np.array([float(value) for item in clusters for value in item.values])
    bounds = np.cumsum([0, *(len(item.values) for item in clusters)])
    picks = [np.arange(bounds[index], bounds[index + 1]) for index in range(len(clusters))]
    rng = np.random.default_rng(seed)
    drawn = rng.integers(len(clusters), size=(resamples, len(clusters)))
    replicates = np.array(
        [np.median(pooled[np.concatenate([picks[index] for index in row])]) for row in drawn]
    )

    theta = float(estimate)
    pivot = replicates - theta
    observed = theta - float(null_value)
    extreme = pivot >= observed if direction == "greater" else pivot <= observed

    low_q, high_q = np.percentile(replicates, [50 * (1 - ci_level), 50 * (1 + ci_level)])
    replicate_spread, degenerate = _replicate_spread(replicates)

    return BootstrapResult(
        estimate=estimate,
        null_value=null_value,
        direction=direction,
        p_value=(1 + int(np.count_nonzero(extreme))) / (resamples + 1),
        ci_level=ci_level,
        ci_low=2 * theta - float(high_q),
        ci_high=2 * theta - float(low_q),
        replicate_spread=replicate_spread,
        degenerate=degenerate,
        n_clusters=len(clusters),
        resamples=resamples,
        seed=seed,
    )


def day_blocks(observations: Sequence[CorridorDayAggregate], block_days: int) -> tuple[int, ...]:
    if block_days < 1:
        raise ValueError(f"block_days must be >= 1, got {block_days}")
    block_of: dict[tuple[str, date], int] = {}
    corridor_now = None
    position = 0
    index = -1
    for corridor, day in sorted({(item.corridor, item.day) for item in observations}):
        if corridor != corridor_now:
            corridor_now = corridor
            position = 0
        if position % block_days == 0:
            index += 1
        block_of[(corridor, day)] = index
        position += 1
    return tuple(block_of[(item.corridor, item.day)] for item in observations)


def _block_se(residual: np.ndarray, weight_total: float, design: np.ndarray) -> np.ndarray:
    block_sums = np.atleast_2d(residual) @ design
    return np.sqrt(np.square(block_sums).sum(axis=1)) / weight_total


def _vanishes(block_se: np.ndarray, scale: np.ndarray, n_terms: int) -> np.ndarray:
    # A flat panel and a degenerate replicate both difference to zero in real arithmetic, so all
    # that reaches float64 is summation noise, which only reads as zero against the scale it
    # cancelled from. Testing == 0 instead lets a rate like 1/3 carry a 1e-29 standard error.
    return block_se <= n_terms * np.finfo(float).eps * scale


def _replicate_spread(replicates: np.ndarray) -> tuple[float, bool]:
    spread = float(np.std(replicates))
    scale = float(np.mean(np.abs(replicates)))
    degenerate = _vanishes(np.array([spread]), np.array([scale]), replicates.size)[0]
    return spread, bool(degenerate)


def _wild_t(
    residual: np.ndarray,
    weights: np.ndarray,
    weight_total: float,
    design: np.ndarray,
    signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shifted = (signs @ design.T) * residual
    offset = shifted.sum(axis=1) / weight_total
    center = offset[:, None] * weights
    replicate_se = _block_se(shifted - center, weight_total, design)
    scale = _block_se(np.abs(shifted) + np.abs(center), weight_total, design)
    degenerate = _vanishes(replicate_se, scale, residual.size)
    return offset / np.where(degenerate, 1.0, replicate_se), degenerate


def wild_cluster_bootstrap(
    observations: Sequence[CorridorDayAggregate],
    *,
    null_value: Decimal,
    direction: Direction,
    block_days: int,
    resamples: int,
    seed: int,
    ci_level: float,
) -> WildBootstrapResult:
    if not observations:
        raise ValueError("a wild cluster bootstrap needs at least one observation")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    unweighted = [item.corridor for item in observations if item.weight <= 0]
    if unweighted:
        raise ValueError("observations carry a non-positive weight: " + ", ".join(unweighted))

    estimate = sum((item.total for item in observations), Decimal(0)) / sum(
        (item.weight for item in observations), Decimal(0)
    )

    blocks = day_blocks(observations, block_days)
    n_blocks = max(blocks) + 1
    design = np.zeros((len(observations), n_blocks))
    design[np.arange(len(observations)), blocks] = 1.0

    weights = np.array([float(item.weight) for item in observations])
    weight_total = float(weights.sum())
    theta = float(estimate)
    null = float(null_value)

    # Residuals are differenced in Decimal and cast once rather than subtracted in float64, where
    # a near-flat panel loses every significant digit it has to the cancellation.
    residual = np.array([float(item.total - estimate * item.weight) for item in observations])
    null_residual = np.array(
        [float(item.total - null_value * item.weight) for item in observations]
    )
    magnitude = np.array([float(abs(item.total)) for item in observations]) + np.abs(
        theta * weights
    )

    sample_se = _block_se(residual, weight_total, design)
    if _vanishes(sample_se, _block_se(magnitude, weight_total, design), len(observations))[0]:
        raise ValueError("the block-clustered standard error vanishes, so no statistic is defined")
    standard_error = float(sample_se[0])
    observed = (theta - null) / standard_error

    rng = np.random.default_rng(seed)
    null_t, degenerate = _wild_t(
        null_residual,
        weights,
        weight_total,
        design,
        rng.integers(2, size=(resamples, n_blocks)) * 2.0 - 1.0,
    )
    beyond = null_t >= observed if direction == "greater" else null_t <= observed
    # A degenerate replicate carries no studentized value, so counting it as extreme in either
    # direction keeps the p-value conservative rather than optimistic.
    extreme = degenerate | beyond

    free_t, free_degenerate = _wild_t(
        residual,
        weights,
        weight_total,
        design,
        rng.integers(2, size=(resamples, n_blocks)) * 2.0 - 1.0,
    )
    usable = free_t[~free_degenerate]
    if usable.size == 0:
        raise ValueError("every resample is degenerate, so no percentile-t interval exists")

    # The interval comes off unrestricted replicates so it does not move with the null the p-value
    # tests, and the percentile-t inversion puts the upper quantile on the lower bound.
    high_q, low_q = np.percentile(usable, [50 * (1 + ci_level), 50 * (1 - ci_level)])

    return WildBootstrapResult(
        estimate=estimate,
        null_value=null_value,
        direction=direction,
        standard_error=standard_error,
        t_statistic=observed,
        p_value=(1 + int(np.count_nonzero(extreme))) / (resamples + 1),
        ci_level=ci_level,
        ci_low=theta - standard_error * float(high_q),
        ci_high=theta - standard_error * float(low_q),
        n_observations=len(observations),
        n_blocks=n_blocks,
        block_days=block_days,
        degenerate_resamples=int(np.count_nonzero(degenerate)),
        resamples=resamples,
        seed=seed,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class GateVerdict:
    economic: bool
    significant: bool
    powered: bool
    undecidable: bool
    passed: bool
    estimate: Decimal
    threshold: Decimal
    direction: Direction
    p_value: float
    alpha: float
    n: int
    n_min: int
    n_unit: str


@dataclass(frozen=True, slots=True, kw_only=True)
class HoldoutVerdict:
    same_sign: bool
    magnitude: bool
    significant: bool
    powered: bool
    undecidable: bool
    replicated: bool
    discovery_estimate: Decimal
    holdout_estimate: Decimal
    holdout_p_value: float
    alpha: float
    holdout_n: int
    holdout_n_min: int
    n_unit: str


def evaluate_gate(
    *,
    estimate: Decimal,
    p_value: float,
    result: BootstrapResult,
    threshold: Decimal,
    direction: Direction,
    alpha: float,
    n_min: int,
    n_unit: str,
    undecidable: bool,
    strict: bool = False,
) -> GateVerdict:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    # Equality means the bar has been met unless the bar is zero, where it means no edge at all,
    # so the inclusive comparison stays the default and a zero bar asks for the strict one.
    if direction == "greater":
        economic = estimate > threshold if strict else estimate >= threshold
    else:
        economic = estimate < threshold if strict else estimate <= threshold
    # An undecidable estimate carries no p-value worth comparing, and refusing at significance
    # rather than at passed is what makes the refusal bind on every condition downstream of it.
    significant = not undecidable and p_value < alpha
    powered = result.n_clusters >= n_min
    return GateVerdict(
        economic=economic,
        significant=significant,
        powered=powered,
        undecidable=undecidable,
        passed=economic and significant and powered,
        estimate=estimate,
        threshold=threshold,
        direction=direction,
        p_value=p_value,
        alpha=alpha,
        n=result.n_clusters,
        n_min=n_min,
        n_unit=n_unit,
    )


def evaluate_holdout(
    *,
    discovery_estimate: Decimal,
    holdout_estimate: Decimal,
    holdout_p_value: float,
    holdout_result: BootstrapResult,
    discovery_n_min: int,
    alpha: float,
    n_unit: str,
    undecidable: bool,
) -> HoldoutVerdict:
    if discovery_estimate == 0:
        raise ValueError("a discovery estimate of zero fixes no direction for holdout to replicate")
    same_sign = (holdout_estimate > 0) == (discovery_estimate > 0) and holdout_estimate != 0
    magnitude = abs(holdout_estimate) >= abs(discovery_estimate) / 2
    significant = not undecidable and holdout_p_value < alpha
    holdout_n_min = (discovery_n_min + 1) // 2
    powered = holdout_result.n_clusters >= holdout_n_min
    return HoldoutVerdict(
        same_sign=same_sign,
        magnitude=magnitude,
        significant=significant,
        powered=powered,
        undecidable=undecidable,
        replicated=same_sign and magnitude and significant and powered,
        discovery_estimate=discovery_estimate,
        holdout_estimate=holdout_estimate,
        holdout_p_value=holdout_p_value,
        alpha=alpha,
        holdout_n=holdout_result.n_clusters,
        holdout_n_min=holdout_n_min,
        n_unit=n_unit,
    )
