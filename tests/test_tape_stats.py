from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from bot.lag.tape_stats import (
    ClusterAggregate,
    CorridorDayAggregate,
    cluster_bootstrap,
    day_blocks,
    evaluate_gate,
    evaluate_holdout,
    wild_cluster_bootstrap,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CHILD_SOURCE = (
    "import sys; sys.path[:0] = ['.', 'tests']; "
    "import test_tape_stats; print(test_tape_stats.reference_digest())"
)


FIXTURE_ALPHA = 0.05
FIXTURE_CI_LEVEL = 0.90
FIXTURE_RESAMPLES = 999


FIRST_DAY = date(2026, 7, 1)


FLAT_PANELS: dict[str, list[tuple[str, str]]] = {
    "binary rate": [("2", "1"), ("4", "2"), ("6", "3"), ("8", "4")],
    "terminating rate": [("0.1", "1"), ("0.2", "2"), ("0.3", "3"), ("0.4", "4")],
    "repeating rate": [("1", "3"), ("2", "6"), ("3", "9"), ("4", "12")],
}


def _unit_clusters(values: np.ndarray, prefix: str = "city") -> list[ClusterAggregate]:
    return [
        ClusterAggregate(cluster=f"{prefix}-{i}", total=Decimal(str(value)), weight=Decimal("1"))
        for i, value in enumerate(values)
    ]


def _corridor(
    corridor: str, offsets: list[int], totals: list[Decimal]
) -> list[CorridorDayAggregate]:
    return [
        CorridorDayAggregate(
            corridor=corridor,
            day=FIRST_DAY + timedelta(days=offset),
            total=total,
            weight=Decimal("1"),
        )
        for offset, total in zip(offsets, totals, strict=True)
    ]


def _panel(values: np.ndarray) -> list[CorridorDayAggregate]:
    return [
        CorridorDayAggregate(
            corridor=f"corridor-{corridor}",
            day=FIRST_DAY + timedelta(days=offset),
            total=Decimal(str(value)),
            weight=Decimal("1"),
        )
        for corridor, row in enumerate(values)
        for offset, value in enumerate(row)
    ]


def test_cluster_bootstrap_holds_its_size_under_the_null() -> None:
    rng = np.random.default_rng(20260812)
    draws = rng.standard_normal((400, 40))
    rejected = 0
    for trial, row in enumerate(draws):
        result = cluster_bootstrap(
            _unit_clusters(row),
            null_value=Decimal("0"),
            direction="greater",
            resamples=FIXTURE_RESAMPLES,
            seed=trial,
            ci_level=FIXTURE_CI_LEVEL,
        )
        rejected += result.p_value < FIXTURE_ALPHA
    rate = rejected / len(draws)
    assert 0.02 <= rate <= 0.09, f"false positive rate {rate:.4f} against nominal {FIXTURE_ALPHA}"


def test_cluster_bootstrap_recovers_a_planted_effect() -> None:
    rng = np.random.default_rng(4041)
    planted = Decimal("0.50")
    weights = [int(value) for value in rng.integers(1, 6, size=60)]
    noise = rng.standard_normal(60) * 0.5
    clusters = [
        ClusterAggregate(
            cluster=f"city-{i}",
            total=Decimal(weight) * planted + Decimal(str(shock)),
            weight=Decimal(weight),
        )
        for i, (weight, shock) in enumerate(zip(weights, noise, strict=True))
    ]
    result = cluster_bootstrap(
        clusters,
        null_value=Decimal("0"),
        direction="greater",
        resamples=FIXTURE_RESAMPLES,
        seed=11,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert abs(result.estimate - planted) < Decimal("0.10"), (
        f"estimate {result.estimate} against planted {planted}"
    )
    assert result.p_value < FIXTURE_ALPHA
    assert result.n_clusters == 60
    assert result.resamples == FIXTURE_RESAMPLES
    assert result.seed == 11
    assert result.ci_low < float(result.estimate) < result.ci_high


def test_cluster_bootstrap_interval_inverts_the_pivot_the_p_value_tests() -> None:
    clusters = _unit_clusters(np.array([1.0] * 19 + [21.0]))
    result = cluster_bootstrap(
        clusters,
        null_value=Decimal("0"),
        direction="greater",
        resamples=FIXTURE_RESAMPLES,
        seed=21,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert result.estimate == Decimal("2")
    assert result.p_value > FIXTURE_ALPHA
    assert result.ci_low <= float(result.null_value), (
        f"interval excludes a null the test does not reject: {result.ci_low}"
    )
    below = float(result.estimate) - result.ci_low
    above = result.ci_high - float(result.estimate)
    assert below > above, f"right-skewed replicates should reflect leftward: {below} vs {above}"


def test_point_estimate_stays_exact_where_float_addition_would_not() -> None:
    clusters = [
        ClusterAggregate(cluster="a", total=Decimal("0.1"), weight=Decimal("0.5")),
        ClusterAggregate(cluster="b", total=Decimal("0.2"), weight=Decimal("0.5")),
    ]
    result = cluster_bootstrap(
        clusters,
        null_value=Decimal("1"),
        direction="less",
        resamples=FIXTURE_RESAMPLES,
        seed=3,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert isinstance(result.estimate, Decimal)
    assert result.estimate == Decimal("0.3")
    assert float(clusters[0].total) + float(clusters[1].total) > 0.3

    verdict = evaluate_gate(
        estimate=result.estimate,
        p_value=result.p_value,
        n=result.n_clusters,
        threshold=Decimal("0.3"),
        direction="less",
        alpha=FIXTURE_ALPHA,
        n_min=2,
        n_unit="city event-days",
    )
    assert verdict.economic


def test_cluster_bootstrap_rejects_unusable_input() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap(
            [],
            null_value=Decimal("0"),
            direction="greater",
            resamples=FIXTURE_RESAMPLES,
            seed=1,
            ci_level=FIXTURE_CI_LEVEL,
        )
    with pytest.raises(ValueError):
        cluster_bootstrap(
            [
                ClusterAggregate(cluster="a", total=Decimal("1"), weight=Decimal("1")),
                ClusterAggregate(cluster="b", total=Decimal("0"), weight=Decimal("0")),
            ],
            null_value=Decimal("0"),
            direction="greater",
            resamples=FIXTURE_RESAMPLES,
            seed=1,
            ci_level=FIXTURE_CI_LEVEL,
        )
    with pytest.raises(ValueError):
        cluster_bootstrap(
            [ClusterAggregate(cluster="a", total=Decimal("1"), weight=Decimal("1"))],
            null_value=Decimal("0"),
            direction="sideways",
            resamples=FIXTURE_RESAMPLES,
            seed=1,
            ci_level=FIXTURE_CI_LEVEL,
        )


def test_p_value_is_never_zero() -> None:
    clusters = [
        ClusterAggregate(cluster=f"city-{i}", total=Decimal("1"), weight=Decimal("1"))
        for i in range(20)
    ]
    result = cluster_bootstrap(
        clusters,
        null_value=Decimal("0"),
        direction="greater",
        resamples=FIXTURE_RESAMPLES,
        seed=5,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert result.p_value == pytest.approx(1 / (FIXTURE_RESAMPLES + 1))


def test_cluster_bootstrap_directions_split_the_replicate_mass() -> None:
    rng = np.random.default_rng(77)
    clusters = _unit_clusters(rng.standard_normal(24) + 0.6)
    upward = cluster_bootstrap(
        clusters,
        null_value=Decimal("0"),
        direction="greater",
        resamples=FIXTURE_RESAMPLES,
        seed=13,
        ci_level=FIXTURE_CI_LEVEL,
    )
    downward = cluster_bootstrap(
        clusters,
        null_value=Decimal("0"),
        direction="less",
        resamples=FIXTURE_RESAMPLES,
        seed=13,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert upward.estimate == downward.estimate
    assert upward.p_value < downward.p_value
    assert upward.p_value + downward.p_value >= 1 + 1 / (FIXTURE_RESAMPLES + 1)


def test_cluster_labels_are_opaque() -> None:
    rng = np.random.default_rng(99)
    values = rng.standard_normal(25)
    first = cluster_bootstrap(
        _unit_clusters(values, prefix="KXHIGHDEN-2026-07-01"),
        null_value=Decimal("0"),
        direction="greater",
        resamples=FIXTURE_RESAMPLES,
        seed=8,
        ci_level=FIXTURE_CI_LEVEL,
    )
    second = cluster_bootstrap(
        _unit_clusters(values, prefix="zzz"),
        null_value=Decimal("0"),
        direction="greater",
        resamples=FIXTURE_RESAMPLES,
        seed=8,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert first == second


def test_blocks_are_contiguous_days_that_never_span_a_corridor() -> None:
    observations = _corridor("north", [0, 1, 2, 3, 4, 5, 6], [Decimal("1")] * 7) + _corridor(
        "south", [3], [Decimal("1")]
    )
    assert day_blocks(observations, block_days=3) == (0, 0, 0, 1, 1, 1, 2, 3)
    assert day_blocks(observations, block_days=7) == (0, 0, 0, 0, 0, 0, 0, 1)


def test_blocks_follow_day_order_and_ignore_calendar_gaps() -> None:
    observations = _corridor("north", [9, 0, 1], [Decimal("1")] * 3)
    assert day_blocks(observations, block_days=2) == (1, 0, 0)


def test_block_length_changes_the_block_count() -> None:
    totals = [Decimal(value) for value in (3, 1, 4, 1, 5, 9, 2)]
    observations = _corridor("north", [0, 1, 2, 3, 4, 5, 6], totals) + _corridor(
        "south", [3], [Decimal("6")]
    )
    wide = wild_cluster_bootstrap(
        observations,
        null_value=Decimal("0"),
        direction="greater",
        block_days=3,
        resamples=FIXTURE_RESAMPLES,
        seed=2,
        ci_level=FIXTURE_CI_LEVEL,
    )
    narrow = wild_cluster_bootstrap(
        observations,
        null_value=Decimal("0"),
        direction="greater",
        block_days=2,
        resamples=FIXTURE_RESAMPLES,
        seed=2,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert wide.n_blocks == 4
    assert narrow.n_blocks == 5
    assert wide.n_observations == narrow.n_observations == 8


def test_wild_bootstrap_holds_its_size_under_the_null() -> None:
    rng = np.random.default_rng(20260812)
    draws = rng.standard_normal((200, 4, 6))
    rejected = 0
    for trial, values in enumerate(draws):
        result = wild_cluster_bootstrap(
            _panel(values),
            null_value=Decimal("0"),
            direction="greater",
            block_days=2,
            resamples=FIXTURE_RESAMPLES,
            seed=trial,
            ci_level=FIXTURE_CI_LEVEL,
        )
        rejected += result.p_value < FIXTURE_ALPHA
    rate = rejected / len(draws)
    assert 0.005 <= rate <= 0.09, f"false positive rate {rate:.4f} against nominal {FIXTURE_ALPHA}"


def test_wild_bootstrap_recovers_a_planted_effect() -> None:
    rng = np.random.default_rng(515)
    planted = Decimal("0.50")
    weights = [int(value) for value in rng.integers(1, 6, size=(4, 6)).ravel()]
    noise = rng.standard_normal((4, 6)).ravel() * 0.5
    observations = [
        CorridorDayAggregate(
            corridor=f"corridor-{index // 6}",
            day=FIRST_DAY + timedelta(days=index % 6),
            total=Decimal(weight) * planted + Decimal(str(shock)),
            weight=Decimal(weight),
        )
        for index, (weight, shock) in enumerate(zip(weights, noise, strict=True))
    ]
    result = wild_cluster_bootstrap(
        observations,
        null_value=Decimal("0"),
        direction="greater",
        block_days=2,
        resamples=FIXTURE_RESAMPLES,
        seed=17,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert abs(result.estimate - planted) < Decimal("0.10"), (
        f"estimate {result.estimate} against planted {planted}"
    )
    assert result.p_value < FIXTURE_ALPHA
    assert result.n_blocks == 12
    assert result.degenerate_resamples == 0
    assert result.ci_low < float(result.estimate) < result.ci_high


def test_wild_bootstrap_interval_does_not_move_with_the_null() -> None:
    totals = [Decimal("3"), Decimal("2"), Decimal("0"), Decimal("-1"), Decimal("4"), Decimal("1")]
    observations = _corridor("north", [0, 1, 2, 3, 4, 5], totals)
    results = [
        wild_cluster_bootstrap(
            observations,
            null_value=Decimal(null),
            direction="greater",
            block_days=2,
            resamples=FIXTURE_RESAMPLES,
            seed=6,
            ci_level=FIXTURE_CI_LEVEL,
        )
        for null in ("0", "0.5", "1")
    ]
    intervals = {(result.ci_low, result.ci_high) for result in results}
    p_values = {result.p_value for result in results}
    assert len(intervals) == 1, f"interval moved with the null: {intervals}"
    assert len({result.standard_error for result in results}) == 1
    assert len(p_values) > 1, f"the p-value should still move with the null: {p_values}"
    assert results[0].ci_low < float(results[0].estimate) < results[0].ci_high


@pytest.mark.parametrize("null", ["0", "1"])
@pytest.mark.parametrize("panel", FLAT_PANELS.values(), ids=list(FLAT_PANELS))
def test_wild_bootstrap_rejects_a_degenerate_sample(
    panel: list[tuple[str, str]], null: str
) -> None:
    observations = [
        CorridorDayAggregate(
            corridor="north",
            day=FIRST_DAY + timedelta(days=offset),
            total=Decimal(total),
            weight=Decimal(weight),
        )
        for offset, (total, weight) in enumerate(panel)
    ]
    with pytest.raises(ValueError):
        wild_cluster_bootstrap(
            observations,
            null_value=Decimal(null),
            direction="greater",
            block_days=2,
            resamples=FIXTURE_RESAMPLES,
            seed=1,
            ci_level=FIXTURE_CI_LEVEL,
        )


def test_wild_bootstrap_keeps_a_two_block_interval_on_the_data_scale() -> None:
    totals = [Decimal("0.13"), Decimal("0.41"), Decimal("-0.07"), Decimal("-0.29")]
    result = wild_cluster_bootstrap(
        _corridor("north", [0, 1, 2, 3], totals),
        null_value=Decimal("0"),
        direction="greater",
        block_days=2,
        resamples=FIXTURE_RESAMPLES,
        seed=4,
        ci_level=FIXTURE_CI_LEVEL,
    )
    span = result.ci_high - result.ci_low
    assert result.n_blocks == 2
    assert span <= 10 * result.standard_error, (
        f"interval spans {span} against se {result.standard_error}"
    )


def test_degenerate_replicates_are_counted_and_treated_as_extreme() -> None:
    totals = [Decimal("3"), Decimal("2"), Decimal("0"), Decimal("-1"), Decimal("4"), Decimal("1")]
    observations = _corridor("north", [0, 1, 2, 3, 4, 5], totals)
    upward = wild_cluster_bootstrap(
        observations,
        null_value=Decimal("1"),
        direction="greater",
        block_days=2,
        resamples=FIXTURE_RESAMPLES,
        seed=6,
        ci_level=FIXTURE_CI_LEVEL,
    )
    downward = wild_cluster_bootstrap(
        observations,
        null_value=Decimal("1"),
        direction="less",
        block_days=2,
        resamples=FIXTURE_RESAMPLES,
        seed=6,
        ci_level=FIXTURE_CI_LEVEL,
    )
    share = upward.degenerate_resamples / (FIXTURE_RESAMPLES + 1)
    assert upward.degenerate_resamples == downward.degenerate_resamples
    assert share == pytest.approx(0.25, abs=0.05), f"degenerate share {share:.4f}"
    assert upward.p_value + downward.p_value >= 1 + share


def test_wild_bootstrap_rejects_a_replicate_set_with_no_usable_interval() -> None:
    totals = [Decimal("1"), Decimal("1"), Decimal("-1"), Decimal("-1")]
    observations = _corridor("north", [0, 1, 2, 3], totals)
    healthy = wild_cluster_bootstrap(
        observations,
        null_value=Decimal("0"),
        direction="greater",
        block_days=2,
        resamples=FIXTURE_RESAMPLES,
        seed=6,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert healthy.standard_error > 0
    assert 0 < healthy.degenerate_resamples < FIXTURE_RESAMPLES

    with pytest.raises(ValueError):
        wild_cluster_bootstrap(
            observations,
            null_value=Decimal("0"),
            direction="greater",
            block_days=2,
            resamples=1,
            seed=6,
            ci_level=FIXTURE_CI_LEVEL,
        )


def reference_digest() -> str:
    clusters = [
        ClusterAggregate(cluster=name, total=Decimal(total), weight=Decimal(weight))
        for name, total, weight in (
            ("den", "3", "2"),
            ("aus", "-1", "1"),
            ("nyc", "4", "3"),
            ("chi", "0", "1"),
            ("mia", "-2", "2"),
        )
    ]
    observations = _corridor(
        "north", [0, 1, 2, 3], [Decimal("3"), Decimal("2"), Decimal("0"), Decimal("-1")]
    ) + _corridor("south", [1, 2], [Decimal("4"), Decimal("1")])
    payload = [
        asdict(
            cluster_bootstrap(
                clusters,
                null_value=Decimal("0"),
                direction="greater",
                resamples=FIXTURE_RESAMPLES,
                seed=424242,
                ci_level=FIXTURE_CI_LEVEL,
            )
        ),
        asdict(
            wild_cluster_bootstrap(
                observations,
                null_value=Decimal("0"),
                direction="greater",
                block_days=2,
                resamples=FIXTURE_RESAMPLES,
                seed=424242,
                ci_level=FIXTURE_CI_LEVEL,
            )
        ),
    ]
    return hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
    ).hexdigest()


def test_a_seed_repeats_in_process_and_across_processes() -> None:
    digest = reference_digest()
    assert reference_digest() == digest
    child = subprocess.run(
        [sys.executable, "-c", CHILD_SOURCE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": "1"},
        timeout=120,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == digest


def test_a_different_seed_moves_the_p_value() -> None:
    rng = np.random.default_rng(31)
    clusters = _unit_clusters(rng.standard_normal(12) + 0.3)
    first = cluster_bootstrap(
        clusters,
        null_value=Decimal("0"),
        direction="greater",
        resamples=FIXTURE_RESAMPLES,
        seed=1,
        ci_level=FIXTURE_CI_LEVEL,
    )
    second = cluster_bootstrap(
        clusters,
        null_value=Decimal("0"),
        direction="greater",
        resamples=FIXTURE_RESAMPLES,
        seed=2,
        ci_level=FIXTURE_CI_LEVEL,
    )
    assert first.estimate == second.estimate
    assert first.p_value != second.p_value
    assert first.ci_low != second.ci_low


def test_gate_reports_its_conditions_separately() -> None:
    verdict = evaluate_gate(
        estimate=Decimal("0.04"),
        p_value=0.01,
        n=120,
        threshold=Decimal("0.02"),
        direction="greater",
        alpha=FIXTURE_ALPHA,
        n_min=100,
        n_unit="market-days",
    )
    assert (verdict.economic, verdict.significant, verdict.powered, verdict.passed) == (
        True,
        True,
        True,
        True,
    )
    assert verdict.n_unit == "market-days"
    assert verdict.n_min == 100

    thin = evaluate_gate(
        estimate=Decimal("0.04"),
        p_value=0.01,
        n=99,
        threshold=Decimal("0.02"),
        direction="greater",
        alpha=FIXTURE_ALPHA,
        n_min=100,
        n_unit="market-days",
    )
    assert thin.economic
    assert thin.significant
    assert not thin.powered
    assert not thin.passed

    exact = evaluate_gate(
        estimate=Decimal("0.04"),
        p_value=0.01,
        n=100,
        threshold=Decimal("0.02"),
        direction="greater",
        alpha=FIXTURE_ALPHA,
        n_min=100,
        n_unit="market-days",
    )
    assert exact.powered
    assert exact.passed


def test_gate_significance_is_strict() -> None:
    verdict = evaluate_gate(
        estimate=Decimal("0.04"),
        p_value=FIXTURE_ALPHA,
        n=120,
        threshold=Decimal("0.02"),
        direction="greater",
        alpha=FIXTURE_ALPHA,
        n_min=100,
        n_unit="market-days",
    )
    assert not verdict.significant
    assert not verdict.passed


def test_gate_economics_run_in_both_directions() -> None:
    upward = evaluate_gate(
        estimate=Decimal("0.02"),
        p_value=0.01,
        n=10,
        threshold=Decimal("0.02"),
        direction="greater",
        alpha=FIXTURE_ALPHA,
        n_min=10,
        n_unit="clusters",
    )
    downward = evaluate_gate(
        estimate=Decimal("0.02"),
        p_value=0.01,
        n=10,
        threshold=Decimal("0.02"),
        direction="less",
        alpha=FIXTURE_ALPHA,
        n_min=10,
        n_unit="clusters",
    )
    short = evaluate_gate(
        estimate=Decimal("0.019"),
        p_value=0.01,
        n=10,
        threshold=Decimal("0.02"),
        direction="greater",
        alpha=FIXTURE_ALPHA,
        n_min=10,
        n_unit="clusters",
    )
    over = evaluate_gate(
        estimate=Decimal("0.021"),
        p_value=0.01,
        n=10,
        threshold=Decimal("0.02"),
        direction="less",
        alpha=FIXTURE_ALPHA,
        n_min=10,
        n_unit="clusters",
    )
    assert upward.economic
    assert downward.economic
    assert upward.passed
    assert downward.passed
    assert not short.economic
    assert not over.economic
    assert short.significant
    assert short.powered
    assert not short.passed
    assert not over.passed
    with pytest.raises(ValueError):
        evaluate_gate(
            estimate=Decimal("0.02"),
            p_value=0.01,
            n=10,
            threshold=Decimal("0.02"),
            direction="sideways",
            alpha=FIXTURE_ALPHA,
            n_min=10,
            n_unit="clusters",
        )


def test_holdout_needs_the_discovery_sign() -> None:
    verdict = evaluate_holdout(
        discovery_estimate=Decimal("0.04"),
        holdout_estimate=Decimal("-0.04"),
        holdout_p_value=0.001,
        holdout_n=100,
        discovery_n_min=100,
        alpha=FIXTURE_ALPHA,
        n_unit="clusters",
    )
    assert not verdict.same_sign
    assert verdict.magnitude
    assert not verdict.replicated


def test_holdout_replicates_a_negative_discovery_estimate() -> None:
    down = evaluate_holdout(
        discovery_estimate=Decimal("-0.04"),
        holdout_estimate=Decimal("-0.03"),
        holdout_p_value=0.001,
        holdout_n=100,
        discovery_n_min=100,
        alpha=FIXTURE_ALPHA,
        n_unit="clusters",
    )
    up = evaluate_holdout(
        discovery_estimate=Decimal("-0.04"),
        holdout_estimate=Decimal("0.03"),
        holdout_p_value=0.001,
        holdout_n=100,
        discovery_n_min=100,
        alpha=FIXTURE_ALPHA,
        n_unit="clusters",
    )
    flat = evaluate_holdout(
        discovery_estimate=Decimal("-0.04"),
        holdout_estimate=Decimal("0"),
        holdout_p_value=0.001,
        holdout_n=100,
        discovery_n_min=100,
        alpha=FIXTURE_ALPHA,
        n_unit="clusters",
    )
    assert down.same_sign
    assert down.replicated
    assert not up.same_sign
    assert up.magnitude
    assert not up.replicated
    assert not flat.same_sign
    assert not flat.replicated


def test_holdout_magnitude_bar_is_half_the_discovery_estimate() -> None:
    short = evaluate_holdout(
        discovery_estimate=Decimal("0.04"),
        holdout_estimate=Decimal("0.0196"),
        holdout_p_value=0.001,
        holdout_n=100,
        discovery_n_min=100,
        alpha=FIXTURE_ALPHA,
        n_unit="clusters",
    )
    exact = evaluate_holdout(
        discovery_estimate=Decimal("0.04"),
        holdout_estimate=Decimal("0.02"),
        holdout_p_value=0.001,
        holdout_n=100,
        discovery_n_min=100,
        alpha=FIXTURE_ALPHA,
        n_unit="clusters",
    )
    assert not short.magnitude
    assert not short.replicated
    assert exact.magnitude
    assert exact.replicated


def test_holdout_minimum_rounds_up_and_keeps_the_caller_unit() -> None:
    verdict = evaluate_holdout(
        discovery_estimate=Decimal("0.04"),
        holdout_estimate=Decimal("0.04"),
        holdout_p_value=0.001,
        holdout_n=2,
        discovery_n_min=5,
        alpha=FIXTURE_ALPHA,
        n_unit="prints",
    )
    assert verdict.holdout_n_min == 3
    assert not verdict.powered
    assert not verdict.replicated
    assert verdict.n_unit == "prints"

    powered = evaluate_holdout(
        discovery_estimate=Decimal("0.04"),
        holdout_estimate=Decimal("0.04"),
        holdout_p_value=0.001,
        holdout_n=3,
        discovery_n_min=5,
        alpha=FIXTURE_ALPHA,
        n_unit="prints",
    )
    assert powered.powered
    assert powered.replicated


def test_holdout_significance_uses_the_caller_alpha() -> None:
    verdict = evaluate_holdout(
        discovery_estimate=Decimal("0.04"),
        holdout_estimate=Decimal("0.04"),
        holdout_p_value=FIXTURE_ALPHA,
        holdout_n=100,
        discovery_n_min=100,
        alpha=FIXTURE_ALPHA,
        n_unit="clusters",
    )
    loose = evaluate_holdout(
        discovery_estimate=Decimal("0.04"),
        holdout_estimate=Decimal("0.04"),
        holdout_p_value=FIXTURE_ALPHA,
        holdout_n=100,
        discovery_n_min=100,
        alpha=0.10,
        n_unit="clusters",
    )
    assert not verdict.significant
    assert loose.significant


def test_holdout_rejects_a_zero_discovery_estimate() -> None:
    with pytest.raises(ValueError):
        evaluate_holdout(
            discovery_estimate=Decimal("0"),
            holdout_estimate=Decimal("0.04"),
            holdout_p_value=0.001,
            holdout_n=100,
            discovery_n_min=100,
            alpha=FIXTURE_ALPHA,
            n_unit="clusters",
        )
