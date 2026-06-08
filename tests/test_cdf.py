from __future__ import annotations

import math

import numpy as np
import pytest

from bot.forecast.cdf import EnsembleCDF


def _ensemble() -> np.ndarray:
    return np.random.default_rng(0).normal(72.0, 5.0, size=31)


def test_six_brackets_sum_to_one() -> None:
    cdf = EnsembleCDF.from_members(_ensemble(), smoothing=1.0)
    edges = [
        (-math.inf, 70.5),
        (70.5, 72.5),
        (72.5, 74.5),
        (74.5, 76.5),
        (76.5, 78.5),
        (78.5, math.inf),
    ]
    total = sum(cdf.prob_range(lo, hi) for lo, hi in edges)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_cdf_is_monotone_nondecreasing() -> None:
    cdf = EnsembleCDF.from_members(_ensemble(), smoothing=1.0)
    points = np.arange(70.0, 80.5, 1.0)
    values = cdf.cdf(points)
    diffs = np.diff(values)
    assert np.all(diffs >= -1e-12)


def test_smaller_smoothing_gives_more_peaked_pdf() -> None:
    members = np.array([72.0, 72.0, 72.0])
    sharp = EnsembleCDF.from_members(members, smoothing=0.5)
    soft = EnsembleCDF.from_members(members, smoothing=2.0)
    assert sharp.pdf(72.0) > soft.pdf(72.0)


def test_vectorized_cdf_and_pdf_preserve_shape() -> None:
    cdf = EnsembleCDF.from_members(_ensemble(), smoothing=1.0)
    pts = np.array([60.0, 70.0, 75.0, 90.0])
    out_cdf = cdf.cdf(pts)
    out_pdf = cdf.pdf(pts)
    assert isinstance(out_cdf, np.ndarray) and out_cdf.shape == pts.shape
    assert isinstance(out_pdf, np.ndarray) and out_pdf.shape == pts.shape


def test_scalar_input_returns_scalar() -> None:
    cdf = EnsembleCDF.from_members(_ensemble(), smoothing=1.0)
    v = cdf.cdf(72.0)
    assert isinstance(v, float)
    p = cdf.pdf(72.0)
    assert isinstance(p, float)


def test_prob_range_with_infinite_bounds() -> None:
    cdf = EnsembleCDF.from_members(_ensemble(), smoothing=1.0)
    full = cdf.prob_range(-math.inf, math.inf)
    assert full == pytest.approx(1.0, abs=1e-9)


def test_empty_members_rejected() -> None:
    with pytest.raises(ValueError):
        EnsembleCDF.from_members(np.array([]), smoothing=1.0)


def test_members_property_returns_input_array() -> None:
    members = _ensemble()
    cdf = EnsembleCDF.from_members(members, smoothing=1.0)
    assert np.array_equal(cdf.members, members)


def test_nonpositive_smoothing_rejected() -> None:
    with pytest.raises(ValueError):
        EnsembleCDF.from_members(np.array([72.0]), smoothing=0.0)
    with pytest.raises(ValueError):
        EnsembleCDF.from_members(np.array([72.0]), smoothing=-1.0)
