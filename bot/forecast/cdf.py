from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


class EnsembleCDF:
    def __init__(self, members: np.ndarray, smoothing: float) -> None:
        self._members = members
        self._sigma = smoothing

    @property
    def members(self) -> np.ndarray:
        return self._members

    @classmethod
    def from_members(cls, members: np.ndarray, smoothing: float = 1.0) -> EnsembleCDF:
        if not isinstance(members, np.ndarray) or members.ndim != 1 or members.size == 0:
            raise ValueError("members must be a non-empty 1-D ndarray")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        return cls(members.astype(np.float64, copy=False), float(smoothing))

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        arr = np.asarray(t, dtype=np.float64)
        z = (arr[..., np.newaxis] - self._members) / self._sigma
        out = norm.cdf(z).mean(axis=-1)
        if arr.ndim == 0:
            return float(out)
        return out

    def pdf(self, t: float | np.ndarray) -> float | np.ndarray:
        arr = np.asarray(t, dtype=np.float64)
        z = (arr[..., np.newaxis] - self._members) / self._sigma
        out = norm.pdf(z).mean(axis=-1) / self._sigma
        if arr.ndim == 0:
            return float(out)
        return out

    def prob_range(self, lo: float, hi: float) -> float:
        """P(lo < X <= hi); infinite bounds clamp to 0 and 1."""
        hi_p = 1.0 if math.isinf(hi) and hi > 0 else float(self.cdf(hi))
        lo_p = 0.0 if math.isinf(lo) and lo < 0 else float(self.cdf(lo))
        return hi_p - lo_p
