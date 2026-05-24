from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from bot.strategy.sizing import SIGMA_T_MEDIAN_BY_LEAD_H


SNAPSHOT_PATH = Path(__file__).parent / "data" / "sigma_t_snapshot.json"
TOLERANCE = Decimal("0.02")


def _load() -> dict[int, dict[str, float | int]]:
    raw = json.loads(SNAPSHOT_PATH.read_text())
    return {int(k): v for k, v in raw.items()}


def test_snapshot_file_exists() -> None:
    assert SNAPSHOT_PATH.exists()


def test_each_tabulated_value_within_tolerance_of_snapshot() -> None:
    snapshot = _load()
    for lead_h, expected in SIGMA_T_MEDIAN_BY_LEAD_H.items():
        if lead_h == 168:
            continue
        assert lead_h in snapshot, f"lead={lead_h}h missing from snapshot"
        snap_median = Decimal(str(snapshot[lead_h]["median"]))
        assert abs(snap_median - expected) <= TOLERANCE, (
            f"lead={lead_h}h table={expected} snapshot={snap_median} diff={abs(snap_median - expected)}"
        )


def test_key_144_present_in_both() -> None:
    snapshot = _load()
    assert 144 in snapshot
    assert 144 in SIGMA_T_MEDIAN_BY_LEAD_H


def test_key_168_mirrors_144_in_table() -> None:
    assert SIGMA_T_MEDIAN_BY_LEAD_H[168] == SIGMA_T_MEDIAN_BY_LEAD_H[144]


def test_snapshot_buckets_have_nonzero_samples() -> None:
    snapshot = _load()
    for k, v in snapshot.items():
        assert v["n"] > 0, f"bucket {k} has zero samples"
