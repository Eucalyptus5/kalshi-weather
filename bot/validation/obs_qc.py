from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import Decimal
from typing import Literal

from bot.observations.metar import StationObservation

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QCConfig:
    plausible_low_f: Decimal = Decimal("-40")
    plausible_high_f: Decimal = Decimal("140")
    max_rate_f_per_minute: Decimal = Decimal("1.5")
    stuck_window: timedelta = timedelta(minutes=60)
    confirmation_window: timedelta = timedelta(minutes=30)
    confirmation_tolerance_f: Decimal = Decimal("1.0")
    neighbor_window: timedelta = timedelta(minutes=30)
    neighbor_max_delta_f: Decimal = Decimal("5.0")


Verdict = Literal["accepted", "rejected", "needs_confirmation"]


@dataclass(frozen=True, slots=True)
class ObsQCResult:
    verdict: Verdict
    reason: str
    calibrated_obs: StationObservation


def apply_station_calibration(
    obs: StationObservation,
    station_calibration_f: Decimal | None,
) -> StationObservation:
    if station_calibration_f is None:
        return obs
    return replace(obs, temp_f=obs.temp_f - station_calibration_f)


def qc_observation(
    prev: StationObservation | None,
    candidate: StationObservation,
    *,
    station_calibration_f: Decimal | None = None,
    config: QCConfig | None = None,
) -> ObsQCResult:
    cfg = config if config is not None else QCConfig()
    calibrated = apply_station_calibration(candidate, station_calibration_f)

    if calibrated.temp_f < cfg.plausible_low_f:
        reason = f"temp_f {calibrated.temp_f} below plausible lower bound {cfg.plausible_low_f}"
        logger.warning("obs_qc verdict=rejected station=%s %s", calibrated.station, reason)
        return ObsQCResult(verdict="rejected", reason=reason, calibrated_obs=calibrated)
    if calibrated.temp_f > cfg.plausible_high_f:
        reason = f"temp_f {calibrated.temp_f} above plausible upper bound {cfg.plausible_high_f}"
        logger.warning("obs_qc verdict=rejected station=%s %s", calibrated.station, reason)
        return ObsQCResult(verdict="rejected", reason=reason, calibrated_obs=calibrated)

    if prev is None or prev.valid_time >= candidate.valid_time:
        return ObsQCResult(verdict="accepted", reason="ok", calibrated_obs=calibrated)

    calibrated_prev = apply_station_calibration(prev, station_calibration_f)
    dt_seconds = Decimal(str((candidate.valid_time - prev.valid_time).total_seconds()))
    dt_min = dt_seconds / Decimal("60")
    delta_f = calibrated.temp_f - calibrated_prev.temp_f
    rate = abs(delta_f) / dt_min
    if rate > cfg.max_rate_f_per_minute:
        reason = f"rate {rate} F/min exceeds max_rate_f_per_minute {cfg.max_rate_f_per_minute}"
        logger.warning(
            "obs_qc verdict=needs_confirmation station=%s %s",
            calibrated.station,
            reason,
        )
        return ObsQCResult(verdict="needs_confirmation", reason=reason, calibrated_obs=calibrated)

    return ObsQCResult(verdict="accepted", reason="ok", calibrated_obs=calibrated)


def confirm_pending(
    pending: StationObservation,
    candidate: StationObservation,
    *,
    station_calibration_f: Decimal | None = None,
    config: QCConfig | None = None,
) -> ObsQCResult:
    cfg = config if config is not None else QCConfig()
    calibrated_pending = apply_station_calibration(pending, station_calibration_f)
    calibrated_candidate = apply_station_calibration(candidate, station_calibration_f)

    gap = candidate.valid_time - pending.valid_time
    if gap > cfg.confirmation_window:
        reason = (
            f"candidate {gap} after pending exceeds confirmation_window {cfg.confirmation_window}"
        )
        return ObsQCResult(
            verdict="needs_confirmation",
            reason=reason,
            calibrated_obs=calibrated_pending,
        )

    diff = abs(calibrated_candidate.temp_f - calibrated_pending.temp_f)
    if diff > cfg.confirmation_tolerance_f:
        reason = (
            f"candidate diverges {diff} F from pending, exceeds tolerance "
            f"{cfg.confirmation_tolerance_f}"
        )
        logger.warning("obs_qc verdict=rejected station=%s %s", calibrated_pending.station, reason)
        return ObsQCResult(verdict="rejected", reason=reason, calibrated_obs=calibrated_pending)

    return ObsQCResult(verdict="accepted", reason="ok", calibrated_obs=calibrated_pending)


def detect_stuck_sensor(
    window: list[StationObservation],
    *,
    config: QCConfig | None = None,
) -> bool:
    """Operates on whatever temp_f values are passed; does not apply station calibration."""
    cfg = config if config is not None else QCConfig()
    if len(window) < 2:
        return False
    first = window[0].temp_f
    for ob in window[1:]:
        if ob.temp_f != first:
            return False
    span = window[-1].valid_time - window[0].valid_time
    return span >= cfg.stuck_window


def neighbor_cross_check(
    candidate: StationObservation,
    neighbors: list[StationObservation],
    *,
    max_delta_f: Decimal | None = None,
    config: QCConfig | None = None,
) -> bool:
    cfg = config if config is not None else QCConfig()
    limit = max_delta_f if max_delta_f is not None else cfg.neighbor_max_delta_f
    window = cfg.neighbor_window
    for neighbor in neighbors:
        gap = neighbor.valid_time - candidate.valid_time
        if abs(gap) > window:
            continue
        if abs(candidate.temp_f - neighbor.temp_f) <= limit:
            return True
    return False
