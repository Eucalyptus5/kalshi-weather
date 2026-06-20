from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from bot.observations.metar import StationObservation
from bot.validation.obs_qc import (
    QCConfig,
    apply_station_calibration,
    confirm_pending,
    detect_stuck_sensor,
    neighbor_cross_check,
    qc_observation,
)


def _obs(
    station: str,
    valid_time: datetime,
    temp_f: Decimal,
    publication_time: datetime | None = None,
    is_special: bool = False,
    raw: str = "",
    source: str = "metar",
) -> StationObservation:
    return StationObservation(
        station=station,
        valid_time=valid_time,
        publication_time=publication_time if publication_time is not None else valid_time,
        temp_f=temp_f,
        is_special=is_special,
        raw=raw,
        source=source,
    )


def test_calibration_applied_kmia_plus_one() -> None:
    t = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    obs = _obs("KMIA", t, Decimal("88.0"), raw="rawline", is_special=True)
    out = apply_station_calibration(obs, Decimal("1.0"))
    assert out.temp_f == Decimal("87.0")
    assert isinstance(out.temp_f, Decimal)
    assert out.station == "KMIA"
    assert out.valid_time == t
    assert out.publication_time == obs.publication_time
    assert out.is_special is True
    assert out.raw == "rawline"
    assert out.source == "metar"


def test_calibration_none_passes_through() -> None:
    t = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    obs = _obs("KDEN", t, Decimal("80.0"))
    out = apply_station_calibration(obs, None)
    assert out is obs or out == obs


def test_plausibility_lower_bound_rejection() -> None:
    t = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    obs = _obs("KDEN", t, Decimal("-50"))
    result = qc_observation(prev=None, candidate=obs)
    assert result.verdict == "rejected"
    assert "-40" in result.reason


def test_plausibility_upper_bound_rejection() -> None:
    t = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    obs = _obs("KDEN", t, Decimal("150"))
    result = qc_observation(prev=None, candidate=obs)
    assert result.verdict == "rejected"
    assert "140" in result.reason


def test_in_bounds_no_prev_accepted() -> None:
    t = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    obs = _obs("KDEN", t, Decimal("80"))
    result = qc_observation(prev=None, candidate=obs)
    assert result.verdict == "accepted"
    assert result.calibrated_obs.temp_f == Decimal("80")
    assert isinstance(result.calibrated_obs.temp_f, Decimal)


def test_rate_of_change_kden_spike_needs_confirmation() -> None:
    t0 = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 17, 12, 1, tzinfo=timezone.utc)
    prev = _obs("KDEN", t0, Decimal("78.0"))
    candidate = _obs("KDEN", t1, Decimal("80.0"))
    result = qc_observation(prev=prev, candidate=candidate)
    assert result.verdict == "needs_confirmation"
    assert "rate" in result.reason.lower()


def test_rate_of_change_kmia_after_calibration() -> None:
    t0 = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 17, 14, 1, tzinfo=timezone.utc)
    prev = _obs("KMIA", t0, Decimal("89"))
    candidate = _obs("KMIA", t1, Decimal("92"))
    result = qc_observation(prev=prev, candidate=candidate, station_calibration_f=Decimal("1.0"))
    assert result.verdict == "needs_confirmation"
    assert result.calibrated_obs.temp_f == Decimal("91")


def test_slow_legitimate_rise_accepted() -> None:
    t0 = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 17, 13, 0, tzinfo=timezone.utc)
    prev = _obs("KDEN", t0, Decimal("80"))
    candidate = _obs("KDEN", t1, Decimal("85"))
    result = qc_observation(prev=prev, candidate=candidate)
    assert result.verdict == "accepted"
    assert result.calibrated_obs.temp_f == Decimal("85")


def test_confirmation_accepted() -> None:
    pending_t = datetime(2026, 6, 17, 12, 1, tzinfo=timezone.utc)
    confirm_t = datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc)
    pending = _obs("KDEN", pending_t, Decimal("80"))
    candidate = _obs("KDEN", confirm_t, Decimal("80.5"))
    result = confirm_pending(pending=pending, candidate=candidate)
    assert result.verdict == "accepted"
    assert result.calibrated_obs.temp_f == Decimal("80")
    assert result.calibrated_obs.valid_time == pending_t


def test_confirmation_rejected_blip() -> None:
    pending_t = datetime(2026, 6, 17, 12, 1, tzinfo=timezone.utc)
    confirm_t = datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc)
    pending = _obs("KDEN", pending_t, Decimal("80"))
    candidate = _obs("KDEN", confirm_t, Decimal("78"))
    result = confirm_pending(pending=pending, candidate=candidate)
    assert result.verdict == "rejected"


def test_confirmation_outside_window_still_waiting() -> None:
    pending_t = datetime(2026, 6, 17, 12, 1, tzinfo=timezone.utc)
    confirm_t = datetime(2026, 6, 17, 12, 35, tzinfo=timezone.utc)
    pending = _obs("KDEN", pending_t, Decimal("80"))
    candidate = _obs("KDEN", confirm_t, Decimal("80.2"))
    result = confirm_pending(pending=pending, candidate=candidate)
    assert result.verdict == "needs_confirmation"


def test_stuck_sensor_positive() -> None:
    base = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    window = [_obs("KDEN", base + timedelta(minutes=i * 10), Decimal("75")) for i in range(8)]
    assert detect_stuck_sensor(window) is True


def test_stuck_sensor_negative_varying_values() -> None:
    base = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    window = [_obs("KDEN", base + timedelta(minutes=i * 10), Decimal("75")) for i in range(7)]
    window.append(_obs("KDEN", base + timedelta(minutes=70), Decimal("76")))
    assert detect_stuck_sensor(window) is False


def test_stuck_sensor_negative_short_window() -> None:
    base = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    window = [_obs("KDEN", base + timedelta(minutes=i * 10), Decimal("75")) for i in range(3)]
    assert detect_stuck_sensor(window) is False


def test_stuck_sensor_single_obs() -> None:
    t = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    window = [_obs("KDEN", t, Decimal("75"))]
    assert detect_stuck_sensor(window) is False


def test_neighbor_cross_check_pass() -> None:
    t_cand = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    t_neighbor = datetime(2026, 6, 17, 14, 15, tzinfo=timezone.utc)
    candidate = _obs("KDEN", t_cand, Decimal("80"))
    neighbors = [_obs("KBKF", t_neighbor, Decimal("79"))]
    assert neighbor_cross_check(candidate, neighbors) is True


def test_neighbor_cross_check_fails_on_delta() -> None:
    t_cand = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    t_neighbor = datetime(2026, 6, 17, 14, 15, tzinfo=timezone.utc)
    candidate = _obs("KDEN", t_cand, Decimal("80"))
    neighbors = [_obs("KBKF", t_neighbor, Decimal("60"))]
    assert neighbor_cross_check(candidate, neighbors) is False


def test_neighbor_cross_check_fails_on_time_gap() -> None:
    t_cand = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    t_neighbor = datetime(2026, 6, 17, 16, 0, tzinfo=timezone.utc)
    candidate = _obs("KDEN", t_cand, Decimal("80"))
    neighbors = [_obs("KBKF", t_neighbor, Decimal("80"))]
    assert neighbor_cross_check(candidate, neighbors) is False


def test_neighbor_cross_check_empty_list() -> None:
    t_cand = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    candidate = _obs("KDEN", t_cand, Decimal("80"))
    assert neighbor_cross_check(candidate, []) is False


def test_truncated_morning_window_no_rejections() -> None:
    base = datetime(2026, 6, 17, 7, 0, tzinfo=timezone.utc)
    seq = [
        _obs("KDEN", base, Decimal("60")),
        _obs("KDEN", base + timedelta(hours=1), Decimal("65")),
        _obs("KDEN", base + timedelta(hours=2), Decimal("70")),
    ]
    prev: StationObservation | None = None
    for ob in seq:
        result = qc_observation(prev=prev, candidate=ob)
        assert result.verdict == "accepted"
        prev = ob


def test_kmia_calibration_round_trip_accepted() -> None:
    t0 = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 17, 14, 5, tzinfo=timezone.utc)
    prev = _obs("KMIA", t0, Decimal("80"))
    candidate = _obs("KMIA", t1, Decimal("80.5"))
    result = qc_observation(prev=prev, candidate=candidate, station_calibration_f=Decimal("1.0"))
    assert result.verdict == "accepted"
    assert result.calibrated_obs.temp_f == Decimal("79.5")


def test_returned_temp_f_is_decimal_type() -> None:
    t = datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc)
    obs = _obs("KDEN", t, Decimal("80"))
    result = qc_observation(prev=None, candidate=obs)
    assert isinstance(result.calibrated_obs.temp_f, Decimal)
    assert result.calibrated_obs.temp_f == Decimal("80")


def test_config_override_threshold() -> None:
    t0 = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 17, 12, 1, tzinfo=timezone.utc)
    prev = _obs("KDEN", t0, Decimal("78.0"))
    candidate = _obs("KDEN", t1, Decimal("80.0"))
    config = QCConfig(max_rate_f_per_minute=Decimal("3.0"))
    result = qc_observation(prev=prev, candidate=candidate, config=config)
    assert result.verdict == "accepted"
