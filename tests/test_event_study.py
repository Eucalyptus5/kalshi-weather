from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal
from typing import Literal

from bot.lag.event_study import (
    LagBucket,
    LagReport,
    OrderbookSnapshotRow,
    study_lag,
)
from bot.lag.lock_events import LockEvent


UTC = _timezone.utc


def _event(
    ticker: str,
    *,
    side: Literal["yes", "no"] = "yes",
    t0: datetime,
    strike: Decimal | str | int = 85,
    crossing: Decimal | str | int | None = None,
    lock_ambiguous: bool = False,
) -> LockEvent:
    strike_d = Decimal(str(strike))
    crossing_d = Decimal(str(crossing)) if crossing is not None else strike_d + Decimal("2")
    return LockEvent(
        ticker=ticker,
        side_locked=side,
        t0=t0,
        strike=strike_d,
        crossing_temp_f=crossing_d,
        lock_ambiguous=lock_ambiguous,
    )


def _snap(
    ticker: str,
    t: datetime,
    yes_bid: Decimal | str,
    yes_ask: Decimal | str,
) -> OrderbookSnapshotRow:
    return OrderbookSnapshotRow(
        ticker=ticker,
        snapshot_at=t,
        yes_bid=Decimal(str(yes_bid)),
        yes_ask=Decimal(str(yes_ask)),
    )


def test_clean_yes_lock_lag_60s() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=30), "0.40", "0.50"),
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.94", "0.96"),
    ]

    report = study_lag([ev], snaps)

    assert isinstance(report, LagReport)
    assert len(report.raw) == 1
    b = report.raw[0]
    assert isinstance(b, LagBucket)
    assert b.series == "KXHIGHDEN"
    assert b.n == 1
    assert b.median_lag_s == 60
    assert b.p90_lag_s == 60
    assert b.never_repriced_n == 0
    assert b.mislock_n == 0
    assert report.net_of_floor == report.raw


def test_no_in_band_snapshot_never_repriced() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=30), "0.45", "0.55"),
    ]

    report = study_lag([ev], snaps)

    assert report.raw[0].n == 1
    assert report.raw[0].never_repriced_n == 1
    assert report.raw[0].median_lag_s is None
    assert report.raw[0].p90_lag_s is None


def test_no_lock_band_condition() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event(
        "KXHIGHCHI-26JUN17-T75-80",
        side="no",
        t0=t0,
        strike=80,
        crossing=82,
    )
    snaps = [
        _snap("KXHIGHCHI-26JUN17-T75-80", t0 + timedelta(seconds=120), "0.08", "0.12"),
        _snap("KXHIGHCHI-26JUN17-T75-80", t0 + timedelta(seconds=240), "0.03", "0.05"),
    ]

    report = study_lag([ev], snaps)

    assert report.raw[0].median_lag_s == 240


def test_per_city_bucketing_pinned_quantile() -> None:
    t0_a = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    t0_b = datetime(2026, 6, 17, 21, 0, tzinfo=UTC)
    t0_c = datetime(2026, 6, 17, 22, 0, tzinfo=UTC)

    ev_den_a = _event("KXHIGHDEN-26JUN17-T85", t0=t0_a)
    ev_den_b = _event("KXHIGHDEN-26JUN17-T90", t0=t0_b)
    ev_chi = _event("KXHIGHCHI-26JUN17-T75-80", side="no", t0=t0_c, strike=80, crossing=82)

    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0_a + timedelta(seconds=60), "0.95", "0.96"),
        _snap("KXHIGHDEN-26JUN17-T90", t0_b + timedelta(seconds=120), "0.95", "0.96"),
        _snap("KXHIGHCHI-26JUN17-T75-80", t0_c + timedelta(seconds=90), "0.03", "0.05"),
    ]

    report = study_lag([ev_chi, ev_den_a, ev_den_b], snaps)

    assert [b.series for b in report.raw] == ["KXHIGHCHI", "KXHIGHDEN"]
    chi_bucket = report.raw[0]
    den_bucket = report.raw[1]
    assert chi_bucket.n == 1
    assert chi_bucket.median_lag_s == 90
    assert chi_bucket.p90_lag_s == 90
    assert den_bucket.n == 2
    assert den_bucket.median_lag_s == 60
    assert den_bucket.p90_lag_s == 120


def test_yes_mislock_settle_below_strike() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
    ]
    settle = {"KXHIGHDEN-26JUN17-T85": Decimal("83")}

    report = study_lag([ev], snaps, settle_by_event=settle)

    b = report.raw[0]
    assert b.mislock_n == 1
    assert b.mislock_rate == Decimal("1") / Decimal("1")


def test_no_mislock_settle_at_or_below_high() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev_under = _event("KXHIGHCHI-26JUN17-T75-80", side="no", t0=t0, strike=80, crossing=82)
    snaps = [
        _snap("KXHIGHCHI-26JUN17-T75-80", t0 + timedelta(seconds=60), "0.03", "0.05"),
    ]
    settle = {"KXHIGHCHI-26JUN17-T75-80": Decimal("78")}

    report = study_lag([ev_under], snaps, settle_by_event=settle)
    assert report.raw[0].mislock_n == 1

    settle_eq = {"KXHIGHCHI-26JUN17-T75-80": Decimal("80")}
    report_eq = study_lag([ev_under], snaps, settle_by_event=settle_eq)
    assert report_eq.raw[0].mislock_n == 1


def test_no_mislock_when_settle_clears() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    yes_ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    no_ev = _event("KXHIGHCHI-26JUN17-T75-80", side="no", t0=t0, strike=80, crossing=82)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
        _snap("KXHIGHCHI-26JUN17-T75-80", t0 + timedelta(seconds=60), "0.03", "0.05"),
    ]
    settle = {
        "KXHIGHDEN-26JUN17-T85": Decimal("87"),
        "KXHIGHCHI-26JUN17-T75-80": Decimal("82"),
    }

    report = study_lag([yes_ev, no_ev], snaps, settle_by_event=settle)

    for b in report.raw:
        assert b.mislock_n == 0
        assert b.mislock_rate == Decimal("0") / Decimal("1")


def test_missing_settle_dilutes_mislock_rate() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
    ]

    report = study_lag([ev], snaps, settle_by_event={})

    b = report.raw[0]
    assert b.mislock_n == 0
    assert b.mislock_rate == Decimal("0") / Decimal("1")


def test_net_of_floor_drops_ambiguous_locks() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    clean_ev = _event(
        "KXHIGHDEN-26JUN17-T85",
        t0=t0,
        strike=85,
        crossing=87,
        lock_ambiguous=False,
    )
    amb_ev = _event(
        "KXHIGHDEN-26JUN17-T90",
        t0=t0 + timedelta(hours=1),
        strike=90,
        crossing=90,
        lock_ambiguous=True,
    )
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
        _snap(
            "KXHIGHDEN-26JUN17-T90",
            t0 + timedelta(hours=1, seconds=60),
            "0.95",
            "0.96",
        ),
    ]

    report = study_lag([clean_ev, amb_ev], snaps)

    assert report.raw[0].n == 2
    assert report.net_of_floor[0].n == 1


def test_net_of_floor_drops_rounding_band_by_settle() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=Decimal("85"))
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
    ]

    settle_band = {"KXHIGHDEN-26JUN17-T85": Decimal("86")}
    band_report = study_lag([ev], snaps, settle_by_event=settle_band)
    assert band_report.raw[0].n == 1
    assert band_report.net_of_floor == []

    settle_far = {"KXHIGHDEN-26JUN17-T85": Decimal("87")}
    far_report = study_lag([ev], snaps, settle_by_event=settle_far)
    assert far_report.raw[0].n == 1
    assert far_report.net_of_floor[0].n == 1

    settle_eq = {"KXHIGHDEN-26JUN17-T85": Decimal("85")}
    eq_report = study_lag([ev], snaps, settle_by_event=settle_eq)
    assert eq_report.net_of_floor[0].n == 1


def test_snapshot_floor_flag_trips() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    cadence_snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=0), "0.50", "0.50"),
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=89), "0.50", "0.50"),
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=178), "0.50", "0.50"),
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=267), "0.50", "0.50"),
    ]
    in_band = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
    ]

    report = study_lag([ev], cadence_snaps + in_band)

    b = report.raw[0]
    assert b.median_lag_s == 60
    assert b.snapshot_unreliable is True


def test_snapshot_floor_flag_does_not_trip() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    cadence_snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=0), "0.50", "0.50"),
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=89), "0.50", "0.50"),
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=178), "0.50", "0.50"),
    ]
    in_band = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=300), "0.95", "0.96"),
    ]

    report = study_lag([ev], cadence_snaps + in_band)

    b = report.raw[0]
    assert b.median_lag_s == 300
    assert b.snapshot_unreliable is False


def test_snapshot_floor_flag_undefined_when_no_cadence() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
    ]

    report = study_lag([ev], snaps)

    assert report.raw[0].snapshot_unreliable is False


def test_decimal_strict_on_report() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
    ]

    report = study_lag([ev], snaps)

    assert type(report.raw[0].mislock_rate) is Decimal


def test_stable_ordering_alphabetical() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    den = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    mia = _event("KXHIGHMIA-26JUN17-T85", t0=t0, strike=85, crossing=87)
    chi = _event("KXHIGHCHI-26JUN17-T75-80", side="no", t0=t0, strike=80, crossing=82)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
        _snap("KXHIGHMIA-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
        _snap("KXHIGHCHI-26JUN17-T75-80", t0 + timedelta(seconds=60), "0.03", "0.05"),
    ]

    report = study_lag([mia, den, chi], snaps)

    assert [b.series for b in report.raw] == ["KXHIGHCHI", "KXHIGHDEN", "KXHIGHMIA"]


def test_quantile_k1() -> None:
    t0 = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    ev = _event("KXHIGHDEN-26JUN17-T85", t0=t0, strike=85, crossing=87)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0 + timedelta(seconds=60), "0.95", "0.96"),
    ]

    report = study_lag([ev], snaps)

    assert report.raw[0].median_lag_s == 60
    assert report.raw[0].p90_lag_s == 60


def test_quantile_k3_pinned() -> None:
    t0_a = datetime(2026, 6, 17, 20, 0, tzinfo=UTC)
    t0_b = datetime(2026, 6, 17, 21, 0, tzinfo=UTC)
    t0_c = datetime(2026, 6, 17, 22, 0, tzinfo=UTC)
    ev_a = _event("KXHIGHDEN-26JUN17-T85", t0=t0_a)
    ev_b = _event("KXHIGHDEN-26JUN17-T90", t0=t0_b)
    ev_c = _event("KXHIGHDEN-26JUN17-T95", t0=t0_c)
    snaps = [
        _snap("KXHIGHDEN-26JUN17-T85", t0_a + timedelta(seconds=30), "0.95", "0.96"),
        _snap("KXHIGHDEN-26JUN17-T90", t0_b + timedelta(seconds=60), "0.95", "0.96"),
        _snap("KXHIGHDEN-26JUN17-T95", t0_c + timedelta(seconds=90), "0.95", "0.96"),
    ]

    report = study_lag([ev_a, ev_b, ev_c], snaps)

    b = report.raw[0]
    assert b.median_lag_s == 60
    assert b.p90_lag_s == 90
