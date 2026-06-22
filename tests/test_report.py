from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from bot.lag.capture_sim import CaptureResult
from bot.lag.event_study import LagBucket, LagReport
from bot.lag.report import CaptureAggregate, aggregate_captures, format_report


UTC = timezone.utc


def _capture(
    ticker: str,
    *,
    side: Literal["yes", "no"] = "yes",
    contracts_filled: int = 5,
    pnl: str = "1.50",
    notional: str = "2.00",
    fees: str = "0.05",
    fillable_depth: int = 5,
    depth_lower_bound: bool = False,
) -> CaptureResult:
    return CaptureResult(
        ticker=ticker,
        side_locked=side,
        stale_price=Decimal("0.40"),
        contracts_filled=contracts_filled,
        fillable_depth=fillable_depth,
        notional_spent=Decimal(notional),
        payoff=Decimal("5.00"),
        fees_paid=Decimal(fees),
        pnl=Decimal(pnl),
        depth_lower_bound=depth_lower_bound,
    )


def test_aggregate_captures_empty() -> None:
    assert aggregate_captures([]) == []


def test_aggregate_captures_groups_by_series_alphabetical() -> None:
    captures = [
        _capture(
            "KXHIGHDEN-26JUN17-T85",
            contracts_filled=3,
            pnl="1.00",
            notional="1.50",
            fillable_depth=3,
        ),
        _capture(
            "KXHIGHDEN-26JUN17-T90",
            contracts_filled=4,
            pnl="2.00",
            notional="2.50",
            fillable_depth=4,
        ),
        _capture(
            "KXHIGHCHI-26JUN17-T75-80",
            side="no",
            contracts_filled=2,
            pnl="0.50",
            notional="0.80",
            fillable_depth=2,
        ),
    ]

    aggs = aggregate_captures(captures)

    assert [a.series for a in aggs] == ["KXHIGHCHI", "KXHIGHDEN"]
    chi, den = aggs
    assert chi.n_filled == 1
    assert chi.total_pnl == Decimal("0.50")
    assert den.n_filled == 2
    assert den.total_pnl == Decimal("3.00")
    assert den.total_notional == Decimal("4.00")
    assert den.total_capacity_contracts == 7


def test_aggregate_n_filled_skips_zero_fills() -> None:
    captures = [
        _capture("KXHIGHDEN-26JUN17-T85", contracts_filled=0, pnl="0.00"),
        _capture("KXHIGHDEN-26JUN17-T90", contracts_filled=2, pnl="1.00"),
    ]
    aggs = aggregate_captures(captures)
    assert aggs[0].n_filled == 1


def test_aggregate_depth_lower_bound_any() -> None:
    a = _capture("KXHIGHDEN-26JUN17-T85", depth_lower_bound=False)
    b = _capture("KXHIGHDEN-26JUN17-T90", depth_lower_bound=True)
    aggs = aggregate_captures([a, b])
    assert aggs[0].depth_lower_bound_any is True

    aggs_clean = aggregate_captures([a])
    assert aggs_clean[0].depth_lower_bound_any is False


def test_format_report_smoke() -> None:
    raw_bucket = LagBucket(
        series="KXHIGHDEN",
        n=3,
        median_lag_s=60,
        p90_lag_s=120,
        never_repriced_n=1,
        mislock_n=1,
        mislock_rate=Decimal("0.33333"),
        snapshot_unreliable=True,
    )
    raw_bucket_b = LagBucket(
        series="KXHIGHCHI",
        n=1,
        median_lag_s=None,
        p90_lag_s=None,
        never_repriced_n=1,
        mislock_n=0,
        mislock_rate=Decimal("0"),
        snapshot_unreliable=False,
    )
    net_bucket = LagBucket(
        series="KXHIGHDEN",
        n=2,
        median_lag_s=90,
        p90_lag_s=90,
        never_repriced_n=0,
        mislock_n=0,
        mislock_rate=Decimal("0"),
        snapshot_unreliable=False,
    )
    report = LagReport(raw=[raw_bucket_b, raw_bucket], net_of_floor=[net_bucket])

    agg_raw = [
        CaptureAggregate(
            series="KXHIGHDEN",
            n_filled=3,
            total_pnl=Decimal("12.345678"),
            total_notional=Decimal("8.50"),
            total_fees=Decimal("0.25"),
            total_capacity_contracts=15,
            depth_lower_bound_any=True,
        )
    ]
    agg_net = [
        CaptureAggregate(
            series="KXHIGHDEN",
            n_filled=2,
            total_pnl=Decimal("8.00"),
            total_notional=Decimal("5.00"),
            total_fees=Decimal("0.10"),
            total_capacity_contracts=10,
            depth_lower_bound_any=False,
        )
    ]

    out = format_report(report, agg_raw, agg_net)

    assert "prod_era_start" in out
    assert "RAW (all events)" in out
    assert "NET OF FLOOR" in out
    assert "CAPTURE (RAW)" in out
    assert "CAPTURE (NET OF FLOOR)" in out
    assert "KXHIGHDEN" in out
    assert "KXHIGHCHI" in out
    assert "n=3" in out
    assert "median_lag_s=60" in out
    assert "n/a" in out
    assert "[snapshot_floor]" in out
    assert "[lower bound]" in out


def test_format_report_decimal_precision() -> None:
    raw_bucket = LagBucket(
        series="KXHIGHDEN",
        n=3,
        median_lag_s=60,
        p90_lag_s=120,
        never_repriced_n=0,
        mislock_n=1,
        mislock_rate=Decimal("0.33333333"),
        snapshot_unreliable=False,
    )
    report = LagReport(raw=[raw_bucket], net_of_floor=[])

    agg = [
        CaptureAggregate(
            series="KXHIGHDEN",
            n_filled=3,
            total_pnl=Decimal("12.345678"),
            total_notional=Decimal("8.501"),
            total_fees=Decimal("0.254"),
            total_capacity_contracts=15,
            depth_lower_bound_any=False,
        )
    ]

    out = format_report(report, agg, [])

    assert "12.35" in out
    assert "8.50" in out
    assert "0.25" in out
    assert "0.3333" in out


def test_format_report_handles_empty_buckets() -> None:
    out = format_report(LagReport(raw=[], net_of_floor=[]), [], [])
    assert "RAW (all events)" in out
    assert "NET OF FLOOR" in out


def test_format_report_event_timestamp_header() -> None:
    out = format_report(LagReport(raw=[], net_of_floor=[]), [], [])
    assert datetime(2026, 6, 13, 0, 0, 18, tzinfo=UTC).isoformat() in out
