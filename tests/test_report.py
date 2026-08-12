from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from bot.lag.capture_sim import CaptureResult
from bot.lag.event_study import POOLED, LagBucket, LagReport
from bot.lag.report import (
    CaptureAggregate,
    LatencyPoint,
    SourceCoverage,
    aggregate_captures,
    format_report,
    pool_captures,
)


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


def _bucket(
    series: str,
    *,
    n: int = 3,
    median: int | None = 60,
    p25: int | None = 45,
    p90: int | None = 120,
    cadence: int | None = 60,
    never_repriced_n: int = 0,
    no_mid_n: int = 0,
    mislock_n: int = 0,
    mislock_rate: str = "0",
    snapshot_unreliable: bool = False,
) -> LagBucket:
    return LagBucket(
        series=series,
        n=n,
        median_lag_s=median,
        p25_lag_s=p25,
        p90_lag_s=p90,
        cadence_s=cadence,
        never_repriced_n=never_repriced_n,
        no_mid_n=no_mid_n,
        mislock_n=mislock_n,
        mislock_rate=Decimal(mislock_rate),
        snapshot_unreliable=snapshot_unreliable,
    )


def _agg(
    series: str,
    *,
    n_filled: int = 3,
    pnl: str = "8.00",
    notional: str = "5.00",
    fees: str = "0.10",
    capacity: int = 10,
    lower_bound: bool = False,
) -> CaptureAggregate:
    return CaptureAggregate(
        series=series,
        n_filled=n_filled,
        total_pnl=Decimal(pnl),
        total_notional=Decimal(notional),
        total_fees=Decimal(fees),
        total_capacity_contracts=capacity,
        depth_lower_bound_any=lower_bound,
    )


def _point(total_s: int, *, net_pooled: CaptureAggregate | None = None) -> LatencyPoint:
    return LatencyPoint(
        total_s=total_s,
        raw=[_agg("KXHIGHDEN")],
        net_of_floor=[_agg("KXHIGHDEN", n_filled=2)],
        raw_pooled=_agg(POOLED),
        net_of_floor_pooled=net_pooled or _agg(POOLED, n_filled=2),
    )


def _coverage(**kwargs: object) -> SourceCoverage:
    base: dict[str, object] = {
        "book_source": "ws",
        "events_found": 40,
        "events_used": 38,
        "gap_excluded_n": 1,
        "no_coverage_n": 1,
    }
    base.update(kwargs)
    return SourceCoverage(**base)  # type: ignore[arg-type]


def _report(
    *,
    raw: list[LagBucket] | None = None,
    net: list[LagBucket] | None = None,
    raw_pooled: LagBucket | None = None,
    net_pooled: LagBucket | None = None,
) -> LagReport:
    return LagReport(
        raw=raw if raw is not None else [_bucket("KXHIGHDEN")],
        net_of_floor=net if net is not None else [_bucket("KXHIGHDEN", n=2)],
        raw_pooled=raw_pooled or _bucket(POOLED),
        net_of_floor_pooled=net_pooled or _bucket(POOLED, n=2),
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


def test_pool_captures_spans_every_series() -> None:
    captures = [
        _capture("KXHIGHDEN-26JUN17-T85", contracts_filled=3, pnl="1.00", fillable_depth=3),
        _capture("KXHIGHCHI-26JUN17-T80", contracts_filled=0, pnl="-0.25", fillable_depth=0),
        _capture("KXHIGHLAX-26JUN17-T95", contracts_filled=4, pnl="2.00", fillable_depth=4),
    ]

    pooled = pool_captures(captures)

    assert pooled.series == POOLED
    assert pooled.n_filled == 2
    assert pooled.total_pnl == Decimal("2.75")
    assert pooled.total_capacity_contracts == 7


def test_pool_captures_empty() -> None:
    pooled = pool_captures([])

    assert pooled.series == POOLED
    assert pooled.n_filled == 0
    assert pooled.total_pnl == Decimal("0")
    assert pooled.depth_lower_bound_any is False


def test_format_report_smoke() -> None:
    report = _report(
        raw=[
            _bucket("KXHIGHCHI", n=1, median=None, p25=None, p90=None, never_repriced_n=1),
            _bucket("KXHIGHDEN", mislock_n=1, mislock_rate="0.33333", snapshot_unreliable=True),
        ],
        raw_pooled=_bucket(POOLED, n=4, snapshot_unreliable=True),
    )
    curve = [_point(s) for s in (30, 60, 90, 120)]

    out = format_report(report, curve, _coverage(), gate_stack_s=90)

    assert "prod_era_start" in out
    assert "book_source=ws" in out
    assert "RAW (all events)" in out
    assert "NET OF FLOOR" in out
    assert "CAPTURE (RAW)" in out
    assert "CAPTURE (NET OF FLOOR)" in out
    assert "KXHIGHDEN" in out
    assert "KXHIGHCHI" in out
    assert "n/a" in out
    assert "[snapshot_floor]" in out


def test_format_report_emits_every_latency_point_raw_and_net() -> None:
    curve = [_point(s) for s in (30, 60, 90, 120)]

    out = format_report(_report(), curve, _coverage(), gate_stack_s=90)

    raw_block, net_block = out.split("== CAPTURE (NET OF FLOOR)")
    for stack in (30, 60, 90, 120):
        assert f"-- {stack}s stack" in raw_block
        assert f"-- {stack}s stack" in net_block


def test_format_report_gate_inputs_are_legible_without_arithmetic() -> None:
    report = _report(
        net_pooled=_bucket(POOLED, n=24, median=83, p25=61, mislock_rate="0.0250", cadence=0)
    )
    curve = [
        _point(30),
        _point(60),
        _point(90, net_pooled=_agg(POOLED, n_filled=27, pnl="-3.40")),
        _point(120),
    ]

    out = format_report(report, curve, _coverage(), gate_stack_s=90)
    gate_block = out.split("== GATE INPUTS")[1].split("==")[0]

    assert "pooled, net of floor, 90s stack" in gate_block
    assert "median_lag_s=83" in gate_block
    assert "p25_lag_s=61" in gate_block
    assert "n=24" in gate_block
    assert "mislock_rate=0.0250" in gate_block
    assert "n_filled=27" in gate_block
    assert "pnl=-3.40" in gate_block


def test_format_report_says_so_when_the_gate_stack_was_not_run() -> None:
    out = format_report(_report(), [_point(60)], _coverage(), gate_stack_s=90)

    assert "90s stack not run" in out


def test_format_report_pools_first_in_every_bucket_listing() -> None:
    report = _report(raw=[_bucket("KXHIGHCHI"), _bucket("KXHIGHDEN")])

    out = format_report(report, [_point(90)], _coverage(), gate_stack_s=90)
    raw_block = out.split("== RAW (all events)")[1].split("== NET OF FLOOR")[0]

    assert raw_block.strip().splitlines()[0].strip().startswith(POOLED)


def test_format_report_reports_ws_exclusion_counts() -> None:
    coverage = _coverage(events_found=412, events_used=398, gap_excluded_n=11, no_coverage_n=3)

    out = format_report(_report(), [_point(90)], coverage, gate_stack_s=90)

    assert "events_found=412" in out
    assert "events_used=398" in out
    assert "gap_excluded=11" in out
    assert "no_coverage=3" in out


def test_format_report_shows_p25_and_cadence_per_bucket() -> None:
    out = format_report(_report(), [_point(90)], _coverage(), gate_stack_s=90)

    assert "p25_lag_s=45" in out
    assert "cadence_s=60" in out


def test_format_report_decimal_precision() -> None:
    report = _report(raw=[_bucket("KXHIGHDEN", mislock_n=1, mislock_rate="0.33333333")])
    curve = [
        _point(
            90,
            net_pooled=_agg(POOLED, pnl="12.345678", notional="8.501", fees="0.254"),
        )
    ]

    out = format_report(report, curve, _coverage(), gate_stack_s=90)

    assert "12.35" in out
    assert "8.50" in out
    assert "0.25" in out
    assert "0.3333" in out


def test_format_report_handles_empty_buckets() -> None:
    report = _report(
        raw=[], net=[], raw_pooled=_bucket(POOLED, n=0), net_pooled=_bucket(POOLED, n=0)
    )

    out = format_report(report, [], _coverage(book_source="rest"), gate_stack_s=90)

    assert "RAW (all events)" in out
    assert "NET OF FLOOR" in out
    assert "book_source=rest" in out


def test_format_report_event_timestamp_header() -> None:
    out = format_report(_report(), [_point(90)], _coverage(), gate_stack_s=90)

    assert datetime(2026, 6, 13, 0, 0, 18, tzinfo=UTC).isoformat() in out
