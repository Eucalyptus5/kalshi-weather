from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bot.lag.capture_sim import CaptureResult
from bot.lag.event_study import POOLED, LagBucket, LagReport
from bot.lag.snapshot_loader import PROD_ERA_START
from bot.markets.parser import series_id


_ZERO = Decimal("0")
_DOLLAR_Q = Decimal("0.01")
_RATE_Q = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class CaptureAggregate:
    series: str
    n_filled: int
    total_pnl: Decimal
    total_notional: Decimal
    total_fees: Decimal
    total_capacity_contracts: int
    depth_lower_bound_any: bool


@dataclass(frozen=True, slots=True)
class LatencyPoint:
    total_s: int
    raw: list[CaptureAggregate]
    net_of_floor: list[CaptureAggregate]
    raw_pooled: CaptureAggregate
    net_of_floor_pooled: CaptureAggregate


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    book_source: str
    events_found: int
    events_used: int
    gap_excluded_n: int
    no_coverage_n: int


def aggregate_captures(captures: list[CaptureResult]) -> list[CaptureAggregate]:
    by_series: dict[str, list[CaptureResult]] = {}
    for c in captures:
        by_series.setdefault(series_id(c.ticker), []).append(c)
    return [_aggregate(s, by_series[s]) for s in sorted(by_series)]


def pool_captures(captures: list[CaptureResult]) -> CaptureAggregate:
    return _aggregate(POOLED, captures)


def _aggregate(label: str, group: list[CaptureResult]) -> CaptureAggregate:
    return CaptureAggregate(
        series=label,
        n_filled=sum(1 for c in group if c.contracts_filled > 0),
        total_pnl=sum((c.pnl for c in group), _ZERO),
        total_notional=sum((c.notional_spent for c in group), _ZERO),
        total_fees=sum((c.fees_paid for c in group), _ZERO),
        total_capacity_contracts=sum(c.fillable_depth for c in group),
        depth_lower_bound_any=any(c.depth_lower_bound for c in group),
    )


def format_report(
    report: LagReport,
    curve: list[LatencyPoint],
    coverage: SourceCoverage,
    *,
    gate_stack_s: int,
) -> str:
    lines: list[str] = []
    lines.append(
        f"# retrospective lag readout  prod_era_start={PROD_ERA_START.isoformat()}  "
        f"book_source={coverage.book_source}"
    )
    lines.append(
        f"  events_found={coverage.events_found}  events_used={coverage.events_used}  "
        f"gap_excluded={coverage.gap_excluded_n}  no_coverage={coverage.no_coverage_n}"
    )
    lines.append("")
    lines.append(f"== GATE INPUTS (pooled, net of floor, {gate_stack_s}s stack)")
    lines.extend(_format_gate(report.net_of_floor_pooled, curve, gate_stack_s))
    lines.append("")
    lines.append("== RAW (all events)")
    lines.extend(_format_buckets([report.raw_pooled] + report.raw))
    lines.append("")
    lines.append("== NET OF FLOOR (drops lock_ambiguous and settle-rounding band)")
    lines.extend(_format_buckets([report.net_of_floor_pooled] + report.net_of_floor))
    lines.append("")
    lines.append("== CAPTURE (RAW)")
    lines.extend(_format_curve(curve, net=False))
    lines.append("")
    lines.append("== CAPTURE (NET OF FLOOR)")
    lines.extend(_format_curve(curve, net=True))
    return "\n".join(lines)


def _format_gate(
    pooled: LagBucket,
    curve: list[LatencyPoint],
    gate_stack_s: int,
) -> list[str]:
    out = [
        f"  median_lag_s={_int_or_na(pooled.median_lag_s)}  "
        f"p25_lag_s={_int_or_na(pooled.p25_lag_s)}  n={pooled.n}  "
        f"never_repriced_n={pooled.never_repriced_n}  "
        f"mislock_rate={pooled.mislock_rate.quantize(_RATE_Q)}"
    ]
    point = next((p for p in curve if p.total_s == gate_stack_s), None)
    if point is None:
        out.append(f"  ({gate_stack_s}s stack not run; see --latency-total-s)")
        return out
    a = point.net_of_floor_pooled
    out.append(
        f"  n_filled={a.n_filled}  pnl={a.total_pnl.quantize(_DOLLAR_Q)}  "
        f"notional={a.total_notional.quantize(_DOLLAR_Q)}  "
        f"fees={a.total_fees.quantize(_DOLLAR_Q)}  "
        f"capacity={a.total_capacity_contracts}"
    )
    return out


def _format_curve(curve: list[LatencyPoint], *, net: bool) -> list[str]:
    if not curve:
        return ["  (none)"]
    out: list[str] = []
    for point in curve:
        out.append(f"  -- {point.total_s}s stack")
        pooled = point.net_of_floor_pooled if net else point.raw_pooled
        per_series = point.net_of_floor if net else point.raw
        out.extend(_format_captures([pooled] + per_series))
    return out


def _format_buckets(buckets: list[LagBucket]) -> list[str]:
    out: list[str] = []
    for b in buckets:
        annot = " [snapshot_floor]" if b.snapshot_unreliable else ""
        out.append(
            f"  {b.series}  n={b.n}  median_lag_s={_int_or_na(b.median_lag_s)}  "
            f"p25_lag_s={_int_or_na(b.p25_lag_s)}  p90_lag_s={_int_or_na(b.p90_lag_s)}  "
            f"cadence_s={_int_or_na(b.cadence_s)}  never_repriced_n={b.never_repriced_n}  "
            f"mislock_n={b.mislock_n}  "
            f"mislock_rate={b.mislock_rate.quantize(_RATE_Q)}{annot}"
        )
    return out


def _format_captures(aggregates: list[CaptureAggregate]) -> list[str]:
    out: list[str] = []
    for a in aggregates:
        annot = " [lower bound]" if a.depth_lower_bound_any else ""
        out.append(
            f"    {a.series}  n_filled={a.n_filled}  "
            f"pnl={a.total_pnl.quantize(_DOLLAR_Q)}  "
            f"notional={a.total_notional.quantize(_DOLLAR_Q)}  "
            f"fees={a.total_fees.quantize(_DOLLAR_Q)}  "
            f"capacity={a.total_capacity_contracts}{annot}"
        )
    return out


def _int_or_na(value: int | None) -> str:
    return "n/a" if value is None else str(value)
