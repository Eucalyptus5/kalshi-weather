from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bot.lag.capture_sim import CaptureResult
from bot.lag.event_study import LagBucket, LagReport
from bot.lag.snapshot_loader import PROD_ERA_START
from bot.markets.parser import series_id


_ZERO = Decimal("0")
_PRICE_Q = Decimal("0.0001")
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


def aggregate_captures(captures: list[CaptureResult]) -> list[CaptureAggregate]:
    by_series: dict[str, list[CaptureResult]] = {}
    for c in captures:
        by_series.setdefault(series_id(c.ticker), []).append(c)

    out: list[CaptureAggregate] = []
    for s in sorted(by_series):
        group = by_series[s]
        out.append(
            CaptureAggregate(
                series=s,
                n_filled=sum(1 for c in group if c.contracts_filled > 0),
                total_pnl=sum((c.pnl for c in group), _ZERO),
                total_notional=sum((c.notional_spent for c in group), _ZERO),
                total_fees=sum((c.fees_paid for c in group), _ZERO),
                total_capacity_contracts=sum(c.fillable_depth for c in group),
                depth_lower_bound_any=any(c.depth_lower_bound for c in group),
            )
        )
    return out


def format_report(
    raw: LagReport,
    aggregates_raw: list[CaptureAggregate],
    aggregates_net: list[CaptureAggregate],
) -> str:
    lines: list[str] = []
    lines.append(f"# retrospective lag readout  prod_era_start={PROD_ERA_START.isoformat()}")
    lines.append("")
    lines.append("== RAW (all events)")
    lines.extend(_format_buckets(raw.raw))
    lines.append("")
    lines.append("== NET OF FLOOR (drops lock_ambiguous and settle-rounding band)")
    lines.extend(_format_buckets(raw.net_of_floor))
    lines.append("")
    lines.append("== CAPTURE (RAW)")
    lines.extend(_format_captures(aggregates_raw))
    lines.append("")
    lines.append("== CAPTURE (NET OF FLOOR)")
    lines.extend(_format_captures(aggregates_net))
    return "\n".join(lines)


def _format_buckets(buckets: list[LagBucket]) -> list[str]:
    if not buckets:
        return ["  (none)"]
    out: list[str] = []
    for b in buckets:
        median_s = "n/a" if b.median_lag_s is None else str(b.median_lag_s)
        p90_s = "n/a" if b.p90_lag_s is None else str(b.p90_lag_s)
        rate = b.mislock_rate.quantize(_RATE_Q)
        annot = " [snapshot_floor]" if b.snapshot_unreliable else ""
        out.append(
            f"  {b.series}  n={b.n}  median_lag_s={median_s}  p90_lag_s={p90_s}  "
            f"never_repriced_n={b.never_repriced_n}  mislock_n={b.mislock_n}  "
            f"mislock_rate={rate}{annot}"
        )
    return out


def _format_captures(aggregates: list[CaptureAggregate]) -> list[str]:
    if not aggregates:
        return ["  (none)"]
    out: list[str] = []
    for a in aggregates:
        annot = " [lower bound]" if a.depth_lower_bound_any else ""
        pnl = a.total_pnl.quantize(_DOLLAR_Q)
        notional = a.total_notional.quantize(_DOLLAR_Q)
        fees = a.total_fees.quantize(_DOLLAR_Q)
        out.append(
            f"  {a.series}  n_filled={a.n_filled}  pnl={pnl}  notional={notional}  "
            f"fees={fees}  capacity={a.total_capacity_contracts}{annot}"
        )
    return out
