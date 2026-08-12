from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Mapping

from bot.lag.lock_events import LockEvent
from bot.lag.mid import mid2, ticks, two_sided
from bot.markets.parser import series_id


POOLED = "POOLED"

YES_BAND = Decimal("0.95")
NO_BAND = Decimal("0.05")
YES_BAND2 = 2 * ticks(YES_BAND)
NO_BAND2 = 2 * ticks(NO_BAND)
_FLOOR_FACTOR = Decimal("1.5")


@dataclass(frozen=True, slots=True)
class OrderbookSnapshotRow:
    ticker: str
    snapshot_at: datetime
    yes_bid: Decimal
    yes_ask: Decimal
    no_bid: Decimal | None = None
    no_ask: Decimal | None = None
    yes_ask_depth: int | None = None
    yes_bid_depth: int | None = None
    no_ask_depth: int | None = None
    no_bid_depth: int | None = None


@dataclass(frozen=True, slots=True)
class LagBucket:
    series: str
    n: int
    median_lag_s: int | None
    p25_lag_s: int | None
    p90_lag_s: int | None
    cadence_s: int | None
    never_repriced_n: int
    no_mid_n: int
    mislock_n: int
    mislock_rate: Decimal
    snapshot_unreliable: bool


@dataclass(frozen=True, slots=True)
class LagReport:
    raw: list[LagBucket]
    net_of_floor: list[LagBucket]
    raw_pooled: LagBucket
    net_of_floor_pooled: LagBucket


@dataclass(frozen=True, slots=True)
class EventProbe:
    lag_s: int | None
    cadence_s: int | None
    no_mid_n: int = 0


def study_lag(
    events: list[LockEvent],
    snapshots: list[OrderbookSnapshotRow],
    *,
    settle_by_event: Mapping[str, Decimal] | None = None,
    day_window_seconds: int = 24 * 3600,
) -> LagReport:
    by_ticker: dict[str, list[OrderbookSnapshotRow]] = {}
    for s in snapshots:
        by_ticker.setdefault(s.ticker, []).append(s)
    for rows in by_ticker.values():
        rows.sort(key=lambda r: r.snapshot_at)

    probes: dict[tuple[str, datetime], EventProbe] = {}
    for ev in events:
        ticker_snaps = by_ticker.get(ev.ticker, [])
        lag_s, no_mid_n = _event_lag(ev, ticker_snaps, day_window_seconds)
        probes[(ev.ticker, ev.t0)] = EventProbe(
            lag_s=lag_s,
            cadence_s=_event_cadence(ev, ticker_snaps),
            no_mid_n=no_mid_n,
        )
    return study_lag_from_probes(events, probes, settle_by_event=settle_by_event)


def study_lag_from_probes(
    events: list[LockEvent],
    probes: Mapping[tuple[str, datetime], EventProbe],
    *,
    settle_by_event: Mapping[str, Decimal] | None = None,
) -> LagReport:
    """Missing settles leave mislock unknown for that event; the bucket mislock_rate
    denominator is bucket n, so unknowns dilute the rate downward."""
    raw, raw_pooled = _aggregate(events, probes, settle_by_event)
    filtered = filter_net_of_floor(events, settle_by_event)
    net, net_pooled = _aggregate(filtered, probes, settle_by_event)
    return LagReport(
        raw=raw,
        net_of_floor=net,
        raw_pooled=raw_pooled,
        net_of_floor_pooled=net_pooled,
    )


def _aggregate(
    events: list[LockEvent],
    probes: Mapping[tuple[str, datetime], EventProbe],
    settle_by_event: Mapping[str, Decimal] | None,
) -> tuple[list[LagBucket], LagBucket]:
    by_series: dict[str, list[LockEvent]] = {}
    for ev in events:
        by_series.setdefault(series_id(ev.ticker), []).append(ev)

    buckets = [_bucket(s, by_series[s], probes, settle_by_event) for s in sorted(by_series)]
    return buckets, _bucket(POOLED, events, probes, settle_by_event)


def _bucket(
    label: str,
    events: list[LockEvent],
    probes: Mapping[tuple[str, datetime], EventProbe],
    settle_by_event: Mapping[str, Decimal] | None,
) -> LagBucket:
    lags: list[int | None] = []
    cadences: list[int] = []
    mislock_n = 0
    no_mid_n = 0
    for ev in events:
        probe = probes[(ev.ticker, ev.t0)]
        lags.append(probe.lag_s)
        no_mid_n += probe.no_mid_n
        if probe.cadence_s is not None:
            cadences.append(probe.cadence_s)
        if _is_mislock(ev, settle_by_event):
            mislock_n += 1

    non_none = sorted(v for v in lags if v is not None)
    median_lag = _quantile(non_none, Decimal("0.5"))
    n = len(events)
    bucket_cadence = median_int(cadences)
    unreliable = (
        median_lag is not None
        and bucket_cadence is not None
        and Decimal(median_lag) <= _FLOOR_FACTOR * Decimal(bucket_cadence)
    )

    return LagBucket(
        series=label,
        n=n,
        median_lag_s=median_lag,
        p25_lag_s=_quantile(non_none, Decimal("0.25")),
        p90_lag_s=_quantile(non_none, Decimal("0.9")),
        cadence_s=bucket_cadence,
        never_repriced_n=sum(1 for v in lags if v is None),
        no_mid_n=no_mid_n,
        mislock_n=mislock_n,
        mislock_rate=Decimal(mislock_n) / Decimal(max(1, n)),
        snapshot_unreliable=bool(unreliable),
    )


def _event_lag(
    ev: LockEvent,
    ticker_snaps: list[OrderbookSnapshotRow],
    day_window_seconds: int,
) -> tuple[int | None, int]:
    no_mid_n = 0
    for snap in ticker_snaps:
        delta = (snap.snapshot_at - ev.t0).total_seconds()
        if delta <= 0:
            continue
        if delta > day_window_seconds:
            break
        if snap.no_bid is None:
            no_mid_n += 1
            continue
        yes_bid = ticks(snap.yes_bid)
        no_bid = ticks(snap.no_bid)
        if not two_sided(yes_bid, no_bid, yes_depth=snap.yes_bid_depth, no_depth=snap.no_bid_depth):
            continue
        mid = mid2(yes_bid, no_bid)
        if ev.side_locked == "yes" and mid >= YES_BAND2:
            return int(delta), no_mid_n
        if ev.side_locked == "no" and mid <= NO_BAND2:
            return int(delta), no_mid_n
    return None, no_mid_n


def _event_cadence(ev: LockEvent, ticker_snaps: list[OrderbookSnapshotRow]) -> int | None:
    window: list[datetime] = []
    for snap in ticker_snaps:
        delta = (snap.snapshot_at - ev.t0).total_seconds()
        if delta < 0:
            continue
        if delta > 600:
            break
        window.append(snap.snapshot_at)
    if len(window) < 3:
        return None
    deltas = [int((window[i + 1] - window[i]).total_seconds()) for i in range(len(window) - 1)]
    return median_int(deltas)


def _is_mislock(ev: LockEvent, settle_by_event: Mapping[str, Decimal] | None) -> bool:
    if settle_by_event is None or ev.ticker not in settle_by_event:
        return False
    settle = settle_by_event[ev.ticker]
    if ev.side_locked == "yes":
        return settle < ev.strike
    return settle <= ev.strike


def filter_net_of_floor(
    events: list[LockEvent], settle_by_event: Mapping[str, Decimal] | None
) -> list[LockEvent]:
    kept: list[LockEvent] = []
    for ev in events:
        if ev.lock_ambiguous:
            continue
        if settle_by_event is not None and ev.ticker in settle_by_event:
            settle = settle_by_event[ev.ticker]
            cross_int = int(ev.crossing_temp_f.to_integral_value(rounding=ROUND_HALF_EVEN))
            settle_int = int(settle.to_integral_value(rounding=ROUND_HALF_EVEN))
            if abs(cross_int - settle_int) == 1:
                continue
        kept.append(ev)
    return kept


def _quantile(sorted_values: list[int], p: Decimal) -> int | None:
    k = len(sorted_values)
    if k == 0:
        return None
    idx_dec = (p * Decimal(k - 1)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
    return sorted_values[int(idx_dec)]


def median_int(values: list[int]) -> int | None:
    if not values:
        return None
    s = sorted(values)
    idx_dec = (Decimal("0.5") * Decimal(len(s) - 1)).quantize(
        Decimal("1"), rounding=ROUND_HALF_EVEN
    )
    return s[int(idx_dec)]
