from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Mapping

from bot.lag.lock_events import LockEvent
from bot.markets.parser import series_id


_YES_BAND = Decimal("0.95")
_NO_BAND = Decimal("0.05")
_TWO = Decimal(2)
_FLOOR_FACTOR = Decimal("1.5")


@dataclass(frozen=True, slots=True)
class OrderbookSnapshotRow:
    ticker: str
    snapshot_at: datetime
    yes_bid: Decimal
    yes_ask: Decimal


@dataclass(frozen=True, slots=True)
class LagBucket:
    series: str
    n: int
    median_lag_s: int | None
    p90_lag_s: int | None
    never_repriced_n: int
    mislock_n: int
    mislock_rate: Decimal
    snapshot_unreliable: bool


@dataclass(frozen=True, slots=True)
class LagReport:
    raw: list[LagBucket]
    net_of_floor: list[LagBucket]


def study_lag(
    events: list[LockEvent],
    snapshots: list[OrderbookSnapshotRow],
    *,
    settle_by_event: Mapping[str, Decimal] | None = None,
    day_window_seconds: int = 24 * 3600,
) -> LagReport:
    """Missing settles leave mislock unknown for that event; the bucket mislock_rate
    denominator is bucket n, so unknowns dilute the rate downward."""
    by_ticker: dict[str, list[OrderbookSnapshotRow]] = {}
    for s in snapshots:
        by_ticker.setdefault(s.ticker, []).append(s)
    for rows in by_ticker.values():
        rows.sort(key=lambda r: r.snapshot_at)

    raw = _aggregate(events, by_ticker, settle_by_event, day_window_seconds)
    filtered = _filter_net_of_floor(events, settle_by_event)
    net = _aggregate(filtered, by_ticker, settle_by_event, day_window_seconds)
    return LagReport(raw=raw, net_of_floor=net)


def _aggregate(
    events: list[LockEvent],
    by_ticker: Mapping[str, list[OrderbookSnapshotRow]],
    settle_by_event: Mapping[str, Decimal] | None,
    day_window_seconds: int,
) -> list[LagBucket]:
    by_series: dict[str, list[LockEvent]] = {}
    for ev in events:
        by_series.setdefault(series_id(ev.ticker), []).append(ev)

    buckets: list[LagBucket] = []
    for s in sorted(by_series):
        bucket_events = by_series[s]
        lags: list[int | None] = []
        cadences: list[int] = []
        mislock_n = 0
        for ev in bucket_events:
            ticker_snaps = by_ticker.get(ev.ticker, [])
            lags.append(_event_lag(ev, ticker_snaps, day_window_seconds))
            cadence = _event_cadence(ev, ticker_snaps)
            if cadence is not None:
                cadences.append(cadence)
            if _is_mislock(ev, settle_by_event):
                mislock_n += 1

        non_none = sorted(v for v in lags if v is not None)
        median_lag = _quantile(non_none, Decimal("0.5"))
        p90_lag = _quantile(non_none, Decimal("0.9"))
        never_repriced = sum(1 for v in lags if v is None)
        n = len(bucket_events)
        rate = Decimal(mislock_n) / Decimal(max(1, n))

        bucket_cadence = _median_int(cadences)
        unreliable = (
            median_lag is not None
            and bucket_cadence is not None
            and Decimal(median_lag) <= _FLOOR_FACTOR * Decimal(bucket_cadence)
        )

        buckets.append(
            LagBucket(
                series=s,
                n=n,
                median_lag_s=median_lag,
                p90_lag_s=p90_lag,
                never_repriced_n=never_repriced,
                mislock_n=mislock_n,
                mislock_rate=rate,
                snapshot_unreliable=bool(unreliable),
            )
        )
    return buckets


def _event_lag(
    ev: LockEvent,
    ticker_snaps: list[OrderbookSnapshotRow],
    day_window_seconds: int,
) -> int | None:
    for snap in ticker_snaps:
        delta = (snap.snapshot_at - ev.t0).total_seconds()
        if delta <= 0:
            continue
        if delta > day_window_seconds:
            break
        mid = (snap.yes_bid + snap.yes_ask) / _TWO
        if ev.side_locked == "yes" and mid >= _YES_BAND:
            return int(delta)
        if ev.side_locked == "no" and mid <= _NO_BAND:
            return int(delta)
    return None


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
    return _median_int(deltas)


def _is_mislock(ev: LockEvent, settle_by_event: Mapping[str, Decimal] | None) -> bool:
    if settle_by_event is None or ev.ticker not in settle_by_event:
        return False
    settle = settle_by_event[ev.ticker]
    if ev.side_locked == "yes":
        return settle < ev.strike
    return settle <= ev.strike


def _filter_net_of_floor(
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


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    s = sorted(values)
    idx_dec = (Decimal("0.5") * Decimal(len(s) - 1)).quantize(
        Decimal("1"), rounding=ROUND_HALF_EVEN
    )
    return s[int(idx_dec)]
