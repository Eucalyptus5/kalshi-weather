from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.capture_sim import CaptureResult, LatencyStack, simulate_capture  # noqa: E402
from bot.lag.event_study import (  # noqa: E402
    LagBucket,
    OrderbookSnapshotRow,
    filter_net_of_floor,
    study_lag,
)
from bot.lag.lock_events import LockEvent, detect_lock_events  # noqa: E402
from bot.lag.report import aggregate_captures, format_report  # noqa: E402
from bot.lag.snapshot_loader import load_snapshots  # noqa: E402
from bot.main import STATIONS  # noqa: E402
from bot.markets.parser import ParsedTicker, parse_ticker, resolve_event_kinds, series_id  # noqa: E402
from bot.observations.basis_check import fetch_iem_1min_asos_archive  # noqa: E402
from bot.observations.metar import StationObservation  # noqa: E402
from bot.validation.reconcile import ACISClient  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "state.db"
DEFAULT_START = date(2026, 6, 13)
DEFAULT_SERIES: tuple[str, ...] = ("KXHIGHDEN", "KXHIGHCHI")
EXCLUDED_SERIES: frozenset[str] = frozenset({"KXHIGHMIA"})
DEFAULT_LATENCY_TOTAL_S = 125
DEFAULT_NOTIONAL_CAP = Decimal("100")
STALE_QUOTE_BUFFER = timedelta(hours=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="retrospective lag readout for prod-era snapshots")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=DEFAULT_START,
        help="inclusive UTC lower bound (YYYY-MM-DD); clamped to prod-era cutover",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="exclusive UTC upper bound (YYYY-MM-DD); defaults to today-1d UTC",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        type=lambda s: date.fromisoformat(s),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--series",
        type=lambda s: tuple(p for p in s.split(",") if p),
        default=DEFAULT_SERIES,
    )
    parser.add_argument("--latency-total-s", type=int, default=DEFAULT_LATENCY_TOTAL_S)
    parser.add_argument("--notional-cap", type=Decimal, default=DEFAULT_NOTIONAL_CAP)
    return parser


def filter_series(requested: tuple[str, ...]) -> tuple[tuple[str, ...], list[str]]:
    kept: list[str] = []
    warnings: list[str] = []
    for s in requested:
        if s in EXCLUDED_SERIES:
            warnings.append(f"basis_excluded: {s} (structural +1F offset, see L1.5 verdict)")
            continue
        kept.append(s)
    return tuple(kept), warnings


def fetch_tickers_for_series(
    db_path: Path,
    series: str,
    start: date,
    end: date,
) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path.absolute()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT ticker FROM markets "
            "WHERE series = ? AND event_date >= ? AND event_date < ? "
            "ORDER BY ticker",
            (series, start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


async def _gather_events_for_series(
    series: str,
    tickers: list[str],
    http: httpx.AsyncClient,
) -> list[LockEvent]:
    cfg = STATIONS[series]
    station = cfg.station
    tz = cfg.timezone

    parsed_by_ticker = {t: parse_ticker(t) for t in tickers}
    grouped: dict[tuple[str, date], list[ParsedTicker]] = {}
    for p in parsed_by_ticker.values():
        grouped.setdefault((p.series, p.event_date), []).append(p)
    parsed_by_ticker = {p.raw: p for group in grouped.values() for p in resolve_event_kinds(group)}
    days = sorted({p.event_date for p in parsed_by_ticker.values()})

    obs_by_day: dict[date, list[StationObservation]] = {}
    for day in days:
        obs = await fetch_iem_1min_asos_archive(station, day, day, http)
        obs_by_day[day] = obs

    events: list[LockEvent] = []
    for ticker, parsed in parsed_by_ticker.items():
        day_obs = obs_by_day.get(parsed.event_date, [])
        if not day_obs:
            continue
        events.extend(detect_lock_events(parsed, day_obs, tz_name=tz))
    return events


async def _fetch_settles(
    events: list[LockEvent],
    series_to_station: dict[str, str],
    acis: ACISClient,
) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    seen: dict[str, LockEvent] = {ev.ticker: ev for ev in events}
    for ticker in seen:
        sid = series_id(ticker)
        station = series_to_station[sid]
        settle = await acis.fetch_daily_high(station[1:], parse_ticker(ticker).event_date)
        if settle is not None:
            out[ticker] = settle
    return out


async def run(args: argparse.Namespace) -> int:
    series_kept, warnings = filter_series(tuple(args.series))
    for w in warnings:
        print(w)

    if not series_kept:
        print("no series to process")
        return 0

    today = datetime.now(timezone.utc).date()
    end_date: date = args.end_date if args.end_date is not None else today - timedelta(days=1)
    start_date: date = args.start

    if start_date >= end_date:
        print(f"empty window: start={start_date} end={end_date}")
        return 0

    series_to_station: dict[str, str] = {s: STATIONS[s].station for s in series_kept}

    # latency stack collapses into obs_publication_s so total_s matches the --latency-total-s flag.
    latency_stack = LatencyStack(
        obs_publication_s=args.latency_total_s,
        poll_interval_s=0,
        decision_s=0,
    )

    all_events: list[LockEvent] = []
    all_snapshots: list[OrderbookSnapshotRow] = []

    async with httpx.AsyncClient(timeout=60.0) as http:
        for series in series_kept:
            tickers = fetch_tickers_for_series(args.db, series, start_date, end_date)
            if not tickers:
                continue
            events = await _gather_events_for_series(series, tickers, http)
            all_events.extend(events)

        acis = ACISClient(http_client=http)
        settle_by_event = await _fetch_settles(all_events, series_to_station, acis)

        for ticker in {ev.ticker for ev in all_events}:
            ticker_events = [ev for ev in all_events if ev.ticker == ticker]
            t0_floor = min(ev.t0 for ev in ticker_events)
            t0_ceiling = max(ev.t0 + timedelta(seconds=86400) for ev in ticker_events)
            snaps = load_snapshots(
                args.db,
                ticker,
                t0_floor - STALE_QUOTE_BUFFER,
                t0_ceiling,
            )
            all_snapshots.extend(snaps)

    report = study_lag(all_events, all_snapshots, settle_by_event=settle_by_event)
    raw_buckets: dict[str, LagBucket] = {b.series: b for b in report.raw}
    net_buckets: dict[str, LagBucket] = {b.series: b for b in report.net_of_floor}

    snaps_by_ticker: dict[str, list[OrderbookSnapshotRow]] = {}
    for s in all_snapshots:
        snaps_by_ticker.setdefault(s.ticker, []).append(s)

    net_events = filter_net_of_floor(all_events, settle_by_event)
    net_event_ids = {(ev.ticker, ev.t0) for ev in net_events}

    captures_raw: list[CaptureResult] = []
    captures_net: list[CaptureResult] = []
    for ev in all_events:
        settle = settle_by_event.get(ev.ticker)
        if settle is None:
            continue
        sid = series_id(ev.ticker)
        ticker_snaps = snaps_by_ticker.get(ev.ticker, [])
        captures_raw.append(
            simulate_capture(
                ev,
                ticker_snaps,
                settle_price=settle,
                notional_cap=args.notional_cap,
                latency_stack=latency_stack,
                snapshot_unreliable=raw_buckets[sid].snapshot_unreliable,
            )
        )
        if (ev.ticker, ev.t0) in net_event_ids:
            captures_net.append(
                simulate_capture(
                    ev,
                    ticker_snaps,
                    settle_price=settle,
                    notional_cap=args.notional_cap,
                    latency_stack=latency_stack,
                    snapshot_unreliable=net_buckets[sid].snapshot_unreliable,
                )
            )

    print(
        format_report(
            report,
            aggregate_captures(captures_raw),
            aggregate_captures(captures_net),
        )
    )
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return await run(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
