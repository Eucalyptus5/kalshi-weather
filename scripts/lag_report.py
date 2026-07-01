from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.capture_sim import CaptureResult, LatencyStack, simulate_capture  # noqa: E402
from bot.lag.event_study import (  # noqa: E402
    EventProbe,
    LagReport,
    OrderbookSnapshotRow,
    filter_net_of_floor,
    study_lag,
    study_lag_from_probes,
)
from bot.lag.lock_events import LockEvent, detect_lock_events  # noqa: E402
from bot.lag.report import (  # noqa: E402
    LatencyPoint,
    SourceCoverage,
    aggregate_captures,
    format_report,
    pool_captures,
)
from bot.lag.snapshot_loader import load_snapshots  # noqa: E402
from bot.lag.ws_book import WsGapError, open_book_db, probe_event  # noqa: E402
from bot.main import STATIONS  # noqa: E402
from bot.markets.parser import ParsedTicker, parse_ticker, resolve_event_kinds, series_id  # noqa: E402
from bot.observations.basis_check import fetch_iem_1min_asos_archive  # noqa: E402
from bot.observations.metar import StationObservation  # noqa: E402
from bot.validation.reconcile import ACISClient  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "state.db"
R0_WINDOW_START = date(2026, 7, 18)
R0_WINDOW_END_EXCLUSIVE = date(2026, 8, 2)
R0_FRACTION_INVALID_MAX = Decimal("0.5")
R0_PASSING_SERIES: tuple[str, ...] = (
    "KXHIGHDEN",
    "KXHIGHAUS",
    "KXHIGHCHI",
    "KXHIGHNY",
    "KXHIGHPHIL",
    "KXHIGHTATL",
    "KXHIGHTBOS",
    "KXHIGHTDAL",
    "KXHIGHTDC",
    "KXHIGHTHOU",
    "KXHIGHTLV",
    "KXHIGHTMIN",
    "KXHIGHTNOLA",
    "KXHIGHTOKC",
    "KXHIGHTPHX",
    "KXHIGHTSATX",
    "KXHIGHTSEA",
    "KXHIGHTSFO",
    "KXHIGHLAX",
    "KXHIGHMIA",
)
BOOK_SOURCES: tuple[str, ...] = ("rest", "ws")
DEFAULT_LATENCY_CURVE_S: tuple[int, ...] = (30, 60, 90, 120)
GATE_STACK_S = 90
DEFAULT_NOTIONAL_CAP = Decimal("100")
STALE_QUOTE_BUFFER = timedelta(hours=1)
REPRICE_WINDOW_S = 24 * 3600

RowsForEvent = Callable[[LockEvent, int], list[OrderbookSnapshotRow]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="retrospective lag readout for prod-era snapshots")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=R0_WINDOW_START,
        help="inclusive UTC lower bound (YYYY-MM-DD); clamped to prod-era cutover",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        type=lambda s: date.fromisoformat(s),
        default=R0_WINDOW_END_EXCLUSIVE,
        help="exclusive UTC upper bound (YYYY-MM-DD)",
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
        default=R0_PASSING_SERIES,
    )
    parser.add_argument("--book-source", choices=BOOK_SOURCES, default="rest")
    parser.add_argument(
        "--latency-total-s",
        type=lambda s: tuple(int(p) for p in s.split(",") if p),
        default=DEFAULT_LATENCY_CURVE_S,
        help="comma-separated latency stacks, seconds",
    )
    parser.add_argument("--notional-cap", type=Decimal, default=DEFAULT_NOTIONAL_CAP)
    return parser


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


def _latency_point(
    offset_s: int,
    events: list[LockEvent],
    settle_by_event: dict[str, Decimal],
    report: LagReport,
    notional_cap: Decimal,
    rows_for: RowsForEvent,
) -> LatencyPoint:
    raw_unreliable = {b.series: b.snapshot_unreliable for b in report.raw}
    net_unreliable = {b.series: b.snapshot_unreliable for b in report.net_of_floor}
    net_ids = {(ev.ticker, ev.t0) for ev in filter_net_of_floor(events, settle_by_event)}
    stack = LatencyStack(obs_publication_s=offset_s, poll_interval_s=0, decision_s=0)

    captures_raw: list[CaptureResult] = []
    captures_net: list[CaptureResult] = []
    for ev in events:
        settle = settle_by_event.get(ev.ticker)
        if settle is None:
            continue
        sid = series_id(ev.ticker)
        rows = rows_for(ev, offset_s)
        captures_raw.append(
            simulate_capture(
                ev,
                rows,
                settle_price=settle,
                notional_cap=notional_cap,
                latency_stack=stack,
                snapshot_unreliable=raw_unreliable[sid],
            )
        )
        if (ev.ticker, ev.t0) in net_ids:
            captures_net.append(
                simulate_capture(
                    ev,
                    rows,
                    settle_price=settle,
                    notional_cap=notional_cap,
                    latency_stack=stack,
                    snapshot_unreliable=net_unreliable[sid],
                )
            )

    return LatencyPoint(
        total_s=offset_s,
        raw=aggregate_captures(captures_raw),
        net_of_floor=aggregate_captures(captures_net),
        raw_pooled=pool_captures(captures_raw),
        net_of_floor_pooled=pool_captures(captures_net),
    )


def _readout_rest(
    db_path: Path,
    events: list[LockEvent],
    settle_by_event: dict[str, Decimal],
    offsets: tuple[int, ...],
    notional_cap: Decimal,
) -> tuple[LagReport, list[LatencyPoint], SourceCoverage]:
    snapshots: list[OrderbookSnapshotRow] = []
    for ticker in sorted({ev.ticker for ev in events}):
        ticker_events = [ev for ev in events if ev.ticker == ticker]
        t0_floor = min(ev.t0 for ev in ticker_events)
        t0_ceiling = max(ev.t0 + timedelta(seconds=REPRICE_WINDOW_S) for ev in ticker_events)
        snapshots.extend(load_snapshots(db_path, ticker, t0_floor - STALE_QUOTE_BUFFER, t0_ceiling))

    by_ticker: dict[str, list[OrderbookSnapshotRow]] = {}
    for s in snapshots:
        by_ticker.setdefault(s.ticker, []).append(s)

    report = study_lag(
        events,
        snapshots,
        settle_by_event=settle_by_event,
        day_window_seconds=REPRICE_WINDOW_S,
    )
    curve = [
        _latency_point(
            offset,
            events,
            settle_by_event,
            report,
            notional_cap,
            lambda ev, _offset: by_ticker.get(ev.ticker, []),
        )
        for offset in offsets
    ]
    coverage = SourceCoverage(
        book_source="rest",
        events_found=len(events),
        events_used=len(events),
        gap_excluded_n=0,
        no_coverage_n=0,
    )
    return report, curve, coverage


def _readout_ws(
    db_path: Path,
    events: list[LockEvent],
    settle_by_event: dict[str, Decimal],
    offsets: tuple[int, ...],
    notional_cap: Decimal,
) -> tuple[LagReport, list[LatencyPoint], SourceCoverage]:
    kept: list[LockEvent] = []
    probes: dict[tuple[str, datetime], EventProbe] = {}
    rows: dict[tuple[str, datetime], dict[int, list[OrderbookSnapshotRow]]] = {}
    gap_excluded = 0
    no_coverage = 0

    # sorted by ticker so the (ticker, received_at) index is walked in one direction
    with open_book_db(db_path) as conn:
        for ev in sorted(events, key=lambda e: (e.ticker, e.t0)):
            try:
                probe = probe_event(
                    conn,
                    ev.ticker,
                    ev.t0,
                    ev.side_locked,
                    decision_offsets_s=offsets,
                    window_s=REPRICE_WINDOW_S,
                )
            except WsGapError:
                gap_excluded += 1
                continue
            if probe is None:
                no_coverage += 1
                continue
            kept.append(ev)
            probes[(ev.ticker, ev.t0)] = EventProbe(
                lag_s=probe.lag_s,
                cadence_s=probe.cadence_s,
            )
            rows[(ev.ticker, ev.t0)] = {
                offset: [probe.at_t0, probe.at_decision[offset]] for offset in offsets
            }

    report = study_lag_from_probes(kept, probes, settle_by_event=settle_by_event)
    curve = [
        _latency_point(
            offset,
            kept,
            settle_by_event,
            report,
            notional_cap,
            lambda ev, off: rows[(ev.ticker, ev.t0)][off],
        )
        for offset in offsets
    ]
    coverage = SourceCoverage(
        book_source="ws",
        events_found=len(events),
        events_used=len(kept),
        gap_excluded_n=gap_excluded,
        no_coverage_n=no_coverage,
    )
    return report, curve, coverage


async def run(args: argparse.Namespace) -> int:
    series_kept = tuple(args.series)
    if not series_kept:
        print("no series to process")
        return 0

    if args.start >= args.end_date:
        print(f"empty window: start={args.start} end={args.end_date}")
        return 0

    offsets = tuple(sorted(set(args.latency_total_s)))
    series_to_station: dict[str, str] = {s: STATIONS[s].station for s in series_kept}

    all_events: list[LockEvent] = []
    async with httpx.AsyncClient(timeout=60.0) as http:
        for series in series_kept:
            tickers = fetch_tickers_for_series(args.db, series, args.start, args.end_date)
            if not tickers:
                continue
            all_events.extend(await _gather_events_for_series(series, tickers, http))

        acis = ACISClient(http_client=http)
        settle_by_event = await _fetch_settles(all_events, series_to_station, acis)

    readout = _readout_ws if args.book_source == "ws" else _readout_rest
    report, curve, coverage = readout(
        args.db,
        all_events,
        settle_by_event,
        offsets,
        args.notional_cap,
    )

    print(format_report(report, curve, coverage, gate_stack_s=GATE_STACK_S))
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return await run(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
