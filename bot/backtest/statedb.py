from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from bot.backtest.cli import settle_run
from bot.backtest.depth_table import parse_dt
from bot.backtest.engine import (
    BacktestConfig,
    BacktestOrder,
    ReplayLog,
    ReplaySnapshot,
    fetch_forecast,
    run_replay,
)
from bot.backtest.forecast_replay import ForecastReplay, StationSpec
from bot.backtest.normalize import CanonicalSnapshot
from bot.backtest.pnl import BacktestFill
from bot.backtest.report import (
    REFERENCE_BANKROLL,
    SENSITIVITY_BANKROLL,
    ReportVerdict,
    ScoredRun,
    evaluate_runs,
    write_report,
)
from bot.backtest.scoring import score_run
from bot.forecast.cdf import EnsembleCDF
from bot.markets.parser import event_id


# stored timestamps carry microseconds; isoformat drops zero micros and skews boundary compares
def _bound(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S.%f")


def fetch_decision_snapshots(
    conn: sqlite3.Connection,
    lead: timedelta,
    start: datetime,
    staleness: timedelta,
) -> list[ReplaySnapshot]:
    markets = conn.execute(
        "SELECT ticker, close_time, status FROM markets"
        " WHERE close_time IS NOT NULL ORDER BY ticker"
    ).fetchall()
    out: list[ReplaySnapshot] = []
    for ticker, close_raw, status in markets:
        close_time = parse_dt(close_raw)
        decision_at = close_time - lead
        row = conn.execute(
            """
            SELECT snapshot_at, yes_ask, yes_bid, no_ask, no_bid, yes_bid_depth, no_bid_depth
              FROM orderbook_snapshots
             WHERE ticker = ? AND snapshot_at <= ? AND snapshot_at >= ?
             ORDER BY snapshot_at DESC
             LIMIT 1
            """,
            (ticker, _bound(decision_at), _bound(max(decision_at - staleness, start))),
        ).fetchone()
        if row is None:
            continue
        snapshot_raw, yes_ask, yes_bid, no_ask, no_bid, yes_bid_depth, no_bid_depth = row
        snap = CanonicalSnapshot(
            ticker=ticker,
            event_ticker=event_id(ticker),
            series_ticker=ticker.split("-", 1)[0],
            status=status,
            result="",
            yes_ask=Decimal(str(yes_ask)),
            yes_bid=Decimal(str(yes_bid)),
            no_ask=Decimal(str(no_ask)),
            no_bid=Decimal(str(no_bid)),
            last_price=Decimal(0),
            volume=Decimal(0),
            volume_24h=Decimal(0),
            open_interest=Decimal(0),
            close_time=close_time,
            yes_bid_size=Decimal(int(yes_bid_depth or 0)),
            no_bid_size=Decimal(int(no_bid_depth or 0)),
        )
        out.append(ReplaySnapshot(snapshot_at=parse_dt(snapshot_raw), snap=snap))
    return out


class StateDbForecastReplay:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def replay(self, station: StationSpec, valid_date: date, as_of: datetime) -> EnsembleCDF:
        # forecasts.run_time is naive UTC text; an aware isoformat's +00:00 suffix corrupts the cut
        not_after = as_of.astimezone(timezone.utc).replace(tzinfo=None)
        row = fetch_forecast(self._conn, station.name, valid_date, not_after)
        if row is None:
            raise LookupError(
                f"no forecast for station={station.name}"
                f" valid_date={valid_date.isoformat()} as_of={as_of.isoformat()}"
            )
        return EnsembleCDF.from_members(row.members, smoothing=1.0)


def settlement_rows(settled: Iterable[CanonicalSnapshot]) -> list[ReplaySnapshot]:
    rows: list[ReplaySnapshot] = []
    for snap in settled:
        if snap.close_time is None:
            raise ValueError(f"settled snapshot missing close_time: {snap.ticker}")
        # snapshot_at == close_time can never qualify as a decision row; it only carries the result
        rows.append(ReplaySnapshot(snapshot_at=snap.close_time, snap=snap))
    return rows


async def run_survival(
    snapshots: list[ReplaySnapshot],
    forecast: ForecastReplay,
    lead: timedelta,
    out_dir: Path,
) -> tuple[ReportVerdict, Path, list[ScoredRun], dict[str, ReplayLog]]:
    runs: list[ScoredRun] = []
    logs: dict[str, ReplayLog] = {}
    reference_orders: list[BacktestOrder] = []
    reference_fills: list[BacktestFill] = []
    for strategy in ("tails", "edge"):
        for bankroll in (REFERENCE_BANKROLL, SENSITIVITY_BANKROLL):
            log = ReplayLog()
            config = BacktestConfig(lead=lead, bankroll=bankroll, depth_table={})
            orders = await run_replay(snapshots, forecast, strategy, config, log=log)
            settled, fills, settlements, _ = settle_run(orders, snapshots, lead)
            runs.append(
                ScoredRun(
                    strategy=strategy,
                    window="out_of_sample",
                    bankroll=bankroll,
                    depth_multiplier=Decimal("1.0"),
                    sigma_multiplier=Decimal("1.0"),
                    report=score_run(settled, fills, settlements, log.failures),
                )
            )
            logs[f"{strategy}:{bankroll}"] = log
            if bankroll == REFERENCE_BANKROLL:
                reference_orders.extend(settled)
                reference_fills.extend(fills)
    verdict = evaluate_runs(runs)
    summary = write_report(runs, reference_orders, reference_fills, out_dir)
    return verdict, summary, runs, logs
