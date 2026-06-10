from __future__ import annotations

import argparse
import asyncio
import importlib
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from bot.backtest.depth_table import ALL_SERIES, LEAD_BUCKETS
from bot.backtest.engine import (
    BacktestConfig,
    BacktestOrder,
    ReplayLog,
    ReplaySnapshot,
    iter_canonical_snapshots,
    run_replay,
)
from bot.backtest.forecast_replay import ForecastReplay, GefsGribForecastReplay
from bot.backtest.pnl import BacktestFill, settle_order
from bot.backtest.report import ScoredRun, evaluate_runs, write_report
from bot.backtest.scoring import OrderSettlement, score_run

GEFS_CACHE_DIR = Path("data") / "gefs_cache"
PASSING_VERDICTS = ("survive", "friction_floor_driven")
_WINDOW = "out_of_sample"
_LEAD_FORM = re.compile(r"(\d+)([mhd])")
_LEAD_UNITS = {"m": "minutes", "h": "hours", "d": "days"}
_SNAPSHOT_AT_FIELD = pa.field("snapshot_at", pa.timestamp("us", tz="UTC"))


def parse_lead(raw: str) -> timedelta:
    match = _LEAD_FORM.fullmatch(raw)
    if match is None:
        raise argparse.ArgumentTypeError(f"invalid lead {raw!r}; expected forms like 24h, 90m, 2d")
    return timedelta(**{_LEAD_UNITS[match.group(2)]: int(match.group(1))})


def _bankroll(raw: str) -> Decimal:
    try:
        return Decimal(raw)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid bankroll {raw!r}") from exc


def _forecast_source(spec: str) -> ForecastReplay:
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise argparse.ArgumentTypeError(f"invalid forecast source {spec!r}; expected module:attr")
    return getattr(importlib.import_module(module_name), attr)()


def load_snapshots(parquet_path: Path, lead: timedelta) -> list[ReplaySnapshot]:
    if "snapshot_at" in pq.read_schema(parquet_path).names:
        return list(iter_canonical_snapshots(parquet_path))
    table = pq.read_table(parquet_path)
    derived = pa.array(
        [
            None if close is None else close - lead
            for close in table.column("close_time").to_pylist()
        ],
        type=_SNAPSHOT_AT_FIELD.type,
    )
    augmented = table.add_column(0, _SNAPSHOT_AT_FIELD, derived)
    # roundtrip through a temp parquet so iter_canonical_snapshots stays the only row decoder
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / parquet_path.name
        pq.write_table(augmented, path)
        return list(iter_canonical_snapshots(path))


def settle_run(
    orders: list[BacktestOrder],
    snapshots: list[ReplaySnapshot],
    lead: timedelta,
) -> tuple[list[BacktestOrder], list[BacktestFill], list[OrderSettlement], int]:
    results: dict[str, str] = {}
    decisions: dict[tuple[str, datetime], ReplaySnapshot] = {}
    for row in snapshots:
        if row.snap.result in ("yes", "no"):
            results[row.snap.ticker] = row.snap.result
        close = row.snap.close_time
        if close is None or row.snapshot_at > close - lead:
            continue
        key = (row.snap.ticker, close)
        best = decisions.get(key)
        if best is None or row.snapshot_at > best.snapshot_at:
            decisions[key] = row

    settled: list[BacktestOrder] = []
    fills: list[BacktestFill] = []
    settlements: list[OrderSettlement] = []
    dropped = 0
    for order in orders:
        result = results.get(order.market_ticker)
        if result is None:
            dropped += 1
            continue
        snap = decisions[(order.market_ticker, order.as_of + lead)].snap
        settled.append(order)
        fills.append(settle_order(order, snap, result))
        settlements.append(
            OrderSettlement(result=result, market_mid=(snap.yes_bid + snap.yes_ask) / 2)
        )
    return settled, fills, settlements, dropped


async def _run(args: argparse.Namespace) -> int:
    snapshots = load_snapshots(args.weather_parquet, args.lead)
    config = BacktestConfig(
        lead=args.lead,
        bankroll=args.bankroll,
        # zero fallback depth: snapshots without book sizes skip as depth_zero
        # instead of fabricating liquidity
        depth_table={(ALL_SERIES, bucket): Decimal("0") for bucket in LEAD_BUCKETS},
    )
    log = ReplayLog()
    if args.forecast_source is not None:
        orders = await run_replay(snapshots, args.forecast_source, args.strategy, config, log=log)
    else:
        async with httpx.AsyncClient() as client:
            source = GefsGribForecastReplay(client, GEFS_CACHE_DIR)
            orders = await run_replay(snapshots, source, args.strategy, config, log=log)

    settled, fills, settlements, dropped = settle_run(orders, snapshots, args.lead)
    runs = [
        ScoredRun(
            strategy=args.strategy,
            window=_WINDOW,
            bankroll=args.bankroll,
            depth_multiplier=Decimal("1.0"),
            sigma_multiplier=Decimal("1.0"),
            report=score_run(settled, fills, settlements, log.failures),
        )
    ]
    verdict = evaluate_runs(runs)
    summary = write_report(runs, settled, fills, args.out)
    print(f"orders={len(settled)} unsettled_dropped={dropped} skips={len(log.failures)}")
    print(f"report={summary}")
    print(f"verdict={verdict.final}")
    if args.dry_run:
        return 0
    return 0 if verdict.final in PASSING_VERDICTS else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bot.backtest.cli")
    parser.add_argument("--weather-parquet", type=Path, required=True)
    parser.add_argument("--strategy", choices=("tails", "edge"), required=True)
    parser.add_argument("--bankroll", type=_bankroll, default=Decimal("20000"))
    parser.add_argument("--lead", type=parse_lead, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--forecast-source", type=_forecast_source, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.forecast_source is None:
        parser.error("--dry-run requires --forecast-source")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
