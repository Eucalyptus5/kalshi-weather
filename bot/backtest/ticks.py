from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from bot.backtest.engine import (
    BacktestConfig,
    BacktestOrder,
    ReplayFailure,
    ReplayLog,
    ReplaySnapshot,
    run_replay,
)
from bot.backtest.forecast_replay import ForecastReplay
from bot.backtest.normalize import CanonicalSnapshot, _cents_to_dollars
from bot.backtest.pnl import BacktestFill
from bot.backtest.report import (
    REFERENCE_BANKROLL,
    SENSITIVITY_BANKROLL,
    ReportVerdict,
    ScoredRun,
    evaluate_runs,
    write_report,
)
from bot.backtest.scoring import OrderSettlement, score_run
from bot.execution.fees import taker_fee
from bot.execution.paper import TradeSide
from bot.validation.scoring import brier_score, realized_pnl_for_trade


_TICK_SCHEMA = pa.schema(
    [
        pa.field("trade_id", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("count", pa.int64()),
        pa.field("yes_price", pa.decimal128(10, 2)),
        pa.field("no_price", pa.decimal128(10, 2)),
        pa.field("taker_side", pa.string()),
        pa.field("created_time", pa.timestamp("us", tz="UTC")),
    ]
)


def ingest_ticks(shard_paths: list[Path], out_path: Path, series: list[str]) -> int:
    if not series:
        raise ValueError("series must contain at least one ticker prefix")

    dataset = ds.dataset([str(p) for p in shard_paths], format="parquet")

    ticker_field = pc.field("ticker")
    filter_expr = pc.starts_with(ticker_field, f"{series[0]}-")
    for prefix in series[1:]:
        filter_expr = filter_expr | pc.starts_with(ticker_field, f"{prefix}-")

    raw = dataset.to_table(filter=filter_expr)

    trade_ids = raw.column("trade_id").to_pylist()
    seen: set[str] = set()
    keep: list[int] = []
    for i, tid in enumerate(trade_ids):
        if tid not in seen:
            seen.add(tid)
            keep.append(i)

    deduped = raw.take(keep)

    sorted_indices = pc.sort_indices(
        deduped,
        sort_keys=[("ticker", "ascending"), ("created_time", "ascending")],
    )
    deduped = deduped.take(sorted_indices)

    rows = deduped.to_pylist()
    table = pa.table(
        {
            "trade_id": [r["trade_id"] for r in rows],
            "ticker": [r["ticker"] for r in rows],
            "count": [r["count"] for r in rows],
            "yes_price": [_cents_to_dollars(r["yes_price"]) for r in rows],
            "no_price": [_cents_to_dollars(r["no_price"]) for r in rows],
            "taker_side": [r["taker_side"] for r in rows],
            "created_time": [r["created_time"] for r in rows],
        },
        schema=_TICK_SCHEMA,
    )

    pq.write_table(table, out_path)
    return table.num_rows


def tick_decision_snapshots(
    tick_path: Path,
    market_state: Sequence[CanonicalSnapshot],
    lead: timedelta,
    staleness: timedelta,
    depth_window: timedelta,
) -> tuple[list[ReplaySnapshot], int]:
    table = pq.read_table(tick_path)
    rows = table.to_pylist()

    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)

    out: list[ReplaySnapshot] = []
    omitted = 0

    for snap in market_state:
        if snap.close_time is None:
            continue
        as_of = snap.close_time - lead
        staleness_floor = as_of - staleness
        depth_floor = as_of - depth_window

        ticker_rows = by_ticker.get(snap.ticker, [])
        candidate: dict | None = None
        for row in ticker_rows:
            t: datetime = row["created_time"]
            if t <= as_of and t >= staleness_floor:
                if candidate is None or t > candidate["created_time"]:
                    candidate = row

        if candidate is None:
            omitted += 1
            out.append(ReplaySnapshot(snapshot_at=snap.close_time, snap=snap))
            continue

        last_price = candidate["yes_price"]
        no_price = Decimal("1") - last_price

        trailing_count = sum(1 for row in ticker_rows if depth_floor < row["created_time"] <= as_of)

        decision_snap = snap.model_copy(
            update={
                "result": "",
                "yes_ask": last_price,
                "yes_bid": last_price,
                "no_ask": no_price,
                "no_bid": no_price,
                "last_price": last_price,
                "yes_bid_size": Decimal(trailing_count),
                "no_bid_size": Decimal(trailing_count),
            }
        )
        out.append(ReplaySnapshot(snapshot_at=as_of, snap=decision_snap))
        out.append(ReplaySnapshot(snapshot_at=snap.close_time, snap=snap))

    return out, omitted


def fill_against_prints(
    order: BacktestOrder,
    prints: Sequence[dict],
    close: datetime,
    rule: Literal["conservative", "inclusive"],
) -> int:
    p = (
        order.price_per_contract
        if order.action is TradeSide.BUY_YES
        else Decimal("1") - order.price_per_contract
    )
    as_of = order.as_of
    total = 0
    for row in prints:
        t: datetime = row["created_time"]
        if t <= as_of or t > close:
            continue
        yes_price: Decimal = row["yes_price"]
        if order.action is TradeSide.SELL_YES:
            qualifies = yes_price > p if rule == "conservative" else yes_price >= p
        else:
            qualifies = yes_price < p if rule == "conservative" else yes_price <= p
        if qualifies:
            total += int(row["count"])
    return min(total, order.contracts)


def make_no_prints_failure(order: BacktestOrder) -> ReplayFailure:
    return ReplayFailure(
        market_ticker=order.market_ticker,
        as_of=order.as_of,
        strategy=order.strategy,
        layer="fill",
        name="no_qualifying_prints",
        reason="no_qualifying_prints",
    )


def settle_filled_order(
    order: BacktestOrder,
    filled_contracts: int,
    executed_yes_price: Decimal,
    result: str,
) -> BacktestFill:
    if filled_contracts == 0:
        return BacktestFill(
            gross_pnl=Decimal("0"),
            fee_dollars=Decimal("0"),
            net_pnl=Decimal("0"),
        )
    won = (result == "yes") == (order.action is TradeSide.BUY_YES)
    fee_dollars = taker_fee(filled_contracts, executed_yes_price)
    gross_pnl = realized_pnl_for_trade(
        order.action, executed_yes_price, filled_contracts, Decimal("0"), won
    )
    net_pnl = realized_pnl_for_trade(
        order.action, executed_yes_price, filled_contracts, fee_dollars, won
    )
    return BacktestFill(gross_pnl=gross_pnl, fee_dollars=fee_dollars, net_pnl=net_pnl)


def tick_settle_orders(
    orders: list[BacktestOrder],
    snapshots: list[ReplaySnapshot],
    lead: timedelta,
) -> list[OrderSettlement]:
    results: dict[str, str] = {}
    decisions: dict[tuple[str, datetime], ReplaySnapshot] = {}
    for row in snapshots:
        if row.snap.result in ("yes", "no"):
            results[row.snap.ticker] = row.snap.result
        close = row.snap.close_time
        if close is None or row.snapshot_at >= close:
            continue
        key = (row.snap.ticker, close)
        best = decisions.get(key)
        if best is None or row.snapshot_at > best.snapshot_at:
            decisions[key] = row

    out: list[OrderSettlement] = []
    for order in orders:
        result = results[order.market_ticker]
        close = order.as_of + lead
        snap = decisions[(order.market_ticker, close)].snap
        out.append(OrderSettlement(result=result, market_mid=(snap.yes_bid + snap.yes_ask) / 2))
    return out


def tick_baseline_brier(
    snapshots: list[ReplaySnapshot],
) -> tuple[Decimal, int]:
    results: dict[str, str] = {}
    decisions: dict[str, ReplaySnapshot] = {}
    for row in snapshots:
        if row.snap.result in ("yes", "no"):
            results[row.snap.ticker] = row.snap.result
        close = row.snap.close_time
        if close is None or row.snapshot_at >= close:
            continue
        ticker = row.snap.ticker
        best = decisions.get(ticker)
        if best is None or row.snapshot_at > best.snapshot_at:
            decisions[ticker] = row

    mids: list[Decimal] = []
    outcomes: list[int] = []
    for ticker, dec_snap in decisions.items():
        result = results.get(ticker)
        if result not in ("yes", "no"):
            continue
        mids.append((dec_snap.snap.yes_bid + dec_snap.snap.yes_ask) / 2)
        outcomes.append(1 if result == "yes" else 0)

    if not mids:
        return Decimal("0"), 0
    return brier_score(mids, outcomes), len(mids)


def _executed_yes_price(order: BacktestOrder) -> Decimal:
    if order.action is TradeSide.BUY_YES:
        return order.price_per_contract
    return Decimal("1") - order.price_per_contract


def _qualifying_prints(
    order: BacktestOrder,
    prints: Sequence[dict],
    close: datetime,
    rule: Literal["conservative", "inclusive"],
) -> list[dict]:
    p = _executed_yes_price(order)
    out: list[dict] = []
    for row in prints:
        t: datetime = row["created_time"]
        if t <= order.as_of or t > close:
            continue
        yes_price: Decimal = row["yes_price"]
        if order.action is TradeSide.SELL_YES:
            qualifies = yes_price > p if rule == "conservative" else yes_price >= p
        else:
            qualifies = yes_price < p if rule == "conservative" else yes_price <= p
        if qualifies:
            out.append(row)
    return out


def conservative_execution_bound(
    n_orders: int,
    n_zero_fill: int,
    threshold: Decimal = Decimal("0.60"),
) -> tuple[Decimal, bool]:
    if n_orders == 0:
        return Decimal("0"), False
    unfilled_rate = Decimal(n_zero_fill) / Decimal(n_orders)
    return unfilled_rate, unfilled_rate > threshold


@dataclass(frozen=True, slots=True)
class DecileRow:
    decile: int
    lo_hours: Decimal
    hi_hours: Decimal
    contracts: int
    sell_yes_contracts: int
    buy_yes_contracts: int
    n_prints: int


def build_decile_rows(
    filled: Sequence[tuple[BacktestOrder, Sequence[dict]]],
    lead: timedelta,
) -> list[DecileRow]:
    width = lead / 10
    lead_hours = Decimal(str(lead.total_seconds() / 3600))
    contracts: dict[int, int] = {}
    sell_yes: dict[int, int] = {}
    buy_yes: dict[int, int] = {}
    n_prints: dict[int, int] = {}
    for order, prints in filled:
        close = order.as_of + lead
        for row in prints:
            decile = min(9, int((close - row["created_time"]) / width))
            count = int(row["count"])
            contracts[decile] = contracts.get(decile, 0) + count
            n_prints[decile] = n_prints.get(decile, 0) + 1
            if order.action is TradeSide.SELL_YES:
                sell_yes[decile] = sell_yes.get(decile, 0) + count
            else:
                buy_yes[decile] = buy_yes.get(decile, 0) + count

    rows: list[DecileRow] = []
    for decile in sorted(contracts):
        rows.append(
            DecileRow(
                decile=decile,
                lo_hours=(Decimal(decile) * lead_hours / Decimal("10")),
                hi_hours=(Decimal(decile + 1) * lead_hours / Decimal("10")),
                contracts=contracts[decile],
                sell_yes_contracts=sell_yes.get(decile, 0),
                buy_yes_contracts=buy_yes.get(decile, 0),
                n_prints=n_prints[decile],
            )
        )
    return rows


def _render_decile_file(
    rule: Literal["conservative", "inclusive"],
    n_orders: int,
    n_zero_fill: int,
    decile_rows: list[DecileRow],
) -> str:
    unfilled_rate, is_execution_bound = conservative_execution_bound(n_orders, n_zero_fill)
    lines = [
        f"# fills by hours-to-close decile ({rule})",
        "",
        f"intended_orders={n_orders} zero_fill_orders={n_zero_fill} unfilled_rate={unfilled_rate}",
    ]
    if rule == "conservative" and is_execution_bound:
        lines.append("execution-bound, not edge-bound")
    lines.append("")
    lines.append(
        "| decile | hours_to_close_range | contracts | sell_yes_contracts"
        " | buy_yes_contracts | n_prints |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in decile_rows:
        lines.append(
            f"| {row.decile} | [{row.lo_hours}, {row.hi_hours}) | {row.contracts} |"
            f" {row.sell_yes_contracts} | {row.buy_yes_contracts} | {row.n_prints} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_prints_by_ticker(tick_path: Path) -> dict[str, list[dict]]:
    rows = pq.read_table(tick_path).to_pylist()
    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(
            {
                "yes_price": row["yes_price"],
                "count": int(row["count"]),
                "created_time": row["created_time"],
            }
        )
    return by_ticker


async def run_tick_survival(
    tick_path: Path,
    snapshots: list[ReplaySnapshot],
    forecast: ForecastReplay,
    lead: timedelta,
    out_dir: Path,
) -> tuple[ReportVerdict, Path, list[ScoredRun], dict[str, ReplayLog]]:
    prints_by_ticker = _load_prints_by_ticker(tick_path)
    logs: dict[str, ReplayLog] = {}

    rule_runs: dict[str, list[ScoredRun]] = {"conservative": [], "inclusive": []}
    reference_orders: dict[str, list[BacktestOrder]] = {"conservative": [], "inclusive": []}
    reference_fills: dict[str, list[BacktestFill]] = {"conservative": [], "inclusive": []}
    reference_filled: dict[str, list[tuple[BacktestOrder, list[dict]]]] = {
        "conservative": [],
        "inclusive": [],
    }
    reference_intended: dict[str, int] = {"conservative": 0, "inclusive": 0}
    reference_zero_fill: dict[str, int] = {"conservative": 0, "inclusive": 0}

    for strategy in ("tails", "edge"):
        for bankroll in (REFERENCE_BANKROLL, SENSITIVITY_BANKROLL):
            log = ReplayLog()
            config = BacktestConfig(lead=lead, bankroll=bankroll, depth_table={})
            orders = await run_replay(snapshots, forecast, strategy, config, log=log)
            logs[f"{strategy}:{bankroll}"] = log
            settlements_by_order = {
                id(o): s for o, s in zip(orders, tick_settle_orders(orders, snapshots, lead))
            }

            for rule in ("conservative", "inclusive"):
                scored_orders: list[BacktestOrder] = []
                fills: list[BacktestFill] = []
                settlements: list[OrderSettlement] = []
                zero_fill: list[BacktestOrder] = []
                filled_prints: list[tuple[BacktestOrder, list[dict]]] = []
                for order in orders:
                    prints = prints_by_ticker.get(order.market_ticker, [])
                    close = order.as_of + lead
                    qualifying = _qualifying_prints(order, prints, close, rule)
                    filled = min(sum(r["count"] for r in qualifying), order.contracts)
                    if filled == 0:
                        zero_fill.append(order)
                        continue
                    settlement = settlements_by_order[id(order)]
                    scored_orders.append(order)
                    fills.append(
                        settle_filled_order(
                            order, filled, _executed_yes_price(order), settlement.result
                        )
                    )
                    settlements.append(settlement)
                    filled_prints.append((order, qualifying))

                failures = log.failures + [make_no_prints_failure(o) for o in zero_fill]
                report = score_run(scored_orders, fills, settlements, failures)
                rule_runs[rule].append(
                    ScoredRun(
                        strategy=strategy,
                        window="in_sample",
                        bankroll=bankroll,
                        depth_multiplier=Decimal("1.0"),
                        sigma_multiplier=Decimal("1.0"),
                        report=report,
                    )
                )
                if bankroll == REFERENCE_BANKROLL:
                    reference_orders[rule].extend(scored_orders)
                    reference_fills[rule].extend(fills)
                    reference_filled[rule].extend(filled_prints)
                    reference_intended[rule] += len(scored_orders) + len(zero_fill)
                    reference_zero_fill[rule] += len(zero_fill)

    verdicts: dict[str, ReportVerdict] = {}
    summaries: dict[str, Path] = {}
    for rule in ("conservative", "inclusive"):
        rule_dir = out_dir / rule
        verdicts[rule] = evaluate_runs(rule_runs[rule])
        summaries[rule] = write_report(
            rule_runs[rule], reference_orders[rule], reference_fills[rule], rule_dir
        )
        decile_rows = build_decile_rows(reference_filled[rule], lead)
        (rule_dir / "fills_by_decile.md").write_text(
            _render_decile_file(
                rule, reference_intended[rule], reference_zero_fill[rule], decile_rows
            )
        )

    return (
        verdicts["conservative"],
        summaries["conservative"],
        rule_runs["conservative"],
        logs,
    )
