from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bot.backtest.context import BacktestBudgets, build_edge_context
from bot.backtest.engine import BacktestOrder, ReplayFailure, build_gate_ctx
from bot.backtest.normalize import CanonicalSnapshot
from bot.backtest.pnl import BacktestFill
from bot.backtest.scoring import OrderSettlement
from bot.backtest.ticks import (
    fill_against_prints,
    ingest_ticks,
    make_no_prints_failure,
    settle_filled_order,
    tick_baseline_brier,
    tick_decision_snapshots,
    tick_settle_orders,
)
from bot.backtest.forecast_replay import StationSpec, pick_cycle
from bot.execution.fees import taker_fee
from bot.execution.paper import TradeSide, TradeIntent
from bot.forecast.cdf import EnsembleCDF
from bot.risk.gates import GateMode
from bot.risk.gates import evaluate as evaluate_gates
from bot.strategy.edge import EdgeAction
from bot.strategy.edge import evaluate as evaluate_edge
from bot.validation.scoring import realized_pnl_for_trade


_TRADES_SCHEMA = pa.schema(
    [
        pa.field("trade_id", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("count", pa.int64()),
        pa.field("yes_price", pa.int64()),
        pa.field("no_price", pa.int64()),
        pa.field("taker_side", pa.string()),
        pa.field("created_time", pa.timestamp("us", tz="UTC")),
    ]
)

_BASE_TIME = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)


def _trade(
    trade_id: str,
    ticker: str,
    *,
    count: int = 1,
    yes_price: int = 50,
    no_price: int = 50,
    taker_side: str = "yes",
    created_time: datetime | None = None,
) -> dict:
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "count": count,
        "yes_price": yes_price,
        "no_price": no_price,
        "taker_side": taker_side,
        "created_time": created_time or _BASE_TIME,
    }


def _write_shard(rows: list[dict], path: Path) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=_TRADES_SCHEMA), path)
    return path


def _make_acceptance_fixture(path: Path) -> Path:
    rows = [
        _trade("t01", "KXHIGHNY-25MAR01-T50", yes_price=60, no_price=40),
        _trade("t02", "KXHIGHNY-25MAR01-T55", yes_price=30, no_price=70),
        _trade("t03", "KXHIGHCHI-25MAR01-T50", yes_price=45, no_price=55),
        _trade("t01", "KXHIGHNY-25MAR01-T50", yes_price=60, no_price=40),
        _trade("t04", "KXHIGHMOV-25MAR01-T50"),
        _trade("t05", "KXHIGHHOU-25MAR01-T55"),
        _trade("t06", "KXHIGHMOVFOO-25MAR01-T50"),
        _trade("t07", "KXPRES-26-DEM"),
        _trade("t08", "KXNFL-26W14-SF"),
        _trade("t09", "KXBTCD-26JUN09-T70000"),
    ]
    return _write_shard(rows, path)


def test_acceptance_fixture_writes_3_rows(tmp_path: Path) -> None:
    shard = _make_acceptance_fixture(tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    n = ingest_ticks(
        [shard],
        out,
        series=[
            "KXHIGHNY",
            "KXHIGHCHI",
            "KXHIGHAUS",
            "KXHIGHMIA",
            "KXHIGHDEN",
            "KXHIGHPHIL",
            "KXHIGHLAX",
        ],
    )

    assert n == 3
    written = pq.read_table(out)
    assert written.num_rows == 3


def test_returns_row_count(tmp_path: Path) -> None:
    shard = _make_acceptance_fixture(tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    n = ingest_ticks([shard], out, series=["KXHIGHNY"])
    assert n == 2


def test_output_sorted_by_ticker_then_created_time(tmp_path: Path) -> None:
    t1 = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 3, 1, 11, 0, tzinfo=timezone.utc)
    t3 = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
    t4 = datetime(2025, 3, 1, 9, 0, tzinfo=timezone.utc)
    rows = [
        _trade("a3", "KXHIGHNY-25MAR01-T55", created_time=t3),
        _trade("a1", "KXHIGHCHI-25MAR01-T50", created_time=t1),
        _trade("a2", "KXHIGHNY-25MAR01-T50", created_time=t2),
        _trade("a4", "KXHIGHNY-25MAR01-T55", created_time=t4),
    ]
    shard = _write_shard(rows, tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    ingest_ticks([shard], out, series=["KXHIGHNY", "KXHIGHCHI"])

    written = pq.read_table(out).to_pylist()
    result = [(r["ticker"], r["created_time"]) for r in written]
    assert result == [
        ("KXHIGHCHI-25MAR01-T50", t1),
        ("KXHIGHNY-25MAR01-T50", t2),
        ("KXHIGHNY-25MAR01-T55", t4),
        ("KXHIGHNY-25MAR01-T55", t3),
    ]


def test_price_normalization(tmp_path: Path) -> None:
    rows = [_trade("x1", "KXHIGHNY-25MAR01-T50", yes_price=12, no_price=88)]
    shard = _write_shard(rows, tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    ingest_ticks([shard], out, series=["KXHIGHNY"])

    written = pq.read_table(out).to_pylist()
    assert len(written) == 1
    assert written[0]["yes_price"] == Decimal("0.12")
    assert written[0]["no_price"] == Decimal("0.88")


def test_dedup_across_two_shards(tmp_path: Path) -> None:
    shard1 = _write_shard(
        [_trade("dup01", "KXHIGHNY-25MAR01-T50", yes_price=40, no_price=60)],
        tmp_path / "s1.parquet",
    )
    shard2 = _write_shard(
        [
            _trade("dup01", "KXHIGHNY-25MAR01-T50", yes_price=40, no_price=60),
            _trade("uniq02", "KXHIGHNY-25MAR01-T55", yes_price=55, no_price=45),
        ],
        tmp_path / "s2.parquet",
    )
    out = tmp_path / "ticks.parquet"

    n = ingest_ticks([shard1, shard2], out, series=["KXHIGHNY"])

    assert n == 2
    written = pq.read_table(out)
    assert written.num_rows == 2


def test_overmatch_prefixes_excluded(tmp_path: Path) -> None:
    rows = [
        _trade("m1", "KXHIGHMOV-25MAR01-T50"),
        _trade("m2", "KXHIGHHOU-25MAR01-T50"),
        _trade("m3", "KXHIGHT-25MAR01-T50"),
        _trade("m4", "KXHIGHNY-25MAR01-T50"),
        _trade("m5", "KXHIGHNYFOO-25MAR01-T50"),
    ]
    shard = _write_shard(rows, tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    n = ingest_ticks([shard], out, series=["KXHIGHNY"])

    assert n == 1


def test_output_schema_columns(tmp_path: Path) -> None:
    rows = [_trade("s1", "KXHIGHNY-25MAR01-T50", yes_price=50, no_price=50)]
    shard = _write_shard(rows, tmp_path / "shard.parquet")
    out = tmp_path / "ticks.parquet"

    ingest_ticks([shard], out, series=["KXHIGHNY"])

    schema = pq.read_table(out).schema
    assert set(schema.names) == {
        "trade_id",
        "ticker",
        "count",
        "yes_price",
        "no_price",
        "taker_side",
        "created_time",
    }


_CLOSE = datetime(2025, 3, 2, 12, 0, tzinfo=timezone.utc)
_LEAD = timedelta(hours=24)
_STALENESS = timedelta(hours=6)
_DEPTH_WINDOW = timedelta(hours=6)


def _canonical_snap(
    ticker: str,
    close_time: datetime | None,
    result: str = "",
) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-")[0],
        status="settled" if result else "active",
        result=result,
        yes_ask=Decimal("0.50"),
        yes_bid=Decimal("0.48"),
        no_ask=Decimal("0.52"),
        no_bid=Decimal("0.50"),
        last_price=Decimal("0.50"),
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
        close_time=close_time,
    )


def _write_tick_parquet(rows: list[dict], path: Path) -> Path:
    schema = pa.schema(
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
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _tick_row(
    trade_id: str,
    ticker: str,
    yes_price: Decimal,
    created_time: datetime,
    count: int = 1,
) -> dict:
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "count": count,
        "yes_price": yes_price,
        "no_price": Decimal("1.00") - yes_price,
        "taker_side": "yes",
        "created_time": created_time,
    }


def test_decision_price_picks_last_print_before_as_of(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    rows = [
        _tick_row("t1", ticker, Decimal("0.12"), _CLOSE - timedelta(hours=26)),
        _tick_row("t2", ticker, Decimal("0.09"), _CLOSE - timedelta(hours=23)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE, result="yes")]

    snaps, omitted = tick_decision_snapshots(
        tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW
    )

    decision_rows = [s for s in snaps if s.snapshot_at < _CLOSE]
    assert len(decision_rows) == 1
    assert decision_rows[0].snap.yes_bid == Decimal("0.12")
    assert decision_rows[0].snap.yes_ask == Decimal("0.12")
    assert omitted == 0


def test_staleness_cap_omits_stale_market(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker, Decimal("0.50"), as_of - timedelta(hours=7)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE, result="yes")]

    snaps, omitted = tick_decision_snapshots(
        tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW
    )

    decision_rows = [s for s in snaps if s.snapshot_at < _CLOSE]
    assert len(decision_rows) == 0
    assert omitted == 1


def test_no_print_before_as_of_omitted(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker, Decimal("0.50"), as_of + timedelta(hours=1)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE, result="yes")]

    snaps, omitted = tick_decision_snapshots(
        tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW
    )

    decision_rows = [s for s in snaps if s.snapshot_at < _CLOSE]
    assert len(decision_rows) == 0
    assert omitted == 1


def test_bid_ask_equal_last_price(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T55"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker, Decimal("0.35"), as_of - timedelta(hours=2)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE)]

    snaps, _ = tick_decision_snapshots(tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW)

    decision_rows = [s for s in snaps if s.snapshot_at < _CLOSE]
    assert len(decision_rows) == 1
    snap = decision_rows[0].snap
    assert snap.yes_bid == Decimal("0.35")
    assert snap.yes_ask == Decimal("0.35")
    assert snap.yes_bid == snap.yes_ask


def test_trailing_depth_uses_depth_window_before_as_of(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker, Decimal("0.40"), as_of - timedelta(hours=7)),
        _tick_row("t2", ticker, Decimal("0.42"), as_of - timedelta(hours=5)),
        _tick_row("t3", ticker, Decimal("0.44"), as_of - timedelta(hours=3)),
        _tick_row("t4", ticker, Decimal("0.45"), as_of - timedelta(hours=1)),
        _tick_row("t5", ticker, Decimal("0.46"), as_of + timedelta(hours=1)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE)]

    snaps, _ = tick_decision_snapshots(tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW)

    decision_rows = [s for s in snaps if s.snapshot_at < _CLOSE]
    assert len(decision_rows) == 1
    snap = decision_rows[0].snap
    assert snap.yes_bid_size == Decimal("3")
    assert snap.no_bid_size == Decimal("3")


def test_settlement_rows_pinned_to_close_time(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker, Decimal("0.40"), as_of - timedelta(hours=2)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE, result="yes")]

    snaps, _ = tick_decision_snapshots(tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW)

    settlement_rows = [s for s in snaps if s.snapshot_at == _CLOSE]
    assert len(settlement_rows) == 1
    assert settlement_rows[0].snap.result == "yes"


def test_depth_window_exclusive_lower_bound(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker, Decimal("0.40"), as_of - _DEPTH_WINDOW),
        _tick_row("t2", ticker, Decimal("0.42"), as_of - timedelta(hours=1)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE)]

    snaps, _ = tick_decision_snapshots(tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW)

    decision_rows = [s for s in snaps if s.snapshot_at < _CLOSE]
    snap = decision_rows[0].snap
    assert snap.yes_bid_size == Decimal("1")


def test_depth_window_inclusive_upper_bound(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker, Decimal("0.40"), as_of),
        _tick_row("t2", ticker, Decimal("0.42"), as_of - timedelta(minutes=1)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE)]

    snaps, _ = tick_decision_snapshots(tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW)

    decision_rows = [s for s in snaps if s.snapshot_at < _CLOSE]
    snap = decision_rows[0].snap
    assert snap.yes_bid_size == Decimal("2")


def test_null_close_time_not_counted_in_omitted(tmp_path: Path) -> None:
    ticker_valid = "KXHIGHNY-25MAR01-T50"
    ticker_null = "KXHIGHNY-25MAR02-T50"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker_valid, Decimal("0.50"), as_of - timedelta(hours=7)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [
        _canonical_snap(ticker_valid, _CLOSE, result="yes"),
        _canonical_snap(ticker_null, None),
    ]

    _, omitted = tick_decision_snapshots(tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW)

    assert omitted == 1


def test_decision_row_result_pinned_empty(tmp_path: Path) -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    as_of = _CLOSE - _LEAD
    rows = [
        _tick_row("t1", ticker, Decimal("0.40"), as_of - timedelta(hours=2)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [_canonical_snap(ticker, _CLOSE, result="yes")]

    snaps, _ = tick_decision_snapshots(tick_path, market_state, _LEAD, _STALENESS, _DEPTH_WINDOW)

    decision_rows = [s for s in snaps if s.snapshot_at < _CLOSE]
    assert len(decision_rows) == 1
    assert decision_rows[0].snap.result == ""


_LOOKAHEAD_LEAD = timedelta(hours=30)


def test_lookahead_invariance(tmp_path: Path) -> None:
    # as_of lands before the observation window for each event date so edge.evaluate
    # passes the same_day gate and reaches compute_stake_contracts with nonzero contracts.
    # NY window for Mar 4 starts 2025-03-04 05:00 UTC; as_of_a = 2025-03-03 18:00 UTC.
    close_a = datetime(2025, 3, 5, 0, 0, tzinfo=timezone.utc)
    close_b = datetime(2025, 3, 6, 0, 0, tzinfo=timezone.utc)
    close_c = datetime(2025, 3, 7, 0, 0, tzinfo=timezone.utc)
    ticker_a = "KXHIGHNY-25MAR04-T50"
    ticker_b = "KXHIGHNY-25MAR05-T50"
    ticker_c = "KXHIGHNY-25MAR06-T50"
    as_of_a = close_a - _LOOKAHEAD_LEAD
    as_of_b = close_b - _LOOKAHEAD_LEAD
    as_of_c = close_c - _LOOKAHEAD_LEAD

    rows = [
        _tick_row("a1", ticker_a, Decimal("0.30"), as_of_a - timedelta(hours=2)),
        _tick_row("a2", ticker_a, Decimal("0.31"), as_of_a - timedelta(hours=1)),
        _tick_row("b1", ticker_b, Decimal("0.30"), as_of_b - timedelta(hours=2)),
        _tick_row("b2", ticker_b, Decimal("0.31"), as_of_b - timedelta(hours=1)),
        _tick_row("b3", ticker_b, Decimal("0.35"), as_of_b + timedelta(hours=1)),
        _tick_row("b4", ticker_b, Decimal("0.36"), as_of_b + timedelta(hours=2)),
        _tick_row("c1", ticker_c, Decimal("0.30"), as_of_c - timedelta(hours=2)),
        _tick_row("c2", ticker_c, Decimal("0.31"), as_of_c - timedelta(hours=1)),
        _tick_row("c3", ticker_c, Decimal("0.32"), as_of_c - timedelta(minutes=30)),
        _tick_row("c4", ticker_c, Decimal("0.33"), as_of_c - timedelta(minutes=10)),
    ]
    tick_path = _write_tick_parquet(rows, tmp_path / "ticks.parquet")
    market_state: Sequence[CanonicalSnapshot] = [
        _canonical_snap(ticker_a, close_a),
        _canonical_snap(ticker_b, close_b),
        _canonical_snap(ticker_c, close_c),
    ]

    snaps, _ = tick_decision_snapshots(
        tick_path, market_state, _LOOKAHEAD_LEAD, _STALENESS, _DEPTH_WINDOW
    )

    by_ticker = {s.snap.ticker: s.snap for s in snaps if s.snapshot_at < s.snap.close_time}
    snap_a = by_ticker[ticker_a]
    snap_b = by_ticker[ticker_b]
    snap_c = by_ticker[ticker_c]

    assert snap_a.yes_bid == snap_b.yes_bid
    assert snap_a.yes_bid_size == snap_b.yes_bid_size
    assert snap_c.yes_bid_size > snap_a.yes_bid_size

    members = np.full(31, 52.0)
    cdf = EnsembleCDF.from_members(members, smoothing=1.0)
    station = StationSpec(name="JFK", latitude=40.64, longitude=-73.78, timezone="America/New_York")
    budgets = BacktestBudgets(
        event_budget_remaining=Decimal("1000"),
        market_budget_remaining=Decimal("1000"),
    )
    bankroll = Decimal("10000")
    spread = Decimal("2.5")

    def _contracts_and_verdict(snap: CanonicalSnapshot, as_of: datetime) -> tuple[int, bool]:
        ctx = build_edge_context(
            snap,
            cdf,
            as_of,
            bankroll,
            budgets,
            ensemble_spread=spread,
            buy_yes_depth=int(snap.no_bid_size or 0),
            sell_yes_depth=int(snap.yes_bid_size or 0),
            station_tz=station.timezone,
        )
        sig = evaluate_edge(ctx, mode="paper")
        if sig.action is EdgeAction.SKIP:
            return sig.contracts, False
        side = TradeSide.BUY_YES if sig.action is EdgeAction.BUY_YES else TradeSide.SELL_YES
        assert snap.close_time is not None
        intent = TradeIntent(
            market_ticker=snap.ticker,
            side=side,
            contracts=sig.contracts,
            fair_yes=ctx.fair_yes,
            q_raw=ctx.fair_yes,
            strategy="edge",
            ensemble_spread_sigma_t=spread,
            lead_time_hours=Decimal(str((snap.close_time - as_of).total_seconds() / 3600)),
        )
        gate_ctx = build_gate_ctx(
            intent=intent,
            fair_yes=ctx.fair_yes,
            spread=spread,
            run_time=pick_cycle(as_of),
            now=as_of,
            book=snap,
            close_time=snap.close_time,
            buy_yes_depth=int(snap.no_bid_size or 0),
            sell_yes_depth=int(snap.yes_bid_size or 0),
            market_existing_dollars=Decimal("0"),
            market_position_cap=Decimal("500"),
            event_existing_dollars=Decimal("0"),
            event_position_cap=Decimal("500"),
            series_existing_dollars=Decimal("0"),
            series_position_cap=Decimal("500"),
            aggregate_existing_dollars=Decimal("0"),
            aggregate_exposure_cap=Decimal("5000"),
            account_balance=bankroll,
            required_cushion=Decimal("0"),
            market_status=snap.status,
        )
        check = evaluate_gates(gate_ctx, GateMode.PAPER)
        return sig.contracts, check.overall_passed

    contracts_a, verdict_a = _contracts_and_verdict(snap_a, as_of_a)
    contracts_b, verdict_b = _contracts_and_verdict(snap_b, as_of_b)
    contracts_c, _ = _contracts_and_verdict(snap_c, as_of_c)

    assert contracts_a > 0
    assert contracts_a == contracts_b
    assert verdict_a == verdict_b
    assert contracts_c != contracts_a


_AS_OF = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
_CLOSE_FILL = datetime(2025, 6, 1, 14, 0, tzinfo=timezone.utc)


def _print_row(
    yes_price: Decimal,
    count: int,
    offset: timedelta,
) -> dict:
    return {
        "yes_price": yes_price,
        "count": count,
        "created_time": _AS_OF + offset,
    }


def _sell_yes_order(
    contracts: int = 100,
    yes_bid: Decimal = Decimal("0.10"),
) -> BacktestOrder:
    return BacktestOrder(
        market_ticker="KXHIGHNY-25JUN01-T50",
        as_of=_AS_OF,
        strategy="tails",
        action=TradeSide.SELL_YES,
        contracts=contracts,
        fair_yes=Decimal("0.05"),
        price_per_contract=Decimal("1") - yes_bid,
        order_dollars=(Decimal("1") - yes_bid) * Decimal(contracts),
        depth_at_price=50,
        depth_source="snapshot",
        lead_bucket="2h",
    )


def _buy_yes_order(
    contracts: int = 100,
    yes_ask: Decimal = Decimal("0.30"),
) -> BacktestOrder:
    return BacktestOrder(
        market_ticker="KXHIGHNY-25JUN01-T50",
        as_of=_AS_OF,
        strategy="edge",
        action=TradeSide.BUY_YES,
        contracts=contracts,
        fair_yes=Decimal("0.50"),
        price_per_contract=yes_ask,
        order_dollars=yes_ask * Decimal(contracts),
        depth_at_price=50,
        depth_source="snapshot",
        lead_bucket="2h",
    )


def test_fill_sell_yes_conservative_golden() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.10"))
    prints = [
        _print_row(Decimal("0.11"), 30, timedelta(minutes=5)),
        _print_row(Decimal("0.10"), 50, timedelta(minutes=10)),
        _print_row(Decimal("0.12"), 40, timedelta(minutes=15)),
    ]
    filled = fill_against_prints(order, prints, _CLOSE_FILL, "conservative")
    assert filled == 70


def test_fill_sell_yes_inclusive_golden() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.10"))
    prints = [
        _print_row(Decimal("0.11"), 30, timedelta(minutes=5)),
        _print_row(Decimal("0.10"), 50, timedelta(minutes=10)),
        _print_row(Decimal("0.12"), 40, timedelta(minutes=15)),
    ]
    filled = fill_against_prints(order, prints, _CLOSE_FILL, "inclusive")
    assert filled == 100


def test_fill_zero_qualifying_prints() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.15"))
    prints = [
        _print_row(Decimal("0.10"), 50, timedelta(minutes=5)),
        _print_row(Decimal("0.12"), 40, timedelta(minutes=10)),
    ]
    filled = fill_against_prints(order, prints, _CLOSE_FILL, "conservative")
    assert filled == 0


def test_fill_zero_produces_no_prints_failure() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.15"))
    prints: list[dict] = []
    filled = fill_against_prints(order, prints, _CLOSE_FILL, "conservative")
    assert filled == 0

    failure = make_no_prints_failure(order)
    assert isinstance(failure, ReplayFailure)
    assert failure.market_ticker == order.market_ticker
    assert failure.as_of == order.as_of
    assert failure.strategy == order.strategy
    assert failure.layer == "fill"
    assert failure.name == "no_qualifying_prints"


def test_settle_zero_fill_returns_flat() -> None:
    order = _sell_yes_order(contracts=100)
    fill = settle_filled_order(order, 0, Decimal("0.10"), "no")
    assert fill.gross_pnl == Decimal("0")
    assert fill.fee_dollars == Decimal("0")
    assert fill.net_pnl == Decimal("0")
    assert isinstance(fill, BacktestFill)


def test_price_sign_golden_no_settled() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.10"))
    executed_yes_price = Decimal("0.10")
    fill = settle_filled_order(order, 70, executed_yes_price, "no")

    expected_fee = taker_fee(70, executed_yes_price)
    expected_net = realized_pnl_for_trade(
        TradeSide.SELL_YES, executed_yes_price, 70, expected_fee, won=True
    )
    assert fill.net_pnl == expected_net
    assert fill.net_pnl > Decimal("0")

    collateral_price = Decimal("1") - executed_yes_price
    wrong_fee = taker_fee(70, collateral_price)
    wrong_net = realized_pnl_for_trade(
        TradeSide.SELL_YES, collateral_price, 70, wrong_fee, won=True
    )
    assert fill.net_pnl != wrong_net


def test_price_sign_golden_yes_settled() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.10"))
    executed_yes_price = Decimal("0.10")
    fill = settle_filled_order(order, 70, executed_yes_price, "yes")

    expected_fee = taker_fee(70, executed_yes_price)
    expected_net = realized_pnl_for_trade(
        TradeSide.SELL_YES, executed_yes_price, 70, expected_fee, won=False
    )
    assert fill.net_pnl == expected_net
    assert fill.net_pnl < Decimal("0")

    collateral_price = Decimal("1") - executed_yes_price
    wrong_fee = taker_fee(70, collateral_price)
    wrong_net = realized_pnl_for_trade(
        TradeSide.SELL_YES, collateral_price, 70, wrong_fee, won=False
    )
    assert fill.net_pnl != wrong_net


def test_fill_buy_yes_conservative() -> None:
    order = _buy_yes_order(contracts=50, yes_ask=Decimal("0.30"))
    prints = [
        _print_row(Decimal("0.29"), 20, timedelta(minutes=5)),
        _print_row(Decimal("0.30"), 30, timedelta(minutes=10)),
        _print_row(Decimal("0.28"), 15, timedelta(minutes=15)),
    ]
    filled = fill_against_prints(order, prints, _CLOSE_FILL, "conservative")
    assert filled == 35


def test_fill_buy_yes_inclusive() -> None:
    order = _buy_yes_order(contracts=50, yes_ask=Decimal("0.30"))
    prints = [
        _print_row(Decimal("0.29"), 20, timedelta(minutes=5)),
        _print_row(Decimal("0.30"), 30, timedelta(minutes=10)),
        _print_row(Decimal("0.28"), 15, timedelta(minutes=15)),
    ]
    filled = fill_against_prints(order, prints, _CLOSE_FILL, "inclusive")
    assert filled == 50


def test_fill_horizon_excludes_print_at_as_of() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.10"))
    prints = [
        _print_row(Decimal("0.15"), 50, timedelta(0)),
    ]
    filled = fill_against_prints(order, prints, _CLOSE_FILL, "conservative")
    assert filled == 0


def test_fill_horizon_includes_print_at_close() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.10"))
    close = _AS_OF + timedelta(hours=2)
    prints = [
        {"yes_price": Decimal("0.15"), "count": 50, "created_time": close},
    ]
    filled = fill_against_prints(order, prints, close, "conservative")
    assert filled == 50


def test_fill_horizon_excludes_print_after_close() -> None:
    order = _sell_yes_order(contracts=100, yes_bid=Decimal("0.10"))
    close = _AS_OF + timedelta(hours=2)
    prints = [
        {"yes_price": Decimal("0.15"), "count": 50, "created_time": close + timedelta(seconds=1)},
    ]
    filled = fill_against_prints(order, prints, close, "conservative")
    assert filled == 0


def test_settle_partial_fill_scales_fees() -> None:
    intended = 100
    filled = 40
    yes_price = Decimal("0.20")
    order = _sell_yes_order(contracts=intended, yes_bid=yes_price)

    fill = settle_filled_order(order, filled, yes_price, "no")

    expected_fee = taker_fee(filled, yes_price)
    assert fill.fee_dollars == expected_fee

    wrong_fee = taker_fee(intended, yes_price)
    assert fill.fee_dollars != wrong_fee


_SETTLE_LEAD = timedelta(hours=24)


def _make_decision_and_settlement_snaps(
    ticker: str,
    close: datetime,
    decision_price: Decimal,
    result: str,
) -> tuple[list, list]:
    from bot.backtest.engine import ReplaySnapshot

    as_of = close - _SETTLE_LEAD
    no_price = Decimal("1") - decision_price

    decision_canonical = CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-")[0],
        status="active",
        result="",
        yes_ask=decision_price,
        yes_bid=decision_price,
        no_ask=no_price,
        no_bid=no_price,
        last_price=decision_price,
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
        close_time=close,
    )
    settlement_canonical = CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-")[0],
        status="settled",
        result=result,
        yes_ask=Decimal("0"),
        yes_bid=Decimal("0"),
        no_ask=Decimal("0"),
        no_bid=Decimal("0"),
        last_price=Decimal("0"),
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
        close_time=close,
    )
    decision_snap = ReplaySnapshot(snapshot_at=as_of, snap=decision_canonical)
    settlement_snap = ReplaySnapshot(snapshot_at=close, snap=settlement_canonical)
    return [decision_snap], [settlement_snap]


def test_tick_settle_orders_market_mid_is_decision_price() -> None:
    ticker = "KXHIGHNY-25MAR01-T50"
    close = datetime(2025, 3, 2, 12, 0, tzinfo=timezone.utc)
    decision_price = Decimal("0.12")

    decision_snaps, settlement_snaps = _make_decision_and_settlement_snaps(
        ticker, close, decision_price, "no"
    )

    order = BacktestOrder(
        market_ticker=ticker,
        as_of=close - _SETTLE_LEAD,
        strategy="tails",
        action=TradeSide.SELL_YES,
        contracts=10,
        fair_yes=Decimal("0.05"),
        price_per_contract=Decimal("0.88"),
        order_dollars=Decimal("8.80"),
        depth_at_price=5,
        depth_source="snapshot",
        lead_bucket="24h",
    )

    settlements = tick_settle_orders([order], decision_snaps + settlement_snaps, _SETTLE_LEAD)

    assert len(settlements) == 1
    assert settlements[0] == OrderSettlement(result="no", market_mid=Decimal("0.12"))


def test_tick_settle_orders_market_mid_is_decision_time_not_settlement_time() -> None:
    ticker = "KXHIGHNY-25MAR01-T55"
    close = datetime(2025, 3, 2, 12, 0, tzinfo=timezone.utc)
    decision_price = Decimal("0.12")

    from bot.backtest.engine import ReplaySnapshot

    decision_canonical = CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-")[0],
        status="active",
        result="",
        yes_ask=decision_price,
        yes_bid=decision_price,
        no_ask=Decimal("1") - decision_price,
        no_bid=Decimal("1") - decision_price,
        last_price=decision_price,
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
        close_time=close,
    )
    settlement_canonical = CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-")[0],
        status="settled",
        result="yes",
        yes_ask=Decimal("0.99"),
        yes_bid=Decimal("0.99"),
        no_ask=Decimal("0.01"),
        no_bid=Decimal("0.01"),
        last_price=Decimal("0.99"),
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
        close_time=close,
    )
    snaps = [
        ReplaySnapshot(snapshot_at=close - _SETTLE_LEAD, snap=decision_canonical),
        ReplaySnapshot(snapshot_at=close, snap=settlement_canonical),
    ]

    order = BacktestOrder(
        market_ticker=ticker,
        as_of=close - _SETTLE_LEAD,
        strategy="tails",
        action=TradeSide.BUY_YES,
        contracts=10,
        fair_yes=Decimal("0.90"),
        price_per_contract=Decimal("0.12"),
        order_dollars=Decimal("1.20"),
        depth_at_price=5,
        depth_source="snapshot",
        lead_bucket="24h",
    )

    settlements = tick_settle_orders([order], snaps, _SETTLE_LEAD)

    assert len(settlements) == 1
    assert settlements[0].market_mid == Decimal("0.12")
    assert settlements[0].market_mid != Decimal("0.99")


def test_tick_baseline_brier_golden_three_markets() -> None:
    from bot.backtest.engine import ReplaySnapshot

    close_a = datetime(2025, 3, 2, 12, 0, tzinfo=timezone.utc)
    close_b = datetime(2025, 3, 3, 12, 0, tzinfo=timezone.utc)
    close_c = datetime(2025, 3, 4, 12, 0, tzinfo=timezone.utc)
    lead = timedelta(hours=24)

    def _snaps(ticker: str, close: datetime, price: Decimal, result: str) -> list:
        no_p = Decimal("1") - price
        dec = CanonicalSnapshot(
            ticker=ticker,
            event_ticker="-".join(ticker.split("-")[:2]),
            series_ticker=ticker.split("-")[0],
            status="active",
            result="",
            yes_ask=price,
            yes_bid=price,
            no_ask=no_p,
            no_bid=no_p,
            last_price=price,
            volume=Decimal("0"),
            volume_24h=Decimal("0"),
            open_interest=Decimal("0"),
            close_time=close,
        )
        sett = CanonicalSnapshot(
            ticker=ticker,
            event_ticker="-".join(ticker.split("-")[:2]),
            series_ticker=ticker.split("-")[0],
            status="settled",
            result=result,
            yes_ask=Decimal("0"),
            yes_bid=Decimal("0"),
            no_ask=Decimal("0"),
            no_bid=Decimal("0"),
            last_price=Decimal("0"),
            volume=Decimal("0"),
            volume_24h=Decimal("0"),
            open_interest=Decimal("0"),
            close_time=close,
        )
        return [
            ReplaySnapshot(snapshot_at=close - lead, snap=dec),
            ReplaySnapshot(snapshot_at=close, snap=sett),
        ]

    all_snaps = (
        _snaps("KXHIGHNY-25MAR01-T50", close_a, Decimal("0.10"), "no")
        + _snaps("KXHIGHNY-25MAR02-T50", close_b, Decimal("0.50"), "yes")
        + _snaps("KXHIGHNY-25MAR03-T50", close_c, Decimal("0.90"), "yes")
    )

    brier, n = tick_baseline_brier(all_snaps, lead)

    expected = (
        (Decimal("0.10") - Decimal("0")) ** 2
        + (Decimal("0.50") - Decimal("1")) ** 2
        + (Decimal("0.90") - Decimal("1")) ** 2
    ) / Decimal("3")
    assert n == 3
    assert brier == expected.quantize(Decimal("0.000001"))


def test_tick_baseline_brier_excludes_market_without_decision_snap() -> None:
    from bot.backtest.engine import ReplaySnapshot

    close = datetime(2025, 3, 2, 12, 0, tzinfo=timezone.utc)
    lead = timedelta(hours=24)

    ticker_with = "KXHIGHNY-25MAR01-T50"
    ticker_without = "KXHIGHNY-25MAR01-T55"

    dec = CanonicalSnapshot(
        ticker=ticker_with,
        event_ticker="KXHIGHNY-25MAR01",
        series_ticker="KXHIGHNY",
        status="active",
        result="",
        yes_ask=Decimal("0.40"),
        yes_bid=Decimal("0.40"),
        no_ask=Decimal("0.60"),
        no_bid=Decimal("0.60"),
        last_price=Decimal("0.40"),
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
        close_time=close,
    )
    sett_with = CanonicalSnapshot(
        ticker=ticker_with,
        event_ticker="KXHIGHNY-25MAR01",
        series_ticker="KXHIGHNY",
        status="settled",
        result="yes",
        yes_ask=Decimal("0"),
        yes_bid=Decimal("0"),
        no_ask=Decimal("0"),
        no_bid=Decimal("0"),
        last_price=Decimal("0"),
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
        close_time=close,
    )
    sett_without = CanonicalSnapshot(
        ticker=ticker_without,
        event_ticker="KXHIGHNY-25MAR01",
        series_ticker="KXHIGHNY",
        status="settled",
        result="no",
        yes_ask=Decimal("0"),
        yes_bid=Decimal("0"),
        no_ask=Decimal("0"),
        no_bid=Decimal("0"),
        last_price=Decimal("0"),
        volume=Decimal("0"),
        volume_24h=Decimal("0"),
        open_interest=Decimal("0"),
        close_time=close,
    )

    snaps = [
        ReplaySnapshot(snapshot_at=close - lead, snap=dec),
        ReplaySnapshot(snapshot_at=close, snap=sett_with),
        ReplaySnapshot(snapshot_at=close, snap=sett_without),
    ]

    brier, n = tick_baseline_brier(snaps, lead)

    assert n == 1
