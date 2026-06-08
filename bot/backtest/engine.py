from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np

from bot.backtest.context import (
    BacktestBudgets,
    build_edge_context,
    build_tails_context,
    fair_yes_for,
)
from bot.backtest.depth_table import ALL_SERIES, lead_bucket_for, parse_dt
from bot.backtest.forecast_replay import ForecastReplay, StationSpec, pick_cycle
from bot.backtest.normalize import CanonicalSnapshot
from bot.execution.gate_cost_basis import paper_collateral_per_contract
from bot.execution.paper import TradeIntent, TradeSide
from bot.forecast.cdf import EnsembleCDF
from bot.main import (
    AGGREGATE_EXPOSURE_CAP,
    AGGREGATE_EXPOSURE_FRAC,
    EVENT_POSITION_CAP,
    EVENT_POSITION_FRAC,
    MARKET_POSITION_CAP,
    MARKET_POSITION_FRAC,
    PAPER_BANKROLL,
    REQUIRED_CUSHION,
    SERIES_POSITION_CAP,
    SERIES_POSITION_FRAC,
    STATIONS,
    STRATEGY_BLACKLIST,
)
from bot.markets.parser import event_id, parse_ticker
from bot.risk.gates import (
    CAP_GATE_NAMES,
    GateContext,
    GateMode,
    evaluate as evaluate_gates,
)
from bot.storage.positions import SETTLEMENT_GRACE_DAYS, _effective_cutoff_date
from bot.strategy import edge as edge_strategy
from bot.strategy import tails as tails_strategy
from bot.strategy.sizing import sigma_t_median_for_lead


@dataclass(frozen=True, slots=True)
class TradeRow:
    pt_id: int
    intended_at: datetime
    market_ticker: str
    side: str
    contracts: int
    strategy: str
    outcome: str


@dataclass(frozen=True, slots=True)
class ForecastRow:
    station: str
    run_time: datetime
    valid_date: date
    members: np.ndarray


@dataclass(frozen=True, slots=True)
class BookRow:
    snapshot_at: datetime
    yes_ask: Decimal
    yes_bid: Decimal
    no_ask: Decimal
    no_bid: Decimal
    yes_ask_depth: int
    yes_bid_depth: int
    no_ask_depth: int
    no_bid_depth: int


@dataclass
class Bucket:
    n_replayed: int = 0
    n_current_would_place: int = 0
    n_current_would_skip: int = 0
    placed_wins: int = 0
    placed_total_outcome: int = 0
    skipped_wins: int = 0
    skipped_total_outcome: int = 0
    placed_size_sum: int = 0
    historical_size_sum: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    snapshot_at: datetime
    snap: CanonicalSnapshot


@dataclass(frozen=True, slots=True)
class LedgerFill:
    market_ticker: str
    side: TradeSide
    contracts: int
    delta: Decimal


@dataclass(frozen=True, slots=True)
class BacktestOrder:
    market_ticker: str
    as_of: datetime
    strategy: str
    action: TradeSide
    contracts: int
    fair_yes: Decimal
    price_per_contract: Decimal
    order_dollars: Decimal
    depth_at_price: int
    depth_source: str
    lead_bucket: str


@dataclass(frozen=True, slots=True)
class ReplayFailure:
    market_ticker: str
    as_of: datetime
    strategy: str
    layer: str
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class CycleRecord:
    minute: datetime
    market: dict[str, Decimal]
    event: dict[str, Decimal]
    series: dict[str, Decimal]
    aggregate: Decimal


@dataclass(slots=True)
class ReplayLog:
    failures: list[ReplayFailure] = field(default_factory=list)
    cycles: list[CycleRecord] = field(default_factory=list)


def _default_station_specs() -> dict[str, StationSpec]:
    return {
        series: StationSpec(
            name=cfg.station,
            latitude=cfg.latitude,
            longitude=cfg.longitude,
            timezone=cfg.timezone,
        )
        for series, cfg in STATIONS.items()
    }


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    lead: timedelta
    bankroll: Decimal
    depth_table: Mapping[tuple[str, str], Decimal] = field(default_factory=dict)
    stations: Mapping[str, StationSpec] = field(default_factory=_default_station_specs)
    required_cushion: Decimal = REQUIRED_CUSHION


def fetch_trades(conn: sqlite3.Connection, cutoff: datetime) -> list[TradeRow]:
    rows = conn.execute(
        """
        SELECT p.id, p.intended_at, p.market_ticker, p.side, p.contracts,
               p.strategy, s.outcome
          FROM paper_trades p
          JOIN simulated_pnl s ON s.paper_trade_id = p.id
         WHERE p.intended_at >= ?
         ORDER BY p.intended_at ASC
        """,
        (cutoff.isoformat(sep=" "),),
    ).fetchall()
    out: list[TradeRow] = []
    for r in rows:
        out.append(
            TradeRow(
                pt_id=int(r["id"]),
                intended_at=parse_dt(r["intended_at"]),
                market_ticker=r["market_ticker"],
                side=r["side"],
                contracts=int(r["contracts"]),
                strategy=r["strategy"],
                outcome=r["outcome"],
            )
        )
    return out


def fetch_forecast(
    conn: sqlite3.Connection,
    station: str,
    valid_date: date,
    not_after: datetime,
) -> ForecastRow | None:
    row = conn.execute(
        """
        SELECT station, run_time, valid_date, members_json
          FROM forecasts
         WHERE station = ? AND valid_date = ? AND run_time <= ?
         ORDER BY run_time DESC
         LIMIT 1
        """,
        (station, valid_date.isoformat(), not_after.isoformat(sep=" ")),
    ).fetchone()
    if row is None:
        return None
    members = np.asarray(json.loads(row["members_json"]), dtype=np.float64)
    return ForecastRow(
        station=row["station"],
        run_time=parse_dt(row["run_time"]),
        valid_date=date.fromisoformat(row["valid_date"]),
        members=members,
    )


def fetch_book(
    conn: sqlite3.Connection,
    ticker: str,
    not_after: datetime,
) -> BookRow | None:
    row = conn.execute(
        """
        SELECT snapshot_at, yes_ask, yes_bid, no_ask, no_bid,
               yes_ask_depth, yes_bid_depth, no_ask_depth, no_bid_depth
          FROM orderbook_snapshots
         WHERE ticker = ? AND snapshot_at <= ?
         ORDER BY snapshot_at DESC
         LIMIT 1
        """,
        (ticker, not_after.isoformat(sep=" ")),
    ).fetchone()
    if row is None:
        return None
    return BookRow(
        snapshot_at=parse_dt(row["snapshot_at"]),
        yes_ask=Decimal(str(row["yes_ask"])),
        yes_bid=Decimal(str(row["yes_bid"])),
        no_ask=Decimal(str(row["no_ask"])),
        no_bid=Decimal(str(row["no_bid"])),
        yes_ask_depth=int(row["yes_ask_depth"] or 0),
        yes_bid_depth=int(row["yes_bid_depth"] or 0),
        no_ask_depth=int(row["no_ask_depth"] or 0),
        no_bid_depth=int(row["no_bid_depth"] or 0),
    )


def fetch_market_close(conn: sqlite3.Connection, ticker: str) -> datetime | None:
    row = conn.execute(
        "SELECT close_time FROM markets WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    if row is None or row["close_time"] is None:
        return None
    return parse_dt(row["close_time"])


def run_edge(
    *,
    fair_yes: Decimal,
    book: BookRow,
    spread: Decimal,
    sigma_T_median: Decimal,
    is_blacklisted: bool,
) -> edge_strategy.EdgeSignal:
    if fair_yes > book.yes_ask + edge_strategy.DIRECTION_CUSHION:
        depth = book.no_bid_depth
        price_per_contract = book.yes_ask
    elif fair_yes < book.yes_bid - edge_strategy.DIRECTION_CUSHION:
        depth = book.yes_bid_depth
        price_per_contract = Decimal("1") - book.yes_bid
    else:
        depth = book.no_bid_depth
        price_per_contract = book.yes_ask
    ctx = edge_strategy.EdgeContext(
        yes_ask=book.yes_ask,
        yes_bid=book.yes_bid,
        fair_yes=fair_yes,
        ensemble_spread=spread,
        bankroll=PAPER_BANKROLL,
        is_same_day=False,
        is_blacklisted=is_blacklisted,
        nbm_divergence=None,
        sigma_T_median=sigma_T_median,
        event_budget_remaining=EVENT_POSITION_CAP,
        market_budget_remaining=MARKET_POSITION_CAP,
        depth_at_price=depth,
        price_per_contract=price_per_contract,
    )
    return edge_strategy.evaluate(ctx, mode="paper")


def run_tails(
    *,
    fair_yes: Decimal,
    book: BookRow,
    close_time: datetime,
    now: datetime,
    spread: Decimal,
    sigma_T_median: Decimal,
) -> tails_strategy.TailsSignal:
    price_per_contract = Decimal("1") - book.yes_bid
    ctx = tails_strategy.TailsContext(
        yes_ask=book.yes_ask,
        yes_bid=book.yes_bid,
        no_bid=book.no_bid,
        fair_yes=fair_yes,
        close_time=close_time,
        now=now,
        bankroll=PAPER_BANKROLL,
        is_same_day=False,
        ensemble_spread=spread,
        sigma_T_median=sigma_T_median,
        event_budget_remaining=EVENT_POSITION_CAP,
        market_budget_remaining=MARKET_POSITION_CAP,
        depth_at_price=book.yes_bid_depth,
        price_per_contract=price_per_contract,
    )
    return tails_strategy.evaluate(ctx, mode="paper")


def seed_overlays(
    ledger: Sequence[LedgerFill],
    as_of: datetime,
) -> tuple[dict[str, Decimal], dict[str, Decimal], dict[str, Decimal], Decimal]:
    by_market: dict[str, Decimal] = {}
    by_event: dict[str, Decimal] = {}
    by_series: dict[str, Decimal] = {}
    for fill in ledger:
        parsed = parse_ticker(fill.market_ticker)
        expiry = _effective_cutoff_date(parsed) + timedelta(days=SETTLEMENT_GRACE_DAYS)
        if expiry <= as_of.date():
            continue
        event_key = event_id(fill.market_ticker)
        by_market[fill.market_ticker] = by_market.get(fill.market_ticker, Decimal("0")) + fill.delta
        by_event[event_key] = by_event.get(event_key, Decimal("0")) + fill.delta
        by_series[parsed.series] = by_series.get(parsed.series, Decimal("0")) + fill.delta
    aggregate = sum(by_market.values(), Decimal("0"))
    return by_market, by_event, by_series, aggregate


def build_gate_ctx(
    *,
    intent: TradeIntent,
    fair_yes: Decimal,
    spread: Decimal,
    run_time: datetime,
    now: datetime,
    book: BookRow | CanonicalSnapshot,
    close_time: datetime | None,
    buy_yes_depth: int,
    sell_yes_depth: int,
    market_existing_dollars: Decimal,
    market_position_cap: Decimal,
    event_existing_dollars: Decimal,
    event_position_cap: Decimal,
    series_existing_dollars: Decimal,
    series_position_cap: Decimal,
    aggregate_existing_dollars: Decimal,
    aggregate_exposure_cap: Decimal,
    account_balance: Decimal,
    required_cushion: Decimal,
    market_status: str = "active",
) -> GateContext:
    if intent.side is TradeSide.BUY_YES:
        edge_dollars = fair_yes - book.yes_ask
        price = book.yes_ask
        depth_at_price = buy_yes_depth
    else:
        edge_dollars = book.yes_bid - fair_yes
        price = Decimal("1") - book.yes_bid
        depth_at_price = sell_yes_depth
    order_dollars = price * Decimal(intent.contracts)
    minutes_to_close = 99999
    if close_time is not None:
        delta = (close_time - now).total_seconds()
        minutes_to_close = max(0, int(delta // 60))
    model_age_hours = Decimal(str((now - run_time).total_seconds() / 3600))
    return GateContext(
        fair_yes=fair_yes,
        model_age_hours=model_age_hours,
        ensemble_spread=spread,
        edge=edge_dollars,
        price=price,
        depth_at_price=depth_at_price,
        contracts=intent.contracts,
        order_size_dollars=order_dollars,
        market_existing_dollars=market_existing_dollars,
        market_position_cap=market_position_cap,
        event_existing_dollars=event_existing_dollars,
        event_position_cap=event_position_cap,
        series_existing_dollars=series_existing_dollars,
        series_position_cap=series_position_cap,
        aggregate_existing_dollars=aggregate_existing_dollars,
        aggregate_exposure_cap=aggregate_exposure_cap,
        account_balance=account_balance,
        required_cushion=required_cushion,
        market_status=market_status,
        minutes_to_close=minutes_to_close,
        circuit_breakers_armed=True,
    )


def replay(
    conn: sqlite3.Connection,
    cutoff: datetime,
    strategy_filter: str,
) -> tuple[dict[str, Bucket], int, int]:
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    market_close_cache: dict[str, datetime | None] = {}

    trades = fetch_trades(conn, cutoff)
    if strategy_filter != "all":
        trades = [t for t in trades if t.strategy == strategy_filter]
    n_total = len(trades)
    n_missing = 0

    ledger: list[LedgerFill] = []
    overlay_market: dict[str, Decimal] = {}
    overlay_event: dict[str, Decimal] = {}
    overlay_series: dict[str, Decimal] = {}
    aggregate = Decimal("0")
    cycle_minute: datetime | None = None

    for t in trades:
        try:
            parsed = parse_ticker(t.market_ticker)
        except ValueError:
            n_missing += 1
            continue
        cfg = STATIONS.get(parsed.series)
        if cfg is None:
            n_missing += 1
            continue

        minute = t.intended_at.replace(second=0, microsecond=0)
        if minute != cycle_minute:
            cycle_minute = minute
            overlay_market, overlay_event, overlay_series, aggregate = seed_overlays(
                ledger, t.intended_at
            )

        forecast = fetch_forecast(conn, cfg.station, parsed.event_date, t.intended_at)
        book = fetch_book(conn, t.market_ticker, t.intended_at)
        if forecast is None or book is None:
            n_missing += 1
            continue
        if t.market_ticker not in market_close_cache:
            market_close_cache[t.market_ticker] = fetch_market_close(conn, t.market_ticker)
        close_time = market_close_cache[t.market_ticker]

        cdf = EnsembleCDF.from_members(forecast.members, smoothing=1.0)
        spread = Decimal(str(float(forecast.members.std())))
        fair_yes = fair_yes_for(parsed, cdf)

        lead_hours = 0
        if close_time is not None:
            lead_hours = int((close_time - t.intended_at).total_seconds() / 3600)
        sigma_T_median = sigma_t_median_for_lead(lead_hours)

        bucket = buckets[t.strategy]
        bucket.n_replayed += 1
        bucket.historical_size_sum += t.contracts

        skip_reason: str | None = None
        intent: TradeIntent | None = None

        if t.strategy == "edge":
            if parsed.is_tail:
                skip_reason = "strategy:routed_to_tails_now"
            else:
                sig = run_edge(
                    fair_yes=fair_yes,
                    book=book,
                    spread=spread,
                    sigma_T_median=sigma_T_median,
                    is_blacklisted=parsed.series in STRATEGY_BLACKLIST,
                )
                if sig.action is edge_strategy.EdgeAction.SKIP:
                    skip_reason = f"strategy:{sig.reason}"
                else:
                    side = (
                        TradeSide.BUY_YES
                        if sig.action is edge_strategy.EdgeAction.BUY_YES
                        else TradeSide.SELL_YES
                    )
                    intent = TradeIntent(
                        market_ticker=t.market_ticker,
                        side=side,
                        contracts=sig.contracts,
                        fair_yes=fair_yes,
                        q_raw=fair_yes,
                        strategy="edge",
                        ensemble_spread_sigma_t=spread,
                        lead_time_hours=Decimal(str(lead_hours)),
                        nbm_divergence=None,
                    )
        elif t.strategy == "tails":
            if not parsed.is_tail:
                skip_reason = "strategy:non_tail_market"
            elif parsed.series in STRATEGY_BLACKLIST:
                skip_reason = "routing:tails_not_invoked_blacklisted"
            elif close_time is None:
                skip_reason = "strategy:no_close_time"
            else:
                sig_t = run_tails(
                    fair_yes=fair_yes,
                    book=book,
                    close_time=close_time,
                    now=t.intended_at,
                    spread=spread,
                    sigma_T_median=sigma_T_median,
                )
                if sig_t.action is tails_strategy.TailsAction.SKIP:
                    skip_reason = f"strategy:{sig_t.reason}"
                else:
                    intent = TradeIntent(
                        market_ticker=t.market_ticker,
                        side=TradeSide.SELL_YES,
                        contracts=sig_t.contracts,
                        fair_yes=fair_yes,
                        q_raw=fair_yes,
                        strategy="tails",
                        ensemble_spread_sigma_t=spread,
                        lead_time_hours=Decimal(str(lead_hours)),
                        nbm_divergence=None,
                    )
        else:
            skip_reason = "strategy:unknown"

        if intent is not None:
            event_key = event_id(t.market_ticker)
            gate_ctx = build_gate_ctx(
                intent=intent,
                fair_yes=fair_yes,
                spread=spread,
                run_time=forecast.run_time,
                now=t.intended_at,
                book=book,
                close_time=close_time,
                buy_yes_depth=book.no_bid_depth,
                sell_yes_depth=book.yes_bid_depth,
                market_existing_dollars=overlay_market.get(t.market_ticker, Decimal("0")),
                market_position_cap=MARKET_POSITION_CAP,
                event_existing_dollars=overlay_event.get(event_key, Decimal("0")),
                event_position_cap=EVENT_POSITION_CAP,
                series_existing_dollars=overlay_series.get(parsed.series, Decimal("0")),
                series_position_cap=SERIES_POSITION_CAP,
                aggregate_existing_dollars=aggregate,
                aggregate_exposure_cap=AGGREGATE_EXPOSURE_CAP,
                account_balance=PAPER_BANKROLL,
                required_cushion=REQUIRED_CUSHION,
            )
            check = evaluate_gates(gate_ctx, GateMode.PAPER)
            cap_failure = next((f for f in check.failures if f.name in CAP_GATE_NAMES), None)
            if cap_failure is not None:
                skip_reason = f"gate:{cap_failure.name}"
            elif not check.overall_passed:
                first = check.failures[0]
                skip_reason = f"gate:{first.name}"
            else:
                delta = paper_collateral_per_contract(intent.side, book) * Decimal(intent.contracts)
                ledger.append(
                    LedgerFill(
                        market_ticker=t.market_ticker,
                        side=intent.side,
                        contracts=intent.contracts,
                        delta=delta,
                    )
                )
                overlay_market[t.market_ticker] = (
                    overlay_market.get(t.market_ticker, Decimal("0")) + delta
                )
                overlay_event[event_key] = overlay_event.get(event_key, Decimal("0")) + delta
                overlay_series[parsed.series] = (
                    overlay_series.get(parsed.series, Decimal("0")) + delta
                )
                aggregate = aggregate + delta
                bucket.n_current_would_place += 1
                bucket.placed_size_sum += intent.contracts
                bucket.placed_total_outcome += 1
                if t.outcome == "won":
                    bucket.placed_wins += 1
                continue

        bucket.n_current_would_skip += 1
        bucket.skipped_total_outcome += 1
        if t.outcome == "won":
            bucket.skipped_wins += 1
        if skip_reason is not None:
            bucket.skip_reasons[skip_reason] += 1

    return buckets, n_total, n_missing


async def run_replay(
    snapshots: Iterable[ReplaySnapshot],
    replay: ForecastReplay,
    strategy_name: str,
    config: BacktestConfig,
    *,
    log: ReplayLog | None = None,
) -> list[BacktestOrder]:
    if strategy_name not in ("edge", "tails"):
        raise ValueError(f"unknown strategy {strategy_name!r}")
    if log is None:
        log = ReplayLog()
    market_cap = config.bankroll * MARKET_POSITION_FRAC
    event_cap = config.bankroll * EVENT_POSITION_FRAC
    series_cap = config.bankroll * SERIES_POSITION_FRAC
    aggregate_cap = config.bankroll * AGGREGATE_EXPOSURE_FRAC
    budgets = BacktestBudgets(
        event_budget_remaining=event_cap,
        market_budget_remaining=market_cap,
    )

    decisions: dict[tuple[str, datetime], ReplaySnapshot] = {}
    for row in snapshots:
        close_time = row.snap.close_time
        if close_time is None:
            continue
        if row.snapshot_at > close_time - config.lead:
            continue
        key = (row.snap.ticker, close_time)
        best = decisions.get(key)
        if best is None or row.snapshot_at > best.snapshot_at:
            decisions[key] = row

    ordered = sorted(decisions.items(), key=lambda kv: (kv[0][1], kv[0][0]))

    orders: list[BacktestOrder] = []
    ledger: list[LedgerFill] = []
    overlay_market: dict[str, Decimal] = {}
    overlay_event: dict[str, Decimal] = {}
    overlay_series: dict[str, Decimal] = {}
    aggregate = Decimal("0")
    cycle_minute: datetime | None = None

    for (ticker, close_time), row in ordered:
        as_of = close_time - config.lead
        minute = as_of.replace(second=0, microsecond=0)
        if minute != cycle_minute:
            cycle_minute = minute
            overlay_market, overlay_event, overlay_series, aggregate = seed_overlays(ledger, as_of)
            log.cycles.append(
                CycleRecord(
                    minute=minute,
                    market=dict(overlay_market),
                    event=dict(overlay_event),
                    series=dict(overlay_series),
                    aggregate=aggregate,
                )
            )

        try:
            parsed = parse_ticker(ticker)
        except ValueError:
            continue
        station = config.stations.get(parsed.series)
        if station is None:
            continue

        is_blacklisted = parsed.series in STRATEGY_BLACKLIST
        if strategy_name == "tails":
            if not parsed.is_tail:
                log.failures.append(
                    ReplayFailure(
                        market_ticker=ticker,
                        as_of=as_of,
                        strategy=strategy_name,
                        layer="routing",
                        name="tails_not_invoked_non_tail",
                        reason="tails_not_invoked_non_tail",
                    )
                )
                continue
            if is_blacklisted:
                log.failures.append(
                    ReplayFailure(
                        market_ticker=ticker,
                        as_of=as_of,
                        strategy=strategy_name,
                        layer="routing",
                        name="tails_not_invoked_blacklisted",
                        reason="tails_not_invoked_blacklisted",
                    )
                )
                continue
        elif parsed.is_tail:
            log.failures.append(
                ReplayFailure(
                    market_ticker=ticker,
                    as_of=as_of,
                    strategy=strategy_name,
                    layer="routing",
                    name="edge_not_invoked_tail",
                    reason="edge_not_invoked_tail",
                )
            )
            continue

        snap = row.snap
        cdf = await replay.replay(station, parsed.event_date, as_of)
        spread = Decimal(str(float(cdf.members.std())))
        fair_yes = fair_yes_for(parsed, cdf)

        lead_bucket = lead_bucket_for(close_time - row.snapshot_at)
        if snap.yes_bid_size is not None and snap.no_bid_size is not None:
            buy_yes_depth = int(snap.no_bid_size)
            sell_yes_depth = int(snap.yes_bid_size)
            depth_source = "snapshot"
        else:
            nominal = config.depth_table.get((parsed.series, lead_bucket))
            depth_source = "series_lead_median"
            if nominal is None:
                nominal = config.depth_table[(ALL_SERIES, lead_bucket)]
                depth_source = "all_series_lead_median"
            buy_yes_depth = int(nominal)
            sell_yes_depth = int(nominal)

        if strategy_name == "tails":
            tails_ctx = build_tails_context(
                snap,
                cdf,
                as_of,
                config.bankroll,
                budgets,
                ensemble_spread=spread,
                depth_at_price=sell_yes_depth,
                station_tz=station.timezone,
            )
            tails_sig = tails_strategy.evaluate(tails_ctx, mode="paper")
            if tails_sig.action is tails_strategy.TailsAction.SKIP:
                log.failures.append(
                    ReplayFailure(
                        market_ticker=ticker,
                        as_of=as_of,
                        strategy=strategy_name,
                        layer="strategy",
                        name=tails_sig.reason,
                        reason=tails_sig.reason,
                    )
                )
                continue
            side = TradeSide.SELL_YES
            contracts = tails_sig.contracts
        else:
            edge_ctx = build_edge_context(
                snap,
                cdf,
                as_of,
                config.bankroll,
                budgets,
                ensemble_spread=spread,
                buy_yes_depth=buy_yes_depth,
                sell_yes_depth=sell_yes_depth,
                station_tz=station.timezone,
                is_blacklisted=is_blacklisted,
            )
            edge_sig = edge_strategy.evaluate(edge_ctx, mode="paper")
            if edge_sig.action is edge_strategy.EdgeAction.SKIP:
                log.failures.append(
                    ReplayFailure(
                        market_ticker=ticker,
                        as_of=as_of,
                        strategy=strategy_name,
                        layer="strategy",
                        name=edge_sig.reason,
                        reason=edge_sig.reason,
                    )
                )
                continue
            side = (
                TradeSide.BUY_YES
                if edge_sig.action is edge_strategy.EdgeAction.BUY_YES
                else TradeSide.SELL_YES
            )
            contracts = edge_sig.contracts

        intent = TradeIntent(
            market_ticker=ticker,
            side=side,
            contracts=contracts,
            fair_yes=fair_yes,
            q_raw=fair_yes,
            strategy=strategy_name,
            ensemble_spread_sigma_t=spread,
            lead_time_hours=Decimal(str((close_time - as_of).total_seconds() / 3600)),
            nbm_divergence=None,
        )
        event_key = event_id(ticker)
        gate_ctx = build_gate_ctx(
            intent=intent,
            fair_yes=fair_yes,
            spread=spread,
            run_time=pick_cycle(as_of),
            now=as_of,
            book=snap,
            close_time=close_time,
            buy_yes_depth=buy_yes_depth,
            sell_yes_depth=sell_yes_depth,
            market_existing_dollars=overlay_market.get(ticker, Decimal("0")),
            market_position_cap=market_cap,
            event_existing_dollars=overlay_event.get(event_key, Decimal("0")),
            event_position_cap=event_cap,
            series_existing_dollars=overlay_series.get(parsed.series, Decimal("0")),
            series_position_cap=series_cap,
            aggregate_existing_dollars=aggregate,
            aggregate_exposure_cap=aggregate_cap,
            account_balance=config.bankroll,
            required_cushion=config.required_cushion,
            market_status=snap.status,
        )
        check = evaluate_gates(gate_ctx, GateMode.PAPER)
        for failure in check.failures:
            log.failures.append(
                ReplayFailure(
                    market_ticker=ticker,
                    as_of=as_of,
                    strategy=strategy_name,
                    layer="gate",
                    name=failure.name,
                    reason=failure.reason or "",
                )
            )
        if any(f.name in CAP_GATE_NAMES for f in check.failures):
            continue
        if not check.overall_passed:
            continue

        delta = paper_collateral_per_contract(side, snap) * Decimal(contracts)
        orders.append(
            BacktestOrder(
                market_ticker=ticker,
                as_of=as_of,
                strategy=strategy_name,
                action=side,
                contracts=contracts,
                fair_yes=fair_yes,
                price_per_contract=(
                    snap.yes_ask if side is TradeSide.BUY_YES else Decimal("1") - snap.yes_bid
                ),
                order_dollars=delta,
                depth_at_price=buy_yes_depth if side is TradeSide.BUY_YES else sell_yes_depth,
                depth_source=depth_source,
                lead_bucket=lead_bucket,
            )
        )
        ledger.append(LedgerFill(market_ticker=ticker, side=side, contracts=contracts, delta=delta))
        overlay_market[ticker] = overlay_market.get(ticker, Decimal("0")) + delta
        overlay_event[event_key] = overlay_event.get(event_key, Decimal("0")) + delta
        overlay_series[parsed.series] = overlay_series.get(parsed.series, Decimal("0")) + delta
        aggregate = aggregate + delta
    return orders
