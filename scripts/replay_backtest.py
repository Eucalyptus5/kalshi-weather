from __future__ import annotations

import json
import logging
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.execution.paper import TradeIntent, TradeSide  # noqa: E402
from bot.forecast.cdf import EnsembleCDF  # noqa: E402
from bot.markets.parser import parse_ticker  # noqa: E402
from bot.risk.gates import (  # noqa: E402
    CAP_GATE_NAMES,
    GateContext,
    GateMode,
    evaluate as evaluate_gates,
)
from bot.strategy import edge as edge_strategy  # noqa: E402
from bot.strategy import tails as tails_strategy  # noqa: E402
from bot.strategy.sizing import sigma_t_median_for_lead  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "state.db"
REPORT_PATH = Path("/tmp/replay_backtest_report.txt")
REGIME_CUTOFF = datetime(2026, 5, 18, 0, 0, 0, tzinfo=_timezone.utc)

PAPER_BANKROLL: Decimal = Decimal("500")
REQUIRED_CUSHION: Decimal = Decimal("100")
MARKET_POSITION_CAP: Decimal = PAPER_BANKROLL * Decimal("0.015")
EVENT_POSITION_CAP: Decimal = PAPER_BANKROLL * Decimal("0.03")
SERIES_POSITION_CAP: Decimal = PAPER_BANKROLL * Decimal("0.05")
AGGREGATE_EXPOSURE_CAP: Decimal = PAPER_BANKROLL * Decimal("0.40")

SERIES_TO_STATION: dict[str, str] = {
    "KXHIGHDEN": "KDEN",
    "KXHIGHAUS": "KAUS",
    "KXHIGHCHI": "KMDW",
    "KXHIGHNY": "KNYC",
    "KXHIGHPHIL": "KPHL",
    "KXHIGHTATL": "KATL",
    "KXHIGHTBOS": "KBOS",
    "KXHIGHTDAL": "KDFW",
    "KXHIGHTDC": "KDCA",
    "KXHIGHTHOU": "KIAH",
    "KXHIGHTLV": "KLAS",
    "KXHIGHTMIN": "KMSP",
    "KXHIGHTNOLA": "KMSY",
    "KXHIGHTOKC": "KOKC",
    "KXHIGHTPHX": "KPHX",
    "KXHIGHTSATX": "KSAT",
    "KXHIGHTSEA": "KSEA",
    "KXHIGHTSFO": "KSFO",
    "KXHIGHLAX": "KLAX",
    "KXHIGHMIA": "KMIA",
}
STRATEGY_BLACKLIST: frozenset[str] = frozenset({"KXHIGHLAX", "KXHIGHMIA"})


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
    skip_reasons: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skip_reasons is None:
            self.skip_reasons = Counter()


def _parse_dt(raw: str) -> datetime:
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_timezone.utc)
    return dt


def _open_readonly() -> sqlite3.Connection:
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def fetch_trades(conn: sqlite3.Connection) -> list[TradeRow]:
    rows = conn.execute(
        """
        SELECT p.id, p.intended_at, p.market_ticker, p.side, p.contracts,
               p.strategy, s.outcome
          FROM paper_trades p
          JOIN simulated_pnl s ON s.paper_trade_id = p.id
         WHERE p.intended_at >= ?
         ORDER BY p.intended_at ASC
        """,
        (REGIME_CUTOFF.isoformat(sep=" "),),
    ).fetchall()
    out: list[TradeRow] = []
    for r in rows:
        out.append(
            TradeRow(
                pt_id=int(r["id"]),
                intended_at=_parse_dt(r["intended_at"]),
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
        run_time=_parse_dt(row["run_time"]),
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
        snapshot_at=_parse_dt(row["snapshot_at"]),
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
    return _parse_dt(row["close_time"])


def fair_yes_for(parsed, cdf: EnsembleCDF) -> Decimal:
    if parsed.kind == "bracket":
        lo = float(parsed.strikes[0])
        hi = float(parsed.strikes[1])
        return Decimal(str(cdf.prob_range(lo, hi)))
    if parsed.kind == "above":
        return Decimal(str(1.0 - cdf.cdf(float(parsed.strikes[0]))))
    return Decimal(str(cdf.cdf(float(parsed.strikes[0]))))


def run_edge(
    *,
    fair_yes: Decimal,
    book: BookRow,
    spread: Decimal,
    sigma_T_median: Decimal,
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
        is_blacklisted=False,
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


def build_gate_ctx(
    *,
    intent: TradeIntent,
    fair_yes: Decimal,
    spread: Decimal,
    run_time: datetime,
    now: datetime,
    book: BookRow,
    close_time: datetime | None,
) -> GateContext:
    if intent.side is TradeSide.BUY_YES:
        edge_dollars = fair_yes - book.yes_ask
        price = book.yes_ask
        depth_at_price = book.no_bid_depth
    else:
        edge_dollars = book.yes_bid - fair_yes
        price = Decimal("1") - book.yes_bid
        depth_at_price = book.yes_bid_depth
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
        market_existing_dollars=Decimal("0"),
        market_position_cap=MARKET_POSITION_CAP,
        event_existing_dollars=Decimal("0"),
        event_position_cap=EVENT_POSITION_CAP,
        series_existing_dollars=Decimal("0"),
        series_position_cap=SERIES_POSITION_CAP,
        aggregate_existing_dollars=Decimal("0"),
        aggregate_exposure_cap=AGGREGATE_EXPOSURE_CAP,
        account_balance=PAPER_BANKROLL,
        required_cushion=REQUIRED_CUSHION,
        market_status="active",
        minutes_to_close=minutes_to_close,
        circuit_breakers_armed=True,
    )


def replay(conn: sqlite3.Connection) -> tuple[dict[str, Bucket], int, int]:
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    market_close_cache: dict[str, datetime | None] = {}

    trades = fetch_trades(conn)
    n_total = len(trades)
    n_missing = 0

    for t in trades:
        try:
            parsed = parse_ticker(t.market_ticker)
        except ValueError:
            n_missing += 1
            continue
        station = SERIES_TO_STATION.get(parsed.series)
        if station is None:
            n_missing += 1
            continue
        if parsed.series in STRATEGY_BLACKLIST:
            continue

        forecast = fetch_forecast(conn, station, parsed.event_date, t.intended_at)
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
            gate_ctx = build_gate_ctx(
                intent=intent,
                fair_yes=fair_yes,
                spread=spread,
                run_time=forecast.run_time,
                now=t.intended_at,
                book=book,
                close_time=close_time,
            )
            check = evaluate_gates(gate_ctx, GateMode.PAPER)
            cap_failure = next((f for f in check.failures if f.name in CAP_GATE_NAMES), None)
            if cap_failure is not None:
                skip_reason = f"gate:{cap_failure.name}"
            elif not check.overall_passed:
                first = check.failures[0]
                skip_reason = f"gate:{first.name}"
            else:
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


def _pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "n/a"
    return f"{(num / denom) * 100:.2f}%"


def render_report(buckets: dict[str, Bucket], n_total: int, n_missing: int) -> str:
    lines: list[str] = []
    now = datetime.now(tz=_timezone.utc).isoformat(timespec="seconds")
    lines.append(f"# strategy replay backtest report\tgenerated_at={now}")
    lines.append(f"# regime_filter\tintended_at >= {REGIME_CUTOFF.isoformat()}")
    lines.append(f"# rows_considered\t{n_total}")
    lines.append(f"# missing_inputs\t{n_missing}")
    lines.append("# regime_label\tpost_floor (single bucket; pre-floor excluded)")
    lines.append(
        "# caveat\thistorical realized_pnl dollar amounts not used; "
        "only simulated_pnl.outcome win/loss"
    )
    lines.append(
        "# caveat\tGateContext uses neutral overlay defaults (zero existing exposure); "
        "absolute pass rates are approximate but place-vs-skip comparison is sound"
    )
    lines.append(
        "# caveat\tcalibration brief 07 Commit B not in bet path; raw_q is the model output"
    )
    lines.append("")
    for strategy in sorted(buckets.keys()):
        b = buckets[strategy]
        lines.append(f"== bucket\tstrategy={strategy}\tregime=post_floor")
        lines.append(f"n_replayed\t{b.n_replayed}")
        lines.append(f"n_current_would_place\t{b.n_current_would_place}")
        lines.append(f"n_current_would_skip\t{b.n_current_would_skip}")
        lines.append(
            "win_rate_would_place\t"
            f"{_pct(b.placed_wins, b.placed_total_outcome)}\t"
            f"(n={b.placed_total_outcome})"
        )
        lines.append(
            "win_rate_would_skip\t"
            f"{_pct(b.skipped_wins, b.skipped_total_outcome)}\t"
            f"(n={b.skipped_total_outcome})"
        )
        avg_place = (b.placed_size_sum / b.n_current_would_place) if b.n_current_would_place else 0
        avg_hist = (b.historical_size_sum / b.n_replayed) if b.n_replayed else 0
        lines.append(f"avg_size_would_place_contracts\t{avg_place:.2f}")
        lines.append(f"avg_size_historical_contracts\t{avg_hist:.2f}")
        top3 = b.skip_reasons.most_common(3)
        for i, (reason, count) in enumerate(top3, start=1):
            lines.append(f"top_skip_reason_{i}\t{reason}\t{count}")
        if not top3:
            lines.append("top_skip_reason_1\t(none)\t0")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.disable(logging.CRITICAL)
    if not DB_PATH.exists():
        sys.stderr.write(f"db not found: {DB_PATH}\n")
        return 2
    conn = _open_readonly()
    try:
        buckets, n_total, n_missing = replay(conn)
    finally:
        conn.close()
    report = render_report(buckets, n_total, n_missing)
    REPORT_PATH.write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
