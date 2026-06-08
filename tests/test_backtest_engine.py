from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from bot.backtest.engine import (
    BacktestConfig,
    BookRow,
    LedgerFill,
    ReplayLog,
    ReplaySnapshot,
    run_replay,
    seed_overlays,
)
from bot.backtest.forecast_replay import StationSpec
from bot.backtest.normalize import CanonicalSnapshot
from bot.execution.gate_cost_basis import paper_collateral_per_contract
from bot.execution.paper import TradeSide
from bot.forecast.cdf import EnsembleCDF
from bot.markets.parser import parse_ticker
from bot.risk.gates import CAP_GATE_NAMES
from bot.storage.positions import SETTLEMENT_GRACE_DAYS, _effective_cutoff_date
from bot.strategy import edge as edge_strategy
from bot.strategy import tails as tails_strategy

_LEAD = timedelta(hours=25)
_BANKROLL = Decimal("20000")
_MEMBERS = np.array([59.5, 61.5, 63.5])
_DAY_8_AS_OF = datetime(2025, 11, 13, 17, 0, tzinfo=timezone.utc)
_DAY_9_AS_OF = datetime(2025, 11, 14, 17, 0, tzinfo=timezone.utc)
_MONTHLY_TICKER = "KXHIGHDEN-25NOV-B61.5"
_MONTHLY_CLOSE = datetime(2025, 11, 5, 22, 0, tzinfo=timezone.utc)
_AGREEMENT_TICKER = "KXHIGHDEN-25NOV05-B61.5"
_AGREEMENT_CLOSE = datetime(2025, 11, 5, 23, 0, tzinfo=timezone.utc)


class FixedReplay:
    def __init__(self, members: np.ndarray = _MEMBERS, smoothing: float = 0.65) -> None:
        self._cdf = EnsembleCDF.from_members(members, smoothing=smoothing)

    async def replay(self, station: StationSpec, valid_date: date, as_of: datetime) -> EnsembleCDF:
        return self._cdf


def make_config(**overrides: object) -> BacktestConfig:
    fields: dict[str, object] = {"lead": _LEAD, "bankroll": _BANKROLL}
    fields.update(overrides)
    return BacktestConfig(**fields)


def make_row(
    ticker: str,
    close_time: datetime,
    *,
    yes_ask: Decimal,
    yes_bid: Decimal,
    depth: int | None = 3000,
) -> ReplaySnapshot:
    size = None if depth is None else Decimal(depth)
    snap = CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-", 1)[0],
        status="active",
        result="",
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=Decimal("1") - yes_bid,
        no_bid=Decimal("1") - yes_ask,
        last_price=yes_bid,
        volume=Decimal(0),
        volume_24h=Decimal(0),
        open_interest=Decimal(0),
        close_time=close_time,
        yes_bid_size=size,
        no_bid_size=size,
    )
    return ReplaySnapshot(snapshot_at=close_time - _LEAD - timedelta(minutes=1), snap=snap)


def make_two_market_run() -> list[ReplaySnapshot]:
    close = datetime(2026, 1, 16, 6, 0, tzinfo=timezone.utc)
    market_a = make_row(
        "KXHIGHDEN-26JAN15-B61.5", close, yes_ask=Decimal("0.10"), yes_bid=Decimal("0.08")
    )
    market_b = make_row(
        "KXHIGHDEN-26JAN15-B63.5", close, yes_ask=Decimal("0.20"), yes_bid=Decimal("0.16")
    )
    return [market_a, market_b]


def make_cap_bind_ladder() -> list[ReplaySnapshot]:
    ticker = "KXHIGHDEN-26JAN16-B61.5"
    base_close = datetime(2026, 1, 17, 6, 0, tzinfo=timezone.utc)
    rungs = [
        make_row(
            ticker,
            base_close + timedelta(minutes=i),
            yes_ask=Decimal("0.10"),
            yes_bid=Decimal("0.08"),
        )
        for i in range(3)
    ]
    rungs.append(
        make_row(
            ticker,
            base_close + timedelta(minutes=3),
            yes_ask=Decimal("0.22"),
            yes_bid=Decimal("0.18"),
        )
    )
    return rungs


def make_series_cap_run() -> list[ReplaySnapshot]:
    return [
        make_row(
            f"KXHIGHDEN-26JAN{16 + i}-B61.5",
            datetime(2026, 1, 17 + i, 6, 0, tzinfo=timezone.utc),
            yes_ask=Decimal("0.10"),
            yes_bid=Decimal("0.08"),
            depth=6000,
        )
        for i in range(5)
    ]


def make_sell_fill(ticker: str, yes_bid: Decimal, contracts: int) -> LedgerFill:
    book = BookRow(
        snapshot_at=datetime(2025, 11, 5, tzinfo=timezone.utc),
        yes_ask=yes_bid + Decimal("0.02"),
        yes_bid=yes_bid,
        no_ask=Decimal("1") - yes_bid,
        no_bid=Decimal("0.98") - yes_bid,
        yes_ask_depth=100,
        yes_bid_depth=100,
        no_ask_depth=100,
        no_bid_depth=100,
    )
    delta = paper_collateral_per_contract(TradeSide.SELL_YES, book) * Decimal(contracts)
    return LedgerFill(
        market_ticker=ticker, side=TradeSide.SELL_YES, contracts=contracts, delta=delta
    )


def make_nine_day_ledger() -> tuple[list[LedgerFill], LedgerFill, LedgerFill]:
    daily = [
        make_sell_fill(f"KXHIGHDEN-25NOV{6 + n:02d}-B61.5", yes_bid=Decimal("0.12"), contracts=25)
        for n in range(1, 10)
    ]
    monthly = make_sell_fill(_MONTHLY_TICKER, yes_bid=Decimal("0.10"), contracts=10)
    agreement = make_sell_fill(_AGREEMENT_TICKER, yes_bid=Decimal("0.15"), contracts=20)
    return daily, monthly, agreement


async def test_two_market_run_fills_a_and_skips_b_on_edge_too_small() -> None:
    log = ReplayLog()

    orders = await run_replay(make_two_market_run(), FixedReplay(), "edge", make_config(), log=log)

    assert len(orders) == 1
    assert orders[0].market_ticker == "KXHIGHDEN-26JAN15-B61.5"
    assert orders[0].action is TradeSide.BUY_YES
    assert orders[0].contracts >= 1
    assert orders[0].depth_source == "snapshot"
    assert not [f for f in log.failures if f.name in CAP_GATE_NAMES]
    skips = [f for f in log.failures if f.name == "edge_too_small"]
    assert len(skips) == 1
    assert skips[0].market_ticker == "KXHIGHDEN-26JAN15-B63.5"
    assert skips[0].layer == "strategy"
    assert skips[0].reason == "edge_too_small"


async def test_cap_bind_ladder_market_cap_binds_on_third_rung() -> None:
    log = ReplayLog()

    orders = await run_replay(make_cap_bind_ladder(), FixedReplay(), "edge", make_config(), log=log)

    assert len(orders) == 2
    assert [o.order_dollars for o in orders] == [Decimal("150.00"), Decimal("150.00")]
    market_cap_skips = [f for f in log.failures if f.name == "within_market_cap"]
    assert len(market_cap_skips) == 1
    assert market_cap_skips[0].market_ticker == "KXHIGHDEN-26JAN16-B61.5"
    assert market_cap_skips[0].layer == "gate"
    assert [f.name for f in log.failures if f.layer == "strategy"] == ["edge_too_small"]
    assert log.cycles[2].market["KXHIGHDEN-26JAN16-B61.5"] == Decimal("300.00")


async def test_series_cap_exhaustion_skips_subsequent_snapshots() -> None:
    log = ReplayLog()

    orders = await run_replay(make_series_cap_run(), FixedReplay(), "edge", make_config(), log=log)

    assert len(orders) == 3
    assert [o.order_dollars for o in orders] == [Decimal("300.00")] * 3
    series_skips = [f for f in log.failures if f.name == "within_series_cap"]
    assert [f.market_ticker for f in series_skips] == [
        "KXHIGHDEN-26JAN19-B61.5",
        "KXHIGHDEN-26JAN20-B61.5",
    ]
    assert not [f for f in log.failures if f.name in CAP_GATE_NAMES - {"within_series_cap"}]


def test_nine_day_ledger_day8_carry_keeps_monthly_on_effective_cutoff() -> None:
    daily, monthly, agreement = make_nine_day_ledger()
    ledger = daily[:7] + [monthly, agreement]

    by_market, by_event, by_series, aggregate = seed_overlays(ledger, _DAY_8_AS_OF)

    assert by_series == {"KXHIGHDEN": Decimal("163.00")}
    assert aggregate == Decimal("163.00")
    assert by_market["KXHIGHDEN-25NOV07-B61.5"] == Decimal("22.00")
    assert by_market[_MONTHLY_TICKER] == Decimal("9.00")

    close_keyed_carry = sum((f.delta for f in daily[:7]), Decimal("0"))
    assert close_keyed_carry == Decimal("154.00")
    assert by_series["KXHIGHDEN"] != close_keyed_carry

    grace = timedelta(days=SETTLEMENT_GRACE_DAYS)
    assert _MONTHLY_CLOSE.date() + grace <= _DAY_8_AS_OF.date()
    monthly_cutoff = _effective_cutoff_date(parse_ticker(_MONTHLY_TICKER))
    assert monthly_cutoff == date(2025, 11, 30)
    assert monthly_cutoff + grace > _DAY_8_AS_OF.date()


def test_nine_day_ledger_daily_row_keys_identically_under_both_keyings() -> None:
    daily, monthly, agreement = make_nine_day_ledger()
    ledger = daily[:7] + [monthly, agreement]

    by_market, _, _, _ = seed_overlays(ledger, _DAY_8_AS_OF)

    agreement_cutoff = _effective_cutoff_date(parse_ticker(_AGREEMENT_TICKER))
    assert agreement_cutoff == _AGREEMENT_CLOSE.date()
    grace = timedelta(days=SETTLEMENT_GRACE_DAYS)
    assert (agreement_cutoff + grace <= _DAY_8_AS_OF.date()) == (
        _AGREEMENT_CLOSE.date() + grace <= _DAY_8_AS_OF.date()
    )
    assert _AGREEMENT_TICKER not in by_market


def test_nine_day_ledger_day9_drops_day1_fill() -> None:
    daily, monthly, agreement = make_nine_day_ledger()
    ledger = daily[:8] + [monthly, agreement]

    by_market, _, by_series, aggregate = seed_overlays(ledger, _DAY_9_AS_OF)

    assert "KXHIGHDEN-25NOV07-B61.5" not in by_market
    assert by_market["KXHIGHDEN-25NOV08-B61.5"] == Decimal("22.00")
    assert by_market[_MONTHLY_TICKER] == Decimal("9.00")
    assert by_series == {"KXHIGHDEN": Decimal("163.00")}
    assert aggregate == Decimal("163.00")


async def test_tails_routing_skips_blacklisted_and_non_tail_without_invoking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close = datetime(2026, 1, 16, 6, 0, tzinfo=timezone.utc)
    lax_tail = make_row(
        "KXHIGHLAX-26JAN15-T85", close, yes_ask=Decimal("0.15"), yes_bid=Decimal("0.12"), depth=100
    )
    lax_bracket = make_row(
        "KXHIGHLAX-26JAN15-T70-71",
        close,
        yes_ask=Decimal("0.15"),
        yes_bid=Decimal("0.12"),
        depth=100,
    )
    den_tail = make_row(
        "KXHIGHDEN-26JAN15-T85", close, yes_ask=Decimal("0.15"), yes_bid=Decimal("0.12"), depth=100
    )

    calls: list[tails_strategy.TailsContext] = []
    real_evaluate = tails_strategy.evaluate

    def spy(ctx: tails_strategy.TailsContext, **kwargs: object) -> tails_strategy.TailsSignal:
        calls.append(ctx)
        return real_evaluate(ctx, **kwargs)

    monkeypatch.setattr("bot.strategy.tails.evaluate", spy)
    log = ReplayLog()

    orders = await run_replay(
        [lax_tail, lax_bracket, den_tail], FixedReplay(), "tails", make_config(), log=log
    )

    assert len(calls) == 1
    assert len(orders) == 1
    assert orders[0].market_ticker == "KXHIGHDEN-26JAN15-T85"
    routing = {f.market_ticker: f for f in log.failures if f.layer == "routing"}
    assert routing["KXHIGHLAX-26JAN15-T85"].name == "tails_not_invoked_blacklisted"
    assert routing["KXHIGHLAX-26JAN15-T70-71"].name == "tails_not_invoked_non_tail"
    assert "KXHIGHDEN-26JAN15-T85" not in routing


async def test_edge_on_blacklisted_bracket_reaches_evaluate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close = datetime(2026, 1, 16, 6, 0, tzinfo=timezone.utc)
    lax_bracket = make_row(
        "KXHIGHLAX-26JAN15-T70-71",
        close,
        yes_ask=Decimal("0.10"),
        yes_bid=Decimal("0.08"),
    )

    signals: list[edge_strategy.EdgeSignal] = []
    real_evaluate = edge_strategy.evaluate

    def spy(ctx: edge_strategy.EdgeContext, **kwargs: object) -> edge_strategy.EdgeSignal:
        signal = real_evaluate(ctx, **kwargs)
        signals.append(signal)
        return signal

    monkeypatch.setattr("bot.strategy.edge.evaluate", spy)
    log = ReplayLog()

    orders = await run_replay([lax_bracket], FixedReplay(), "edge", make_config(), log=log)

    assert orders == []
    assert len(signals) == 1
    assert signals[0].reason == "blacklisted"
    skips = [f for f in log.failures if f.reason == "blacklisted"]
    assert len(skips) == 1
    assert skips[0].layer == "strategy"


async def test_edge_blacklist_parity_counts_match_snapshot_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 3
    rows = [
        make_row(
            f"{series}-26JAN{15 + i}-B61.5",
            datetime(2026, 1, 16 + i, 5, 0, tzinfo=timezone.utc),
            yes_ask=Decimal("0.10"),
            yes_bid=Decimal("0.08"),
        )
        for series in ("KXHIGHLAX", "KXHIGHMIA")
        for i in range(n)
    ]

    calls: list[edge_strategy.EdgeContext] = []
    real_evaluate = edge_strategy.evaluate

    def spy(ctx: edge_strategy.EdgeContext, **kwargs: object) -> edge_strategy.EdgeSignal:
        calls.append(ctx)
        return real_evaluate(ctx, **kwargs)

    monkeypatch.setattr("bot.strategy.edge.evaluate", spy)
    log = ReplayLog()

    orders = await run_replay(rows, FixedReplay(), "edge", make_config(), log=log)

    assert orders == []
    assert len(calls) == 2 * n
    assert all(ctx.is_blacklisted for ctx in calls)
    blacklisted = [f for f in log.failures if f.reason == "blacklisted"]
    assert len(blacklisted) == 2 * n
    assert all(f.layer == "strategy" for f in blacklisted)
    by_series: dict[str, int] = {}
    for f in blacklisted:
        series = f.market_ticker.split("-", 1)[0]
        by_series[series] = by_series.get(series, 0) + 1
    assert by_series == {"KXHIGHLAX": n, "KXHIGHMIA": n}
