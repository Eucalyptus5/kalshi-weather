from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from bot.storage.positions import SETTLEMENT_GRACE_DAYS, open_exposures
from bot.storage.sqlite import (
    Base,
    PaperTradeRow,
    SimulatedPnl,
    make_engine,
    make_session_factory,
)


def _session_factory():
    engine = make_engine(":memory:")
    Base.metadata.create_all(engine)
    return make_session_factory(engine), engine


def _insert_trade(
    factory,
    *,
    market_ticker: str,
    side: str,
    contracts: int,
    simulated_price: Decimal,
    intended_at: datetime,
    settled: bool = False,
) -> int:
    with factory() as session:
        row = PaperTradeRow(
            intended_at=intended_at,
            market_ticker=market_ticker,
            side=side,
            contracts=contracts,
            simulated_price=simulated_price,
            fee_dollars=Decimal("0.05"),
            fair_at_entry=Decimal("0.50"),
            q_raw=Decimal("0.50"),
            strategy="edge",
        )
        session.add(row)
        session.commit()
        trade_id = row.id
        if settled:
            session.add(
                SimulatedPnl(
                    paper_trade_id=trade_id,
                    settled_at=intended_at + timedelta(days=1),
                    outcome="won",
                    realized_pnl=Decimal("1.00"),
                )
            )
            session.commit()
        return trade_id


def test_open_exposures_buy_yes_max_loss_is_premium_paid() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session, now=now)
    assert by_market["KXHIGHDEN-26MAY27-T70-75"] == Decimal("4.00")


def test_open_exposures_sell_yes_max_loss_is_one_minus_premium() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T70-75",
        side="sell_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session, now=now)
    assert by_market["KXHIGHDEN-26MAY27-T70-75"] == Decimal("6.00")


def test_open_exposures_excludes_settled_trades() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T75-80",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.30"),
        intended_at=intended,
        settled=True,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session, now=now)
    assert by_market.get("KXHIGHDEN-26MAY27-T70-75") == Decimal("4.00")
    assert "KXHIGHDEN-26MAY27-T75-80" not in by_market


def test_open_exposures_aggregates_per_market_event_series() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T75-80",
        side="buy_yes",
        contracts=20,
        simulated_price=Decimal("0.10"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY28-T70-75",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.20"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, by_event, by_series, _agg = open_exposures(session, now=now)
    assert by_market["KXHIGHDEN-26MAY27-T70-75"] == Decimal("4.00")
    assert by_market["KXHIGHDEN-26MAY27-T75-80"] == Decimal("2.00")
    assert by_market["KXHIGHDEN-26MAY28-T70-75"] == Decimal("1.00")
    assert by_event["KXHIGHDEN-26MAY27"] == Decimal("6.00")
    assert by_event["KXHIGHDEN-26MAY28"] == Decimal("1.00")
    assert by_series["KXHIGHDEN"] == Decimal("7.00")


def test_open_exposures_aggregates_4_part_t_form_bracket_under_canonical_event_key() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T70.5-72.5",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T72.5-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.30"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        _by_market, by_event, _by_series, _agg = open_exposures(session, now=now)
    assert by_event["KXHIGHDEN-26MAY27"] == Decimal("7.00")


def test_open_exposures_returns_zero_for_unknown_ticker() -> None:
    factory, _ = _session_factory()
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session, now=now)
    assert by_market.get("nonexistent", Decimal("0")) == Decimal("0")


def test_open_exposures_excludes_pending_past_grace_window() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 4, 1, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26APR01-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, by_event, by_series, _agg = open_exposures(session, now=now)
    assert "KXHIGHDEN-26APR01-T70-75" not in by_market
    assert "KXHIGHDEN-26APR01" not in by_event
    assert by_series.get("KXHIGHDEN", Decimal("0")) == Decimal("0")


def test_open_exposures_includes_pending_inside_grace_window() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 25, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY25-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session, now=now)
    assert by_market["KXHIGHDEN-26MAY25-T70-75"] == Decimal("4.00")


def test_open_exposures_grace_boundary_flips_with_now() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 20, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY20-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    included_now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    excluded_now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market_in, _, _, _ = open_exposures(session, now=included_now)
        by_market_out, _, _, _ = open_exposures(session, now=excluded_now)
    assert by_market_in["KXHIGHDEN-26MAY20-T70-75"] == Decimal("4.00")
    assert "KXHIGHDEN-26MAY20-T70-75" not in by_market_out


def test_open_exposures_default_grace_matches_module_constant() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 25, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY25-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        default = open_exposures(session, now=now)
        explicit = open_exposures(session, now=now, grace_days=SETTLEMENT_GRACE_DAYS)
    assert default == explicit


def test_open_exposures_monthly_event_uses_end_of_month_cutoff() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXRAINSFOM-26JUN-T1.0",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    now = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session, now=now)
    assert by_market["KXRAINSFOM-26JUN-T1.0"] == Decimal("4.00")


def test_open_exposures_monthly_event_drops_past_grace_after_end_of_month() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXRAINSFOM-26JUN-T1.0",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session, now=now)
    assert "KXRAINSFOM-26JUN-T1.0" not in by_market


def test_open_exposures_skips_unparseable_ticker_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXJUNK-NOTADATE-XYZ",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    caplog.set_level(logging.WARNING, logger="bot.storage.positions")
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session, now=now)
    assert by_market["KXHIGHDEN-26MAY27-T70-75"] == Decimal("4.00")
    matches = [r for r in caplog.records if "unparseable" in r.getMessage()]
    assert matches


def test_open_exposures_uses_default_now_when_none_passed() -> None:
    factory, _ = _session_factory()
    today = date.today()
    intended = datetime(today.year, today.month, today.day, 18, 0, tzinfo=timezone.utc)
    ticker = f"KXHIGHDEN-{today.strftime('%y%b%d').upper()}-T70-75"
    _insert_trade(
        factory,
        market_ticker=ticker,
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    with factory() as session:
        by_market, _by_event, _by_series, _agg = open_exposures(session)
    assert by_market[ticker] == Decimal("4.00")


def test_open_exposures_returns_four_tuple() -> None:
    factory, _ = _session_factory()
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        result = open_exposures(session, now=now)
    assert isinstance(result, tuple)
    assert len(result) == 4
    by_market, by_event, by_series, aggregate = result
    assert isinstance(by_market, dict)
    assert isinstance(by_event, dict)
    assert isinstance(by_series, dict)
    assert isinstance(aggregate, Decimal)


def test_open_exposures_aggregate_equals_sum_of_market_values() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T75-80",
        side="buy_yes",
        contracts=20,
        simulated_price=Decimal("0.10"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T75-80",
        side="sell_yes",
        contracts=5,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY28-T70-75",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.20"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHNY-26MAY27-T80-85",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.30"),
        intended_at=intended,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        by_market, _by_event, _by_series, aggregate = open_exposures(session, now=now)
    assert aggregate == sum(by_market.values(), Decimal("0"))


def test_open_exposures_aggregate_excludes_settled_trades() -> None:
    factory, _ = _session_factory()
    intended = datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        intended_at=intended,
    )
    _insert_trade(
        factory,
        market_ticker="KXHIGHDEN-26MAY27-T75-80",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.30"),
        intended_at=intended,
        settled=True,
    )
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        _by_market, _by_event, _by_series, aggregate = open_exposures(session, now=now)
    assert aggregate == Decimal("4.00")


def test_open_exposures_aggregate_zero_when_no_pending() -> None:
    factory, _ = _session_factory()
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    with factory() as session:
        _by_market, _by_event, _by_series, aggregate = open_exposures(session, now=now)
    assert aggregate == Decimal("0")
    assert isinstance(aggregate, Decimal)
