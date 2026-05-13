from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from bot.storage.sqlite import (
    Base,
    Forecast,
    GateFailure,
    Market,
    OrderbookSnapshot,
    PaperTradeRow,
    SimulatedPnl,
    make_engine,
    make_session_factory,
)


@pytest.fixture
def engine():
    eng = make_engine(":memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    factory = make_session_factory(engine)
    with factory() as s:
        yield s


def test_schema_creates_all_six_tables(engine):
    names = set(inspect(engine).get_table_names())
    assert {
        "forecasts",
        "markets",
        "orderbook_snapshots",
        "paper_trades",
        "simulated_pnl",
        "gate_failures",
    } <= names


def test_forecast_round_trip(session):
    members = [72.1, 73.4, 71.8, 75.0]
    run = datetime(2026, 5, 5, 12, 0, tzinfo=_timezone.utc)
    row = Forecast(
        station="KDEN",
        run_time=run,
        valid_date=date(2026, 5, 6),
        members_json=json.dumps(members),
    )
    session.add(row)
    session.commit()

    got = session.scalars(
        select(Forecast).where(
            Forecast.station == "KDEN",
            Forecast.run_time == run,
            Forecast.valid_date == date(2026, 5, 6),
        )
    ).one()
    assert got.station == "KDEN"
    assert got.run_time == run
    assert got.valid_date == date(2026, 5, 6)
    assert json.loads(got.members_json) == members
    assert got.created_at.tzinfo is not None


def test_forecast_unique_constraint(session):
    run = datetime(2026, 5, 5, 12, 0, tzinfo=_timezone.utc)
    valid = date(2026, 5, 6)
    session.add(Forecast(station="KDEN", run_time=run, valid_date=valid, members_json="[]"))
    session.commit()

    session.add(Forecast(station="KDEN", run_time=run, valid_date=valid, members_json="[1]"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_market_round_trip_with_and_without_high_strike(session):
    now = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    bracket = Market(
        ticker="KXHIGHDEN-26MAY06-T70-75",
        series="KXHIGHDEN",
        event_date=date(2026, 5, 6),
        is_monthly=False,
        is_tail=False,
        strike_low=Decimal("70.0000"),
        strike_high=Decimal("75.0000"),
        close_time=now + timedelta(hours=6),
        status="open",
        last_seen_at=now,
    )
    tail = Market(
        ticker="KXHIGHDEN-26MAY06-T100",
        series="KXHIGHDEN",
        event_date=date(2026, 5, 6),
        is_monthly=False,
        is_tail=True,
        strike_low=Decimal("100.0000"),
        strike_high=None,
        close_time=None,
        status="open",
        last_seen_at=now,
    )
    session.add_all([bracket, tail])
    session.commit()

    got_bracket = session.scalars(
        select(Market).where(Market.ticker == "KXHIGHDEN-26MAY06-T70-75")
    ).one()
    got_tail = session.scalars(
        select(Market).where(Market.ticker == "KXHIGHDEN-26MAY06-T100")
    ).one()

    assert got_bracket.strike_low == Decimal("70.0000")
    assert got_bracket.strike_high == Decimal("75.0000")
    assert got_bracket.close_time == now + timedelta(hours=6)
    assert got_tail.strike_high is None
    assert got_tail.close_time is None
    assert got_tail.is_tail is True


def test_market_ticker_unique(session):
    now = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    base = dict(
        ticker="KXHIGHDEN-26MAY06-T70-75",
        series="KXHIGHDEN",
        event_date=date(2026, 5, 6),
        is_monthly=False,
        is_tail=False,
        strike_low=Decimal("70.0000"),
        strike_high=Decimal("75.0000"),
        status="open",
        last_seen_at=now,
    )
    session.add(Market(**base))
    session.commit()

    session.add(Market(**base))
    with pytest.raises(IntegrityError):
        session.commit()


def test_orderbook_snapshot_decimal_precision(session):
    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    row = OrderbookSnapshot(
        ticker="KXHIGHDEN-26MAY06-T70-75",
        snapshot_at=snap_at,
        yes_ask=Decimal("0.017500"),
        yes_bid=Decimal("0.012345"),
        no_ask=Decimal("0.987655"),
        no_bid=Decimal("0.982500"),
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(OrderbookSnapshot)).one()
    assert got.yes_ask == Decimal("0.017500")
    assert got.yes_bid == Decimal("0.012345")
    assert got.no_ask == Decimal("0.987655")
    assert got.no_bid == Decimal("0.982500")


def test_paper_trade_round_trip(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    row = PaperTradeRow(
        intended_at=when,
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        side="sell_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.011200"),
        fair_at_entry=Decimal("0.50"),
        strategy="density_v1",
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(PaperTradeRow)).one()
    assert got.intended_at == when
    assert got.market_ticker == "KXHIGHDEN-26MAY06-T70-75"
    assert got.side == "sell_yes"
    assert got.contracts == 10
    assert got.simulated_price == Decimal("0.40")
    assert got.fee_dollars == Decimal("0.011200")
    assert got.fair_at_entry == Decimal("0.50")
    assert got.strategy == "density_v1"


def test_simulated_pnl_fk_and_cascade(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    trade = PaperTradeRow(
        intended_at=when,
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        side="buy_yes",
        contracts=5,
        simulated_price=Decimal("0.45"),
        fee_dollars=Decimal("0.005000"),
        fair_at_entry=Decimal("0.55"),
        strategy="density_v1",
    )
    session.add(trade)
    session.commit()

    pnl = SimulatedPnl(
        paper_trade_id=trade.id,
        settled_at=when + timedelta(hours=10),
        outcome="won",
        realized_pnl=Decimal("2.500000"),
    )
    session.add(pnl)
    session.commit()

    got = session.scalars(select(SimulatedPnl).where(SimulatedPnl.paper_trade_id == trade.id)).one()
    assert got.outcome == "won"
    assert got.realized_pnl == Decimal("2.500000")

    session.delete(trade)
    session.commit()

    remaining = session.scalars(select(SimulatedPnl)).all()
    assert remaining == []


def test_gate_failure_round_trip(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    with_ticker = GateFailure(
        evaluated_at=when,
        gate_name="fair_value_sane",
        reason="fair_yes=0.999 outside [0.01, 0.99]",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
    )
    no_ticker = GateFailure(
        evaluated_at=when,
        gate_name="circuit_breakers_armed",
        reason="circuit_breakers_armed=False",
        mode="paper",
        market_ticker=None,
    )
    session.add_all([with_ticker, no_ticker])
    session.commit()

    rows = session.scalars(select(GateFailure).order_by(GateFailure.id)).all()
    assert rows[0].market_ticker == "KXHIGHDEN-26MAY06-T70-75"
    assert rows[1].market_ticker is None
    assert rows[0].reason == "fair_yes=0.999 outside [0.01, 0.99]"


def test_datetime_timezone_preserved(session):
    when = datetime(2026, 5, 5, 18, 30, 45, 123456, tzinfo=_timezone.utc)
    session.add(
        GateFailure(
            evaluated_at=when,
            gate_name="model_fresh",
            reason="stale",
            mode="paper",
            market_ticker=None,
        )
    )
    session.commit()
    got = session.scalars(select(GateFailure)).one()
    assert got.evaluated_at.tzinfo is not None
    assert got.evaluated_at.utcoffset() == _timezone.utc.utcoffset(when)
    assert got.evaluated_at == when


def test_decimal_six_dp_round_trip(session):
    snap = OrderbookSnapshot(
        ticker="KXHIGHDEN-26MAY06-T70-75",
        snapshot_at=datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc),
        yes_ask=Decimal("0.123456"),
        yes_bid=Decimal("0.000001"),
        no_ask=Decimal("0.999999"),
        no_bid=Decimal("0.500000"),
    )
    session.add(snap)
    session.commit()
    got = session.scalars(select(OrderbookSnapshot)).one()
    assert got.yes_ask == Decimal("0.123456")
    assert got.yes_bid == Decimal("0.000001")
    assert got.no_ask == Decimal("0.999999")
    assert got.no_bid == Decimal("0.500000")


def test_named_indexes_exist(engine):
    insp = inspect(engine)
    expected = {
        "forecasts": {"ix_forecasts_station_valid_date"},
        "markets": {"ix_markets_series_event_date"},
        "orderbook_snapshots": {"ix_orderbook_snapshots_ticker_snapshot_at"},
        "paper_trades": {"ix_paper_trades_strategy_intended_at"},
        "gate_failures": {"ix_gate_failures_evaluated_at_gate_name"},
    }
    for table, names in expected.items():
        present = {idx["name"] for idx in insp.get_indexes(table)}
        assert names <= present, f"{table}: missing {names - present}"


def test_make_engine_enables_wal_on_disk_db(tmp_path):
    eng = make_engine(tmp_path / "test.db")
    with eng.connect() as conn:
        mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
    eng.dispose()
    assert mode == "wal"


def test_make_engine_sets_synchronous_normal(tmp_path):
    eng = make_engine(tmp_path / "test.db")
    with eng.connect() as conn:
        sync = conn.exec_driver_sql("PRAGMA synchronous").scalar()
    eng.dispose()
    assert sync == 1


def test_make_engine_sets_busy_timeout(tmp_path):
    eng = make_engine(tmp_path / "test.db")
    with eng.connect() as conn:
        timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()
    eng.dispose()
    assert timeout == 5000


def test_make_engine_sets_connect_timeout(tmp_path):
    eng = make_engine(tmp_path / "test.db")
    creator = eng.pool._creator
    closure = dict(zip(creator.__code__.co_freevars, creator.__closure__ or ()))
    cparams = closure["cparams"].cell_contents
    eng.dispose()
    assert cparams.get("timeout") == 30


def test_make_engine_in_memory_db_pragmas_are_safe():
    eng = make_engine(":memory:")
    with eng.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode").scalar()
    eng.dispose()


def test_alembic_upgrade_head_creates_all_tables(tmp_path):
    db_file = tmp_path / "state.db"
    ini_path = tmp_path / "alembic.ini"
    repo_root = __import__("pathlib").Path(__file__).resolve().parent.parent
    ini_path.write_text(
        f"[alembic]\n"
        f"script_location = {repo_root / 'alembic'}\n"
        f"sqlalchemy.url = sqlite:///{db_file}\n"
    )

    cfg = Config(str(ini_path))
    command.upgrade(cfg, "head")

    eng = make_engine(db_file)
    names = set(inspect(eng).get_table_names())
    eng.dispose()
    assert {
        "forecasts",
        "markets",
        "orderbook_snapshots",
        "paper_trades",
        "simulated_pnl",
        "gate_failures",
        "alembic_version",
    } <= names
