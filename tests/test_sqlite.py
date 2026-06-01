from __future__ import annotations

import inspect as py_inspect
import json
import logging
import re
import shutil
from datetime import date, datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError, OperationalError

from bot.storage.sqlite import (
    Base,
    DemoOrder,
    Forecast,
    GateFailure,
    Market,
    OrderbookSnapshot,
    PaperTradeRow,
    SimulatedPnl,
    ensure_baseline_stamped,
    make_engine,
    make_session_factory,
    upgrade_schema,
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
        q_raw=Decimal("0.50"),
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
        q_raw=Decimal("0.55"),
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
        last_seen_at=when,
    )
    no_ticker = GateFailure(
        evaluated_at=when,
        gate_name="circuit_breakers_armed",
        reason="circuit_breakers_armed=False",
        mode="paper",
        market_ticker=None,
        last_seen_at=when,
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
            last_seen_at=when,
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


REPO_ROOT = Path(__file__).resolve().parent.parent


def _alembic_cfg(tmp_path, db_file: Path) -> Config:
    ini_path = tmp_path / "alembic.ini"
    ini_path.write_text(
        f"[alembic]\n"
        f"script_location = {REPO_ROOT / 'alembic'}\n"
        f"sqlalchemy.url = sqlite:///{db_file}\n"
    )
    return Config(str(ini_path))


def test_paper_trade_persists_sigma_t(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    row = PaperTradeRow(
        intended_at=when,
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        ensemble_spread_sigma_t=Decimal("2.500000"),
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(PaperTradeRow)).one()
    assert got.ensemble_spread_sigma_t == Decimal("2.500000")


def test_paper_trade_persists_attempted_contracts(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    row = PaperTradeRow(
        intended_at=when,
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        side="sell_yes",
        contracts=1,
        simulated_price=Decimal("0.99"),
        fee_dollars=Decimal("0.01"),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        attempted_contracts=7194,
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(PaperTradeRow)).one()
    assert got.contracts == 1
    assert got.attempted_contracts == 7194


def test_paper_trade_lead_time_hours_persists(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    row = PaperTradeRow(
        intended_at=when,
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        lead_time_hours=Decimal("36.5000"),
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(PaperTradeRow)).one()
    assert got.lead_time_hours == Decimal("36.5000")


def test_paper_trade_nbm_divergence_nullable(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    row = PaperTradeRow(
        intended_at=when,
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        nbm_divergence=None,
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(PaperTradeRow)).one()
    assert got.nbm_divergence is None


def test_paper_trade_lead_time_hours_nullable(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    row = PaperTradeRow(
        intended_at=when,
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.40"),
        fee_dollars=Decimal("0.05"),
        fair_at_entry=Decimal("0.50"),
        q_raw=Decimal("0.50"),
        strategy="edge",
        lead_time_hours=None,
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(PaperTradeRow)).one()
    assert got.lead_time_hours is None


def test_orderbook_snapshot_persists_depth(session):
    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    row = OrderbookSnapshot(
        ticker="KXHIGHDEN-26MAY06-T70-75",
        snapshot_at=snap_at,
        yes_ask=Decimal("0.40"),
        yes_bid=Decimal("0.38"),
        no_ask=Decimal("0.62"),
        no_bid=Decimal("0.60"),
        yes_ask_depth=12,
        yes_bid_depth=3,
        no_ask_depth=7,
        no_bid_depth=11,
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(OrderbookSnapshot)).one()
    assert got.yes_ask_depth == 12
    assert got.yes_bid_depth == 3
    assert got.no_ask_depth == 7
    assert got.no_bid_depth == 11


def test_gate_failure_notes_nullable(session):
    when = datetime(2026, 5, 5, 18, 30, tzinfo=_timezone.utc)
    no_note = GateFailure(
        evaluated_at=when,
        gate_name="market_open",
        reason="market_status=closed",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        notes=None,
        last_seen_at=when,
    )
    with_note = GateFailure(
        evaluated_at=when,
        gate_name="market_open",
        reason="market_status=active",
        mode="paper",
        market_ticker="KXHIGHDEN-26MAY06-T70-75",
        notes="pre_fix_status_string_bug",
        last_seen_at=when,
    )
    session.add_all([no_note, with_note])
    session.commit()

    rows = session.scalars(select(GateFailure).order_by(GateFailure.id)).all()
    assert rows[0].notes is None
    assert rows[1].notes == "pre_fix_status_string_bug"


def test_alembic_env_honors_injected_connection(tmp_path):
    decoy_db = tmp_path / "decoy.db"
    target_db = tmp_path / "target.db"
    cfg = _alembic_cfg(tmp_path, decoy_db)

    target_engine = make_engine(target_db)
    with target_engine.connect() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")

    target_inspector = inspect(target_engine)
    tnames = set(target_inspector.get_table_names())
    assert "alembic_version" in tnames
    assert "forecasts" in tnames
    paper_cols = {c["name"] for c in target_inspector.get_columns("paper_trades")}
    assert "attempted_contracts" in paper_cols
    gate_cols = {c["name"] for c in target_inspector.get_columns("gate_failures")}
    assert "notes" in gate_cols
    ob_cols = {c["name"] for c in target_inspector.get_columns("orderbook_snapshots")}
    assert "yes_ask_depth" in ob_cols
    target_engine.dispose()

    assert not decoy_db.exists() or decoy_db.stat().st_size == 0


def _demo_order(**overrides) -> DemoOrder:
    when = datetime(2026, 5, 22, 12, 0, tzinfo=_timezone.utc)
    base = dict(
        client_order_id="kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22",
        exchange_order_id="EX1",
        market_ticker="KXHIGHDEN-26MAY22-T70",
        strategy="edge",
        side="yes",
        requested_contracts=10,
        filled_contracts=0,
        requested_yes_price_dollars=Decimal("0.620000"),
        fair_at_entry=Decimal("0.650000"),
        intended_at=when,
        status="resting",
        placed_at=when,
        last_status_at=when,
    )
    base.update(overrides)
    return DemoOrder(**base)


def test_demo_orders_unique_client_order_id(session):
    session.add(_demo_order(client_order_id="kw-edge-yes-A", exchange_order_id="EX1"))
    session.commit()
    session.add(_demo_order(client_order_id="kw-edge-yes-A", exchange_order_id="EX2"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_demo_orders_client_order_id_is_not_null(session):
    session.add(_demo_order(client_order_id=None))
    with pytest.raises(IntegrityError):
        session.commit()


def test_demo_orders_backfill_row_with_null_intent_fields_commits(session):
    row = _demo_order(
        client_order_id="kw-backfill-EX123",
        exchange_order_id="EX123",
        status="executed",
        strategy=None,
        fair_at_entry=None,
        intended_at=None,
        requested_yes_price_dollars=None,
    )
    session.add(row)
    session.commit()

    got = session.scalars(
        select(DemoOrder).where(DemoOrder.client_order_id == "kw-backfill-EX123")
    ).one()
    assert got.strategy is None
    assert got.fair_at_entry is None
    assert got.intended_at is None
    assert got.requested_yes_price_dollars is None
    assert got.exchange_order_id == "EX123"
    assert got.status == "executed"


def test_demo_orders_natural_key_row_round_trip(session):
    when = datetime(2026, 5, 22, 12, 0, tzinfo=_timezone.utc)
    row = _demo_order(
        client_order_id="kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22",
        exchange_order_id="EX9",
        strategy="edge",
        fair_at_entry=Decimal("0.620000"),
        intended_at=when,
        requested_yes_price_dollars=Decimal("0.580000"),
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(DemoOrder).where(DemoOrder.exchange_order_id == "EX9")).one()
    assert got.strategy == "edge"
    assert got.fair_at_entry == Decimal("0.620000")
    assert got.intended_at == when
    assert got.requested_yes_price_dollars == Decimal("0.580000")
    assert got.side == "yes"
    assert got.market_ticker == "KXHIGHDEN-26MAY22-T70"


def test_paper_trade_demo_link_round_trip(session):
    when = datetime(2026, 5, 22, 18, 30, tzinfo=_timezone.utc)
    row = PaperTradeRow(
        intended_at=when,
        market_ticker="KXHIGHDEN-26MAY22-T70",
        side="buy_yes",
        contracts=10,
        simulated_price=Decimal("0.205000"),
        fee_dollars=Decimal("0.070000"),
        fair_at_entry=Decimal("0.620000"),
        q_raw=Decimal("0.620000"),
        strategy="edge",
        demo_order_client_id="kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22",
    )
    session.add(row)
    session.commit()

    got = session.scalars(select(PaperTradeRow)).one()
    assert got.demo_order_client_id == "kw-edge-yes-KXHIGHDEN-26MAY22-T70-2026-05-22"


def test_paper_trade_demo_link_allows_multiple_nulls(session):
    when = datetime(2026, 5, 22, 18, 30, tzinfo=_timezone.utc)
    for _ in range(2):
        session.add(
            PaperTradeRow(
                intended_at=when,
                market_ticker="KXHIGHDEN-26MAY22-T70",
                side="buy_yes",
                contracts=5,
                simulated_price=Decimal("0.40"),
                fee_dollars=Decimal("0.01"),
                fair_at_entry=Decimal("0.50"),
                q_raw=Decimal("0.50"),
                strategy="edge",
                demo_order_client_id=None,
            )
        )
    session.commit()
    assert len(session.scalars(select(PaperTradeRow)).all()) == 2


def test_ensure_baseline_stamped_stamps_unstamped_create_all_db(tmp_path):
    db_file = tmp_path / "state.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)

    ensure_baseline_stamped(engine, "0001")

    with engine.connect() as connection:
        row = connection.execute(text("SELECT version_num FROM alembic_version")).first()
    assert row is not None
    assert row[0] == "0001"
    engine.dispose()


def test_ensure_baseline_stamped_is_noop_on_already_stamped_db(tmp_path):
    db_file = tmp_path / "state.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('0001')"))

    ensure_baseline_stamped(engine, "0099")

    with engine.connect() as connection:
        rows = connection.execute(text("SELECT version_num FROM alembic_version")).all()
    assert len(rows) == 1
    assert rows[0][0] == "0001"
    engine.dispose()


def test_ensure_baseline_stamped_refuses_empty_db(tmp_path):
    db_file = tmp_path / "state.db"
    engine = make_engine(db_file)

    ensure_baseline_stamped(engine, "0001")

    inspector = inspect(engine)
    assert "alembic_version" not in inspector.get_table_names()
    engine.dispose()


def test_bare_alembic_upgrade_against_create_all_db_raises(tmp_path):
    db_file = tmp_path / "state.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    engine.dispose()

    cfg = _alembic_cfg(tmp_path, db_file)
    with pytest.raises(OperationalError) as excinfo:
        command.upgrade(cfg, "head")
    assert "already exists" in str(excinfo.value)


def test_head_shape_db_stamped_0001_fails_on_upgrade(tmp_path):
    db_file = tmp_path / "state.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    ensure_baseline_stamped(engine, "0001")
    engine.dispose()

    cfg = _alembic_cfg(tmp_path, db_file)
    with pytest.raises(OperationalError) as excinfo:
        command.upgrade(cfg, "head")
    assert "duplicate column name" in str(excinfo.value)


def test_migrate_script_stamps_head_on_head_shape_db(tmp_path):
    from scripts.migrate import _detect_baseline

    db_file = tmp_path / "state.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)

    cfg = _alembic_cfg(tmp_path, db_file)
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    assert baseline == script_dir.get_current_head()
    ensure_baseline_stamped(engine, baseline)
    with engine.connect() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")

    with engine.connect() as connection:
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == script_dir.get_current_head()
    engine.dispose()


def test_migrate_script_stamps_0001_on_baseline_shape_db(tmp_path):
    from scripts.migrate import _detect_baseline

    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    assert baseline == "0001"
    with engine.connect() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")

    with engine.connect() as connection:
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == script_dir.get_current_head()
    engine.dispose()


def test_detect_baseline_raises_on_partial_drift_notes_only(tmp_path):
    from scripts.migrate import _detect_baseline

    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE gate_failures ADD COLUMN notes VARCHAR(64)"))
    script_dir = ScriptDirectory.from_config(cfg)
    with pytest.raises(RuntimeError) as excinfo:
        _detect_baseline(engine, script_dir)
    msg = str(excinfo.value)
    assert "partial 0002 schema" in msg
    assert "notes" in msg
    engine.dispose()


def test_detect_baseline_raises_on_partial_drift_two_of_three(tmp_path):
    from scripts.migrate import _detect_baseline

    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE gate_failures ADD COLUMN notes VARCHAR(64)"))
        conn.execute(text("ALTER TABLE paper_trades ADD COLUMN attempted_contracts INTEGER"))
    script_dir = ScriptDirectory.from_config(cfg)
    with pytest.raises(RuntimeError) as excinfo:
        _detect_baseline(engine, script_dir)
    assert "partial 0002 schema" in str(excinfo.value)
    engine.dispose()


def test_detect_baseline_returns_newest_sentinel_shape_even_when_head_is_unrecognized(tmp_path):
    from scripts.migrate import _detect_baseline

    alembic_dir = tmp_path / "alembic"
    versions = alembic_dir / "versions"
    versions.mkdir(parents=True)
    (alembic_dir / "env.py").write_text((REPO_ROOT / "alembic" / "env.py").read_text())
    (alembic_dir / "script.py.mako").write_text(
        (REPO_ROOT / "alembic" / "script.py.mako").read_text()
    )
    (versions / "0001_initial.py").write_text(
        (REPO_ROOT / "alembic" / "versions" / "0001_initial.py").read_text()
    )
    (versions / "0002_orderbook_depth_and_trade_features.py").write_text(
        (
            REPO_ROOT / "alembic" / "versions" / "0002_orderbook_depth_and_trade_features.py"
        ).read_text()
    )
    (versions / "0003_demo_orders.py").write_text(
        (REPO_ROOT / "alembic" / "versions" / "0003_demo_orders.py").read_text()
    )
    (versions / "0004_paper_trades_q_raw.py").write_text(
        (REPO_ROOT / "alembic" / "versions" / "0004_paper_trades_q_raw.py").read_text()
    )
    (versions / "0005_reconciler_state.py").write_text(
        (REPO_ROOT / "alembic" / "versions" / "0005_reconciler_state.py").read_text()
    )
    (versions / "0006_gate_failures_dedupe.py").write_text(
        (REPO_ROOT / "alembic" / "versions" / "0006_gate_failures_dedupe.py").read_text()
    )
    (versions / "0007_portfolio_snapshots.py").write_text(
        (REPO_ROOT / "alembic" / "versions" / "0007_portfolio_snapshots.py").read_text()
    )
    (versions / "0008_decoy.py").write_text(
        '"""decoy 0008 for forward-compat test\n\n'
        "Revision ID: 0008\n"
        "Revises: 0007\n"
        "Create Date: 2026-06-02 14:00:00.000000\n\n"
        '"""\n\n'
        "from typing import Sequence, Union\n\n"
        "from alembic import op  # noqa: F401\n\n\n"
        'revision: str = "0008"\n'
        'down_revision: Union[str, Sequence[str], None] = "0007"\n'
        "branch_labels: Union[str, Sequence[str], None] = None\n"
        "depends_on: Union[str, Sequence[str], None] = None\n\n\n"
        "def upgrade() -> None:\n"
        "    pass\n\n\n"
        "def downgrade() -> None:\n"
        "    pass\n"
    )

    db_file = tmp_path / "state.db"
    ini_path = tmp_path / "alembic.ini"
    ini_path.write_text(
        f"[alembic]\nscript_location = {alembic_dir}\nsqlalchemy.url = sqlite:///{db_file}\n"
    )
    cfg = Config(str(ini_path))

    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    script_dir = ScriptDirectory.from_config(cfg)
    assert script_dir.get_current_head() == "0008"
    assert _detect_baseline(engine, script_dir) == "0007"
    engine.dispose()


def test_migration_0002_backfills_depth_with_zero_via_server_default(tmp_path):
    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO orderbook_snapshots "
                "(ticker, snapshot_at, yes_ask, yes_bid, no_ask, no_bid, created_at) "
                "VALUES (:ticker, :snapshot_at, :yes_ask, :yes_bid, :no_ask, :no_bid, :created)"
            ),
            {
                "ticker": "KXHIGHDEN-26MAY06-T70-75",
                "snapshot_at": "2026-05-05 18:00:00+00:00",
                "yes_ask": 0.40,
                "yes_bid": 0.38,
                "no_ask": 0.62,
                "no_bid": 0.60,
                "created": "2026-05-05 18:00:00+00:00",
            },
        )
    engine.dispose()

    command.upgrade(cfg, "0002")

    engine = make_engine(db_file)
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT yes_ask_depth, yes_bid_depth, no_ask_depth, no_bid_depth "
                "FROM orderbook_snapshots"
            )
        ).one()
    assert row == (0, 0, 0, 0)
    engine.dispose()


def test_migration_0002_backfills_attempted_contracts_from_contracts(tmp_path):
    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_trades "
                "(intended_at, market_ticker, side, contracts, simulated_price, "
                "fee_dollars, fair_at_entry, strategy, created_at) "
                "VALUES (:intended_at, :ticker, :side, :contracts, :price, :fee, "
                ":fair, :strategy, :created)"
            ),
            {
                "intended_at": "2026-05-05 18:30:00+00:00",
                "ticker": "KXHIGHDEN-26MAY06-T70-75",
                "side": "buy_yes",
                "contracts": 42,
                "price": 0.40,
                "fee": 0.05,
                "fair": 0.50,
                "strategy": "edge",
                "created": "2026-05-05 18:30:00+00:00",
            },
        )
    engine.dispose()

    command.upgrade(cfg, "0002")

    engine = make_engine(db_file)
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT contracts, attempted_contracts FROM paper_trades")
        ).one()
    assert row == (42, 42)
    engine.dispose()


def test_full_migrate_path_runs_brief_02_backfill_end_to_end(tmp_path):
    from scripts.migrate import _detect_baseline

    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_trades "
                "(intended_at, market_ticker, side, contracts, simulated_price, "
                "fee_dollars, fair_at_entry, strategy, created_at) "
                "VALUES (:intended_at, :ticker, :side, :contracts, :price, :fee, "
                ":fair, :strategy, :created)"
            ),
            {
                "intended_at": "2026-05-05 18:30:00+00:00",
                "ticker": "KXHIGHDEN-26MAY06-T70-75",
                "side": "buy_yes",
                "contracts": 42,
                "price": 0.40,
                "fee": 0.05,
                "fair": 0.50,
                "strategy": "edge",
                "created": "2026-05-05 18:30:00+00:00",
            },
        )
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    ensure_baseline_stamped(engine, baseline)
    with engine.connect() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")

    with engine.connect() as connection:
        attempted = connection.execute(
            text("SELECT attempted_contracts FROM paper_trades")
        ).scalar()
    assert attempted == 42
    engine.dispose()


def test_pinned_broken_single_column_detector_silently_stamps_head_on_partial_drift(tmp_path):
    from alembic.script import ScriptDirectory as _ScriptDirectory

    def _broken_single_column_detector(engine, script_dir) -> str:
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("gate_failures")}
        if "notes" in cols:
            return script_dir.get_current_head()
        return "0001"

    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE gate_failures ADD COLUMN notes VARCHAR(64)"))
    script_dir = _ScriptDirectory.from_config(cfg)
    head = script_dir.get_current_head()
    assert _broken_single_column_detector(engine, script_dir) == head
    engine.dispose()


def _resolved_head(tmp_path: Path) -> str:
    db_file = tmp_path / "_probe.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    head = ScriptDirectory.from_config(cfg).get_current_head()
    return head


def _snapshot_real_state_db() -> tuple[float, int] | None:
    real_db = REPO_ROOT / "data" / "state.db"
    if not real_db.exists():
        return None
    stat = real_db.stat()
    return (stat.st_mtime, stat.st_size)


def test_upgrade_schema_fresh_db_creates_all_tables(tmp_path):
    pre_snapshot = _snapshot_real_state_db()
    db_file = tmp_path / "state.db"
    upgrade_schema(db_file)

    engine = make_engine(db_file)
    session_factory = make_session_factory(engine)
    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    with session_factory() as s:
        s.add(
            OrderbookSnapshot(
                ticker="KXHIGHDEN-26MAY06-T70-75",
                snapshot_at=snap_at,
                yes_ask=Decimal("0.40"),
                yes_bid=Decimal("0.38"),
                no_ask=Decimal("0.62"),
                no_bid=Decimal("0.60"),
                yes_ask_depth=12,
                yes_bid_depth=5,
                no_ask_depth=7,
                no_bid_depth=3,
            )
        )
        s.commit()
        got = s.scalars(select(OrderbookSnapshot)).one()
    assert got.yes_bid_depth == 5
    assert got.no_bid_depth == 3
    engine.dispose()

    post_snapshot = _snapshot_real_state_db()
    assert pre_snapshot == post_snapshot


def test_upgrade_schema_fresh_no_tables_walks_full_chain(tmp_path):
    db_file = tmp_path / "state.db"
    upgrade_schema(db_file)
    expected_head = _resolved_head(tmp_path)

    engine = make_engine(db_file)
    tables = set(inspect(engine).get_table_names())
    assert "alembic_version" in tables
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == expected_head

    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    session_factory = make_session_factory(engine)
    with session_factory() as s:
        s.add(
            OrderbookSnapshot(
                ticker="KXHIGHDEN-26MAY06-T70-75",
                snapshot_at=snap_at,
                yes_ask=Decimal("0.40"),
                yes_bid=Decimal("0.38"),
                no_ask=Decimal("0.62"),
                no_bid=Decimal("0.60"),
                yes_ask_depth=12,
                yes_bid_depth=5,
                no_ask_depth=7,
                no_bid_depth=3,
            )
        )
        s.commit()
    engine.dispose()


def test_upgrade_schema_drifted_0001_schema_alembic_version_empty_walks_to_head(tmp_path):
    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM alembic_version"))
    engine.dispose()

    upgrade_schema(db_file)
    expected_head = _resolved_head(tmp_path)

    engine = make_engine(db_file)
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == expected_head

    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    session_factory = make_session_factory(engine)
    with session_factory() as s:
        s.add(
            OrderbookSnapshot(
                ticker="KXHIGHDEN-26MAY06-T70-75",
                snapshot_at=snap_at,
                yes_ask=Decimal("0.40"),
                yes_bid=Decimal("0.38"),
                no_ask=Decimal("0.62"),
                no_bid=Decimal("0.60"),
                yes_ask_depth=12,
                yes_bid_depth=5,
                no_ask_depth=7,
                no_bid_depth=3,
            )
        )
        s.commit()
    engine.dispose()


def test_upgrade_schema_stamped_below_head_advances_to_head(tmp_path):
    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    upgrade_schema(db_file)
    expected_head = _resolved_head(tmp_path)

    engine = make_engine(db_file)
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == expected_head

    cols = {c["name"] for c in inspect(engine).get_columns("orderbook_snapshots")}
    assert {"yes_ask_depth", "yes_bid_depth", "no_ask_depth", "no_bid_depth"} <= cols
    engine.dispose()


def test_upgrade_schema_stamped_below_head_supports_depth_insert(tmp_path):
    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    upgrade_schema(db_file)

    engine = make_engine(db_file)
    session_factory = make_session_factory(engine)
    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    with session_factory() as s:
        s.add(
            OrderbookSnapshot(
                ticker="KXHIGHDEN-26MAY06-T70-75",
                snapshot_at=snap_at,
                yes_ask=Decimal("0.40"),
                yes_bid=Decimal("0.38"),
                no_ask=Decimal("0.62"),
                no_bid=Decimal("0.60"),
                yes_ask_depth=5,
                yes_bid_depth=5,
                no_ask_depth=5,
                no_bid_depth=5,
            )
        )
        s.commit()
    engine.dispose()


def test_upgrade_schema_target_is_dynamic_not_hardcoded():
    src = py_inspect.getsource(upgrade_schema)
    assert "get_current_head()" in src
    assert set(re.findall(r'"\d{4}"', src)) == {'"0001"'}


def test_upgrade_schema_case_b_baseline_branch_uses_literal_0001():
    src = py_inspect.getsource(upgrade_schema)
    assert '"0001"' in src


def test_upgrade_schema_case_b_head_branch_routes_create_all_materialized_db_to_head(tmp_path):
    db_file = tmp_path / "state.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    engine.dispose()

    upgrade_schema(db_file)
    expected_head = _resolved_head(tmp_path)

    engine = make_engine(db_file)
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == expected_head

    session_factory = make_session_factory(engine)
    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    with session_factory() as s:
        s.add(
            OrderbookSnapshot(
                ticker="KXHIGHDEN-26MAY06-T70-75",
                snapshot_at=snap_at,
                yes_ask=Decimal("0.40"),
                yes_bid=Decimal("0.38"),
                no_ask=Decimal("0.62"),
                no_bid=Decimal("0.60"),
                yes_ask_depth=5,
                yes_bid_depth=5,
                no_ask_depth=5,
                no_bid_depth=5,
            )
        )
        s.commit()
    engine.dispose()


def test_upgrade_schema_case_c_create_all_materialized_db_stamped_below_head_routes_to_head_update(
    tmp_path,
):
    db_file = tmp_path / "state.db"
    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)")
        )
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('0001')"))
    engine.dispose()

    upgrade_schema(db_file)
    expected_head = _resolved_head(tmp_path)

    engine = make_engine(db_file)
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == expected_head

    session_factory = make_session_factory(engine)
    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    with session_factory() as s:
        s.add(
            OrderbookSnapshot(
                ticker="KXHIGHDEN-26MAY06-T70-75",
                snapshot_at=snap_at,
                yes_ask=Decimal("0.40"),
                yes_bid=Decimal("0.38"),
                no_ask=Decimal("0.62"),
                no_bid=Decimal("0.60"),
                yes_ask_depth=5,
                yes_bid_depth=5,
                no_ask_depth=5,
                no_bid_depth=5,
            )
        )
        s.commit()
    engine.dispose()


def test_upgrade_schema_case_b_drifted_schema_walks_to_head_columns(tmp_path):
    db_file = tmp_path / "state.db"
    cfg = _alembic_cfg(tmp_path, db_file)
    command.upgrade(cfg, "0001")

    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM alembic_version"))
    engine.dispose()

    upgrade_schema(db_file)

    engine = make_engine(db_file)
    cols = {c["name"] for c in inspect(engine).get_columns("orderbook_snapshots")}
    assert {"yes_ask_depth", "yes_bid_depth", "no_ask_depth", "no_bid_depth"} <= cols
    engine.dispose()


def test_upgrade_schema_uses_engine_begin_not_connect():
    src = py_inspect.getsource(upgrade_schema)
    assert "engine.begin()" in src
    assert "engine.connect()" not in src


def test_create_all_alone_does_not_repair_drifted_schema(tmp_path):
    real_versions = REPO_ROOT / "alembic" / "versions"
    real_listing_before = sorted(p.name for p in real_versions.iterdir())

    alembic_clone = tmp_path / "alembic"
    shutil.copytree(REPO_ROOT / "alembic", alembic_clone)
    versions_dir = alembic_clone / "versions"

    probe_cfg = Config()
    probe_cfg.set_main_option("script_location", str(alembic_clone))
    probe_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path / '_probe.db'}")
    current_head = ScriptDirectory.from_config(probe_cfg).get_current_head()

    drift_file = versions_dir / "9999_drift_test.py"
    drift_file.write_text(
        '"""drift test\n\n'
        "Revision ID: 9999\n"
        f"Revises: {current_head}\n"
        "Create Date: 2026-05-28 13:00:00.000000\n\n"
        '"""\n\n'
        "from typing import Sequence, Union\n\n"
        "from alembic import op\n"
        "import sqlalchemy as sa\n\n\n"
        'revision: str = "9999"\n'
        f'down_revision: Union[str, Sequence[str], None] = "{current_head}"\n'
        "branch_labels: Union[str, Sequence[str], None] = None\n"
        "depends_on: Union[str, Sequence[str], None] = None\n\n\n"
        "def upgrade() -> None:\n"
        "    with op.batch_alter_table('gate_failures') as batch_op:\n"
        "        batch_op.add_column(sa.Column('int_col', sa.Integer(), nullable=True))\n\n\n"
        "def downgrade() -> None:\n"
        "    with op.batch_alter_table('gate_failures') as batch_op:\n"
        "        batch_op.drop_column('int_col')\n"
    )

    db_file = tmp_path / "state.db"
    ini_path = tmp_path / "alembic.ini"
    ini_path.write_text(
        f"[alembic]\nscript_location = {alembic_clone}\nsqlalchemy.url = sqlite:///{db_file}\n"
    )
    drift_cfg = Config(str(ini_path))
    command.upgrade(drift_cfg, current_head)

    engine = make_engine(db_file)
    Base.metadata.create_all(engine)
    cols_before = {c["name"] for c in inspect(engine).get_columns("gate_failures")}
    assert "int_col" not in cols_before
    engine.dispose()

    upgrade_schema(db_file, script_location=alembic_clone)

    engine = make_engine(db_file)
    cols_after = {c["name"] for c in inspect(engine).get_columns("gate_failures")}
    assert "int_col" in cols_after
    engine.dispose()

    real_listing_after = sorted(p.name for p in real_versions.iterdir())
    assert real_listing_before == real_listing_after
    assert "9999_drift_test.py" not in real_listing_after


def test_upgrade_schema_is_idempotent(tmp_path):
    db_file = tmp_path / "state.db"
    upgrade_schema(db_file)
    upgrade_schema(db_file)

    engine = make_engine(db_file)
    session_factory = make_session_factory(engine)
    snap_at = datetime(2026, 5, 5, 18, 0, tzinfo=_timezone.utc)
    with session_factory() as s:
        s.add(
            OrderbookSnapshot(
                ticker="KXHIGHDEN-26MAY06-T70-75",
                snapshot_at=snap_at,
                yes_ask=Decimal("0.40"),
                yes_bid=Decimal("0.38"),
                no_ask=Decimal("0.62"),
                no_bid=Decimal("0.60"),
                yes_ask_depth=5,
                yes_bid_depth=5,
                no_ask_depth=5,
                no_bid_depth=5,
            )
        )
        s.commit()
    engine.dispose()


def test_upgrade_schema_does_not_touch_real_data_state_db(tmp_path):
    pre_snapshot = _snapshot_real_state_db()
    db_file = tmp_path / "foo.db"
    upgrade_schema(db_file)
    post_snapshot = _snapshot_real_state_db()
    assert pre_snapshot == post_snapshot
    assert db_file.exists()
    engine = make_engine(db_file)
    tables = set(inspect(engine).get_table_names())
    assert "alembic_version" in tables
    engine.dispose()


def test_orm_update_cannot_replace_text_for_per_ticker_pnl_update(session):
    from sqlalchemy import Numeric, bindparam, update
    from sqlalchemy.sql.dml import Update

    from bot.storage.sqlite import UtcDateTime

    assert getattr(Update, "order_by", None) is None
    assert getattr(Update, "limit", None) is None

    ticker = "KXHIGHDEN-26JUN01-T70"
    base = datetime(2026, 6, 2, 12, 0, tzinfo=_timezone.utc)
    for cid in ("kw-edge-a", "kw-edge-b"):
        session.add(
            DemoOrder(
                client_order_id=cid,
                exchange_order_id="EX-" + cid,
                market_ticker=ticker,
                strategy="edge",
                side="no",
                requested_contracts=10,
                filled_contracts=10,
                requested_yes_price_dollars=Decimal("0.58"),
                fair_at_entry=Decimal("0.62"),
                intended_at=base,
                avg_fill_price=Decimal("0.205"),
                fee_dollars=Decimal("0.07"),
                status="executed",
                placed_at=base,
                last_status_at=base,
            )
        )
    session.commit()

    orm_result = session.execute(
        update(DemoOrder)
        .where(DemoOrder.market_ticker == ticker)
        .where(DemoOrder.status == "executed")
        .where(DemoOrder.filled_contracts > 0)
        .values(realized_pnl_dollars=Decimal("1.530000"))
    )
    assert orm_result.rowcount == 2
    session.rollback()

    text_sql = text(
        """
        UPDATE demo_orders
           SET realized_pnl_dollars = :pnl
         WHERE id = (
             SELECT id FROM demo_orders
              WHERE market_ticker = :ticker
                AND status = 'executed'
                AND filled_contracts > 0
                AND client_order_id NOT LIKE 'kw-backfill-%'
                AND placed_at <= :snapshot_at
              ORDER BY placed_at DESC, id DESC
              LIMIT 1
         )
        """
    ).bindparams(
        bindparam("pnl", type_=Numeric(10, 6)),
        bindparam("snapshot_at", type_=UtcDateTime()),
    )
    text_result = session.execute(
        text_sql,
        {"pnl": Decimal("1.530000"), "ticker": ticker, "snapshot_at": base + timedelta(hours=1)},
    )
    assert text_result.rowcount == 1
    session.commit()

    pinned = session.scalars(
        select(DemoOrder).where(DemoOrder.realized_pnl_dollars.is_not(None))
    ).all()
    assert len(pinned) == 1
    assert pinned[0].client_order_id == "kw-edge-b"


def test_upgrade_schema_does_not_mutate_root_logger(tmp_path, caplog):
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    propagate_before = root.propagate

    db_file = tmp_path / "state.db"
    upgrade_schema(db_file)

    assert list(root.handlers) == handlers_before
    assert root.level == level_before
    assert root.propagate == propagate_before

    test_logger = logging.getLogger("test_upgrade_schema_caplog_probe")
    with caplog.at_level(logging.INFO, logger="test_upgrade_schema_caplog_probe"):
        test_logger.info("probe message")
    assert any("probe message" in r.getMessage() for r in caplog.records)
