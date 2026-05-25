from __future__ import annotations

from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from bot.storage.sqlite import make_engine


REPO_ROOT = Path(__file__).resolve().parent.parent


def _cfg(tmp_path: Path, db_file: Path) -> Config:
    ini_path = tmp_path / "alembic.ini"
    ini_path.write_text(
        f"[alembic]\n"
        f"script_location = {REPO_ROOT / 'alembic'}\n"
        f"sqlalchemy.url = sqlite:///{db_file}\n"
    )
    return Config(str(ini_path))


def _at_0003(tmp_path: Path) -> tuple[Path, Config]:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0003")
    return db_file, cfg


def _insert_paper_trade(engine, fair: Decimal) -> int:
    with engine.begin() as conn:
        when = datetime(2026, 5, 20, 12, 0, tzinfo=_timezone.utc)
        result = conn.execute(
            text(
                "INSERT INTO paper_trades "
                "(intended_at, market_ticker, side, contracts, simulated_price, "
                "fee_dollars, fair_at_entry, strategy, created_at) "
                "VALUES (:t, :mt, :side, :c, :sp, :fee, :fair, :strat, :ca)"
            ),
            {
                "t": when,
                "mt": "KXHIGHDEN-26MAY20-T70",
                "side": "buy_yes",
                "c": 10,
                "sp": float(Decimal("0.40")),
                "fee": float(Decimal("0.05")),
                "fair": float(fair),
                "strat": "edge",
                "ca": when,
            },
        )
        return int(result.lastrowid)


def _insert_demo_order(engine, fair: Decimal | None) -> int:
    with engine.begin() as conn:
        placed = datetime(2026, 5, 20, 12, 0, tzinfo=_timezone.utc)
        result = conn.execute(
            text(
                "INSERT INTO demo_orders "
                "(client_order_id, exchange_order_id, market_ticker, strategy, side, "
                "requested_contracts, filled_contracts, fair_at_entry, status, "
                "placed_at, last_status_at, created_at) "
                "VALUES (:cid, :eid, :mt, :strat, :side, :rc, :fc, :fair, :stat, "
                ":pl, :ls, :ca)"
            ),
            {
                "cid": "kw-edge-yes-A",
                "eid": "EX1",
                "mt": "KXHIGHDEN-26MAY20-T70",
                "strat": "edge",
                "side": "yes",
                "rc": 10,
                "fc": 10,
                "fair": float(fair) if fair is not None else None,
                "stat": "executed",
                "pl": placed,
                "ls": placed,
                "ca": placed,
            },
        )
        return int(result.lastrowid)


def test_migration_0004_backfills_paper_trades_q_raw_from_fair_at_entry(tmp_path) -> None:
    db_file, cfg = _at_0003(tmp_path)
    engine = make_engine(db_file)
    _insert_paper_trade(engine, Decimal("0.42"))
    engine.dispose()
    command.upgrade(cfg, "0004")
    engine = make_engine(db_file)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT fair_at_entry, q_raw FROM paper_trades")).one()
    engine.dispose()
    assert row[0] == row[1]
    assert Decimal(str(row[1])) == Decimal("0.42")


def test_migration_0004_paper_trades_q_raw_is_not_null_after_upgrade(tmp_path) -> None:
    db_file, cfg = _at_0003(tmp_path)
    command.upgrade(cfg, "0004")
    engine = make_engine(db_file)
    cols = {c["name"]: c for c in inspect(engine).get_columns("paper_trades")}
    engine.dispose()
    assert cols["q_raw"]["nullable"] is False


def test_migration_0004_backfills_demo_orders_q_raw_from_non_null_fair_at_entry(
    tmp_path,
) -> None:
    db_file, cfg = _at_0003(tmp_path)
    engine = make_engine(db_file)
    _insert_demo_order(engine, Decimal("0.62"))
    engine.dispose()
    command.upgrade(cfg, "0004")
    engine = make_engine(db_file)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT fair_at_entry, q_raw FROM demo_orders")).one()
    engine.dispose()
    assert Decimal(str(row[0])) == Decimal("0.62")
    assert Decimal(str(row[1])) == Decimal("0.62")


def test_migration_0004_demo_orders_q_raw_stays_nullable(tmp_path) -> None:
    db_file, cfg = _at_0003(tmp_path)
    command.upgrade(cfg, "0004")
    engine = make_engine(db_file)
    cols = {c["name"]: c for c in inspect(engine).get_columns("demo_orders")}
    engine.dispose()
    assert cols["q_raw"]["nullable"] is True


def test_migration_0004_leaves_orphan_demo_rows_with_null_q_raw(tmp_path) -> None:
    db_file, cfg = _at_0003(tmp_path)
    engine = make_engine(db_file)
    _insert_demo_order(engine, None)
    engine.dispose()
    command.upgrade(cfg, "0004")
    engine = make_engine(db_file)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT fair_at_entry, q_raw FROM demo_orders")).one()
    engine.dispose()
    assert row[0] is None
    assert row[1] is None


def test_migration_0004_widens_gate_failures_reason_to_512(tmp_path) -> None:
    db_file, cfg = _at_0003(tmp_path)
    command.upgrade(cfg, "0004")
    engine = make_engine(db_file)
    cols = {c["name"]: c for c in inspect(engine).get_columns("gate_failures")}
    engine.dispose()
    reason_type = cols["reason"]["type"]
    assert getattr(reason_type, "length", None) == 512


def test_migration_0004_aborts_when_paper_trades_fair_at_entry_out_of_range(
    tmp_path,
) -> None:
    db_file, cfg = _at_0003(tmp_path)
    engine = make_engine(db_file)
    _insert_paper_trade(engine, Decimal("1.5"))
    engine.dispose()
    with pytest.raises(RuntimeError, match="migration_0004_aborted"):
        command.upgrade(cfg, "0004")
    engine = make_engine(db_file)
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()
    assert version == "0003"


def test_migration_0004_aborts_when_demo_orders_fair_at_entry_out_of_range(
    tmp_path,
) -> None:
    db_file, cfg = _at_0003(tmp_path)
    engine = make_engine(db_file)
    _insert_demo_order(engine, Decimal("1.5"))
    engine.dispose()
    with pytest.raises(RuntimeError, match="migration_0004_aborted"):
        command.upgrade(cfg, "0004")
    engine = make_engine(db_file)
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()
    assert version == "0003"
