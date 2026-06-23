from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from bot.storage.sqlite import make_engine
from scripts.migrate import _detect_baseline


REPO_ROOT = Path(__file__).resolve().parent.parent

PRE_0009_TABLES = {
    "forecasts",
    "markets",
    "orderbook_snapshots",
    "paper_trades",
    "demo_orders",
    "simulated_pnl",
    "reconciler_state",
    "portfolio_snapshots",
    "gate_failures",
}


def _cfg(tmp_path: Path, db_file: Path) -> Config:
    ini_path = tmp_path / "alembic.ini"
    ini_path.write_text(
        f"[alembic]\n"
        f"script_location = {REPO_ROOT / 'alembic'}\n"
        f"sqlalchemy.url = sqlite:///{db_file}\n"
    )
    return Config(str(ini_path))


def _at_0008(tmp_path: Path) -> tuple[Path, Config]:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0008")
    return db_file, cfg


def test_migration_0009_creates_ws_tables_and_indexes(tmp_path) -> None:
    db_file, cfg = _at_0008(tmp_path)

    command.upgrade(cfg, "0009")

    engine = make_engine(db_file)
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    book_indexes = {i["name"] for i in insp.get_indexes("ws_book_events")}
    trade_indexes = {i["name"] for i in insp.get_indexes("ws_trades")}
    gap_indexes = {i["name"] for i in insp.get_indexes("ws_gaps")}
    engine.dispose()

    assert {"ws_book_events", "ws_trades", "ws_gaps"} <= tables
    assert "ix_ws_book_events_ticker_received_at" in book_indexes
    assert "ix_ws_trades_ticker_received_at" in trade_indexes
    assert "ix_ws_gaps_ticker_detected_at" in gap_indexes


def test_migration_0009_downgrade_drops_ws_tables_only(tmp_path) -> None:
    db_file, cfg = _at_0008(tmp_path)
    command.upgrade(cfg, "0009")

    command.downgrade(cfg, "0008")

    engine = make_engine(db_file)
    tables = set(inspect(engine).get_table_names())
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()

    assert not {"ws_book_events", "ws_trades", "ws_gaps"} & tables
    assert PRE_0009_TABLES <= tables
    assert version == "0008"


def test_detect_baseline_returns_0009_on_fresh_head(tmp_path) -> None:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0009")
    engine = make_engine(db_file)
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    engine.dispose()
    assert baseline == "0009"


def test_detect_baseline_raises_on_partial_0009_schema(tmp_path) -> None:
    db_file, cfg = _at_0008(tmp_path)
    command.upgrade(cfg, "0009")
    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE ws_gaps"))
    script_dir = ScriptDirectory.from_config(cfg)
    with pytest.raises(RuntimeError, match="partial 0009 schema"):
        _detect_baseline(engine, script_dir)
    engine.dispose()
