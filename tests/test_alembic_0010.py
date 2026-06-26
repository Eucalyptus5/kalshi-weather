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


def _cfg(tmp_path: Path, db_file: Path) -> Config:
    ini_path = tmp_path / "alembic.ini"
    ini_path.write_text(
        f"[alembic]\n"
        f"script_location = {REPO_ROOT / 'alembic'}\n"
        f"sqlalchemy.url = sqlite:///{db_file}\n"
    )
    return Config(str(ini_path))


def _at_0009(tmp_path: Path) -> tuple[Path, Config]:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0009")
    return db_file, cfg


def _cols(engine, table: str) -> dict[str, dict]:
    return {c["name"]: c for c in inspect(engine).get_columns(table)}


def test_migration_0010_adds_ts_ms_and_trade_id(tmp_path) -> None:
    db_file, cfg = _at_0009(tmp_path)

    command.upgrade(cfg, "0010")

    engine = make_engine(db_file)
    book = _cols(engine, "ws_book_events")
    trade = _cols(engine, "ws_trades")
    engine.dispose()

    assert "ts_ms" in book
    assert book["ts_ms"]["nullable"] is True
    assert str(book["ts_ms"]["type"]).upper().startswith("INTEGER")

    assert "ts_ms" in trade
    assert trade["ts_ms"]["nullable"] is False
    assert str(trade["ts_ms"]["type"]).upper().startswith("INTEGER")

    assert "trade_id" in trade
    assert trade["trade_id"]["nullable"] is False
    assert (
        "VARCHAR" in str(trade["trade_id"]["type"]).upper()
        or "TEXT" in str(trade["trade_id"]["type"]).upper()
    )


def test_migration_0010_downgrade_drops_new_columns(tmp_path) -> None:
    db_file, cfg = _at_0009(tmp_path)
    command.upgrade(cfg, "0010")

    command.downgrade(cfg, "0009")

    engine = make_engine(db_file)
    book = _cols(engine, "ws_book_events")
    trade = _cols(engine, "ws_trades")
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()

    assert "ts_ms" not in book
    assert "ts_ms" not in trade
    assert "trade_id" not in trade
    assert version == "0009"


def test_detect_baseline_returns_0010_on_fresh_head(tmp_path) -> None:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0010")
    engine = make_engine(db_file)
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    engine.dispose()
    assert baseline == "0010"


def test_detect_baseline_raises_on_partial_0010_schema(tmp_path) -> None:
    db_file, cfg = _at_0009(tmp_path)
    command.upgrade(cfg, "0010")
    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE ws_book_events DROP COLUMN ts_ms"))
    script_dir = ScriptDirectory.from_config(cfg)
    with pytest.raises(RuntimeError, match="partial 0010 schema"):
        _detect_baseline(engine, script_dir)
    engine.dispose()
