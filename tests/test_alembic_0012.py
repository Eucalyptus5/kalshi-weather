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


def _at_0011(tmp_path: Path) -> tuple[Path, Config]:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0011")
    return db_file, cfg


def _cols(engine, table: str) -> dict[str, dict]:
    return {c["name"]: c for c in inspect(engine).get_columns(table)}


def test_migration_0012_creates_ws_obs_arrivals(tmp_path) -> None:
    db_file, cfg = _at_0011(tmp_path)

    command.upgrade(cfg, "0012")

    engine = make_engine(db_file)
    cols = _cols(engine, "ws_obs_arrivals")
    indexes = {idx["name"]: idx for idx in inspect(engine).get_indexes("ws_obs_arrivals")}
    pk = inspect(engine).get_pk_constraint("ws_obs_arrivals")
    engine.dispose()

    assert pk["constrained_columns"] == ["id"]

    for name in ("station", "source", "tmpf"):
        assert cols[name]["nullable"] is False
        assert "CHAR" in str(cols[name]["type"]).upper()

    for name in ("obs_time", "received_at", "created_at"):
        assert cols[name]["nullable"] is False
        assert "DATETIME" in str(cols[name]["type"]).upper()

    assert "ix_ws_obs_arrivals_station_obs_time" in indexes
    assert indexes["ix_ws_obs_arrivals_station_obs_time"]["column_names"] == [
        "station",
        "obs_time",
    ]
    assert not indexes["ix_ws_obs_arrivals_station_obs_time"]["unique"]


def test_migration_0012_downgrade_drops_table(tmp_path) -> None:
    db_file, cfg = _at_0011(tmp_path)
    command.upgrade(cfg, "0012")

    command.downgrade(cfg, "0011")

    engine = make_engine(db_file)
    tables = set(inspect(engine).get_table_names())
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()

    assert "ws_obs_arrivals" not in tables
    assert version == "0011"


def test_detect_baseline_returns_0012_on_fresh_head(tmp_path) -> None:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0012")
    engine = make_engine(db_file)
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    engine.dispose()
    assert baseline == "0012"


def test_detect_baseline_raises_on_partial_0012_schema(tmp_path) -> None:
    db_file, cfg = _at_0011(tmp_path)
    command.upgrade(cfg, "0012")
    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE ws_obs_arrivals DROP COLUMN received_at"))
    script_dir = ScriptDirectory.from_config(cfg)
    with pytest.raises(RuntimeError, match="partial 0012 schema"):
        _detect_baseline(engine, script_dir)
    engine.dispose()
