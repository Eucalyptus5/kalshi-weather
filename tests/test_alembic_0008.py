from __future__ import annotations

from datetime import datetime
from datetime import timezone as _timezone
from decimal import Decimal
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


def _at_0007(tmp_path: Path) -> tuple[Path, Config]:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0007")
    return db_file, cfg


def _seed_legacy_snapshot(engine, *, portfolio_value_dollars: Decimal) -> None:
    when = datetime(2026, 6, 2, 12, 0, tzinfo=_timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO portfolio_snapshots "
                "(snapshot_at, cash_dollars, portfolio_value_dollars, "
                "total_exposure_dollars, realized_pnl_dollars, fees_paid_dollars, "
                "open_positions_count, created_at) "
                "VALUES (:sa, :cd, :pv, :te, :rp, :fp, :op, :ca)"
            ),
            {
                "sa": when,
                "cd": float(Decimal("1020.000000")),
                "pv": float(portfolio_value_dollars),
                "te": float(Decimal("4.380000")),
                "rp": float(Decimal("0.000000")),
                "fp": float(Decimal("0.000000")),
                "op": 2,
                "ca": when,
            },
        )


def test_migration_0008_renames_column_and_preserves_value(tmp_path) -> None:
    db_file, cfg = _at_0007(tmp_path)
    engine = make_engine(db_file)
    _seed_legacy_snapshot(engine, portfolio_value_dollars=Decimal("1024.380000"))
    engine.dispose()

    command.upgrade(cfg, "0008")

    engine = make_engine(db_file)
    cols = {c["name"]: c for c in inspect(engine).get_columns("portfolio_snapshots")}
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT total_collateral_dollars, portfolio_value_mtm_dollars "
                "FROM portfolio_snapshots"
            )
        ).one()
    engine.dispose()

    assert "total_collateral_dollars" in cols
    assert "portfolio_value_dollars" not in cols
    assert "portfolio_value_mtm_dollars" in cols
    assert cols["portfolio_value_mtm_dollars"]["nullable"] is True
    assert Decimal(str(row[0])) == Decimal("1024.380000")
    assert row[1] is None


def test_migration_0008_downgrade_restores_legacy_column(tmp_path) -> None:
    db_file, cfg = _at_0007(tmp_path)
    engine = make_engine(db_file)
    _seed_legacy_snapshot(engine, portfolio_value_dollars=Decimal("1024.380000"))
    engine.dispose()

    command.upgrade(cfg, "0008")
    command.downgrade(cfg, "0007")

    engine = make_engine(db_file)
    cols = {c["name"] for c in inspect(engine).get_columns("portfolio_snapshots")}
    with engine.connect() as conn:
        row = conn.execute(text("SELECT portfolio_value_dollars FROM portfolio_snapshots")).one()
    engine.dispose()

    assert "portfolio_value_dollars" in cols
    assert "total_collateral_dollars" not in cols
    assert "portfolio_value_mtm_dollars" not in cols
    assert Decimal(str(row[0])) == Decimal("1024.380000")


def test_detect_baseline_returns_0008_on_fresh_head(tmp_path) -> None:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0008")
    engine = make_engine(db_file)
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    engine.dispose()
    assert baseline == "0008"


def test_detect_baseline_returns_0007_when_renamed_column_only(tmp_path) -> None:
    db_file, cfg = _at_0007(tmp_path)
    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE portfolio_snapshots RENAME COLUMN "
                "portfolio_value_dollars TO total_collateral_dollars"
            )
        )
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    engine.dispose()
    assert baseline == "0007"


def test_detect_baseline_raises_on_partial_0008_schema(tmp_path) -> None:
    db_file, cfg = _at_0007(tmp_path)
    engine = make_engine(db_file)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE portfolio_snapshots DROP COLUMN portfolio_value_dollars"))
    script_dir = ScriptDirectory.from_config(cfg)
    with pytest.raises(RuntimeError, match="partial 0008 schema"):
        _detect_baseline(engine, script_dir)
    engine.dispose()
