from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as _timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

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


def _at_0005(tmp_path: Path) -> tuple[Path, Config]:
    db_file = tmp_path / "state.db"
    cfg = _cfg(tmp_path, db_file)
    command.upgrade(cfg, "0005")
    return db_file, cfg


def _insert_gate_failure(
    engine,
    *,
    evaluated_at: datetime,
    gate_name: str,
    reason: str,
    market_ticker: str | None,
    created_at: datetime,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO gate_failures "
                "(evaluated_at, gate_name, reason, mode, market_ticker, created_at) "
                "VALUES (:e, :g, :r, :m, :mt, :ca)"
            ),
            {
                "e": evaluated_at,
                "g": gate_name,
                "r": reason,
                "m": "paper",
                "mt": market_ticker,
                "ca": created_at,
            },
        )


def test_migration_0006_adds_count_and_last_seen_at_columns(tmp_path) -> None:
    db_file, cfg = _at_0005(tmp_path)
    command.upgrade(cfg, "0006")
    engine = make_engine(db_file)
    cols = {c["name"]: c for c in inspect(engine).get_columns("gate_failures")}
    engine.dispose()
    assert "count" in cols
    assert "last_seen_at" in cols
    assert cols["last_seen_at"]["nullable"] is False


def test_migration_0006_creates_unique_dedup_index(tmp_path) -> None:
    db_file, cfg = _at_0005(tmp_path)
    command.upgrade(cfg, "0006")
    engine = make_engine(db_file)
    indexes = inspect(engine).get_indexes("gate_failures")
    engine.dispose()
    dedup = [ix for ix in indexes if ix["name"] == "ux_gate_failures_dedup"]
    assert len(dedup) == 1
    assert bool(dedup[0]["unique"]) is True
    assert dedup[0]["column_names"] == ["gate_name", "market_ticker", "reason"]


def test_migration_0006_collapses_duplicates_across_multiple_tickers(tmp_path) -> None:
    db_file, cfg = _at_0005(tmp_path)
    engine = make_engine(db_file)
    base = datetime(2026, 5, 30, 12, 0, tzinfo=_timezone.utc)
    for i in range(5):
        _insert_gate_failure(
            engine,
            evaluated_at=base + timedelta(minutes=i),
            gate_name="fair_value_sane",
            reason="fair_yes=9.649e-09 outside [0.01, 0.99]",
            market_ticker="KXHIGHTPHX-26JUN01-B106.5",
            created_at=base + timedelta(minutes=i),
        )
    for i in range(3):
        _insert_gate_failure(
            engine,
            evaluated_at=base + timedelta(minutes=10 + i),
            gate_name="fair_value_sane",
            reason="fair_yes=9.649e-09 outside [0.01, 0.99]",
            market_ticker="KXHIGHTNY-26JUN01-B72.5",
            created_at=base + timedelta(minutes=10 + i),
        )
    _insert_gate_failure(
        engine,
        evaluated_at=base + timedelta(minutes=20),
        gate_name="model_fresh",
        reason="model_age_hours=7.5 > 6",
        market_ticker="KXHIGHTLAX-26JUN01-B72.5",
        created_at=base + timedelta(minutes=20),
    )
    engine.dispose()

    command.upgrade(cfg, "0006")

    engine = make_engine(db_file)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT gate_name, market_ticker, count, last_seen_at "
                "FROM gate_failures ORDER BY gate_name, market_ticker"
            )
        ).all()
    engine.dispose()

    by_key = {(r[0], r[1]): (int(r[2]), r[3]) for r in rows}
    assert len(by_key) == 3
    assert by_key[("fair_value_sane", "KXHIGHTPHX-26JUN01-B106.5")][0] == 5
    assert by_key[("fair_value_sane", "KXHIGHTNY-26JUN01-B72.5")][0] == 3
    assert by_key[("model_fresh", "KXHIGHTLAX-26JUN01-B72.5")][0] == 1

    phx_last = by_key[("fair_value_sane", "KXHIGHTPHX-26JUN01-B106.5")][1]
    ny_last = by_key[("fair_value_sane", "KXHIGHTNY-26JUN01-B72.5")][1]
    assert datetime.fromisoformat(str(phx_last).replace(" ", "T")).replace(
        tzinfo=_timezone.utc
    ) == base + timedelta(minutes=4)
    assert datetime.fromisoformat(str(ny_last).replace(" ", "T")).replace(
        tzinfo=_timezone.utc
    ) == base + timedelta(minutes=12)


def test_migration_0006_unique_index_rejects_post_upgrade_duplicate(tmp_path) -> None:
    db_file, cfg = _at_0005(tmp_path)
    engine = make_engine(db_file)
    base = datetime(2026, 5, 30, 12, 0, tzinfo=_timezone.utc)
    _insert_gate_failure(
        engine,
        evaluated_at=base,
        gate_name="fair_value_sane",
        reason="fair_yes=9.649e-09 outside [0.01, 0.99]",
        market_ticker="KXHIGHTPHX-26JUN01-B106.5",
        created_at=base,
    )
    engine.dispose()
    command.upgrade(cfg, "0006")

    engine = make_engine(db_file)
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO gate_failures "
                    "(evaluated_at, gate_name, reason, mode, market_ticker, "
                    "count, last_seen_at, created_at) "
                    "VALUES (:e, :g, :r, 'paper', :mt, 1, :lsa, :ca)"
                ),
                {
                    "e": base,
                    "g": "fair_value_sane",
                    "r": "fair_yes=9.649e-09 outside [0.01, 0.99]",
                    "mt": "KXHIGHTPHX-26JUN01-B106.5",
                    "lsa": base,
                    "ca": base,
                },
            )
    engine.dispose()


def test_regression_phoenix_only_delete_then_index_raises_integrity_error(
    tmp_path,
) -> None:
    db_file, cfg = _at_0005(tmp_path)
    engine = make_engine(db_file)
    base = datetime(2026, 5, 30, 12, 0, tzinfo=_timezone.utc)
    for i in range(5):
        _insert_gate_failure(
            engine,
            evaluated_at=base + timedelta(minutes=i),
            gate_name="fair_value_sane",
            reason="fair_yes=9.649e-09 outside [0.01, 0.99]",
            market_ticker="KXHIGHTPHX-26JUN01-B106.5",
            created_at=base + timedelta(minutes=i),
        )
    for i in range(3):
        _insert_gate_failure(
            engine,
            evaluated_at=base + timedelta(minutes=10 + i),
            gate_name="fair_value_sane",
            reason="fair_yes=9.649e-09 outside [0.01, 0.99]",
            market_ticker="KXHIGHTNY-26JUN01-B72.5",
            created_at=base + timedelta(minutes=10 + i),
        )

    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM gate_failures "
                "WHERE market_ticker = 'KXHIGHTPHX-26JUN01-B106.5' "
                "AND id NOT IN (SELECT MIN(id) FROM gate_failures "
                "WHERE market_ticker = 'KXHIGHTPHX-26JUN01-B106.5' "
                "GROUP BY gate_name, market_ticker, reason)"
            )
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX ux_gate_failures_dedup_probe "
                    "ON gate_failures (gate_name, market_ticker, reason)"
                )
            )
    engine.dispose()
