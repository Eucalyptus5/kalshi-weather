from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector

from bot.storage.sqlite import ensure_baseline_stamped, make_engine, make_session_factory
from scripts.annotate_market_open_noise import annotate


SENTINELS_0002: tuple[tuple[str, str], ...] = (
    ("gate_failures", "notes"),
    ("paper_trades", "attempted_contracts"),
    ("orderbook_snapshots", "yes_ask_depth"),
)

SENTINELS_0003: tuple[tuple[str, str], ...] = (
    ("demo_orders", "client_order_id"),
    ("paper_trades", "demo_order_client_id"),
)

SENTINELS_0004: tuple[tuple[str, str], ...] = (
    ("paper_trades", "q_raw"),
    ("demo_orders", "q_raw"),
)


def _present_sentinels(
    inspector: Inspector, tables: set[str], sentinels: tuple[tuple[str, str], ...]
) -> set[tuple[str, str]]:
    present: set[tuple[str, str]] = set()
    for table, column in sentinels:
        if table not in tables:
            continue
        if column in {c["name"] for c in inspector.get_columns(table)}:
            present.add((table, column))
    return present


def _detect_baseline(engine: Engine, script_dir: ScriptDirectory) -> str:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    present_0002 = _present_sentinels(inspector, tables, SENTINELS_0002)
    if not present_0002:
        return "0001"
    if len(present_0002) != len(SENTINELS_0002):
        missing = set(SENTINELS_0002) - present_0002
        raise RuntimeError(
            f"partial 0002 schema detected; present={sorted(present_0002)} missing={sorted(missing)}"
        )

    present_0003 = _present_sentinels(inspector, tables, SENTINELS_0003)
    if not present_0003:
        return "0002"
    if len(present_0003) != len(SENTINELS_0003):
        missing = set(SENTINELS_0003) - present_0003
        raise RuntimeError(
            f"partial 0003 schema detected; present={sorted(present_0003)} missing={sorted(missing)}"
        )

    present_0004 = _present_sentinels(inspector, tables, SENTINELS_0004)
    if not present_0004:
        return "0003"
    if len(present_0004) != len(SENTINELS_0004):
        missing = set(SENTINELS_0004) - present_0004
        raise RuntimeError(
            f"partial 0004 schema detected; present={sorted(present_0004)} missing={sorted(missing)}"
        )
    return "0004"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    db_path = repo_root / "data" / "state.db"
    engine = make_engine(str(db_path))
    cfg = Config(str(repo_root / "alembic.ini"))
    script_dir = ScriptDirectory.from_config(cfg)
    baseline = _detect_baseline(engine, script_dir)
    ensure_baseline_stamped(engine, baseline)
    with engine.connect() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")
    session_factory = make_session_factory(engine)
    rows_annotated = annotate(session_factory)
    print(f"migrate_ok rows_annotated={rows_annotated}")


if __name__ == "__main__":
    main()
