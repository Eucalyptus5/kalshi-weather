from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from bot.storage.sqlite import ensure_baseline_stamped, make_engine, make_session_factory
from scripts.annotate_market_open_noise import annotate


SENTINELS_0002: tuple[tuple[str, str], ...] = (
    ("gate_failures", "notes"),
    ("paper_trades", "attempted_contracts"),
    ("orderbook_snapshots", "yes_ask_depth"),
)


def _detect_baseline(engine: Engine, script_dir: ScriptDirectory) -> str:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    present: set[tuple[str, str]] = set()
    for table, column in SENTINELS_0002:
        if table not in tables:
            continue
        if column in {c["name"] for c in inspector.get_columns(table)}:
            present.add((table, column))
    if not present:
        return "0001"
    if len(present) == len(SENTINELS_0002):
        return "0002"
    missing = set(SENTINELS_0002) - present
    raise RuntimeError(
        f"partial 0002 schema detected; present={sorted(present)} missing={sorted(missing)}"
    )


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
