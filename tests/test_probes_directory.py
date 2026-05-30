from __future__ import annotations

import ast
from pathlib import Path

PROBES_DIR = Path(__file__).resolve().parent.parent / "scripts" / "probes"
FORBIDDEN = frozenset({"bot.main", "bot.execution.order_placer"})


def _imported_modules(tree: ast.AST) -> set[str]:
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                seen.add(node.module)
    return seen


def test_probes_directory_exists() -> None:
    assert PROBES_DIR.is_dir(), f"missing directory: {PROBES_DIR}"


def test_probes_do_not_import_bot_main_or_order_placer() -> None:
    for py in PROBES_DIR.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        imports = _imported_modules(tree)
        offending = imports & FORBIDDEN
        assert not offending, (
            f"{py.relative_to(PROBES_DIR.parent.parent)} imports {sorted(offending)}"
        )
