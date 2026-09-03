import ast
import inspect
from pathlib import Path
from types import ModuleType

import pytest

from bot.lag import (
    depth_continuity,
    depth_map_run,
    ladder_run,
    lead_lag_run,
    lock_convergence,
    maker_edge_run,
    near_lock,
    settlement_run,
    taker_flow_run,
)
from bot.lag.tape_studies import LADDER, TOUCH, TRADES
from tests import test_maker_edge_run, test_settlement_run


LAG = Path(__file__).resolve().parent.parent / "bot" / "lag"
DECLARED = (
    (ladder_run, (TOUCH,)),
    (lead_lag_run, (TOUCH,)),
    (depth_continuity, (LADDER,)),
    (settlement_run, (LADDER,)),
    (depth_map_run, (LADDER, TRADES)),
    (maker_edge_run, (LADDER, TRADES)),
    (lock_convergence, (TOUCH, TRADES)),
    (near_lock, (TOUCH, TRADES)),
    (taker_flow_run, (TOUCH, TRADES)),
)
RUN_NAMES = [module.__name__ for module, _ in DECLARED]
UNREAD = "unread"
ASSEMBLER = "assemble_run_inputs"


class Reached(Exception):
    """The run reached the assembler, so the recorder holds the kinds it declared."""


def calls_the_assembler(source: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == ASSEMBLER)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == ASSEMBLER)
        )
        for node in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize(("module", "kinds"), DECLARED, ids=RUN_NAMES)
def test_every_run_declares_the_kinds_it_reads(
    module: ModuleType,
    kinds: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, ...]] = []

    def recorder(**passed: object) -> None:
        seen.append(passed["kinds"])
        raise Reached

    monkeypatch.setattr(module, "assemble_run_inputs", recorder)
    # The seven runs that pre-flight nothing reach the assembler on the first statement of execute,
    # so what they forward is never read and one stand-in serves for every argument; the two that
    # check a sidecar first need the tree their own suite builds.
    with pytest.raises(Reached):
        if module is settlement_run:
            test_settlement_run.run_at(tmp_path, test_settlement_run.run_paths(tmp_path))
        elif module is maker_edge_run:
            test_maker_edge_run.run_at(tmp_path, test_maker_edge_run.run_paths(tmp_path))
        else:
            module.execute(**dict.fromkeys(inspect.signature(module.execute).parameters, UNREAD))

    assert seen == [kinds]


def test_every_run_that_assembles_inputs_names_its_kinds_here() -> None:
    assembling = sorted(
        f"bot.lag.{path.stem}" for path in LAG.glob("*.py") if calls_the_assembler(path.read_text())
    )

    assert assembling == sorted(RUN_NAMES)
