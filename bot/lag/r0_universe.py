import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq


# Not bot.main.STRATEGY_BLACKLIST: that frozenset also drops KXHIGHLAX on a strategy verdict, and
# the carve-out here is the KMIA basis offset alone.
LOCK_CARVE_OUT = "KXHIGHMIA"

AGREE = "agree"
DISAGREE = "disagree"


@dataclass(frozen=True, slots=True)
class Coverage:
    cities: tuple[str, ...]
    ladder_widths: tuple[int, ...]
    in_scope_city_days: int


@dataclass(frozen=True, slots=True)
class R0Universe:
    fraction_invalid_max: Decimal
    passing: tuple[str, ...]
    lock_dependent: tuple[str, ...]
    recorded: tuple[str, ...]
    ladder_widths: tuple[int, ...]
    in_scope_city_days: int
    reconciliation: str
    recorded_not_passing: tuple[str, ...]
    passing_not_recorded: tuple[str, ...]


def read_recorded_coverage(path: Path) -> Coverage:
    rows = pq.read_table(path).to_pylist()
    scoped = [row for row in rows if row["in_scope"]]
    return Coverage(
        cities=tuple(sorted({row["series"] for row in rows})),
        ladder_widths=tuple(sorted({row["tickers"] for row in scoped})),
        in_scope_city_days=len(scoped),
    )


def lock_dependent_series(passing: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(series for series in passing if series != LOCK_CARVE_OUT))


def freeze_universe(
    *, fraction_invalid_max: Decimal, passing: Sequence[str], coverage: Coverage
) -> R0Universe:
    ordered = tuple(sorted(passing))
    recorded = tuple(sorted(coverage.cities))
    extra = tuple(series for series in recorded if series not in set(ordered))
    missing = tuple(series for series in ordered if series not in set(recorded))
    return R0Universe(
        fraction_invalid_max=fraction_invalid_max,
        passing=ordered,
        lock_dependent=lock_dependent_series(ordered),
        recorded=recorded,
        ladder_widths=coverage.ladder_widths,
        in_scope_city_days=coverage.in_scope_city_days,
        reconciliation=AGREE if not extra and not missing else DISAGREE,
        recorded_not_passing=extra,
        passing_not_recorded=missing,
    )


def universe_payload(universe: R0Universe) -> dict:
    return {
        "fraction_invalid_max": str(universe.fraction_invalid_max),
        "passing": list(universe.passing),
        "lock_carve_out": LOCK_CARVE_OUT,
        "lock_dependent": list(universe.lock_dependent),
        "recorded": list(universe.recorded),
        "ladder_widths": list(universe.ladder_widths),
        "in_scope_city_days": universe.in_scope_city_days,
        "reconciliation": universe.reconciliation,
        "recorded_not_passing": list(universe.recorded_not_passing),
        "passing_not_recorded": list(universe.passing_not_recorded),
    }


# Deliberately not imported from bot.replay.run_scope, for the reason taker_side.Split gives: that
# module pulls bot.main, and bot/replay already depends on bot/lag.
def freeze_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_universe(path: Path, universe: R0Universe) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = universe_payload(universe)
    digest = freeze_digest(payload)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=1))
    return digest
