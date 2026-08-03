from collections.abc import Iterable
from dataclasses import replace

from bot.main import RECORDING_SERIES, STATIONS, StationConfig


# Not folded into bot.main.STATIONS: that map drives what the live bot forecasts, evaluates and
# persists, so widening it would change the bot rather than what analysis is allowed to read.
LOW_TO_HIGH: dict[str, str] = {
    "KXLOWTBOS": "KXHIGHTBOS",
    "KXLOWTDEN": "KXHIGHDEN",
    "KXLOWTAUS": "KXHIGHAUS",
    "KXLOWTCHI": "KXHIGHCHI",
    "KXLOWTNYC": "KXHIGHNY",
    "KXLOWTPHIL": "KXHIGHPHIL",
    "KXLOWTATL": "KXHIGHTATL",
    "KXLOWTDAL": "KXHIGHTDAL",
    "KXLOWTDC": "KXHIGHTDC",
    "KXLOWTHOU": "KXHIGHTHOU",
    "KXLOWTLV": "KXHIGHTLV",
    "KXLOWTMIN": "KXHIGHTMIN",
    "KXLOWTNOLA": "KXHIGHTNOLA",
    "KXLOWTOKC": "KXHIGHTOKC",
    "KXLOWTPHX": "KXHIGHTPHX",
    "KXLOWTSATX": "KXHIGHTSATX",
    "KXLOWTSEA": "KXHIGHTSEA",
    "KXLOWTSFO": "KXHIGHTSFO",
    "KXLOWTLAX": "KXHIGHLAX",
    "KXLOWTMIA": "KXHIGHMIA",
}

assert set(LOW_TO_HIGH) == {root for root in RECORDING_SERIES if root.startswith("KXLOW")}
assert set(LOW_TO_HIGH.values()) == set(STATIONS)

# Each low series settles on exactly its paired high series' station, so the station, timezone and
# coordinates are derived from the high side and cannot drift away from it.
LOW_STATIONS: dict[str, StationConfig] = {
    low: replace(STATIONS[high], series=low) for low, high in LOW_TO_HIGH.items()
}

ANALYSIS_STATIONS: dict[str, StationConfig] = {**STATIONS, **LOW_STATIONS}

assert len(ANALYSIS_STATIONS) == 40

HIGH = "high"
LOW = "low"


def ladder_of(series: str) -> str:
    return LOW if series in LOW_TO_HIGH else HIGH


# The two ladders settle off the same stations on the same days, so a sweep handed both counts
# every city twice and reads as twice the evidence it holds.
def in_cohort(series: Iterable[str], cohort: str | None) -> tuple[str, ...]:
    names = tuple(sorted(set(series)))
    if cohort is None:
        if len({ladder_of(name) for name in names}) > 1:
            raise ValueError(
                "the frozen scope spans both ladders and the run names no cohort: "
                + ", ".join(names)
            )
        return names
    kept = tuple(name for name in names if ladder_of(name) == cohort)
    if not kept:
        raise ValueError(f"the frozen scope holds no {cohort} series: " + ", ".join(names))
    return kept
