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
