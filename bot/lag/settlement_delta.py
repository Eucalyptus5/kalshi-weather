from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from bot.lag.settlement_straddle import Straddle
from bot.markets.observation_window import observation_window
from bot.observations.metar import StationObservation


SURVIVES = "rounding survives the check"
FALSIFIED = "rounding falsified"


@dataclass(frozen=True, slots=True, kw_only=True)
class DeltaPartition:
    abs_delta_gt_1: int
    abs_delta_le_1: int
    delta_histogram: Mapping[int, int]
    coverage_minutes: Mapping[tuple[str, date], int]
    abs_delta_gt_1_rows: tuple[tuple[str, date], ...]


# Both sources report whole degrees, so an integral delta is the invariant the histogram key rests
# on. Truncating a fractional one would key it under a magnitude nobody measured, and a fractional
# delta means an upstream encoding changed rather than that this day is unusual.
def delta_of(straddle: Straddle) -> int:
    delta = straddle.observed_f - straddle.acis_f
    if delta != delta.to_integral_value():
        raise ValueError(
            f"{straddle.station} {straddle.event_date} carries a fractional delta of {delta}"
        )
    return int(delta)


# Half-open, matching the entry walk: the next event day's window opens where this one closes, and
# a boundary minute counted twice would credit one reading to two days' coverage.
def coverage_of(straddle: Straddle, observations: Sequence[StationObservation]) -> int:
    start, end = observation_window(straddle.timezone, straddle.event_date)
    return len(
        {
            observation.valid_time.replace(second=0, microsecond=0)
            for observation in observations
            if start <= observation.valid_time < end
        }
    )


def delta_partition(
    rows: Sequence[tuple[Straddle, Sequence[StationObservation]]],
) -> DeltaPartition:
    deltas = [(straddle, delta_of(straddle)) for straddle, _ in rows]
    wide = tuple(
        (straddle.station, straddle.event_date) for straddle, delta in deltas if abs(delta) > 1
    )
    return DeltaPartition(
        abs_delta_gt_1=len(wide),
        abs_delta_le_1=sum(1 for _, delta in deltas if abs(delta) <= 1),
        delta_histogram=dict(Counter(delta for _, delta in deltas)),
        coverage_minutes={
            (straddle.station, straddle.event_date): coverage_of(straddle, observations)
            for straddle, observations in rows
        },
        abs_delta_gt_1_rows=wide,
    )


# A rounding difference between two whole-degree sources cannot exceed one, so a wider delta rules
# rounding out. Nothing here can rule it in: an all-narrow set is equally what an aggregation
# difference produces on days where both sides cover the window.
def supported_reading(partition: DeltaPartition) -> str:
    return FALSIFIED if partition.abs_delta_gt_1 else SURVIVES
