import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from bot.lag.lock_convergence import ROUNDING_MARGIN_F
from bot.lag.lock_events import detect_lock_events, is_low_ladder
from bot.lag.observation_freeze import ObservationIndex, read_observation_sidecar
from bot.lag.placement_grid import CloseSidecar, read_sidecar
from bot.lag.tape_studies import RunScope, split_of
from bot.markets.observation_window import observation_window
from bot.markets.parser import ParsedTicker, parse_ticker, resolve_event_kinds
from bot.observations.metar import StationObservation


logger = logging.getLogger(__name__)

POOLED = "pooled"

# Pre-registered for this question alone. lock_convergence.STATION_DAY_MIN and
# settlement_run.DISCOVERY_N_MIN were fixed for other questions, so importing either would let a
# revision over there move this bar.
DISCOVERY_STATION_DAY_MIN = 30
HOLDOUT_STATION_DAY_MIN = 15


@dataclass(frozen=True, slots=True, kw_only=True)
class StationDaySupply:
    series: str
    station: str
    event_date: date
    split: str
    readings: int
    markets: int
    clean: int
    ambiguous: int
    no_lock: int
    window_open: int
    mid_day: int


@dataclass(frozen=True, slots=True, kw_only=True)
class LockSupply:
    roots: tuple[str, ...]
    markets: int
    station_days: tuple[StationDaySupply, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitSupply:
    split: str
    station_days: int
    station_days_with_readings: int
    markets: int
    clean: int
    ambiguous: int
    no_lock: int
    window_open: int
    mid_day: int
    window_open_station_days: int
    mid_day_station_days: int
    ambiguous_to_clean: Decimal | None
    ambiguous_to_mid_day: Decimal | None


def read_ladders(directory: Path) -> dict[str, CloseSidecar]:
    sidecars = [read_sidecar(path) for path in sorted(directory.glob("*.json"))]
    return {sidecar.root: sidecar for sidecar in sidecars}


def read_observations(
    directory: Path, index: ObservationIndex
) -> dict[tuple[str, date], list[StationObservation]]:
    grouped: dict[tuple[str, date], list[StationObservation]] = {}
    for station in sorted(index.stations):
        sidecar = read_observation_sidecar(directory / f"{station}.json")
        for event_date, day in sorted(sidecar.days.items()):
            grouped[(station, event_date)] = list(day.readings)
    return grouped


def ladder_events(sidecar: CloseSidecar) -> dict[tuple[str, date], list[ParsedTicker]]:
    grouped: dict[tuple[str, date], list[ParsedTicker]] = {}
    for ticker in sorted(sidecar.markets):
        market = parse_ticker(ticker)
        grouped.setdefault((market.series, market.event_date), []).append(market)
    return grouped


def survey_lock_supply(
    scope: RunScope,
    ladders: Mapping[str, CloseSidecar],
    observations: Mapping[tuple[str, date], Sequence[StationObservation]],
) -> LockSupply:
    roots = tuple(sorted(ladders))
    # ladder_of answers high for a series on neither ladder and would sweep it silently, so the
    # kinds are derived through is_low_ladder, which raises. The two ladders settle off the same
    # stations on the same days, so a sweep handed both counts every station-day twice.
    if len({is_low_ladder(root) for root in roots}) > 1:
        raise ValueError("the swept roots span both ladders: " + ", ".join(roots))

    rows: list[StationDaySupply] = []
    for root in roots:
        for key, legs in sorted(ladder_events(ladders[root]).items()):
            series, event_date = key
            day = scope.event_days.get(key)
            if day is None:
                continue
            recorded = list(observations.get((day.station, event_date), ()))
            start, end = observation_window(day.timezone, event_date)
            in_window = sorted(
                (row for row in recorded if start <= row.valid_time < end),
                key=lambda row: row.valid_time,
            )
            opened_at = in_window[0].publication_time if in_window else None

            clean = 0
            ambiguous = 0
            no_lock = 0
            window_open = 0
            mid_day = 0
            # A tail read on its own parses as a bracket, so the kinds are settled across the whole
            # event ladder before the detector reads any leg's strike.
            for market in resolve_event_kinds(legs):
                found = detect_lock_events(
                    market, recorded, tz_name=day.timezone, rounding_margin_f=ROUNDING_MARGIN_F
                )
                if not found:
                    no_lock += 1
                    continue
                if found[0].lock_ambiguous:
                    ambiguous += 1
                    continue
                clean += 1
                if found[0].t0 == opened_at:
                    window_open += 1
                else:
                    mid_day += 1

            rows.append(
                StationDaySupply(
                    series=series,
                    station=day.station,
                    event_date=event_date,
                    split=split_of(scope, series, event_date),
                    readings=len(in_window),
                    markets=len(legs),
                    clean=clean,
                    ambiguous=ambiguous,
                    no_lock=no_lock,
                    window_open=window_open,
                    mid_day=mid_day,
                )
            )

    supply = LockSupply(
        roots=roots,
        markets=sum(row.markets for row in rows),
        station_days=tuple(rows),
    )
    pooled = summarize(supply, None)
    logger.info(
        "lock_supply roots=%d station_days=%d markets=%d clean=%d ambiguous=%d no_lock=%d "
        "window_open=%d mid_day=%d",
        len(roots),
        pooled.station_days,
        pooled.markets,
        pooled.clean,
        pooled.ambiguous,
        pooled.no_lock,
        pooled.window_open,
        pooled.mid_day,
    )
    return supply


def summarize(supply: LockSupply, split: str | None) -> SplitSupply:
    rows = [row for row in supply.station_days if split is None or row.split == split]
    clean = sum(row.clean for row in rows)
    ambiguous = sum(row.ambiguous for row in rows)
    mid_day = sum(row.mid_day for row in rows)
    return SplitSupply(
        split=POOLED if split is None else split,
        station_days=len(rows),
        station_days_with_readings=sum(1 for row in rows if row.readings),
        markets=sum(row.markets for row in rows),
        clean=clean,
        ambiguous=ambiguous,
        no_lock=sum(row.no_lock for row in rows),
        window_open=sum(row.window_open for row in rows),
        mid_day=mid_day,
        window_open_station_days=sum(1 for row in rows if row.window_open),
        mid_day_station_days=sum(1 for row in rows if row.mid_day),
        ambiguous_to_clean=None if clean == 0 else Decimal(ambiguous) / Decimal(clean),
        ambiguous_to_mid_day=None if mid_day == 0 else Decimal(ambiguous) / Decimal(mid_day),
    )
