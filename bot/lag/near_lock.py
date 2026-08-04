import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow.compute as pc

from bot.lag.lock_events import detect_lock_events
from bot.lag.read_rtt import FloorSource
from bot.lag.run_manifest import BOOTSTRAP_RESAMPLES, MANIFEST_NAME, write_manifest
from bot.lag.taker_flow import PRIMARY_HORIZON_S
from bot.lag.taker_flow_run import UNDERPOWERED, HorizonReadout, Sweep, readout, sweep_prints
from bot.lag.tape_studies import (
    SELF_CHARGED_BAR,
    SELF_CHARGED_BAR_SOURCE,
    TRADES,
    RunScope,
    assemble_run_inputs,
    load_run_scope,
    read_window,
)
from bot.markets.parser import parse_ticker
from bot.observations.metar import StationObservation
from bot.replay.analysis_stations import in_cohort
from bot.replay.run_scope import DISCOVERY, HOLDOUT


logger = logging.getLogger(__name__)

STRATUM = "near_lock"
REPORTED = "REPORTED"
REPORTED_ONLY = "this stratum is reported only and never gates the question"
PRINT_MIN_STRATUM = 200
LOCK_HALF_WIDTH_S = 900

_HALF_WIDTH = timedelta(seconds=LOCK_HALF_WIDTH_S)


@dataclass(frozen=True, slots=True, kw_only=True)
class LockScan:
    cities: tuple[str, ...]
    windows: Mapping[str, tuple[datetime, datetime]]
    markets: int
    locked: int
    ambiguous: int
    no_lock: int
    no_observations: int


@dataclass(frozen=True, slots=True, kw_only=True)
class NearLockRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    seed: int
    locks: LockScan
    sweep: Sweep
    discovery: HorizonReadout
    holdout: HorizonReadout

    @property
    def status(self) -> str:
        return status_of(self.discovery)


def read_observations(path: Path) -> dict[str, list[StationObservation]]:
    grouped: dict[str, list[StationObservation]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        grouped.setdefault(row["station"], []).append(
            StationObservation(
                station=row["station"],
                valid_time=datetime.fromisoformat(row["obs_time"]),
                publication_time=datetime.fromisoformat(row["received_at"]),
                temp_f=Decimal(row["tmpf"]),
                # ws_obs_arrivals records neither the report text nor its SPECI flag, so these two
                # carry no evidence here. detect_lock_events reads neither.
                is_special=False,
                raw="",
                source=row["source"],
            )
        )
    return grouped


def scan_locks(
    scope: RunScope,
    artifacts: Path,
    observations: Mapping[str, list[StationObservation]],
    *,
    cohort: str | None = None,
) -> LockScan:
    scoped = set(in_cohort({series for series, _ in scope.event_days}, cohort))
    lock_dependent = set(scope.universe.lock_dependent)
    cities = sorted(scoped & lock_dependent)
    # Zero locks would otherwise read as an underpowered run rather than a freeze paired with the
    # wrong ladder.
    if not cities:
        raise ValueError(
            f"the scope's series {sorted(scoped)} share nothing with the universe's "
            f"lock-dependent series {sorted(lock_dependent)}"
        )
    windows: dict[str, tuple[datetime, datetime]] = {}
    markets = 0
    ambiguous = 0
    no_lock = 0
    no_observations = 0

    for series_root in cities:
        table = read_window(artifacts, TRADES, series_root, scope.scope_start, scope.scope_end)
        for ticker in sorted(pc.unique(table.column("ticker")).to_pylist()):
            market = parse_ticker(ticker)
            day = scope.event_days.get((market.series, market.event_date))
            if day is None:
                continue
            markets += 1
            recorded = observations.get(day.station)
            if recorded is None:
                no_observations += 1
                continue
            events = detect_lock_events(market, recorded, tz_name=day.timezone)
            if not events:
                no_lock += 1
                continue
            lock = events[0]
            ambiguous += int(lock.lock_ambiguous)
            windows[ticker] = (lock.t0 - _HALF_WIDTH, lock.t0 + _HALF_WIDTH)
        # The next root is read at the top of the loop and the host cannot hold two roots at once.
        del table

    logger.info(
        "near_lock markets=%d locked=%d ambiguous=%d no_lock=%d no_observations=%d",
        markets,
        len(windows),
        ambiguous,
        no_lock,
        no_observations,
    )
    return LockScan(
        cities=tuple(cities),
        windows=windows,
        markets=markets,
        locked=len(windows),
        ambiguous=ambiguous,
        no_lock=no_lock,
        no_observations=no_observations,
    )


def status_of(item: HorizonReadout) -> str:
    if item.bootstrap is None or item.result.n_prints < PRINT_MIN_STRATUM:
        return UNDERPOWERED
    return REPORTED


def execute(
    *,
    run_id: str,
    preregistration: Path,
    repo: Path,
    run_scope: Path,
    artifacts: Path,
    observations: Path,
    rtt_samples: Path,
    floor_source: FloorSource,
    seed: int,
    run_root: Path,
) -> NearLockRun:
    inputs = assemble_run_inputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        run_scope=run_scope,
        artifacts=artifacts,
        rtt_samples=rtt_samples,
        floor_source=floor_source,
        economic_bar_size=SELF_CHARGED_BAR,
        economic_bar_price=SELF_CHARGED_BAR,
        economic_bar_price_source=SELF_CHARGED_BAR_SOURCE,
        bootstrap_seed=seed,
    )
    scope = load_run_scope(run_scope)
    locks = scan_locks(scope, artifacts, read_observations(observations))
    swept = sweep_prints(scope, artifacts, lock_windows=locks.windows)
    # Every refusal above this line leaves the run root untouched, so a mis-paired freeze writes no
    # manifest for a run that never happened.
    digest = write_manifest(run_root, inputs)
    run = NearLockRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        seed=seed,
        locks=locks,
        sweep=swept,
        discovery=readout(
            swept.tallies[(DISCOVERY, PRIMARY_HORIZON_S)],
            split=DISCOVERY,
            horizon_s=PRIMARY_HORIZON_S,
            seed=seed,
        ),
        holdout=readout(
            swept.tallies[(HOLDOUT, PRIMARY_HORIZON_S)],
            split=HOLDOUT,
            horizon_s=PRIMARY_HORIZON_S,
            seed=seed,
        ),
    )
    logger.info(
        "near_lock status=%s n=%d locked=%d ambiguous=%d",
        run.status,
        run.discovery.result.n_prints,
        locks.locked,
        locks.ambiguous,
    )
    return run


def reading_payload(item: HorizonReadout) -> dict:
    bootstrap = item.bootstrap
    status = status_of(item)
    reported = status == REPORTED
    fraction = item.excluded_fraction
    return {
        "split": item.result.split,
        "horizon_s": item.result.horizon_s,
        "status": status,
        "mean_net_cents": str(bootstrap.estimate) if reported else None,
        "ci_low": bootstrap.ci_low if reported else None,
        "ci_high": bootstrap.ci_high if reported else None,
        "ci_level": bootstrap.ci_level if reported else None,
        "p_value": bootstrap.p_value if reported else None,
        "n_prints": item.result.n_prints,
        "n_min": PRINT_MIN_STRATUM,
        "contracts": str(item.result.contracts),
        "clusters": len(item.result.clusters),
        "candidates": item.candidates,
        "excluded": item.excluded,
        "excluded_fraction": None if fraction is None else str(fraction),
        "out_of_window": item.out_of_window,
        "by_class": dict(sorted(item.by_class.items())),
        "unresolved": item.result.counts.unresolved,
        "uncovered": item.result.counts.uncovered,
        "one_sided": item.result.counts.one_sided,
        "host_clock": item.result.counts.host_clock,
    }


def result_payload(run: NearLockRun) -> dict:
    discovery = run.discovery
    holdout = run.holdout
    locks = run.locks
    candidates = discovery.candidates + holdout.candidates
    excluded = discovery.excluded + holdout.excluded
    counts = discovery.result.counts + holdout.result.counts
    in_scope = dict(run.sweep.in_scope)
    scanned = set(locks.cities)
    return {
        "run_id": run.run_id,
        "stratum": STRATUM,
        "gating": False,
        "reported_only": REPORTED_ONLY,
        "stratum_status": run.status,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "bootstrap_seed": run.seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "primary_horizon_s": PRIMARY_HORIZON_S,
        "lock_half_width_s": LOCK_HALF_WIDTH_S,
        "stratum_n_min": PRINT_MIN_STRATUM,
        "locks": {
            "markets": locks.markets,
            "locked": locks.locked,
            "ambiguous": locks.ambiguous,
            "ambiguous_fraction": (
                None if locks.locked == 0 else str(Decimal(locks.ambiguous) / Decimal(locks.locked))
            ),
            "no_lock": locks.no_lock,
            "no_observations": locks.no_observations,
        },
        "prints": {
            "in_window_pooled": sum(in_scope.values()),
            "in_window_discovery": in_scope[DISCOVERY],
            "in_window_holdout": in_scope[HOLDOUT],
            "screened_pooled": discovery.result.n_prints + holdout.result.n_prints,
            "screened_discovery": discovery.result.n_prints,
            "screened_holdout": holdout.result.n_prints,
            "empty_side": run.sweep.empty_side,
            "duplicate_trade_id": run.sweep.duplicates,
            "out_of_scope": run.sweep.out_of_scope,
            "out_of_window": discovery.out_of_window + holdout.out_of_window,
            "outside_lock_window": run.sweep.outside_lock_window,
            "on_a_market_that_never_locked": run.sweep.no_lock_prints,
            "on_a_series_outside_the_lock_universe": run.sweep.off_universe_prints,
        },
        "discovery": reading_payload(discovery),
        "holdout": reading_payload(holdout),
        "exclusions": {
            "candidates": candidates,
            "excluded": excluded,
            "excluded_fraction": (
                None if candidates == 0 else str(Decimal(excluded) / Decimal(candidates))
            ),
            "by_class": {
                name: discovery.by_class.get(name, 0) + holdout.by_class.get(name, 0)
                for name in sorted(set(discovery.by_class) | set(holdout.by_class))
            },
        },
        "kernel_drops": {
            "unresolved": counts.unresolved,
            "uncovered": counts.uncovered,
            "one_sided": counts.one_sided,
            "host_clock": counts.host_clock,
            "read_ts_violations": run.sweep.read_ts_violations,
            "fractional_size_prints": run.sweep.fractional_size_prints,
        },
        "cities": sorted({series for series, _ in run.sweep.tickers} & scanned),
        "tickers_per_city_day": {
            f"{series} {event_date.isoformat()}": len(names)
            for (series, event_date), names in sorted(run.sweep.tickers.items())
            if series in scanned
        },
    }
