import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.lock_convergence import (  # noqa: E402
    ECONOMIC_BAR_PRICE,
    ECONOMIC_BAR_PRICE_SOURCE,
    ECONOMIC_BAR_SIZE,
    MAKER_RATE,
    MAKER_RATE_SOURCE,
    PROVIDING,
    TAKING,
    execute,
    result_payload,
)
from bot.lag.read_rtt import FloorSource  # noqa: E402
from bot.lag.run_manifest import ManifestIncomplete  # noqa: E402
from bot.lag.taker_flow_run import RESULTS_NAME  # noqa: E402
from bot.lag.tape_studies import RunScope, load_run_scope  # noqa: E402
from bot.markets.observation_window import observation_window  # noqa: E402
from bot.observations.basis_check import fetch_iem_1min_asos_archive  # noqa: E402
from bot.observations.metar import StationObservation  # noqa: E402
from bot.replay.analysis_stations import HIGH, LOW  # noqa: E402
from bot.validation.reconcile import ACISClient  # noqa: E402


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "tape_studies"
FLOOR_SOURCES = tuple(source.value for source in FloorSource)
ARRIVALS_QUERY = (
    "SELECT station, source, obs_time, tmpf, received_at FROM ws_obs_arrivals "
    "WHERE station = ? AND obs_time >= ? AND obs_time <= ? ORDER BY obs_time"
)
SETTLES_NAME = "acis_settles.json"
# obs_time is TEXT and the range bound is compared lexicographically, so a bound separating the
# date from the time with anything but a space sorts past every stored row of its own date.
STORED_STAMP = "%Y-%m-%d %H:%M:%S.%f"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read how fast a locked book converges off the frozen run scope"
    )
    parser.add_argument("--run-id", required=True, help="names the run directory under --run-root")
    parser.add_argument(
        "--preregistration", type=Path, required=True, help="the file the manifest hashes"
    )
    parser.add_argument(
        "--run-scope", type=Path, required=True, help="the frozen run-scope directory"
    )
    parser.add_argument(
        "--artifacts", type=Path, required=True, help="the forward-pass artifact root"
    )
    parser.add_argument(
        "--state-db", type=Path, required=True, help="the recorder state db, opened read-only"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="where the archive and settle pulls are kept so a rerun refetches neither",
    )
    parser.add_argument("--rtt-samples", type=Path, required=True, help="the read-RTT sample file")
    parser.add_argument(
        "--floor-source",
        required=True,
        choices=FLOOR_SOURCES,
        help="which round trip supplied the latency floor",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="the bootstrap seed every resample runs under"
    )
    parser.add_argument(
        "--cohort",
        choices=(HIGH, LOW),
        default=None,
        help="which ladder of the frozen scope the run reads",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    return parser


def stations_of(scope: RunScope) -> list[str]:
    return sorted({day.station for day in scope.event_days.values()})


# The endpoint takes a date range, so one pull per station covers every event-day it carries and
# the days are cut out of the reply afterwards.
async def gather_archive(
    scope: RunScope, cache: Path, http: httpx.AsyncClient
) -> dict[str, list[StationObservation]]:
    start = scope.scope_start.date()
    end = scope.scope_end.date()
    stations = stations_of(scope)
    out: dict[str, list[StationObservation]] = {}
    for index, station in enumerate(stations, start=1):
        path = cache / f"{station}-{start.isoformat()}-{end.isoformat()}.jsonl"
        if path.exists():
            readings = read_archive_cache(path)
            source = "cache"
        else:
            readings = await fetch_iem_1min_asos_archive(station, start, end, http)
            write_archive_cache(path, readings)
            source = "iem"
        out[station] = readings
        logger.info(
            "q4 archive station=%s from=%s rows=%d (%d/%d)",
            station,
            source,
            len(readings),
            index,
            len(stations),
        )
    return out


def slice_observations(
    scope: RunScope, readings: Mapping[str, Sequence[StationObservation]]
) -> dict[tuple[str, date], list[StationObservation]]:
    out: dict[tuple[str, date], list[StationObservation]] = {}
    for day in scope.event_days.values():
        start, end = observation_window(day.timezone, day.event_date)
        bucket = [row for row in readings[day.station] if start <= row.valid_time < end]
        if bucket:
            out[(day.station, day.event_date)] = bucket
    logger.info(
        "q4 observations station_days=%d readings=%d",
        len(out),
        sum(len(bucket) for bucket in out.values()),
    )
    return out


# A published high never changes but an unpublished one can still arrive, so the cache carries the
# settles that exist and the days it has nothing for are asked again.
async def gather_settles(
    scope: RunScope, cache: Path, http: httpx.AsyncClient
) -> dict[tuple[str, date], Decimal]:
    path = cache / SETTLES_NAME
    known = read_settles_cache(path) if path.exists() else {}
    acis = ACISClient(http_client=http)
    wanted = sorted({(day.station, day.event_date) for day in scope.event_days.values()})
    out: dict[tuple[str, date], Decimal] = {}
    fetched = 0
    for index, (station, event_date) in enumerate(wanted, start=1):
        settle = known.get((station, event_date))
        source = "cache"
        if settle is None:
            settle = await acis.fetch_daily_high(station[1:], event_date)
            source = "acis"
            fetched += 1
        if settle is not None:
            out[(station, event_date)] = settle
        logger.info(
            "q4 settle station=%s event_date=%s from=%s maxt=%s (%d/%d)",
            station,
            event_date.isoformat(),
            source,
            settle,
            index,
            len(wanted),
        )
    write_settles_cache(path, known | out)
    logger.info(
        "q4 settles wanted=%d settled=%d fetched=%d cached=%d",
        len(wanted),
        len(out),
        fetched,
        len(wanted) - fetched,
    )
    return out


# The recorder writes this database while the read runs, so the read is one indexed seek per
# station and nothing else: no pragma that touches the WAL, no cursor held open past the fetch.
def read_arrivals(state_db: Path, scope: RunScope) -> dict[str, list[StationObservation]]:
    stations = stations_of(scope)
    low = scope.scope_start.strftime(STORED_STAMP)
    high = scope.scope_end.strftime(STORED_STAMP)
    grouped: dict[str, list[StationObservation]] = {}
    conn = sqlite3.connect(f"file:{state_db.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        plan = conn.execute(
            f"EXPLAIN QUERY PLAN {ARRIVALS_QUERY}", (stations[0], low, high)
        ).fetchall()
        logger.info("q4 arrivals plan=%s", " | ".join(row[3] for row in plan))
        for station in stations:
            rows = conn.execute(ARRIVALS_QUERY, (station, low, high)).fetchall()
            grouped[station] = [
                StationObservation(
                    station=name,
                    valid_time=_utc(obs_time),
                    publication_time=_utc(received_at),
                    temp_f=Decimal(tmpf),
                    # ws_obs_arrivals records neither the report text nor its SPECI flag, so these
                    # two carry no evidence here.
                    is_special=False,
                    raw="",
                    source=source,
                )
                for name, source, obs_time, tmpf, received_at in rows
            ]
            logger.info("q4 arrivals station=%s rows=%d", station, len(rows))
    finally:
        conn.close()
    return grouped


# The 1-minute archive publishes neither the report text nor a SPECI flag, so neither crosses the
# cache and both come back as the constants the fetch itself hands out.
def write_archive_cache(path: Path, readings: Sequence[StationObservation]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "station": row.station,
                    "valid_time": row.valid_time.isoformat(),
                    "publication_time": row.publication_time.isoformat(),
                    "tmpf": str(row.temp_f),
                    "source": row.source,
                }
            )
            + "\n"
            for row in readings
        )
    )


def read_archive_cache(path: Path) -> list[StationObservation]:
    return [
        StationObservation(
            station=row["station"],
            valid_time=datetime.fromisoformat(row["valid_time"]),
            publication_time=datetime.fromisoformat(row["publication_time"]),
            temp_f=Decimal(row["tmpf"]),
            is_special=False,
            raw="",
            source=row["source"],
        )
        for row in (json.loads(line) for line in path.read_text().splitlines())
    ]


def write_settles_cache(path: Path, settles: Mapping[tuple[str, date], Decimal]) -> None:
    path.write_text(
        json.dumps(
            {
                f"{station} {event_date.isoformat()}": str(value)
                for (station, event_date), value in sorted(settles.items())
            },
            indent=1,
        )
    )


def read_settles_cache(path: Path) -> dict[tuple[str, date], Decimal]:
    out: dict[tuple[str, date], Decimal] = {}
    for key, value in json.loads(path.read_text()).items():
        station, event_date = key.split(" ")
        out[(station, date.fromisoformat(event_date))] = Decimal(value)
    return out


async def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    scope = load_run_scope(args.run_scope)
    args.cache.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120.0) as http:
        readings = await gather_archive(scope, args.cache, http)
        settles = await gather_settles(scope, args.cache, http)

    try:
        result = execute(
            run_id=args.run_id,
            preregistration=args.preregistration,
            repo=args.repo,
            run_scope=args.run_scope,
            artifacts=args.artifacts,
            observations=slice_observations(scope, readings),
            arrivals=read_arrivals(args.state_db, scope),
            settles=settles,
            rtt_samples=args.rtt_samples,
            floor_source=FloorSource(args.floor_source),
            maker_rate=MAKER_RATE,
            maker_rate_source=MAKER_RATE_SOURCE,
            economic_bar_size=ECONOMIC_BAR_SIZE,
            economic_bar_price=ECONOMIC_BAR_PRICE,
            economic_bar_price_source=ECONOMIC_BAR_PRICE_SOURCE,
            seed=args.seed,
            run_root=args.run_root,
            cohort=args.cohort,
        )
    except ManifestIncomplete as exc:
        print(exc, file=sys.stderr)
        return 1

    payload = result_payload(result) | {"elapsed_s": round(time.monotonic() - started, 1)}
    (args.run_root / args.run_id / RESULTS_NAME).write_text(json.dumps(payload, indent=1))
    print(format_report(payload))
    return 0


def format_report(payload: dict) -> str:
    locks = payload["locks"]
    half_life = payload["half_life"]
    anchor = payload["arrival_anchor"]
    exclusions = payload["exclusions"]
    fills = payload["post_lock_fills"]
    settles = payload["settles"]
    book = payload["book"]
    return "\n".join(
        [
            f"== Q4 LOCK CONVERGENCE  run_id={payload['run_id']}  verdict={payload['verdict']}",
            f"manifest={payload['manifest']}  sha256={payload['manifest_sha256']}",
            f"seed={payload['bootstrap_seed']}  resamples={payload['bootstrap_resamples']}  "
            f"lock_band={payload['lock_band']}  persist_s={payload['persist_s']}  "
            f"threshold_s={payload['half_life_threshold_s']}",
            "",
            "== STATION-DAYS",
            f"  discovery={locks['clean_station_days_discovery']} "
            f"holdout={locks['clean_station_days_holdout']} "
            f"pooled={locks['clean_station_days']} "
            f"ceiling={locks['population_ceiling']} n_min={payload['station_day_min']}",
            f"  markets={locks['markets']} clean_events={locks['clean_events']} "
            f"lock_rate={locks['lock_rate']} ambiguous={locks['ambiguous']} "
            f"no_lock={locks['no_lock']} no_observations={locks['no_observations']}",
            f"  cities={','.join(locks['cities'])}",
            "",
            "== HALF-LIFE",
            f"  gate_ran={half_life['gate_ran']} note={half_life['note']} "
            f"censored={half_life['censored']}",
            *_format_split(payload["discovery"]),
            *_format_split(payload["holdout"]),
            f"  pooled {_format_summary(half_life['pooled'])}",
            *_format_gate(payload["gate"]),
            *_format_replication(payload["replication"], payload["replication_skipped"]),
            f"  {payload['censoring']}",
            f"  {payload['length_bias']}",
            "",
            "== ARRIVAL ANCHOR (reported, not gating)",
            f"  no_arrival_rows={anchor['no_arrival_rows']} "
            f"never_clears={anchor['never_clears']} "
            f"arrival_precedes_lock={anchor['arrival_precedes_lock']}",
            f"  delta_s {_format_summary(anchor)}",
            "",
            "== EXCLUSIONS",
            f"  candidates={exclusions['candidates']} excluded={exclusions['excluded']} "
            f"fraction={exclusions['excluded_fraction']} "
            f"out_of_window={exclusions['out_of_window']} "
            f"out_of_scope={exclusions['out_of_scope']}",
            "  " + " ".join(f"{name}={count}" for name, count in exclusions["by_class"].items()),
            "",
            "== POST-LOCK FILLS (reported, not gating)",
            f"  dropped={fills['dropped']} "
            f"unclassified_taker_side={fills['unclassified_taker_side']}",
            *_format_fill(TAKING, fills[TAKING]),
            *_format_fill(PROVIDING, fills[PROVIDING]),
            "",
            "== SETTLES",
            f"  last_settled_event_day={settles['last_settled_event_day']} "
            f"usable_event_days={settles['usable_event_days']} "
            f"unsettled_station_days={settles['unsettled_station_days']} "
            f"settle_contradicts={settles['settle_contradicts']}",
            "",
            "== BOOK",
            f"  one_sided_rows={book['one_sided_rows']} no_book_at_lock={book['no_book_at_lock']}",
        ]
    )


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value + "+00:00")


def _fixed(value: float | None, places: int) -> str:
    return "None" if value is None else f"{value:.{places}f}"


def _format_summary(item: Mapping[str, object]) -> str:
    return f"count={item['count']} min={item['min']} median={item['median']} max={item['max']}"


def _format_split(item: dict) -> list[str]:
    return [
        f"  {item['split']} median_half_life_s={item['median_half_life_s']}  "
        f"ci{item['ci_level']}=[{_fixed(item['ci_low'], 4)}, {_fixed(item['ci_high'], 4)}]  "
        f"p_value={_fixed(item['p_value'], 5)}  n_station_days={item['n_station_days']}  "
        f"n_unit={item['n_unit']}  events={item['events']}"
    ]


def _format_gate(gate: dict | None) -> list[str]:
    if gate is None:
        return ["  gate did not run"]
    return [
        f"  gate estimate={gate['estimate']} threshold={gate['threshold']} "
        f"direction={gate['direction']} p_value={_fixed(gate['p_value'], 5)} "
        f"alpha={gate['alpha']} n={gate['n']} n_min={gate['n_min']} "
        f"economic={gate['economic']} significant={gate['significant']} "
        f"powered={gate['powered']} undecidable={gate['undecidable']} "
        f"passed={gate['passed']}"
    ]


def _format_replication(replication: dict | None, skipped: str) -> list[str]:
    if replication is None:
        return [f"  replication did not run: {skipped}"]
    return [
        f"  replication discovery={replication['discovery_estimate']} "
        f"holdout={replication['holdout_estimate']} "
        f"p_value={_fixed(replication['holdout_p_value'], 5)} alpha={replication['alpha']} "
        f"holdout_n={replication['holdout_n']} holdout_n_min={replication['holdout_n_min']} "
        f"same_sign={replication['same_sign']} magnitude={replication['magnitude']} "
        f"significant={replication['significant']} powered={replication['powered']} "
        f"undecidable={replication['undecidable']} replicated={replication['replicated']}"
    ]


def _format_fill(name: str, tally: dict) -> list[str]:
    price = tally["invalidated_price"]
    return [
        f"  {name} fills={tally['fills']} contracts={tally['contracts']} "
        f"notional={tally['notional']} fees={tally['fees']}",
        f"    invalidated_price {_format_summary(price)}",
        "    " + " ".join(f"{cent}={count}" for cent, count in price["by_cent"].items()),
    ]


async def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return await run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
