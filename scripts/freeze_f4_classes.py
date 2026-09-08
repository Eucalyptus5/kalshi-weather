from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.backtest.hrrr import (  # noqa: E402
    DecodedFieldCache,
    leg_fields,
    pull_fields,
    run_directory_url,
)
from bot.backtest.nbm import daily_high_row, fetch_mos_archive  # noqa: E402
from bot.backtest.previous_runs import (  # noqa: E402
    CLASS_A_MEMBERS,
    PREVIOUS_DAY_OFFSET,
    PREVIOUS_DAY_VARIABLE,
    fetch_previous_runs,
)
from bot.lag.forecast_classes import (  # noqa: E402
    CLASS_A,
    CLASS_B,
    CLASS_B_MEMBER,
    CLASS_C,
    CLASS_C_MEMBER,
    ClassRecord,
    class_a_record,
    class_b_record,
    class_c_record,
    class_freeze_path,
    leg_index,
    write_class_freeze,
)
from bot.lag.forecast_sample import (  # noqa: E402
    F4_LEADS,
    SampleLeg,
    read_sample_freeze,
    sidecar_path,
)
from bot.main import STATIONS  # noqa: E402
from scripts.freeze_f4_sample import FREEZE_NAME  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "tape_studies" / "f4_inputs"
DEFAULT_SAMPLE = DEFAULT_OUT / FREEZE_NAME
DEFAULT_CONCURRENCY = 8
OPEN_METEO_TIMEOUT = 300.0
IEM_TIMEOUT = 600.0
BUCKET_TIMEOUT = 120.0
REPORTED_REFUSALS = 20

STATION_CONFIGS = {config.station: config for config in STATIONS.values()}

logger = logging.getLogger("freeze_f4_classes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="retrieve one forecast class and freeze its daily highs against the sample"
    )
    parser.add_argument(
        "--class", dest="forecast_class", required=True, choices=(CLASS_A, CLASS_B, CLASS_C)
    )
    parser.add_argument(
        "--lead",
        dest="leads",
        action="append",
        type=int,
        choices=F4_LEADS,
        help="repeat for both leads; omit to retrieve 24 and 36",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--cache", type=Path, default=None, help="the resumable decoded-field cache Class C writes"
    )
    parser.add_argument(
        "--event-date",
        dest="event_dates",
        action="append",
        type=date.fromisoformat,
        help="restrict the pull to these event days",
    )
    return parser


def wanted_legs(args: argparse.Namespace) -> list[SampleLeg]:
    leads = set(args.leads or F4_LEADS)
    days = set(args.event_dates or ())
    return [
        leg
        for leg in leg_index(read_sample_freeze(args.sample)).values()
        if leg.lead_hours in leads and (not days or leg.event_date in days)
    ]


async def pull_class_a(legs: Sequence[SampleLeg]) -> tuple[list[ClassRecord], dict]:
    span = (
        min(leg.event_date for leg in legs),
        max(leg.event_date for leg in legs) + timedelta(days=1),
    )
    series = {}
    async with httpx.AsyncClient(timeout=OPEN_METEO_TIMEOUT) as client:
        for station in sorted({leg.station for leg in legs}):
            config = STATION_CONFIGS[station]
            for member in CLASS_A_MEMBERS:
                series[(station, member)] = await fetch_previous_runs(
                    latitude=config.latitude,
                    longitude=config.longitude,
                    member=member,
                    start_date=span[0],
                    end_date=span[1],
                    client=client,
                )
                logger.info("class a station=%s member=%s", station, member)

    records = []
    uncovered: Counter[str] = Counter()
    for leg in legs:
        variable = PREVIOUS_DAY_VARIABLE[leg.lead_hours]
        for member in CLASS_A_MEMBERS:
            retrieved = series[(leg.station, member)]
            built = class_a_record(
                leg,
                member=member,
                hourly=retrieved.hourly[variable],
                issue_offset=PREVIOUS_DAY_OFFSET[variable],
                latitude=retrieved.latitude,
                longitude=retrieved.longitude,
                source_url=retrieved.source_url,
            )
            if built is None:
                uncovered[member] += 1
                continue
            records.append(built)
    return records, {"uncovered": dict(uncovered), "requests": len(series)}


async def pull_class_b(legs: Sequence[SampleLeg]) -> tuple[list[ClassRecord], dict]:
    span = (
        min(leg.event_date for leg in legs) - timedelta(days=2),
        max(leg.event_date for leg in legs) + timedelta(days=2),
    )
    archives = {}
    async with httpx.AsyncClient(timeout=IEM_TIMEOUT) as client:
        for station in sorted({leg.station for leg in legs}):
            archives[station] = await fetch_mos_archive(
                station=station, start_date=span[0], end_date=span[1], client=client
            )
            logger.info("class b station=%s rows=%d", station, len(archives[station].rows))

    records = []
    uncovered = 0
    without_sigma = 0
    for leg in legs:
        archive = archives[leg.station]
        row = daily_high_row(archive.rows, leg.event_date, leg.as_of)
        if row is None or row.txn is None:
            uncovered += 1
            continue
        if row.xnd is None:
            without_sigma += 1
        records.append(
            class_b_record(
                leg,
                daily_high_f=row.txn,
                native_sigma_f=row.xnd,
                runtime=row.runtime,
                source_url=archive.source_url,
            )
        )
    return records, {
        "uncovered": {CLASS_B_MEMBER: uncovered},
        "records_without_sigma": without_sigma,
    }


async def pull_class_c(
    legs: Sequence[SampleLeg], concurrency: int, cache_path: Path | None
) -> tuple[list[ClassRecord], dict]:
    plans = {(leg.station, leg.event_date, leg.lead_hours): leg_fields(leg) for leg in legs}
    requests = sorted({(plan.run, fxx) for plan in plans.values() for fxx in plan.fxx})
    stations = {
        station: (STATION_CONFIGS[station].latitude, STATION_CONFIGS[station].longitude)
        for station in sorted({leg.station for leg in legs})
    }
    cache = DecodedFieldCache(cache_path)
    async with httpx.AsyncClient(timeout=BUCKET_TIMEOUT) as client:
        pulled = await pull_fields(requests, stations, cache, client, concurrency)
    if not pulled.points:
        raise RuntimeError(f"no hrrr field decoded out of {len(requests)} requested")

    records = []
    uncovered = 0
    for leg in legs:
        plan = plans[(leg.station, leg.event_date, leg.lead_hours)]
        hourly = {
            hour: pulled.values[(plan.run, fxx)][leg.station]
            for hour, fxx in zip(plan.hours, plan.fxx)
            if (plan.run, fxx) in pulled.values
        }
        cell = pulled.points[leg.station]
        built = class_c_record(
            leg,
            hourly=hourly,
            run=plan.run,
            latitude=cell.latitude,
            longitude=cell.longitude,
            source_url=run_directory_url(plan.run),
        )
        if built is None:
            uncovered += 1
            continue
        records.append(built)
    return records, {
        "uncovered": {CLASS_C_MEMBER: uncovered},
        "fields": len(requests),
        "downloaded_fields": pulled.downloaded,
        "cache_hits": pulled.cache_hits,
        "transferred_bytes": pulled.transferred_bytes,
        "missing_fields": len(pulled.missing),
        "missing": list(pulled.missing[:REPORTED_REFUSALS]),
    }


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    legs = wanted_legs(args)
    if args.forecast_class == CLASS_A:
        records, detail = asyncio.run(pull_class_a(legs))
    elif args.forecast_class == CLASS_B:
        records, detail = asyncio.run(pull_class_b(legs))
    else:
        records, detail = asyncio.run(pull_class_c(legs, args.concurrency, args.cache))

    args.out.mkdir(parents=True, exist_ok=True)
    path = class_freeze_path(args.out, args.forecast_class)
    frozen = write_class_freeze(records, path, leg_index(legs))

    print(
        json.dumps(
            {
                "class": args.forecast_class,
                "leads": sorted({leg.lead_hours for leg in legs}),
                "city_days": len(legs),
                "event_days": len({leg.event_date for leg in legs}),
                "out": str(path),
                "sidecar": str(sidecar_path(path)),
                "sha256": frozen.sha256,
                "bytes": path.stat().st_size,
                "records": frozen.written,
                "per_member": dict(sorted(Counter(row.member for row in records).items())),
                "per_basis": dict(sorted(Counter(row.window_basis for row in records).items())),
                "refused": len(frozen.refused),
                "refusals": list(frozen.refused[:REPORTED_REFUSALS]),
                "seconds": round(time.monotonic() - started, 1),
                **detail,
            },
            indent=1,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
