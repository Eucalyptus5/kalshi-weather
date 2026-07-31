import logging
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa

from bot.lag.depth_map import scaled_units, screen_cells, weighted_quantile
from bot.lag.depth_map_run import (
    NO,
    PRICE_DECIMALS,
    SIDES,
    SIZE_DECIMALS,
    TOUCH_COLUMNS,
    YES,
    _last_at_each,
    _read_legs,
)
from bot.lag.ladder_consistency import PRICE_TICKS, SIZE_UNITS
from bot.lag.read_rtt import FloorSource, LatencyFloor
from bot.lag.run_manifest import MANIFEST_NAME, write_manifest
from bot.lag.tape_studies import LADDER, RunScope, assemble_run_inputs, load_run_scope, window_dates
from bot.replay.run_scope import EventDay


logger = logging.getLogger(__name__)

RESULTS_NAME = "results.json"
SAMPLE_MAX = 20
# Market reads moved from demo to prod Kalshi at this instant. Demo books are a different
# liquidity population, so a comparison window that straddles it is not one population noisily
# measured twice.
ERA_START = datetime(2026, 6, 12, 18, 18, 14, tzinfo=timezone.utc)

SNAPSHOT_ROWS = (
    "SELECT snapshot_at, yes_bid, no_bid, yes_bid_depth, no_bid_depth "
    "FROM orderbook_snapshots WHERE ticker = ? AND snapshot_at >= ? AND snapshot_at <= ? "
    "ORDER BY snapshot_at"
)
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROSECOND = timedelta(microseconds=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class Snapshot:
    at: datetime
    yes_bid: Decimal
    no_bid: Decimal
    yes_bid_depth: int | None
    no_bid_depth: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class LegBook:
    times: np.ndarray
    yes_bid: np.ndarray
    no_bid: np.ndarray
    yes_bid_depth: np.ndarray
    no_bid_depth: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class Disagreement:
    ticker: str
    snapshot_at: datetime
    rest_yes_bid: Decimal
    rest_no_bid: Decimal
    rest_yes_bid_depth: Decimal | None
    rest_no_bid_depth: Decimal | None
    ws_yes_bid: Decimal
    ws_no_bid: Decimal
    ws_yes_bid_depth: Decimal
    ws_no_bid_depth: Decimal


@dataclass(slots=True)
class RowTally:
    returned: int = 0
    era: int = 0
    out_of_window: int = 0
    excluded: int = 0
    no_coverage: int = 0
    null_depth: int = 0
    compared: int = 0
    by_class: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class Tally:
    compared: int = 0
    price_agree: int = 0
    depth_compared: int = 0
    depth_agree: int = 0
    depth_truncated: int = 0
    fractional: int = 0
    within_one: dict[str, int] = field(default_factory=dict)
    differences: dict[str, dict[int, int]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class SideReadout:
    p25: Decimal | None
    p50: Decimal | None
    p75: Decimal | None
    within_one_contract: int
    within_one_fraction: Decimal | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Agreement:
    compared: int
    price_agree: int
    price_fraction: Decimal | None
    depth_compared: int
    depth_agree: int
    depth_fraction: Decimal | None
    depth_truncated: int
    depth_truncated_fraction: Decimal | None
    fractional: int
    fractional_fraction: Decimal | None
    sides: Mapping[str, SideReadout]


@dataclass(frozen=True, slots=True, kw_only=True)
class Continuity:
    cities: tuple[str, ...]
    rows: RowTally
    tallies: Mapping[str, Tally]
    disagreements: tuple[Disagreement, ...]

    def pooled(self) -> Tally:
        out = Tally()
        for tally in self.tallies.values():
            out.compared += tally.compared
            out.price_agree += tally.price_agree
            out.depth_compared += tally.depth_compared
            out.depth_agree += tally.depth_agree
            out.depth_truncated += tally.depth_truncated
            out.fractional += tally.fractional
            for side, count in tally.within_one.items():
                out.within_one[side] = out.within_one.get(side, 0) + count
            for side, bins in tally.differences.items():
                merged = out.differences.setdefault(side, {})
                for units, count in bins.items():
                    merged[units] = merged.get(units, 0) + count
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuityRun:
    run_id: str
    manifest: Path
    manifest_sha256: str
    floor: LatencyFloor
    scope_start: datetime
    scope_end: datetime
    event_days: tuple[date, ...]
    sweep: Continuity
    pooled: Agreement
    by_city: Mapping[str, Agreement]


# str on a float gives the shortest repr that round trips, so a stored cent price recovers
# exactly and no arithmetic ever touches the float.
def read_snapshots(
    conn: sqlite3.Connection, ticker: str, start: datetime, end: datetime, *, rows: RowTally
) -> list[Snapshot]:
    bounds = (ticker, start.strftime(STAMP_FORMAT), end.strftime(STAMP_FORMAT))
    out = []
    stale = 0
    for at, yes_bid, no_bid, yes_bid_depth, no_bid_depth in conn.execute(SNAPSHOT_ROWS, bounds):
        rows.returned += 1
        stamp = datetime.strptime(at, STAMP_FORMAT).replace(tzinfo=timezone.utc)
        if stamp < ERA_START:
            stale += 1
            continue
        out.append(
            Snapshot(
                at=stamp,
                yes_bid=Decimal(str(yes_bid)),
                no_bid=Decimal(str(no_bid)),
                yes_bid_depth=yes_bid_depth,
                no_bid_depth=no_bid_depth,
            )
        )
    rows.era += stale
    if stale:
        raise ValueError(
            f"{ticker} carries {stale} snapshot rows before {ERA_START.isoformat()}: the frozen "
            f"scope and the era boundary disagree"
        )
    return out


def day_books(artifacts: Path, series: str, event_date: date, day: EventDay) -> dict[str, LegBook]:
    table = _read_legs(
        artifacts,
        LADDER,
        series,
        window_dates(day.window_start, day.window_end),
        event_date,
        TOUCH_COLUMNS,
    )
    legs = table.column("ticker").combine_chunks().dictionary_encode()
    seat_of_row = np.asarray(legs.indices)
    times = np.asarray(table.column("received_at").combine_chunks().cast(pa.int64()))
    yes_bid = _units(table, "yes_bid", PRICE_DECIMALS)
    no_bid = _units(table, "no_bid", PRICE_DECIMALS)
    yes_bid_depth = _units(table, "yes_bid_depth", SIZE_DECIMALS)
    no_bid_depth = _units(table, "no_bid_depth", SIZE_DECIMALS)

    books = {}
    for seat, ticker in enumerate(legs.dictionary.to_pylist()):
        of_leg = np.flatnonzero(seat_of_row == seat)
        stamps = times[of_leg]
        if np.any(np.diff(stamps) < 0):
            raise ValueError(f"{ticker} ladder rows are not in received_at order")
        # A snapshot batch emits one row per level under one received_at, so every row of a batch
        # but the last is a partially rebuilt book that never held.
        held = of_leg[_last_at_each(stamps)]
        books[ticker] = LegBook(
            times=times[held],
            yes_bid=yes_bid[held],
            no_bid=no_bid[held],
            yes_bid_depth=yes_bid_depth[held],
            no_bid_depth=no_bid_depth[held],
        )
    return books


def compare_leg(
    scope: RunScope,
    series: str,
    event_date: date,
    ticker: str,
    snapshots: Sequence[Snapshot],
    book: LegBook,
    *,
    rows: RowTally,
    tally: Tally,
) -> list[Disagreement]:
    at_us = np.fromiter(
        (_micros(item.at) for item in snapshots), dtype=np.int64, count=len(snapshots)
    )
    cells = screen_cells(scope, series, event_date, at_us, at_us)
    rows.out_of_window += cells.out_of_window
    rows.excluded += cells.excluded
    for name, count in cells.by_class.items():
        rows.by_class[name] = rows.by_class.get(name, 0) + count

    state = np.searchsorted(book.times, at_us, side="right") - 1
    rows.no_coverage += int(np.count_nonzero(cells.kept & (state < 0)))

    found: list[Disagreement] = []
    for index in np.flatnonzero(cells.kept & (state >= 0)):
        item = snapshots[index]
        seat = int(state[index])
        ws_yes_bid = int(book.yes_bid[seat])
        ws_no_bid = int(book.no_bid[seat])
        ws_yes_depth = int(book.yes_bid_depth[seat])
        ws_no_depth = int(book.no_bid_depth[seat])
        rows.compared += 1
        tally.compared += 1

        priced = item.yes_bid * PRICE_TICKS == ws_yes_bid and item.no_bid * PRICE_TICKS == ws_no_bid
        if priced:
            tally.price_agree += 1

        if item.yes_bid_depth is None or item.no_bid_depth is None:
            rows.null_depth += 1
            matched = priced
        else:
            rest_yes_depth = item.yes_bid_depth * SIZE_UNITS
            rest_no_depth = item.no_bid_depth * SIZE_UNITS
            truncated = (
                item.yes_bid_depth == ws_yes_depth // SIZE_UNITS
                and item.no_bid_depth == ws_no_depth // SIZE_UNITS
            )
            tally.depth_compared += 1
            if rest_yes_depth == ws_yes_depth and rest_no_depth == ws_no_depth:
                tally.depth_agree += 1
            if truncated:
                tally.depth_truncated += 1
            if ws_yes_depth % SIZE_UNITS or ws_no_depth % SIZE_UNITS:
                tally.fractional += 1
            for side, difference in (
                (YES, rest_yes_depth - ws_yes_depth),
                (NO, rest_no_depth - ws_no_depth),
            ):
                bins = tally.differences.setdefault(side, {})
                bins[difference] = bins.get(difference, 0) + 1
                if abs(difference) <= SIZE_UNITS:
                    tally.within_one[side] = tally.within_one.get(side, 0) + 1
            matched = priced and truncated

        if matched or len(found) >= SAMPLE_MAX:
            continue
        found.append(
            Disagreement(
                ticker=ticker,
                snapshot_at=item.at,
                rest_yes_bid=item.yes_bid,
                rest_no_bid=item.no_bid,
                rest_yes_bid_depth=_whole(item.yes_bid_depth),
                rest_no_bid_depth=_whole(item.no_bid_depth),
                ws_yes_bid=Decimal(ws_yes_bid) / PRICE_TICKS,
                ws_no_bid=Decimal(ws_no_bid) / PRICE_TICKS,
                ws_yes_bid_depth=Decimal(ws_yes_depth) / SIZE_UNITS,
                ws_no_bid_depth=Decimal(ws_no_depth) / SIZE_UNITS,
            )
        )
    return found


def sweep_continuity(scope: RunScope, artifacts: Path, db: Path) -> Continuity:
    cities = tuple(sorted({series for series, _ in scope.event_days}))
    rows = RowTally()
    tallies = {series: Tally() for series in cities}
    sample: list[Disagreement] = []

    conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        for (series, event_date), day in sorted(scope.event_days.items()):
            started = time.monotonic()
            before = rows.compared
            books = day_books(artifacts, series, event_date, day)
            found: list[Disagreement] = []
            for ticker in sorted(books):
                snapshots = read_snapshots(
                    conn, ticker, day.window_start, day.window_end, rows=rows
                )
                found.extend(
                    compare_leg(
                        scope,
                        series,
                        event_date,
                        ticker,
                        snapshots,
                        books[ticker],
                        rows=rows,
                        tally=tallies[series],
                    )
                )
            sample = sorted(sample + found, key=lambda item: (item.ticker, item.snapshot_at))
            del sample[SAMPLE_MAX:]
            logger.info(
                "depth_continuity series=%s event_date=%s legs=%d compared=%d elapsed_s=%.1f",
                series,
                event_date.isoformat(),
                len(books),
                rows.compared - before,
                time.monotonic() - started,
            )
    finally:
        conn.close()

    return Continuity(cities=cities, rows=rows, tallies=tallies, disagreements=tuple(sample))


def agreement(tally: Tally) -> Agreement:
    return Agreement(
        compared=tally.compared,
        price_agree=tally.price_agree,
        price_fraction=_fraction(tally.price_agree, tally.compared),
        depth_compared=tally.depth_compared,
        depth_agree=tally.depth_agree,
        depth_fraction=_fraction(tally.depth_agree, tally.depth_compared),
        depth_truncated=tally.depth_truncated,
        depth_truncated_fraction=_fraction(tally.depth_truncated, tally.depth_compared),
        fractional=tally.fractional,
        fractional_fraction=_fraction(tally.fractional, tally.depth_compared),
        sides={side: _side_readout(tally, side) for side in SIDES},
    )


def _side_readout(tally: Tally, side: str) -> SideReadout:
    bins = tally.differences.get(side, {})
    within = tally.within_one.get(side, 0)
    return SideReadout(
        p25=_contracts(weighted_quantile(bins, 1, 4)),
        p50=_contracts(weighted_quantile(bins, 1, 2)),
        p75=_contracts(weighted_quantile(bins, 3, 4)),
        within_one_contract=within,
        within_one_fraction=_fraction(within, tally.depth_compared),
    )


def execute(
    *,
    run_id: str,
    preregistration: Path,
    repo: Path,
    run_scope: Path,
    artifacts: Path,
    db: Path,
    rtt_samples: Path,
    floor_source: FloorSource,
    seed: int,
    run_root: Path,
) -> ContinuityRun:
    inputs = assemble_run_inputs(
        run_id=run_id,
        preregistration=preregistration,
        repo=repo,
        run_scope=run_scope,
        artifacts=artifacts,
        rtt_samples=rtt_samples,
        floor_source=floor_source,
        bootstrap_seed=seed,
    )
    digest = write_manifest(run_root, inputs)

    scope = load_run_scope(run_scope)
    swept = sweep_continuity(scope, artifacts, db)
    pooled = agreement(swept.pooled())
    run = ContinuityRun(
        run_id=run_id,
        manifest=run_root / run_id / MANIFEST_NAME,
        manifest_sha256=digest,
        floor=inputs.floor,
        scope_start=scope.scope_start,
        scope_end=scope.scope_end,
        event_days=tuple(sorted(scope.discovery_days | scope.holdout_days)),
        sweep=swept,
        pooled=pooled,
        by_city={series: agreement(swept.tallies[series]) for series in swept.cities},
    )
    logger.info(
        "depth_continuity compared=%d price=%s depth=%s truncated=%s fractional=%s",
        pooled.compared,
        pooled.price_fraction,
        pooled.depth_fraction,
        pooled.depth_truncated_fraction,
        pooled.fractional_fraction,
    )
    return run


def result_payload(run: ContinuityRun) -> dict:
    rows = run.sweep.rows
    return {
        "run_id": run.run_id,
        "manifest": str(run.manifest),
        "manifest_sha256": run.manifest_sha256,
        "latency_floor_source": run.floor.source.value,
        "scope_start": run.scope_start.isoformat(),
        "scope_end": run.scope_end.isoformat(),
        "era_start": ERA_START.isoformat(),
        "sample_max": SAMPLE_MAX,
        "event_days": [day.isoformat() for day in run.event_days],
        "cities": list(run.sweep.cities),
        "rows": {
            "returned": rows.returned,
            "era_dropped": rows.era,
            "out_of_window": rows.out_of_window,
            "excluded": rows.excluded,
            "by_class": dict(sorted(rows.by_class.items())),
            "no_coverage": rows.no_coverage,
            "null_depth": rows.null_depth,
            "compared": rows.compared,
        },
        "pooled": _agreement_payload(run.pooled),
        "by_city": {
            series: _agreement_payload(block) for series, block in sorted(run.by_city.items())
        },
        "disagreements": [
            {
                "ticker": item.ticker,
                "snapshot_at": item.snapshot_at.isoformat(),
                "rest": {
                    "yes_bid": str(item.rest_yes_bid),
                    "no_bid": str(item.rest_no_bid),
                    "yes_bid_depth": _text(item.rest_yes_bid_depth),
                    "no_bid_depth": _text(item.rest_no_bid_depth),
                },
                "ws": {
                    "yes_bid": str(item.ws_yes_bid),
                    "no_bid": str(item.ws_no_bid),
                    "yes_bid_depth": str(item.ws_yes_bid_depth),
                    "no_bid_depth": str(item.ws_no_bid_depth),
                },
            }
            for item in run.sweep.disagreements
        ],
    }


def _agreement_payload(block: Agreement) -> dict:
    return {
        "compared": block.compared,
        "price_agree": block.price_agree,
        "price_agree_fraction": _text(block.price_fraction),
        "depth_compared": block.depth_compared,
        "depth_agree": block.depth_agree,
        "depth_agree_fraction": _text(block.depth_fraction),
        "depth_agree_truncated": block.depth_truncated,
        "depth_agree_truncated_fraction": _text(block.depth_truncated_fraction),
        "fractional_ws_depth": block.fractional,
        "fractional_ws_depth_fraction": _text(block.fractional_fraction),
        "depth_difference_contracts": {
            side: {
                "p25": _text(readout.p25),
                "p50": _text(readout.p50),
                "p75": _text(readout.p75),
                "within_one_contract": readout.within_one_contract,
                "within_one_contract_fraction": _text(readout.within_one_fraction),
            }
            for side, readout in block.sides.items()
        },
    }


def _units(table: pa.Table, name: str, decimals: int) -> np.ndarray:
    return scaled_units(table.column(name).combine_chunks(), decimals)


def _fraction(part: int, whole: int) -> Decimal | None:
    return None if whole == 0 else Decimal(part) / Decimal(whole)


def _contracts(units: int | None) -> Decimal | None:
    return None if units is None else Decimal(units) / SIZE_UNITS


def _whole(contracts: int | None) -> Decimal | None:
    return None if contracts is None else Decimal(contracts)


def _text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _micros(stamp: datetime) -> int:
    return (stamp - _EPOCH) // _MICROSECOND
