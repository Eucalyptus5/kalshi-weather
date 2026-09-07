from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bot.main import STATIONS
from bot.markets.parser import parse_ticker, resolve_event_kinds


F4_SERIES: tuple[str, ...] = (
    "KXHIGHAUS",
    "KXHIGHCHI",
    "KXHIGHDEN",
    "KXHIGHLAX",
    "KXHIGHMIA",
    "KXHIGHNY",
    "KXHIGHPHIL",
)
F4_LEADS: tuple[int, ...] = (24, 36)
SPLIT_BOUNDARY: date = date(2025, 8, 15)
DEPTH_WINDOW: timedelta = timedelta(hours=6)
WEEKLY_ORDER: tuple[str, ...] = (
    "0016",
    "0017",
    "0018",
    "0019",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
)
PRE_WEEKLY_ERA = "0010"
SIDECAR_SUFFIX = ".sha256.json"

_MICROS_PER_MINUTE = Decimal(60_000_000)


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplePlan:
    candidate_days: list[date]
    sampled_days: list[date]
    unread_days: list[date]
    discovery_days: list[date]
    holdout_days: list[date]


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleLeg:
    ticker: str
    series: str
    station: str
    timezone: str
    event_date: date
    split: str
    lead_hours: int
    close_time: datetime
    as_of: datetime
    entry_price: Decimal
    staleness_minutes: Decimal
    era: str
    trailing_prints: int
    trailing_contracts: Decimal
    strike_lo: Decimal
    strike_hi: Decimal | None
    kind: str
    result: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TickTape:
    rows: Mapping[str, np.ndarray]
    created_time: np.ndarray
    yes_price: pa.Array
    count: np.ndarray


def read_sample_plan(path: Path) -> SamplePlan:
    payload = json.loads(path.read_text())
    candidate = [date.fromisoformat(day) for day in payload["candidate_days"]]
    sampled = [date.fromisoformat(day) for day in payload["sampled_days"]]
    if candidate[::4] != sampled:
        raise ValueError(f"{path} sampled_days is not candidate_days[::4]")
    read = set(sampled)
    unread = [day for day in candidate if day not in read]
    return SamplePlan(
        candidate_days=candidate,
        sampled_days=sampled,
        unread_days=unread,
        discovery_days=[day for day in unread if day < SPLIT_BOUNDARY],
        holdout_days=[day for day in unread if day >= SPLIT_BOUNDARY],
    )


def era_caps(era_report_path: Path) -> dict[str, timedelta | None]:
    eras = json.loads(era_report_path.read_text(), parse_float=Decimal)["eras"]
    caps: dict[str, timedelta | None] = {}
    for era, row in eras.items():
        if "cap_minutes" not in row:
            continue
        minutes = row["cap_minutes"]
        caps[era] = (
            None
            if minutes is None
            else timedelta(microseconds=int(Decimal(minutes) * _MICROS_PER_MINUTE))
        )
    return caps


def era_index(ingest_report_path: Path) -> tuple[tuple[str, datetime], ...]:
    per_shard = json.loads(ingest_report_path.read_text())["per_shard"]
    return tuple(
        (shard, datetime.fromisoformat(per_shard[shard]["min_created"])) for shard in WEEKLY_ORDER
    )


def era_of(as_of: datetime, eras: Sequence[tuple[str, datetime]]) -> str:
    if as_of < eras[0][1]:
        return PRE_WEEKLY_ERA
    era = eras[0][0]
    for shard, start in eras:
        if as_of >= start:
            era = shard
    return era


def read_tick_tape(path: Path) -> TickTape:
    table = pq.read_table(path, columns=["ticker", "created_time", "yes_price", "count"])
    created_time = table.column("created_time").combine_chunks().to_numpy()
    positions: dict[str, list[int]] = {}
    for row, ticker in enumerate(table.column("ticker").to_pylist()):
        positions.setdefault(ticker, []).append(row)
    rows = {}
    for ticker, found in positions.items():
        ordered = np.asarray(found)
        rows[ticker] = ordered[np.argsort(created_time[ordered], kind="stable")]
    return TickTape(
        rows=rows,
        created_time=created_time,
        yes_price=table.column("yes_price").combine_chunks(),
        count=table.column("count").combine_chunks().to_numpy(),
    )


def build_sample(
    markets_path: Path,
    tape: TickTape,
    plan: SamplePlan,
    caps: Mapping[str, timedelta | None],
    eras: Sequence[tuple[str, datetime]],
    lead_hours: int,
) -> list[SampleLeg]:
    unread = set(plan.unread_days)
    discovery = set(plan.discovery_days)
    lead = timedelta(hours=lead_hours)

    ladders: dict[tuple[str, date], list[dict]] = {}
    for row in pq.read_table(markets_path).to_pylist():
        if row["series_ticker"] not in F4_SERIES:
            continue
        parsed = parse_ticker(row["ticker"])
        if parsed.event_date not in unread:
            continue
        ladders.setdefault((parsed.series, parsed.event_date), []).append(row)

    legs: list[SampleLeg] = []
    for key in sorted(ladders):
        rungs = sorted(ladders[key], key=lambda row: row["ticker"])
        tagged = resolve_event_kinds([parse_ticker(row["ticker"]) for row in rungs])
        for row, parsed in zip(rungs, tagged):
            if row["close_time"] is None or row["result"] not in ("yes", "no"):
                continue
            as_of = row["close_time"] - lead
            era = era_of(as_of, eras)
            positions = tape.rows.get(row["ticker"])
            if positions is None:
                continue
            times = tape.created_time[positions]
            as_of64 = _to64(as_of)
            last = int(np.searchsorted(times, as_of64, side="right")) - 1
            if last < 0:
                continue
            micros = int((as_of64 - times[last]) / np.timedelta64(1, "us"))
            cap = caps[era]
            if cap is not None and timedelta(microseconds=micros) > cap:
                continue
            first = int(np.searchsorted(times, _to64(as_of - DEPTH_WINDOW), side="right"))
            station = STATIONS[parsed.series]
            legs.append(
                SampleLeg(
                    ticker=row["ticker"],
                    series=parsed.series,
                    station=station.station,
                    timezone=station.timezone,
                    event_date=parsed.event_date,
                    split="discovery" if parsed.event_date in discovery else "holdout",
                    lead_hours=lead_hours,
                    close_time=row["close_time"],
                    as_of=as_of,
                    entry_price=tape.yes_price[int(positions[last])].as_py(),
                    staleness_minutes=Decimal(micros) / _MICROS_PER_MINUTE,
                    era=era,
                    trailing_prints=last + 1 - first,
                    trailing_contracts=Decimal(int(tape.count[positions[first : last + 1]].sum())),
                    strike_lo=parsed.strikes[0],
                    strike_hi=parsed.strikes[1] if len(parsed.strikes) == 2 else None,
                    kind=parsed.kind,
                    result=row["result"],
                )
            )
    return legs


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def write_sample_freeze(legs: Sequence[SampleLeg], path: Path) -> str:
    sidecar = sidecar_path(path)
    for target in (path, sidecar):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
    body = "".join(json.dumps(_leg_payload(leg)) + "\n" for leg in legs).encode()
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    sidecar.write_text(
        json.dumps(
            {
                "file": path.name,
                "legs": len(legs),
                "leads": sorted({leg.lead_hours for leg in legs}),
                "sha256": digest,
            },
            indent=1,
        )
    )
    return digest


def read_sample_freeze(path: Path) -> list[SampleLeg]:
    body = path.read_bytes()
    stored = json.loads(sidecar_path(path).read_text())["sha256"]
    if hashlib.sha256(body).hexdigest() != stored:
        raise ValueError(f"{path} does not match the sha256 it carries")
    return [_leg_from_payload(json.loads(line)) for line in body.decode().splitlines()]


def _to64(moment: datetime) -> np.datetime64:
    return np.datetime64(moment.astimezone(timezone.utc).replace(tzinfo=None), "us")


def _leg_payload(leg: SampleLeg) -> dict:
    return {
        "ticker": leg.ticker,
        "series": leg.series,
        "station": leg.station,
        "timezone": leg.timezone,
        "event_date": leg.event_date.isoformat(),
        "split": leg.split,
        "lead_hours": leg.lead_hours,
        "close_time": leg.close_time.isoformat(),
        "as_of": leg.as_of.isoformat(),
        "entry_price": str(leg.entry_price),
        "staleness_minutes": str(leg.staleness_minutes),
        "era": leg.era,
        "trailing_prints": leg.trailing_prints,
        "trailing_contracts": str(leg.trailing_contracts),
        "strike_lo": str(leg.strike_lo),
        "strike_hi": None if leg.strike_hi is None else str(leg.strike_hi),
        "kind": leg.kind,
        "result": leg.result,
    }


def _leg_from_payload(payload: dict) -> SampleLeg:
    strike_hi = payload["strike_hi"]
    return SampleLeg(
        ticker=payload["ticker"],
        series=payload["series"],
        station=payload["station"],
        timezone=payload["timezone"],
        event_date=date.fromisoformat(payload["event_date"]),
        split=payload["split"],
        lead_hours=payload["lead_hours"],
        close_time=datetime.fromisoformat(payload["close_time"]),
        as_of=datetime.fromisoformat(payload["as_of"]),
        entry_price=Decimal(payload["entry_price"]),
        staleness_minutes=Decimal(payload["staleness_minutes"]),
        era=payload["era"],
        trailing_prints=payload["trailing_prints"],
        trailing_contracts=Decimal(payload["trailing_contracts"]),
        strike_lo=Decimal(payload["strike_lo"]),
        strike_hi=None if strike_hi is None else Decimal(strike_hi),
        kind=payload["kind"],
        result=payload["result"],
    )
