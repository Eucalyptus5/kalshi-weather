from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot.lag.forecast_sample import SampleLeg, sidecar_path
from bot.markets.observation_window import observation_window


CLASS_A = "a"
CLASS_B = "b"
CLASS_C = "c"

LST_FULL = "lst_full"
LST_FULL_LESS_LAST_HOUR = "lst_full_less_last_hour"
UTC_12Z_00Z = "utc_12z_00z"

CLASS_B_MEMBER = "nbm_nbs"
CLASS_C_MEMBER = "hrrr"

LEAD_ANCHORED_COMPOSITE = "lead_anchored_composite"
MODEL_RUN = "model_run"

WINDOW_HOURS = 24

LegKey = tuple[str, date, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassRecord:
    station: str
    event_date: date
    lead_hours: int
    forecast_class: str
    member: str
    daily_high_f: Decimal
    issue_time: datetime
    issue_rule: str
    window_basis: str
    native_sigma_f: Decimal | None
    grid_latitude: float | None
    grid_longitude: float | None
    source_url: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassFreeze:
    path: Path
    sha256: str
    written: int
    refused: tuple[str, ...]


def window_basis_for(forecast_class: str, lead_hours: int) -> str:
    if forecast_class == CLASS_B:
        return UTC_12Z_00Z
    if forecast_class == CLASS_A and lead_hours == 24:
        return LST_FULL_LESS_LAST_HOUR
    return LST_FULL


def lst_window_hours(timezone: str, event_date: date, basis: str) -> tuple[datetime, ...]:
    start, _ = observation_window(timezone, event_date)
    points = WINDOW_HOURS - 1 if basis == LST_FULL_LESS_LAST_HOUR else WINDOW_HOURS
    return tuple(start + timedelta(hours=offset) for offset in range(points))


def class_freeze_path(directory: Path, forecast_class: str) -> Path:
    return directory / f"class_{forecast_class}.jsonl"


def leg_index(legs: Sequence[SampleLeg]) -> dict[LegKey, SampleLeg]:
    return {(leg.station, leg.event_date, leg.lead_hours): leg for leg in legs}


def class_a_record(
    leg: SampleLeg,
    *,
    member: str,
    hourly: Mapping[datetime, Decimal],
    issue_offset: timedelta,
    latitude: float,
    longitude: float,
    source_url: str,
) -> ClassRecord | None:
    basis = window_basis_for(CLASS_A, leg.lead_hours)
    hours = lst_window_hours(leg.timezone, leg.event_date, basis)
    if not all(hour in hourly for hour in hours):
        return None
    return ClassRecord(
        station=leg.station,
        event_date=leg.event_date,
        lead_hours=leg.lead_hours,
        forecast_class=CLASS_A,
        member=member,
        daily_high_f=max(hourly[hour] for hour in hours),
        issue_time=hours[-1] - issue_offset,
        issue_rule=LEAD_ANCHORED_COMPOSITE,
        window_basis=basis,
        native_sigma_f=None,
        grid_latitude=latitude,
        grid_longitude=longitude,
        source_url=source_url,
    )


def class_b_record(
    leg: SampleLeg,
    *,
    daily_high_f: Decimal,
    native_sigma_f: Decimal | None,
    runtime: datetime,
    source_url: str,
) -> ClassRecord:
    return ClassRecord(
        station=leg.station,
        event_date=leg.event_date,
        lead_hours=leg.lead_hours,
        forecast_class=CLASS_B,
        member=CLASS_B_MEMBER,
        daily_high_f=daily_high_f,
        issue_time=runtime,
        issue_rule=MODEL_RUN,
        window_basis=UTC_12Z_00Z,
        native_sigma_f=native_sigma_f,
        grid_latitude=None,
        grid_longitude=None,
        source_url=source_url,
    )


def class_c_record(
    leg: SampleLeg,
    *,
    hourly: Mapping[datetime, Decimal],
    run: datetime,
    latitude: float,
    longitude: float,
    source_url: str,
) -> ClassRecord | None:
    hours = lst_window_hours(leg.timezone, leg.event_date, LST_FULL)
    if not all(hour in hourly for hour in hours):
        return None
    return ClassRecord(
        station=leg.station,
        event_date=leg.event_date,
        lead_hours=leg.lead_hours,
        forecast_class=CLASS_C,
        member=CLASS_C_MEMBER,
        daily_high_f=max(hourly[hour] for hour in hours),
        issue_time=run,
        issue_rule=MODEL_RUN,
        window_basis=LST_FULL,
        native_sigma_f=None,
        grid_latitude=latitude,
        grid_longitude=longitude,
        source_url=source_url,
    )


def write_class_freeze(
    records: Sequence[ClassRecord], path: Path, legs: Mapping[LegKey, SampleLeg]
) -> ClassFreeze:
    sidecar = sidecar_path(path)
    for target in (path, sidecar):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    kept: list[ClassRecord] = []
    refused: list[str] = []
    for row in records:
        as_of = legs[(row.station, row.event_date, row.lead_hours)].as_of
        if row.issue_time > as_of:
            refused.append(
                f"{row.station} {row.event_date.isoformat()} lead={row.lead_hours} "
                f"{row.forecast_class}/{row.member} issued {row.issue_time.isoformat()} "
                f"after as_of {as_of.isoformat()}"
            )
            continue
        kept.append(row)

    body = "".join(json.dumps(_record_payload(row)) + "\n" for row in kept).encode()
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    sidecar.write_text(
        json.dumps(
            {
                "file": path.name,
                "records": len(kept),
                "refused": len(refused),
                "leads": sorted({row.lead_hours for row in kept}),
                "members": sorted({row.member for row in kept}),
                "sha256": digest,
            },
            indent=1,
        )
    )
    return ClassFreeze(path=path, sha256=digest, written=len(kept), refused=tuple(refused))


def read_class_freeze(path: Path) -> list[ClassRecord]:
    body = path.read_bytes()
    stored = json.loads(sidecar_path(path).read_text())["sha256"]
    if hashlib.sha256(body).hexdigest() != stored:
        raise ValueError(f"{path} does not match the sha256 it carries")
    return [_record_from_payload(json.loads(line)) for line in body.decode().splitlines()]


def _record_payload(row: ClassRecord) -> dict:
    return {
        "station": row.station,
        "event_date": row.event_date.isoformat(),
        "lead_hours": row.lead_hours,
        "forecast_class": row.forecast_class,
        "member": row.member,
        "daily_high_f": str(row.daily_high_f),
        "issue_time": row.issue_time.isoformat(),
        "issue_rule": row.issue_rule,
        "window_basis": row.window_basis,
        "native_sigma_f": None if row.native_sigma_f is None else str(row.native_sigma_f),
        "grid_latitude": row.grid_latitude,
        "grid_longitude": row.grid_longitude,
        "source_url": row.source_url,
    }


def _record_from_payload(payload: dict) -> ClassRecord:
    sigma = payload["native_sigma_f"]
    return ClassRecord(
        station=payload["station"],
        event_date=date.fromisoformat(payload["event_date"]),
        lead_hours=payload["lead_hours"],
        forecast_class=payload["forecast_class"],
        member=payload["member"],
        daily_high_f=Decimal(payload["daily_high_f"]),
        issue_time=datetime.fromisoformat(payload["issue_time"]),
        issue_rule=payload["issue_rule"],
        window_basis=payload["window_basis"],
        native_sigma_f=None if sigma is None else Decimal(sigma),
        grid_latitude=payload["grid_latitude"],
        grid_longitude=payload["grid_longitude"],
        source_url=payload["source_url"],
    )
