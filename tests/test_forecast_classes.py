from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.forecast_classes import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    LEAD_ANCHORED_COMPOSITE,
    LST_FULL,
    LST_FULL_LESS_LAST_HOUR,
    MODEL_RUN,
    UTC_12Z_00Z,
    WINDOW_HOURS,
    ClassRecord,
    class_a_record,
    class_b_record,
    class_c_record,
    class_freeze_path,
    leg_index,
    lst_window_hours,
    read_class_freeze,
    sidecar_path,
    window_basis_for,
    write_class_freeze,
)
from bot.lag.forecast_sample import SampleLeg
from bot.markets.observation_window import observation_window


UTC = timezone.utc
EVENT_DATE = date(2025, 6, 8)
CLOSE = datetime(2025, 6, 9, 3, 59, tzinfo=UTC)


def sample_leg(
    *, lead_hours: int = 24, station: str = "KNYC", tz: str = "America/New_York"
) -> SampleLeg:
    close = CLOSE
    return SampleLeg(
        ticker=f"KXHIGHNY-25JUN08-T80-{lead_hours}",
        series="KXHIGHNY",
        station=station,
        timezone=tz,
        event_date=EVENT_DATE,
        split="discovery",
        lead_hours=lead_hours,
        close_time=close,
        as_of=close - timedelta(hours=lead_hours),
        entry_price=Decimal("0.42"),
        staleness_minutes=Decimal("3.0"),
        era="0016",
        trailing_prints=4,
        trailing_contracts=Decimal(12),
        strike_lo=Decimal(80),
        strike_hi=None,
        kind="threshold_above",
        result="yes",
    )


def hourly_for(
    leg: SampleLeg, basis: str, peak: Decimal = Decimal("77.4")
) -> dict[datetime, Decimal]:
    hours = lst_window_hours(leg.timezone, leg.event_date, basis)
    values = {hour: Decimal("60.0") for hour in hours}
    values[hours[12]] = peak
    return values


def record(
    *,
    member: str = "ecmwf_ifs025",
    issue_time: datetime = CLOSE - timedelta(hours=25),
    native_sigma_f: Decimal | None = None,
) -> ClassRecord:
    return ClassRecord(
        station="KNYC",
        event_date=EVENT_DATE,
        lead_hours=24,
        forecast_class=CLASS_A,
        member=member,
        daily_high_f=Decimal("77.4"),
        issue_time=issue_time,
        issue_rule=LEAD_ANCHORED_COMPOSITE,
        window_basis=LST_FULL_LESS_LAST_HOUR,
        native_sigma_f=native_sigma_f,
        grid_latitude=40.75,
        grid_longitude=-74.0,
        source_url="https://previous-runs-api.open-meteo.com/v1/forecast?x=1",
    )


def test_window_hours_stop_one_hour_short_of_the_window_end() -> None:
    start, end = observation_window("America/New_York", EVENT_DATE)
    hours = lst_window_hours("America/New_York", EVENT_DATE, LST_FULL)

    assert WINDOW_HOURS == 24
    assert len(hours) == 24
    assert hours[0] == start
    assert hours[-1] == start + timedelta(hours=23)
    assert end not in hours


def test_less_last_hour_basis_drops_the_twenty_fourth_point() -> None:
    hours = lst_window_hours("America/New_York", EVENT_DATE, LST_FULL_LESS_LAST_HOUR)
    full = lst_window_hours("America/New_York", EVENT_DATE, LST_FULL)

    assert len(hours) == 23
    assert hours == full[:-1]


@pytest.mark.parametrize(
    ("forecast_class", "lead_hours", "basis"),
    [
        (CLASS_A, 24, LST_FULL_LESS_LAST_HOUR),
        (CLASS_A, 36, LST_FULL),
        (CLASS_B, 24, UTC_12Z_00Z),
        (CLASS_B, 36, UTC_12Z_00Z),
        (CLASS_C, 24, LST_FULL),
        (CLASS_C, 36, LST_FULL),
    ],
)
def test_window_basis_is_pinned_per_class_and_lead(
    forecast_class: str, lead_hours: int, basis: str
) -> None:
    assert window_basis_for(forecast_class, lead_hours) == basis


def test_class_a_record_takes_the_max_and_the_latest_component_issue() -> None:
    leg = sample_leg(lead_hours=24)
    basis = window_basis_for(CLASS_A, 24)
    hourly = hourly_for(leg, basis)

    built = class_a_record(
        leg,
        member="ecmwf_ifs025",
        hourly=hourly,
        issue_offset=timedelta(hours=24),
        latitude=40.75,
        longitude=-74.0,
        source_url="https://previous-runs-api.open-meteo.com/v1/forecast",
    )

    hours = lst_window_hours(leg.timezone, leg.event_date, basis)
    assert built is not None
    assert built.daily_high_f == Decimal("77.4")
    assert built.issue_time == hours[-1] - timedelta(hours=24)
    assert built.issue_rule == LEAD_ANCHORED_COMPOSITE
    assert built.window_basis == LST_FULL_LESS_LAST_HOUR
    assert built.native_sigma_f is None
    assert built.forecast_class == CLASS_A


def test_class_a_record_is_none_when_a_window_hour_is_absent() -> None:
    leg = sample_leg(lead_hours=36)
    basis = window_basis_for(CLASS_A, 36)
    hourly = hourly_for(leg, basis)
    del hourly[lst_window_hours(leg.timezone, leg.event_date, basis)[3]]

    assert (
        class_a_record(
            leg,
            member="ukmo_global_deterministic_10km",
            hourly=hourly,
            issue_offset=timedelta(hours=48),
            latitude=40.78125,
            longitude=-73.96875,
            source_url="https://previous-runs-api.open-meteo.com/v1/forecast",
        )
        is None
    )


def test_class_b_record_is_station_granularity_and_utc_anchored() -> None:
    leg = sample_leg(lead_hours=36)
    runtime = datetime(2025, 6, 7, 19, tzinfo=UTC)

    built = class_b_record(
        leg,
        daily_high_f=Decimal("88.0"),
        native_sigma_f=Decimal("2.0"),
        runtime=runtime,
        source_url="https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station=KNYC",
    )

    assert built.forecast_class == CLASS_B
    assert built.member == "nbm_nbs"
    assert built.window_basis == UTC_12Z_00Z
    assert built.issue_rule == MODEL_RUN
    assert built.issue_time == runtime
    assert built.native_sigma_f == Decimal("2.0")
    assert built.grid_latitude is None
    assert built.grid_longitude is None


def test_class_c_record_carries_the_cycle_and_the_grid_cell() -> None:
    leg = sample_leg(lead_hours=24)
    hourly = hourly_for(leg, LST_FULL, peak=Decimal("91.22"))
    run = datetime(2025, 6, 7, 18, tzinfo=UTC)

    built = class_c_record(
        leg,
        hourly=hourly,
        run=run,
        latitude=40.788,
        longitude=-73.965,
        source_url="https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250607/conus",
    )

    assert built is not None
    assert built.forecast_class == CLASS_C
    assert built.member == "hrrr"
    assert built.window_basis == LST_FULL
    assert built.issue_rule == MODEL_RUN
    assert built.issue_time == run
    assert built.daily_high_f == Decimal("91.22")
    assert built.native_sigma_f is None
    assert built.grid_latitude == pytest.approx(40.788)


def test_freeze_round_trips_every_field(tmp_path: Path) -> None:
    leg = sample_leg()
    records = [record(), record(member="icon_global", native_sigma_f=Decimal("1.5"))]
    path = tmp_path / "class_a.jsonl"

    frozen = write_class_freeze(records, path, leg_index([leg]))

    assert frozen.written == 2
    assert frozen.refused == ()
    assert read_class_freeze(path) == records
    assert json.loads(sidecar_path(path).read_text())["sha256"] == frozen.sha256


def test_freeze_refuses_to_overwrite(tmp_path: Path) -> None:
    leg = sample_leg()
    path = tmp_path / "class_a.jsonl"
    write_class_freeze([record()], path, leg_index([leg]))

    with pytest.raises(FileExistsError):
        write_class_freeze([record()], path, leg_index([leg]))


def test_read_freeze_raises_when_the_sidecar_disagrees(tmp_path: Path) -> None:
    leg = sample_leg()
    path = tmp_path / "class_a.jsonl"
    write_class_freeze([record()], path, leg_index([leg]))
    path.write_bytes(path.read_bytes().replace(b"77.4", b"88.4"))

    with pytest.raises(ValueError, match="sha256"):
        read_class_freeze(path)


def test_freeze_refuses_a_record_issued_after_its_leg_decision_instant(tmp_path: Path) -> None:
    leg = sample_leg()
    late = record(issue_time=leg.as_of + timedelta(seconds=60))
    path = tmp_path / "class_a.jsonl"

    frozen = write_class_freeze([record(), late], path, leg_index([leg]))

    assert frozen.written == 1
    assert len(frozen.refused) == 1
    named = frozen.refused[0]
    assert "KNYC" in named
    assert EVENT_DATE.isoformat() in named
    assert "lead=24" in named
    assert leg.as_of.isoformat() in named
    assert [row.issue_time for row in read_class_freeze(path)] == [record().issue_time]


def test_class_freeze_path_names_one_file_per_class(tmp_path: Path) -> None:
    assert class_freeze_path(tmp_path, CLASS_A).name == "class_a.jsonl"
    assert class_freeze_path(tmp_path, CLASS_B).name == "class_b.jsonl"
    assert class_freeze_path(tmp_path, CLASS_C).name == "class_c.jsonl"


def test_leg_index_keys_on_the_triple_the_classes_retrieve_for() -> None:
    legs: Sequence[SampleLeg] = [sample_leg(lead_hours=24), sample_leg(lead_hours=36)]

    index = leg_index(legs)

    assert sorted(index) == [("KNYC", EVENT_DATE, 24), ("KNYC", EVENT_DATE, 36)]
    assert index[("KNYC", EVENT_DATE, 36)].as_of == CLOSE - timedelta(hours=36)
