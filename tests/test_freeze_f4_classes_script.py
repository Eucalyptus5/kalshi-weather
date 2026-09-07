from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from bot.lag.forecast_classes import (
    LST_FULL,
    LST_FULL_LESS_LAST_HOUR,
    UTC_12Z_00Z,
    read_class_freeze,
)
from bot.lag.forecast_sample import SampleLeg, write_sample_freeze
from scripts.freeze_f4_classes import (
    DEFAULT_OUT,
    DEFAULT_SAMPLE,
    DEFAULT_THREADS,
    REPO_ROOT,
    build_parser,
    main,
)
from tests.test_hrrr import MESSAGE, bucket_handler, idx_body


UTC = timezone.utc
EVENT_DATE = date(2025, 1, 15)
CLOSE = datetime(2025, 1, 16, 4, 59, tzinfo=UTC)
ECMWF_FIXTURE = Path(__file__).parent / "data" / "previous_runs_knyc_ecmwf.json"
MOS_FIXTURE = Path(__file__).parent / "data" / "nbs_knyc_20250128_01z.csv"


def leg(lead_hours: int) -> SampleLeg:
    return SampleLeg(
        ticker=f"KXHIGHNY-25JAN15-T40-{lead_hours}",
        series="KXHIGHNY",
        station="KNYC",
        timezone="America/New_York",
        event_date=EVENT_DATE,
        split="discovery",
        lead_hours=lead_hours,
        close_time=CLOSE,
        as_of=CLOSE - timedelta(hours=lead_hours),
        entry_price=Decimal("0.30"),
        staleness_minutes=Decimal("2.0"),
        era="0016",
        trailing_prints=3,
        trailing_contracts=Decimal(9),
        strike_lo=Decimal(40),
        strike_hi=None,
        kind="above",
        result="no",
    )


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable[..., httpx.Response]], None]:
    real = httpx.AsyncClient

    def install(handler: Callable[..., httpx.Response]) -> None:
        monkeypatch.setattr(
            "scripts.freeze_f4_classes.httpx.AsyncClient",
            lambda **kwargs: real(transport=httpx.MockTransport(handler)),
        )

    return install


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.jsonl"
    write_sample_freeze([leg(24), leg(36)], path)
    return path


def previous_runs_body(start: datetime, hours: int) -> bytes:
    times = [
        (start + timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M") for offset in range(hours)
    ]
    return json.dumps(
        {
            "latitude": 40.75,
            "longitude": -74.0,
            "hourly": {
                "time": times,
                "temperature_2m_previous_day1": [40.5] * hours,
                "temperature_2m_previous_day2": [41.5] * hours,
            },
        }
    ).encode()


def mos_body() -> str:
    header, *rows = MOS_FIXTURE.read_text().splitlines()
    shifted = []
    for runtime in ("2025-01-14 13:00:00", "2025-01-15 01:00:00"):
        for row in rows:
            columns = row.split(",")
            columns[0] = runtime
            columns[1] = (datetime.fromisoformat(columns[1]) - timedelta(days=13)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            shifted.append(",".join(columns))
    return "\n".join([header, *shifted]) + "\n"


def test_the_parser_defaults_match_the_frozen_sample_layout() -> None:
    parsed = build_parser().parse_args(["--class", "a"])

    assert parsed.out == DEFAULT_OUT
    assert parsed.sample == DEFAULT_SAMPLE
    assert parsed.threads == DEFAULT_THREADS
    assert parsed.leads is None
    assert parsed.cache is None
    assert DEFAULT_OUT == REPO_ROOT / "data" / "tape_studies" / "f4_inputs"
    assert DEFAULT_SAMPLE == DEFAULT_OUT / "sample.jsonl"


def test_the_parser_takes_a_repeated_lead() -> None:
    assert build_parser().parse_args(["--class", "c", "--lead", "36"]).leads == [36]
    assert build_parser().parse_args(["--class", "c", "--lead", "24", "--lead", "36"]).leads == [
        24,
        36,
    ]


def test_class_a_freezes_one_record_per_member_and_lead(
    sample: Path,
    tmp_path: Path,
    offline: Callable[[Callable[..., httpx.Response]], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = previous_runs_body(datetime(2025, 1, 15, tzinfo=UTC), 48)
    offline(lambda request: httpx.Response(200, content=body))

    assert main(["--class", "a", "--sample", str(sample), "--out", str(tmp_path / "out")]) == 0
    summary = json.loads(capsys.readouterr().out)

    records = read_class_freeze(Path(summary["out"]))
    assert summary["class"] == "a"
    assert summary["records"] == 6
    assert summary["per_member"] == {
        "ecmwf_ifs025": 2,
        "icon_global": 2,
        "ukmo_global_deterministic_10km": 2,
    }
    assert summary["per_basis"] == {LST_FULL: 3, LST_FULL_LESS_LAST_HOUR: 3}
    assert summary["refused"] == 0
    assert {row.daily_high_f for row in records if row.lead_hours == 24} == {Decimal("40.5")}
    assert {row.daily_high_f for row in records if row.lead_hours == 36} == {Decimal("41.5")}


def test_class_b_freezes_the_twelve_hour_max_and_its_sigma(
    sample: Path,
    tmp_path: Path,
    offline: Callable[[Callable[..., httpx.Response]], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = mos_body()
    offline(lambda request: httpx.Response(200, text=text))

    assert main(["--class", "b", "--sample", str(sample), "--out", str(tmp_path / "out")]) == 0
    summary = json.loads(capsys.readouterr().out)

    records = read_class_freeze(Path(summary["out"]))
    assert summary["records"] == 2
    assert summary["per_basis"] == {UTC_12Z_00Z: 2}
    assert summary["records_without_sigma"] == 0
    assert {row.daily_high_f for row in records} == {Decimal("38.0")}
    assert {row.native_sigma_f for row in records} == {Decimal("2.0")}
    assert {row.grid_latitude for row in records} == {None}
    assert {row.lead_hours: row.issue_time for row in records} == {
        24: datetime(2025, 1, 15, 1, tzinfo=UTC),
        36: datetime(2025, 1, 14, 13, tzinfo=UTC),
    }


def test_class_c_downloads_each_field_once_and_resumes_from_the_cache(
    sample: Path,
    tmp_path: Path,
    offline: Callable[[Callable[..., httpx.Response]], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[str] = []
    offline(bucket_handler(seen))
    cache = tmp_path / "fields.jsonl"
    flags = [
        "--class",
        "c",
        "--sample",
        str(sample),
        "--lead",
        "24",
        "--cache",
        str(cache),
        "--threads",
        "4",
    ]

    assert main([*flags, "--out", str(tmp_path / "first")]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main([*flags, "--out", str(tmp_path / "second")]) == 0
    second = json.loads(capsys.readouterr().out)

    records = read_class_freeze(Path(first["out"]))
    assert first["records"] == 1
    assert first["downloaded_fields"] == 24
    assert first["cache_hits"] == 0
    assert first["missing_fields"] == 0
    assert first["transferred_bytes"] == 24 * len(MESSAGE)
    assert second["downloaded_fields"] == 0
    assert second["cache_hits"] == 24
    assert second["sha256"] == first["sha256"]
    assert records[0].window_basis == LST_FULL
    assert records[0].grid_latitude == pytest.approx(40.788, abs=0.01)
    assert records[0].daily_high_f == Decimal("62.49")


def test_a_missing_field_costs_the_city_day_rather_than_the_run(
    sample: Path,
    tmp_path: Path,
    offline: Callable[[Callable[..., httpx.Response]], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "wrfsfcf10" in url:
            return httpx.Response(404)
        if url.endswith(".idx"):
            return httpx.Response(200, content=idx_body(int(url.split("wrfsfcf")[1].split(".")[0])))
        return httpx.Response(206, content=MESSAGE)

    offline(handler)

    assert (
        main(
            [
                "--class",
                "c",
                "--sample",
                str(sample),
                "--lead",
                "24",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)

    assert summary["records"] == 0
    assert summary["missing_fields"] == 1
    assert summary["missing"] == ["2025-01-15T00Z/f010"]
    assert summary["uncovered"] == {"hrrr": 1}


def test_the_event_date_filter_narrows_the_pull(sample: Path, tmp_path: Path) -> None:
    parsed = build_parser().parse_args(
        ["--class", "b", "--event-date", "2025-01-15", "--event-date", "2025-01-16"]
    )

    assert parsed.event_dates == [date(2025, 1, 15), date(2025, 1, 16)]
