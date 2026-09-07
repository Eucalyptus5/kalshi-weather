from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from bot.backtest.nbm import (
    IEM_MOS_URL,
    NBS_MODEL,
    MosRow,
    daily_high_row,
    fetch_mos_archive,
    parse_mos_csv,
)
from bot.lag.forecast_classes import CLASS_B_MEMBER


UTC = timezone.utc
RUN_FIXTURE = Path(__file__).parent / "data" / "nbs_knyc_20250128_01z.csv"
DUPLICATE_FIXTURE = Path(__file__).parent / "data" / "nbs_knyc_20251014_07z_duplicated.csv"
SPAN = (date(2024, 10, 20), date(2026, 1, 30))
RUN = datetime(2025, 1, 28, 1, tzinfo=UTC)
HIGH_FTIME = datetime(2025, 1, 29, tzinfo=UTC)
LOW_FTIME = datetime(2025, 1, 29, 12, tzinfo=UTC)


@pytest.fixture(scope="module")
def worked_run() -> tuple[MosRow, ...]:
    return parse_mos_csv(RUN_FIXTURE.read_text())


@pytest.fixture(scope="module")
def knyc_csv() -> str:
    response = httpx.get(
        IEM_MOS_URL,
        params={
            "station": "KNYC",
            "model": NBS_MODEL,
            "sts": f"{SPAN[0].isoformat()}T00:00Z",
            "ets": f"{SPAN[1].isoformat()}T00:00Z",
            "format": "csv",
        },
        timeout=300.0,
    )
    response.raise_for_status()
    return response.text


@pytest.fixture(scope="module")
def knyc_archive(knyc_csv: str) -> tuple[MosRow, ...]:
    return parse_mos_csv(knyc_csv)


def test_the_member_and_the_model_the_seam_publishes_are_pinned() -> None:
    assert CLASS_B_MEMBER == "nbm_nbs"
    assert NBS_MODEL == "NBS"


def test_the_worked_run_carries_a_twelve_hour_max_the_hourly_grid_misses(
    worked_run: tuple[MosRow, ...],
) -> None:
    high = next(row for row in worked_run if row.ftime == HIGH_FTIME)
    window = [
        row.tmp for row in worked_run if HIGH_FTIME - timedelta(hours=12) <= row.ftime <= HIGH_FTIME
    ]

    assert high.txn == Decimal("38.0")
    assert high.xnd == Decimal("2.0")
    assert high.tsd == Decimal("3")
    assert window == [Decimal(value) for value in (34, 36, 36, 34, 31)]
    assert max(window) == Decimal(36)
    assert high.txn - max(window) == Decimal("2.0")


def test_the_twelve_z_ftime_is_the_overnight_minimum(worked_run: tuple[MosRow, ...]) -> None:
    overnight = next(row for row in worked_run if row.ftime == LOW_FTIME)

    assert overnight.txn == Decimal("30.0")
    picked = daily_high_row(
        worked_run, event_date=date(2025, 1, 28), as_of=datetime(2025, 1, 28, 4, 59, tzinfo=UTC)
    )
    assert picked is not None
    assert picked.ftime == HIGH_FTIME
    assert picked.txn == Decimal("38.0")


def test_the_run_grid_is_three_hourly_out_to_a_seventy_one_hour_lead(
    worked_run: tuple[MosRow, ...],
) -> None:
    leads = sorted(int((row.ftime - RUN).total_seconds() // 3600) for row in worked_run)

    assert len(worked_run) == 23
    assert leads == list(range(5, 72, 3))
    assert {row.runtime for row in worked_run} == {RUN}


def test_daily_high_row_never_reads_a_run_issued_after_the_decision_instant(
    worked_run: tuple[MosRow, ...],
) -> None:
    assert (
        daily_high_row(
            worked_run,
            event_date=date(2025, 1, 28),
            as_of=RUN - timedelta(minutes=1),
        )
        is None
    )


def test_the_duplicated_run_collapses_to_one_row_per_ftime() -> None:
    text = DUPLICATE_FIXTURE.read_text()

    rows = parse_mos_csv(text)

    assert len(text.splitlines()) - 1 == 46
    assert len(rows) == 23
    assert len({row.ftime for row in rows}) == 23


async def test_fetch_asks_iem_for_one_station_over_the_whole_span() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=RUN_FIXTURE.read_text())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        archive = await fetch_mos_archive(
            station="KNYC", start_date=SPAN[0], end_date=SPAN[1], client=client
        )

    assert len(seen) == 1
    url = seen[0].url
    assert str(url).startswith(IEM_MOS_URL)
    assert url.params["station"] == "KNYC"
    assert url.params["model"] == "NBS"
    assert url.params["format"] == "csv"
    assert archive.station == "KNYC"
    assert archive.source_url == str(url)
    assert len(archive.rows) == 23


def test_the_live_csv_carries_four_runs_a_day_with_no_missing_runtime_day(knyc_csv: str) -> None:
    raw = [line.split(",") for line in knyc_csv.splitlines()[1:]]
    days = {line[0][:10] for line in raw}
    span = [
        (date(2024, 10, 20) + timedelta(days=offset)).isoformat()
        for offset in range((date(2026, 1, 29) - date(2024, 10, 20)).days + 1)
    ]

    assert len(raw) == 42964
    assert len(days) == 467
    assert (min(days), max(days)) == ("2024-10-20", "2026-01-29")
    assert len(span) == 467
    assert [day for day in span if day not in days] == []
    assert dict(sorted(Counter(line[0][11:13] for line in raw).items())) == {
        "01": 10741,
        "07": 10741,
        "13": 10741,
        "19": 10741,
    }


def test_the_live_archive_publishes_a_sigma_beside_every_twelve_hour_extreme(
    knyc_archive: tuple[MosRow, ...],
) -> None:
    extremes = [row for row in knyc_archive if row.txn is not None]

    assert len(knyc_archive) == 42941
    assert len(extremes) == 9335
    assert Counter(f"{row.ftime:%H}" for row in extremes) == {"00": 4667, "12": 4668}
    assert sum(1 for row in knyc_archive if row.tsd is None) == 0
    assert sum(1 for row in extremes if row.xnd is None) == 0


def test_the_archive_skips_one_run_and_duplicates_another(
    knyc_csv: str, knyc_archive: tuple[MosRow, ...]
) -> None:
    raw_per_day = Counter(line[:10] for line in knyc_csv.splitlines()[1:])
    runs_per_day = Counter(runtime.date() for runtime in {row.runtime for row in knyc_archive})

    assert raw_per_day["2025-06-20"] == 69
    assert {
        f"{row.runtime:%H}" for row in knyc_archive if row.runtime.date() == date(2025, 6, 20)
    } == {"01", "13", "19"}
    assert raw_per_day["2025-10-14"] == 115
    assert (
        sum(1 for row in knyc_archive if row.runtime == datetime(2025, 10, 14, 7, tzinfo=UTC)) == 23
    )
    assert Counter(runs_per_day.values()) == {4: 466, 3: 1}
