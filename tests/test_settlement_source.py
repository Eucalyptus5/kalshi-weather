from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from bot.lag import settlement_source
from bot.lag.r0_universe import freeze_digest
from bot.lag.settlement_source import (
    BOUNDARY_SOURCE,
    OBSERVATION_SOURCE,
    BoundaryOutsideWindow,
    SeriesSettlementSource,
    SettlementNoticeMisdated,
    SettlementNoticeUnreadable,
    SettlementProvenance,
    SettlementScopeShort,
    SettlementSourceUnreadable,
    boundary_split,
    check_settlement_scope,
    distinct_notice_bodies,
    pull_settlement_sources,
    read_settlement_sources,
    settlement_payload,
    source_root_counts,
)


UTC = timezone.utc
OBSERVED_AT = datetime(2026, 8, 19, 17, 30, tzinfo=UTC)
MOVED_AT = "2026-08-14T17:48:38Z"
WEATHER_COMPANY = "The Weather Company"
WEATHER_COMPANY_URL = "https://weather.com/kalshi"
NWS = "National Weather Service"
NWS_URL = "https://www.weather.gov"
BANNER = (
    "**Important information:** \n\nEffective Friday, August 14th, daily temperature markets "
    "will transition their settlement source from the National Weather Service (NWS) to "
    "The Weather Company."
)
LIVE_BANNER = (
    "**Important information:** \n\nEffective Friday, August 14th, daily temperature markets "
    "will transition their settlement source from the National Weather Service (NWS) to "
    "The Weather Company. The Weather Company utilizes NWS as its primary underlying source, "
    "and official settlement data will be accessible at https://weather.com/kalshi."
)
LIVE_BANNER_SHA256 = "8479086f1ed3267b42acdd6a9aa3a14bd9d7e271573be96a5c433d1f43e2c865"
BULK_INFO_ID = "GLOBALTEMPERATURE-bulk-2026-08-11"
LAX_INFO_ID = "KXHIGHLAX-2026-08-11"
RATCHETED_AT = "2026-08-20T21:30:00Z"
DEN = "KXHIGHDEN"
NY = "KXHIGHNY"
SFO = "KXHIGHTSFO"
MIA = "KXLOWTMIA"
LOWT_DEN = "KXLOWTDEN"
LAX = "KXHIGHLAX"
ROOTS = (DEN, NY, SFO, MIA)
LIVE_ROOTS = (DEN, LOWT_DEN, LAX)
LIVE_STAMPS = {
    DEN: "2026-08-20T21:30:00.537864Z",
    LOWT_DEN: "2026-08-20T21:30:00.599192Z",
}
TWENTY_ROOTS = tuple(f"KXHIGH{index:02d}" for index in range(20))
WINDOW = tuple(date(2026, 8, 2) + timedelta(days=offset) for offset in range(14))
LIVE_ONLY = pytest.mark.skipif(
    os.environ.get("KW_LIVE_SERIES") != "1",
    reason="set KW_LIVE_SERIES=1 to read the live series endpoint",
)


def banner_dated(named: str) -> str:
    return BANNER.replace("Friday, August 14th", named)


def series_body(
    root: str,
    name: str = WEATHER_COMPANY,
    url: str = WEATHER_COMPANY_URL,
    last_updated_ts: str = MOVED_AT,
    markdown: str = BANNER,
    info_id: str = BULK_INFO_ID,
) -> dict:
    return {
        "series": {
            "ticker": root,
            "category": "Climate and Weather",
            "fee_type": "quadratic",
            "fee_multiplier": 1,
            "last_updated_ts": last_updated_ts,
            "settlement_sources": [{"name": name, "url": url}],
            "product_metadata": {"important_info": {"id": info_id, "markdown": markdown}},
        }
    }


def transport_for(bodies: Mapping[str, dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        root = request.url.path.rsplit("/", 1)[-1]
        if root not in bodies:
            return httpx.Response(404, json={"error": {"code": "not_found"}})
        return httpx.Response(200, json=bodies[root])

    return httpx.MockTransport(handler)


def moved_bodies(roots: Sequence[str] = ROOTS) -> dict[str, dict]:
    return {root: series_body(root) for root in roots}


def frozen(path: Path, bodies: Mapping[str, dict]) -> SettlementProvenance:
    pull_settlement_sources(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))
    return read_settlement_sources(path)


@pytest.fixture
def moved(tmp_path: Path) -> SettlementProvenance:
    return frozen(tmp_path / "settlement_source.json", moved_bodies())


@pytest.fixture
def host_zone_behind_utc(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_a_sweep_freezes_the_source_every_root_names(moved: SettlementProvenance) -> None:
    assert sorted(moved.series) == sorted(ROOTS)
    assert {item.settlement_source for item in moved.series.values()} == {WEATHER_COMPANY}
    assert {item.settlement_source_url for item in moved.series.values()} == {WEATHER_COMPANY_URL}
    assert moved.observed_at == OBSERVED_AT


def test_the_sidecar_reproduces_its_digest_on_a_second_read(tmp_path: Path) -> None:
    path = tmp_path / "settlement_source.json"
    bodies = moved_bodies()

    written = pull_settlement_sources(
        sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies)
    )

    assert read_settlement_sources(path).sha256 == written
    assert read_settlement_sources(path).sha256 == read_settlement_sources(path).sha256


def test_the_order_the_roots_were_asked_for_does_not_move_the_digest(tmp_path: Path) -> None:
    bodies = moved_bodies()
    forward = tmp_path / "forward.json"
    reverse = tmp_path / "reverse.json"

    ahead = pull_settlement_sources(list(ROOTS), OBSERVED_AT, forward, transport_for(bodies))
    behind = pull_settlement_sources(
        list(reversed(ROOTS)), OBSERVED_AT, reverse, transport_for(bodies)
    )

    assert ahead == behind
    assert forward.read_text() == reverse.read_text()


def test_the_sweep_refuses_to_overwrite_a_frozen_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "settlement_source.json"
    bodies = moved_bodies()
    pull_settlement_sources(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))

    with pytest.raises(FileExistsError):
        pull_settlement_sources(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))


def test_a_sidecar_carrying_no_sha256_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "settlement_source.json"
    bodies = moved_bodies()
    pull_settlement_sources(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))
    payload = json.loads(path.read_text())
    payload.pop("sha256")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="sha256"):
        read_settlement_sources(path)


def test_a_sidecar_whose_digest_was_edited_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "settlement_source.json"
    bodies = moved_bodies()
    digest = pull_settlement_sources(
        sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies)
    )
    path.write_text(path.read_text().replace(digest, "0" * 64))

    with pytest.raises(ValueError, match="sha256"):
        read_settlement_sources(path)


def test_a_sidecar_whose_source_was_edited_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "settlement_source.json"
    bodies = moved_bodies()
    pull_settlement_sources(sorted(bodies), OBSERVED_AT, path, transport=transport_for(bodies))
    path.write_text(path.read_text().replace(WEATHER_COMPANY, NWS))

    with pytest.raises(ValueError, match="sha256"):
        read_settlement_sources(path)


def test_the_window_carries_twelve_days_before_the_boundary_and_two_on_or_after(
    moved: SettlementProvenance,
) -> None:
    split = boundary_split(moved, WINDOW)

    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_the_two_sides_of_the_boundary_are_separate_counts_not_a_ratio(
    moved: SettlementProvenance,
) -> None:
    split = boundary_split(moved, WINDOW)

    assert isinstance(split.days_before_boundary, int)
    assert isinstance(split.days_on_or_after_boundary, int)
    assert split.days_before_boundary + split.days_on_or_after_boundary == len(WINDOW)
    assert isinstance(split.boundary_date, date)


def test_a_window_wholly_before_the_boundary_is_refused_not_reported_clean(
    moved: SettlementProvenance,
) -> None:
    with pytest.raises(BoundaryOutsideWindow) as excinfo:
        boundary_split(moved, WINDOW[:12])

    assert excinfo.value.boundary == date(2026, 8, 14)
    assert (excinfo.value.first, excinfo.value.last) == (WINDOW[0], WINDOW[11])


def test_a_day_named_twice_lands_twice_on_its_side_of_the_boundary(tmp_path: Path) -> None:
    bodies = {DEN: series_body(DEN, markdown=banner_dated("Monday, August 10th"))}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)
    days = [date(2026, 8, 5), date(2026, 8, 12), date(2026, 8, 12), date(2026, 8, 15)]

    split = boundary_split(provenance, days)

    assert len(set(days)) == 3
    assert split.boundary_date == date(2026, 8, 10)
    assert split.days_before_boundary == 1
    assert split.days_on_or_after_boundary == 3


def test_twenty_roots_freeze_twenty_rows(tmp_path: Path) -> None:
    provenance = frozen(tmp_path / "settlement_source.json", moved_bodies(TWENTY_ROOTS))

    assert len(provenance.series) == 20
    assert len(json.loads((tmp_path / "settlement_source.json").read_text())["series"]) == 20
    check_settlement_scope(provenance, TWENTY_ROOTS)


def test_a_scope_naming_a_root_the_sidecar_lacks_aborts(moved: SettlementProvenance) -> None:
    with pytest.raises(SettlementScopeShort) as excinfo:
        check_settlement_scope(moved, (*ROOTS, "KXHIGHCHI"))

    assert "KXHIGHCHI" in str(excinfo.value)
    assert excinfo.value.offenders == ("KXHIGHCHI is not named in the frozen sidecar",)


def test_a_scope_missing_two_roots_names_both(moved: SettlementProvenance) -> None:
    with pytest.raises(SettlementScopeShort) as excinfo:
        check_settlement_scope(moved, (*ROOTS, "KXHIGHCHI", "KXHIGHAUS"))

    message = str(excinfo.value)
    assert "KXHIGHCHI" in message
    assert "KXHIGHAUS" in message
    assert len(excinfo.value.offenders) == 2


def test_one_root_left_on_the_old_source_is_recorded_and_counted(tmp_path: Path) -> None:
    roots = TWENTY_ROOTS
    bodies = moved_bodies(roots) | {roots[7]: series_body(roots[7], name=NWS, url=NWS_URL)}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    assert source_root_counts(provenance) == {WEATHER_COMPANY: 19, NWS: 1}
    assert provenance.series[roots[7]].settlement_source == NWS
    assert provenance.series[roots[7]].settlement_source_url == NWS_URL
    assert boundary_split(provenance, WINDOW).boundary_date == date(2026, 8, 14)
    check_settlement_scope(provenance, roots)


def test_one_root_carrying_a_different_notice_body_is_recorded_and_counted(
    tmp_path: Path,
) -> None:
    roots = TWENTY_ROOTS
    bodies = moved_bodies(roots) | {roots[7]: series_body(roots[7], markdown=LIVE_BANNER)}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    split = boundary_split(provenance, WINDOW)

    assert distinct_notice_bodies(provenance) == 2
    assert provenance.series[roots[7]].important_info == LIVE_BANNER
    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_two_roots_whose_notice_ids_differ_carry_one_body_and_one_boundary(
    tmp_path: Path,
) -> None:
    bodies = {
        DEN: series_body(DEN, markdown=LIVE_BANNER, info_id=BULK_INFO_ID),
        LAX: series_body(LAX, markdown=LIVE_BANNER, info_id=LAX_INFO_ID),
    }
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    split = boundary_split(provenance, WINDOW)

    ids = {body["series"]["product_metadata"]["important_info"]["id"] for body in bodies.values()}
    assert ids == {BULK_INFO_ID, LAX_INFO_ID}
    assert distinct_notice_bodies(provenance) == 1
    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_the_stamp_lands_as_a_tz_aware_utc_datetime(moved: SettlementProvenance) -> None:
    stamp = moved.series[DEN].last_updated_ts

    assert stamp.tzinfo is not None
    assert stamp.utcoffset() == timedelta(0)
    assert stamp == datetime(2026, 8, 14, 17, 48, 38, tzinfo=UTC)


def test_a_stamp_carrying_fractional_seconds_parses_without_loss(tmp_path: Path) -> None:
    bodies = {DEN: series_body(DEN, last_updated_ts="2026-08-18T19:15:56.178852Z")}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)
    stamp = provenance.series[DEN].last_updated_ts

    assert stamp == datetime(2026, 8, 18, 19, 15, 56, 178852, tzinfo=UTC)
    assert stamp.microsecond == 178852
    assert stamp.utcoffset() == timedelta(0)


def test_the_earliest_notice_across_disagreeing_roots_sets_the_boundary(tmp_path: Path) -> None:
    bodies = {
        DEN: series_body(DEN, markdown=banner_dated("Thursday, August 20th")),
        NY: series_body(NY),
    }
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    assert next(iter(provenance.series)) == DEN
    assert distinct_notice_bodies(provenance) == 2

    split = boundary_split(provenance, WINDOW)

    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_the_earliest_notice_sets_the_boundary_when_its_root_also_sorts_first(
    tmp_path: Path,
) -> None:
    bodies = {
        DEN: series_body(DEN),
        NY: series_body(NY, markdown=banner_dated("Thursday, August 20th")),
    }
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    assert next(iter(provenance.series)) == DEN

    split = boundary_split(provenance, WINDOW)

    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_the_boundary_is_the_notice_date_whatever_zone_the_host_keeps(
    tmp_path: Path, host_zone_behind_utc: None
) -> None:
    bodies = {DEN: series_body(DEN, last_updated_ts="2026-08-14T02:00:00Z")}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    assert provenance.series[DEN].last_updated_ts.astimezone().date() == date(2026, 8, 13)

    split = boundary_split(provenance, WINDOW)

    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_a_series_carrying_no_stamp_is_refused(tmp_path: Path) -> None:
    body = series_body(DEN)
    body["series"].pop("last_updated_ts")

    with pytest.raises(SettlementSourceUnreadable) as excinfo:
        pull_settlement_sources(
            [DEN], OBSERVED_AT, tmp_path / "s.json", transport=transport_for({DEN: body})
        )

    assert excinfo.value.root == DEN
    assert excinfo.value.key == "last_updated_ts"
    assert DEN in str(excinfo.value)
    assert "last_updated_ts" in str(excinfo.value)
    assert not (tmp_path / "s.json").exists()


@pytest.mark.parametrize("sources", (None, []))
def test_a_series_naming_no_settlement_source_is_refused(
    tmp_path: Path, sources: list | None
) -> None:
    body = series_body(DEN)
    body["series"]["settlement_sources"] = sources

    with pytest.raises(SettlementSourceUnreadable) as excinfo:
        pull_settlement_sources(
            [DEN], OBSERVED_AT, tmp_path / "s.json", transport=transport_for({DEN: body})
        )

    assert excinfo.value.root == DEN
    assert excinfo.value.key == "settlement_sources"
    assert DEN in str(excinfo.value)


def test_a_series_omitting_the_settlement_source_key_is_refused(tmp_path: Path) -> None:
    body = series_body(DEN)
    body["series"].pop("settlement_sources")

    with pytest.raises(SettlementSourceUnreadable, match=DEN):
        pull_settlement_sources(
            [DEN], OBSERVED_AT, tmp_path / "s.json", transport=transport_for({DEN: body})
        )


def test_the_first_of_several_named_sources_is_the_one_recorded(tmp_path: Path) -> None:
    body = series_body(DEN)
    body["series"]["settlement_sources"] = [
        {"name": WEATHER_COMPANY, "url": WEATHER_COMPANY_URL},
        {"name": NWS, "url": NWS_URL},
    ]
    provenance = frozen(tmp_path / "settlement_source.json", {DEN: body})

    assert provenance.series[DEN].settlement_source == WEATHER_COMPANY
    assert provenance.series[DEN].settlement_source_url == WEATHER_COMPANY_URL
    assert source_root_counts(provenance) == {WEATHER_COMPANY: 1}


def test_the_banner_text_survives_the_freeze_verbatim(moved: SettlementProvenance) -> None:
    assert moved.series[DEN].important_info == BANNER
    assert moved.series[NY].important_info == BANNER


def test_a_series_carrying_no_banner_freezes_an_empty_string(tmp_path: Path) -> None:
    body = series_body(DEN)
    body["series"].pop("product_metadata")
    provenance = frozen(tmp_path / "settlement_source.json", {DEN: body})

    assert provenance.series[DEN].important_info == ""


def test_the_boundary_is_read_from_the_notice_and_never_from_the_stamp() -> None:
    body = inspect.getsource(settlement_source.boundary_split)

    assert BOUNDARY_SOURCE == "product_metadata.important_info.markdown"
    assert "important_info" in body
    assert "last_updated_ts" not in body


def test_the_split_names_the_field_the_boundary_was_read_from(
    moved: SettlementProvenance,
) -> None:
    assert boundary_split(moved, WINDOW).boundary_source == BOUNDARY_SOURCE


def test_the_byte_exact_live_notice_derives_the_same_three_literals(tmp_path: Path) -> None:
    bodies = {
        root: series_body(root, last_updated_ts=stamp, markdown=LIVE_BANNER)
        for root, stamp in LIVE_STAMPS.items()
    }
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    split = boundary_split(provenance, WINDOW)

    assert hashlib.sha256(LIVE_BANNER.encode()).hexdigest() == LIVE_BANNER_SHA256
    assert LIVE_BANNER != BANNER
    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


@LIVE_ONLY
def test_a_sidecar_pulled_from_the_live_endpoint_derives_the_same_three_literals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settlement_source.json"
    pull_settlement_sources(LIVE_ROOTS, OBSERVED_AT, path)
    provenance = read_settlement_sources(path)

    split = boundary_split(provenance, WINDOW)

    assert sorted(provenance.series) == sorted(LIVE_ROOTS)
    assert distinct_notice_bodies(provenance) == 1
    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_a_stamp_six_days_past_the_change_still_derives_the_notice_date(tmp_path: Path) -> None:
    bodies = {root: series_body(root, last_updated_ts=RATCHETED_AT) for root in ROOTS}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    split = boundary_split(provenance, WINDOW)

    assert {item.last_updated_ts for item in provenance.series.values()} == {
        datetime(2026, 8, 20, 21, 30, tzinfo=UTC)
    }
    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_the_stamp_the_old_derivation_read_lands_past_the_window(tmp_path: Path) -> None:
    bodies = {root: series_body(root, last_updated_ts=RATCHETED_AT) for root in ROOTS}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    ratcheted = min(item.last_updated_ts for item in provenance.series.values()).date()
    before = sum(1 for day in WINDOW if day < ratcheted)

    assert ratcheted == date(2026, 8, 20)
    assert before == 14
    assert len(WINDOW) - before == 0
    assert boundary_split(provenance, WINDOW).boundary_date == date(2026, 8, 14)


@pytest.mark.parametrize(
    ("named", "boundary"),
    (("Thursday, August 20th", "2026-08-20"), ("Sunday, August 2nd", "2026-08-02")),
)
def test_a_boundary_outside_the_window_is_refused_not_reported_clean(
    tmp_path: Path, named: str, boundary: str
) -> None:
    bodies = {DEN: series_body(DEN, markdown=banner_dated(named))}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    with pytest.raises(BoundaryOutsideWindow) as excinfo:
        boundary_split(provenance, WINDOW)

    message = str(excinfo.value)
    assert boundary in message
    assert "2026-08-02" in message
    assert "2026-08-15" in message
    assert excinfo.value.boundary == date.fromisoformat(boundary)
    assert (excinfo.value.first, excinfo.value.last) == (WINDOW[0], WINDOW[-1])


@pytest.mark.parametrize(
    ("named", "boundary", "before", "after"),
    (
        ("Monday, August 3rd", date(2026, 8, 3), 1, 13),
        ("Saturday, August 15th", date(2026, 8, 15), 13, 1),
    ),
)
def test_a_boundary_on_either_admissible_extreme_still_splits_the_window(
    tmp_path: Path, named: str, boundary: date, before: int, after: int
) -> None:
    bodies = {DEN: series_body(DEN, markdown=banner_dated(named))}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    split = boundary_split(provenance, WINDOW)

    assert split.boundary_date == boundary
    assert split.days_before_boundary == before
    assert split.days_on_or_after_boundary == after


def test_a_notice_whose_weekday_misses_its_date_in_the_sidecars_year_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settlement_source.json"
    bodies = {DEN: series_body(DEN, markdown=LIVE_BANNER)}
    pull_settlement_sources(
        [DEN], datetime(2025, 8, 19, 17, 30, tzinfo=UTC), path, transport_for(bodies)
    )
    provenance = read_settlement_sources(path)

    with pytest.raises(SettlementNoticeMisdated) as excinfo:
        boundary_split(provenance, WINDOW)

    assert excinfo.value.root == DEN
    assert excinfo.value.named == "Friday"
    assert excinfo.value.moment == date(2025, 8, 14)
    assert "Thursday" in str(excinfo.value)


def test_a_sidecar_whose_notices_name_no_effective_date_derives_no_boundary(
    tmp_path: Path,
) -> None:
    bodies = {
        DEN: series_body(DEN, markdown=""),
        NY: series_body(NY, markdown="settlement moves to The Weather Company"),
    }
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    with pytest.raises(SettlementNoticeUnreadable) as excinfo:
        boundary_split(provenance, WINDOW)

    assert excinfo.value.roots == (DEN, NY)
    assert DEN in str(excinfo.value)
    assert NY in str(excinfo.value)


def test_a_root_carrying_no_notice_contributes_nothing_to_the_boundary(tmp_path: Path) -> None:
    bodies = {DEN: series_body(DEN, markdown=""), NY: series_body(NY)}
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    split = boundary_split(provenance, WINDOW)

    assert distinct_notice_bodies(provenance) == 2
    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_the_observation_side_comparison_value_is_frozen_alongside(
    moved: SettlementProvenance,
) -> None:
    assert OBSERVATION_SOURCE == "ACIS"
    assert moved.observation_source == "ACIS"
    assert moved.observation_source not in (WEATHER_COMPANY, NWS)


def test_a_stamp_carrying_no_zone_is_refused(tmp_path: Path) -> None:
    bodies = {DEN: series_body(DEN, last_updated_ts="2026-08-14T17:48:38")}

    with pytest.raises(SettlementSourceUnreadable) as excinfo:
        pull_settlement_sources(
            [DEN], OBSERVED_AT, tmp_path / "s.json", transport=transport_for(bodies)
        )

    assert excinfo.value.key == "last_updated_ts"


def test_a_series_the_venue_refuses_is_never_frozen(tmp_path: Path) -> None:
    path = tmp_path / "settlement_source.json"

    with pytest.raises(httpx.HTTPStatusError):
        pull_settlement_sources(
            [DEN, "KXHIGHNOWHERE"], OBSERVED_AT, path, transport_for(moved_bodies())
        )

    assert not path.exists()


def test_a_stored_stamp_carrying_an_offset_zone_is_read_back_in_utc(tmp_path: Path) -> None:
    row = SeriesSettlementSource(
        root=DEN,
        settlement_source=WEATHER_COMPANY,
        settlement_source_url=WEATHER_COMPANY_URL,
        last_updated_ts=datetime(2026, 8, 13, 19, 0, tzinfo=timezone(timedelta(hours=-7))),
        important_info=BANNER,
    )
    payload = settlement_payload(OBSERVED_AT, [row])
    path = tmp_path / "settlement_source.json"
    path.write_text(json.dumps({**payload, "sha256": freeze_digest(payload)}, indent=1))
    assert payload["series"][0]["last_updated_ts"] == "2026-08-13T19:00:00-07:00"

    stamp = read_settlement_sources(path).series[DEN].last_updated_ts

    assert stamp.utcoffset() == timedelta(0)
    assert stamp.tzinfo is UTC
    assert stamp == datetime(2026, 8, 14, 2, 0, tzinfo=UTC)
    assert stamp.date() == date(2026, 8, 14)
