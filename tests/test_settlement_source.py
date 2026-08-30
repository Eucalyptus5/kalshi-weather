from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from bot.lag import settlement_source
from bot.lag.r0_universe import freeze_digest
from bot.lag.settlement_source import (
    OBSERVATION_SOURCE,
    SeriesSettlementSource,
    SettlementProvenance,
    SettlementScopeShort,
    SettlementSourceUnreadable,
    boundary_split,
    check_settlement_scope,
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
DEN = "KXHIGHDEN"
NY = "KXHIGHNY"
SFO = "KXHIGHTSFO"
MIA = "KXLOWTMIA"
ROOTS = (DEN, NY, SFO, MIA)
TWENTY_ROOTS = tuple(f"KXHIGH{index:02d}" for index in range(20))
WINDOW = tuple(date(2026, 8, 2) + timedelta(days=offset) for offset in range(14))


def series_body(
    root: str,
    name: str = WEATHER_COMPANY,
    url: str = WEATHER_COMPANY_URL,
    last_updated_ts: str = MOVED_AT,
    markdown: str = BANNER,
) -> dict:
    return {
        "series": {
            "ticker": root,
            "category": "Climate and Weather",
            "fee_type": "quadratic",
            "fee_multiplier": 1,
            "last_updated_ts": last_updated_ts,
            "settlement_sources": [{"name": name, "url": url}],
            "product_metadata": {"important_info": {"markdown": markdown}},
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


def test_a_window_wholly_before_the_boundary_puts_nothing_on_the_far_side(
    moved: SettlementProvenance,
) -> None:
    split = boundary_split(moved, WINDOW[:12])

    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 0


def test_a_day_named_twice_lands_twice_on_its_side_of_the_boundary(tmp_path: Path) -> None:
    bodies = {DEN: series_body(DEN, last_updated_ts="2026-08-10T17:48:38Z")}
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
    check_settlement_scope(provenance, roots)


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


def test_the_earliest_stamp_across_disagreeing_roots_sets_the_boundary(tmp_path: Path) -> None:
    bodies = {
        DEN: series_body(DEN, last_updated_ts="2026-08-18T19:15:56.178852Z"),
        NY: series_body(NY, last_updated_ts="2026-08-14T17:48:38Z"),
    }
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    assert next(iter(provenance.series)) == DEN
    assert provenance.series[DEN].last_updated_ts > provenance.series[NY].last_updated_ts

    split = boundary_split(provenance, WINDOW)

    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_the_earliest_stamp_sets_the_boundary_when_its_root_also_sorts_first(
    tmp_path: Path,
) -> None:
    bodies = {
        DEN: series_body(DEN, last_updated_ts="2026-08-14T17:48:38Z"),
        NY: series_body(NY, last_updated_ts="2026-08-18T19:15:56.178852Z"),
    }
    provenance = frozen(tmp_path / "settlement_source.json", bodies)

    assert next(iter(provenance.series)) == DEN

    split = boundary_split(provenance, WINDOW)

    assert split.boundary_date == date(2026, 8, 14)
    assert split.days_before_boundary == 12
    assert split.days_on_or_after_boundary == 2


def test_the_boundary_takes_the_utc_date_of_the_stamp_not_the_local_one(
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


def test_the_banner_text_is_carried_and_never_read(moved: SettlementProvenance) -> None:
    source = Path(settlement_source.__file__).read_text()
    readers = ("strptime", "fromisoformat", "re.search", "re.match", "split", "August")

    assert "import re" not in source
    for line in source.splitlines():
        if "important_info" in line:
            assert not [reader for reader in readers if reader in line]


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
