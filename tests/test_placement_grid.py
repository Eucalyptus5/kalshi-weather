from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from bot.lag.placement_grid import (
    PLACEMENT_STEP_S,
    PLACEMENTS,
    WINDOW_CLOSE_HOURS,
    WINDOW_OPEN_HOURS,
    CloseSidecar,
    close_of,
    grid_for,
    placement_grid,
    pull_closes,
    read_sidecar,
)
from bot.lag.r0_universe import R0Universe
from bot.lag.tape_studies import (
    EXCLUSION_CLASSES,
    EvidenceWindow,
    RunScope,
    merge_intervals,
    screen_windows,
)
from bot.markets.observation_window import observation_window
from bot.replay.run_scope import DISCOVERY, EventDay


_DATA = Path(__file__).parent / "data"
_DEN_CLOSES = _DATA / "kalshi_settled_den_closes.json"
_EMPTY = _DATA / "kalshi_settled_page2_empty.json"

SERIES = "KXHIGHDEN"
AUG13 = "KXHIGHDEN-26AUG13-T58"
AUG14 = "KXHIGHDEN-26AUG14-T58"
BRACKET = "KXHIGHDEN-26AUG13-B8889"
VOID = "KXHIGHDEN-26AUG14-T99"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _scripted_handler(pages: list[str]) -> Callable[[httpx.Request], httpx.Response]:
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = pages[len(seen) % len(pages)]
        seen.append(1)
        return httpx.Response(200, content=body)

    return handler


@pytest.fixture
def sidecar(tmp_path: Path) -> CloseSidecar:
    handler = _scripted_handler([_DEN_CLOSES.read_text()])
    pull_closes(
        [SERIES],
        min_ts=1754000000,
        max_ts=1756000000,
        directory=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    return read_sidecar(tmp_path / f"{SERIES}.json")


def _scope(event_date: date) -> RunScope:
    window_start, window_end = observation_window("America/Denver", event_date)
    day = EventDay(
        series=SERIES,
        station="KDEN",
        timezone="America/Denver",
        event_date=event_date,
        window_start=window_start,
        window_end=window_end,
        tickers=6,
        ladder_rows=900,
        first_event_at=window_start,
        last_event_at=window_end,
        covered=True,
        evaluable=True,
        in_scope=True,
        day_index=1,
        split=DISCOVERY,
        excluded_us=0,
        span_us=0,
    )
    return RunScope(
        exclusions=(),
        merged=merge_intervals(()),
        by_class={name: merge_intervals(()) for name in EXCLUSION_CLASSES},
        event_days={(SERIES, event_date): day},
        discovery_days=frozenset({event_date}),
        holdout_days=frozenset(),
        scope_start=window_start,
        scope_end=window_end,
        universe=R0Universe(
            fraction_invalid_max=Decimal("0.4"),
            passing=(SERIES,),
            lock_dependent=(),
            recorded=(SERIES,),
            ladder_widths=(6,),
            in_scope_city_days=14,
            reconciliation="agree",
            recorded_not_passing=(),
            passing_not_recorded=(),
        ),
    )


def test_the_grid_is_forty_eight_instants_inside_the_window() -> None:
    grid = placement_grid(_utc("2026-08-11T06:59:00Z"))

    assert PLACEMENTS == 48
    assert len(grid) == 48
    assert grid[0] == _utc("2026-08-10T07:00:00Z")
    assert grid[47] == _utc("2026-08-10T18:45:00Z")


def test_the_first_placement_is_the_event_days_own_window_start() -> None:
    window_start, _ = observation_window("America/Denver", date(2026, 8, 10))

    assert placement_grid(_utc("2026-08-11T06:59:00Z"))[0] == window_start


def test_a_fifty_nine_close_and_an_on_the_hour_close_share_one_grid() -> None:
    legacy = placement_grid(_utc("2026-08-11T06:59:00Z"))
    moved = placement_grid(_utc("2026-08-11T07:00:00Z"))

    assert legacy == moved
    assert moved[0] == _utc("2026-08-10T07:00:00Z")
    assert moved[47] == _utc("2026-08-10T18:45:00Z")


def test_adjacent_placements_are_nine_hundred_seconds_apart() -> None:
    grid = placement_grid(_utc("2026-08-11T06:59:00Z"))

    assert PLACEMENT_STEP_S == 900
    assert {(later - earlier).total_seconds() for earlier, later in zip(grid, grid[1:])} == {900}


def test_the_window_edges_are_the_preregistered_hours() -> None:
    assert WINDOW_OPEN_HOURS == 24
    assert WINDOW_CLOSE_HOURS == 12


def test_the_placements_sit_inside_the_window_edges() -> None:
    for iso in ("2026-08-11T06:59:00Z", "2026-08-11T07:00:00Z"):
        close = _utc(iso)
        grid = placement_grid(close)

        assert grid[0] >= close - timedelta(hours=WINDOW_OPEN_HOURS)
        assert grid[47] < close - timedelta(hours=WINDOW_CLOSE_HOURS)
        assert grid[0] - timedelta(seconds=PLACEMENT_STEP_S) < close - timedelta(
            hours=WINDOW_OPEN_HOURS
        )


def test_every_placement_survives_the_frozen_screen() -> None:
    scope = _scope(date(2026, 8, 10))
    windows = [
        EvidenceWindow(
            series=SERIES,
            event_date=date(2026, 8, 10),
            start=instant,
            end=instant + timedelta(seconds=600),
        )
        for instant in placement_grid(_utc("2026-08-11T06:59:00Z"))
    ]

    screened = screen_windows(scope, windows)

    assert screened.candidates == 48
    assert len(screened.kept) == 48
    assert screened.out_of_window == 0
    assert screened.out_of_scope == 0


def test_a_close_anchored_grid_loses_its_first_placement() -> None:
    scope = _scope(date(2026, 8, 10))
    anchor = _utc("2026-08-11T06:59:00Z") - timedelta(hours=24)
    windows = [
        EvidenceWindow(
            series=SERIES,
            event_date=date(2026, 8, 10),
            start=anchor + timedelta(seconds=900 * index),
            end=anchor + timedelta(seconds=900 * index + 600),
        )
        for index in range(48)
    ]

    screened = screen_windows(scope, windows)

    assert screened.out_of_window == 1
    assert len(screened.kept) == 47


def test_the_sidecar_stores_a_close_per_ticker_not_per_root(sidecar: CloseSidecar) -> None:
    assert close_of(sidecar, AUG13) == _utc("2026-08-14T06:59:00Z")
    assert close_of(sidecar, AUG14) == _utc("2026-08-15T07:00:00Z")


def test_the_stored_closes_carry_a_grid_apart_inside_one_root(sidecar: CloseSidecar) -> None:
    assert grid_for(sidecar, AUG13)[0] == _utc("2026-08-13T07:00:00Z")
    assert grid_for(sidecar, AUG14)[0] == _utc("2026-08-14T07:00:00Z")
    assert grid_for(sidecar, AUG13)[47] == _utc("2026-08-13T18:45:00Z")
    assert grid_for(sidecar, AUG14)[47] == _utc("2026-08-14T18:45:00Z")


def test_a_ticker_the_sidecar_does_not_name_is_refused(sidecar: CloseSidecar) -> None:
    with pytest.raises(ValueError, match="KXHIGHDEN-26AUG15-T58"):
        close_of(sidecar, "KXHIGHDEN-26AUG15-T58")

    with pytest.raises(ValueError, match="KXHIGHDEN-26AUG15-T58"):
        grid_for(sidecar, "KXHIGHDEN-26AUG15-T58")


def test_a_void_is_excluded_from_the_grid_and_counted(sidecar: CloseSidecar) -> None:
    assert sidecar.voided == (VOID,)
    assert VOID not in sidecar.markets

    with pytest.raises(ValueError, match=VOID):
        grid_for(sidecar, VOID)


def test_the_venue_reports_finalized_and_a_settled_screen_drops_every_row(
    sidecar: CloseSidecar,
) -> None:
    assert [market.status for market in sidecar.markets.values()] == ["finalized"] * 3
    assert [
        ticker for ticker, market in sidecar.markets.items() if market.status == "settled"
    ] == []


def test_the_sidecar_reproduces_its_digest_on_a_second_read(tmp_path: Path) -> None:
    handler = _scripted_handler([_DEN_CLOSES.read_text()])
    written = pull_closes(
        [SERIES],
        min_ts=1754000000,
        max_ts=1756000000,
        directory=tmp_path,
        transport=httpx.MockTransport(handler),
    )

    first = read_sidecar(tmp_path / f"{SERIES}.json")
    second = read_sidecar(tmp_path / f"{SERIES}.json")

    assert first.sha256 == second.sha256
    assert written[SERIES] == first.sha256


def test_a_tampered_sidecar_is_refused(tmp_path: Path) -> None:
    handler = _scripted_handler([_DEN_CLOSES.read_text()])
    pull_closes(
        [SERIES],
        min_ts=1754000000,
        max_ts=1756000000,
        directory=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    path = tmp_path / f"{SERIES}.json"
    path.write_text(path.read_text().replace("2026-08-15T07:00:00", "2026-08-15T08:00:00"))

    with pytest.raises(ValueError, match="sha256"):
        read_sidecar(path)


def test_the_sidecar_carries_the_bracket_fields(sidecar: CloseSidecar) -> None:
    bracket = sidecar.markets[BRACKET]

    assert bracket.strike_type == "between"
    assert bracket.floor_strike == 88
    assert bracket.cap_strike == 89
    assert sidecar.markets[AUG13].cap_strike is None


def test_the_pull_writes_one_sidecar_per_root(tmp_path: Path) -> None:
    handler = _scripted_handler([_DEN_CLOSES.read_text(), _EMPTY.read_text()])

    written = pull_closes(
        [SERIES, "KXHIGHNY"],
        min_ts=1754000000,
        max_ts=1756000000,
        directory=tmp_path,
        transport=httpx.MockTransport(handler),
    )

    assert sorted(written) == ["KXHIGHDEN", "KXHIGHNY"]
    assert (tmp_path / "KXHIGHDEN.json").exists()
    assert (tmp_path / "KXHIGHNY.json").exists()
    assert read_sidecar(tmp_path / "KXHIGHNY.json").markets == {}


def test_the_pull_refuses_to_overwrite_a_frozen_sidecar(tmp_path: Path) -> None:
    handler = _scripted_handler([_DEN_CLOSES.read_text()])
    kwargs = {
        "min_ts": 1754000000,
        "max_ts": 1756000000,
        "directory": tmp_path,
        "transport": httpx.MockTransport(handler),
    }
    pull_closes([SERIES], **kwargs)

    with pytest.raises(FileExistsError):
        pull_closes([SERIES], **kwargs)


def test_the_stored_close_is_utc_aware(sidecar: CloseSidecar) -> None:
    assert close_of(sidecar, AUG13).tzinfo == timezone.utc
