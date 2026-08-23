from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.placement_grid import close_of, read_sidecar
from bot.lag.r0_universe import Coverage, freeze_universe, write_universe
from bot.replay.analysis_stations import HIGH, LOW
from bot.replay.run_scope import (
    DISCOVERY,
    EVENT_DAYS_SCHEMA,
    EXCLUSIONS_SCHEMA,
    HOLDOUT,
    Split,
    write_split,
)
from scripts.pull_closes import REPO_ROOT, build_parser, main
from tests.test_tape_studies import argument_flags, event_day_row


SCRIPT = REPO_ROOT / "scripts" / "pull_closes.py"
UTC = timezone.utc

DEN = "KXHIGHDEN"
NY = "KXHIGHNY"
LOW_SERIES = "KXLOWTDEN"

DISCOVERY_DAY = date(2026, 8, 13)
HOLDOUT_DAY = date(2026, 8, 14)
OPENS = {
    DEN: timedelta(hours=7),
    NY: timedelta(hours=5),
    LOW_SERIES: timedelta(hours=7),
}

SCOPE_START = datetime(2026, 8, 13, 5, tzinfo=UTC)
SCOPE_END = datetime(2026, 8, 15, 7, tzinfo=UTC)

DEN_AUG13 = "KXHIGHDEN-26AUG13-T58"
DEN_AUG13_BRACKET = "KXHIGHDEN-26AUG13-B8889"
DEN_AUG14 = "KXHIGHDEN-26AUG14-T58"
DEN_VOID = "KXHIGHDEN-26AUG14-T99"
NY_AUG13 = "KXHIGHNY-26AUG13-T88"

LEGACY_CLOSE = "2026-08-14T06:59:00Z"
MOVED_CLOSE = "2026-08-15T07:00:00Z"
NY_CLOSE = "2026-08-14T04:59:00Z"


def market_row(ticker: str, close: str, *, result: str = "yes") -> dict:
    return {
        "ticker": ticker,
        "event_ticker": ticker.rsplit("-", 1)[0],
        "close_time": close,
        "status": "finalized",
        "result": result,
        "floor_strike": 58,
        "cap_strike": None,
        "strike_type": "greater",
    }


def page(markets: Sequence[dict], cursor: str = "") -> dict:
    return {"markets": list(markets), "cursor": cursor}


def den_pages() -> list[dict]:
    return [
        page(
            [
                market_row(DEN_AUG13, LEGACY_CLOSE),
                market_row(DEN_AUG14, MOVED_CLOSE),
                market_row(DEN_VOID, MOVED_CLOSE, result=""),
            ]
        )
    ]


def ny_pages() -> list[dict]:
    return [page([market_row(NY_AUG13, NY_CLOSE)])]


def both_roots() -> dict[str, list[dict]]:
    return {DEN: den_pages(), NY: ny_pages()}


def handler_for(pages: Mapping[str, Sequence[dict]]) -> Callable[[httpx.Request], httpx.Response]:
    served: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        series = request.url.params["series_ticker"]
        index = served.get(series, 0)
        served[series] = index + 1
        return httpx.Response(200, json=pages[series][index])

    return handler


Offline = Callable[[Mapping[str, Sequence[dict]]], list[httpx.Request]]


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Offline:
    client = httpx.AsyncClient

    def install(pages: Mapping[str, Sequence[dict]]) -> list[httpx.Request]:
        seen: list[httpx.Request] = []
        served = handler_for(pages)

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return served(request)

        def factory(**kwargs: object) -> httpx.AsyncClient:
            return client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr("bot.lag.placement_grid.httpx.AsyncClient", factory)
        return seen

    return install


def scope_dir(tmp_path: Path, *, series: Sequence[str] = (DEN, NY), name: str = "scope") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    pq.write_table(
        pa.Table.from_pylist([], schema=EXCLUSIONS_SCHEMA), directory / "exclusions.parquet"
    )
    rows = [
        event_day_row(
            event_date,
            in_scope=True,
            split=split,
            day_index=index,
            opens=OPENS[root],
            series=root,
        )
        for root in series
        for index, (event_date, split) in enumerate(
            ((DISCOVERY_DAY, DISCOVERY), (HOLDOUT_DAY, HOLDOUT)), start=1
        )
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=EVENT_DAYS_SCHEMA), directory / "event_days.parquet"
    )
    write_split(
        directory / "split.json",
        Split(
            cities=tuple(series),
            discovery_days=(DISCOVERY_DAY,),
            holdout_days=(HOLDOUT_DAY,),
            boundary_event_day=HOLDOUT_DAY,
            scope_start=SCOPE_START,
            scope_end=SCOPE_END,
        ),
    )
    write_universe(
        directory / "r0_universe.json",
        freeze_universe(
            fraction_invalid_max=Decimal("0.4"),
            passing=tuple(series),
            coverage=Coverage(
                cities=tuple(series), ladder_widths=(6,), in_scope_city_days=2 * len(series)
            ),
        ),
    )
    return directory


def argv_parts(scope: Path, out: Path, cohort: str | None = HIGH) -> dict[str, list[str]]:
    parts = {"--run-scope": [str(scope)], "--out": [str(out)]}
    if cohort is not None:
        parts["--cohort"] = [cohort]
    return parts


def argv_for(scope: Path, out: Path, cohort: str | None = HIGH) -> list[str]:
    return [
        item for flag, values in argv_parts(scope, out, cohort).items() for item in (flag, *values)
    ]


def without(scope: Path, out: Path, flag: str) -> list[str]:
    parts = argv_parts(scope, out)
    del parts[flag]
    return [item for name, values in parts.items() for item in (name, *values)]


def pulled(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
    *,
    pages: Mapping[str, Sequence[dict]] | None = None,
) -> dict:
    offline(both_roots() if pages is None else pages)
    assert main(argv_for(scope_dir(tmp_path), tmp_path / "closes")) == 0
    return json.loads(capsys.readouterr().out)


def test_the_pull_writes_one_sidecar_per_root_and_prints_their_digests(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    printed = pulled(tmp_path, offline, capsys)

    assert sorted(printed["roots"]) == [DEN, NY]
    for root in (DEN, NY):
        sidecar = read_sidecar(tmp_path / "closes" / f"{root}.json")
        assert printed["roots"][root]["sha256"] == sidecar.sha256


def test_the_bracket_is_the_frozen_scopes_own_bounds(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    printed = pulled(tmp_path, offline, capsys)

    assert printed["min_ts"] == 1786597200
    assert printed["max_ts"] == 1786777200
    assert printed["min_ts_iso"] == "2026-08-13T05:00:00+00:00"
    assert printed["max_ts_iso"] == "2026-08-15T07:00:00+00:00"


def test_the_bracket_is_the_query_the_venue_is_asked(tmp_path: Path, offline: Offline) -> None:
    seen = offline(both_roots())

    assert main(argv_for(scope_dir(tmp_path), tmp_path / "closes")) == 0

    assert [request.url.params["min_close_ts"] for request in seen] == ["1786597200"] * 2
    assert [request.url.params["max_close_ts"] for request in seen] == ["1786777200"] * 2


def test_a_finalized_row_survives_into_the_sidecar(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pulled(tmp_path, offline, capsys)

    sidecar = read_sidecar(tmp_path / "closes" / f"{DEN}.json")

    assert DEN_AUG13 in sidecar.markets
    assert sidecar.markets[DEN_AUG13].status == "finalized"


def test_two_tickers_of_one_root_carry_their_own_closes(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pulled(tmp_path, offline, capsys)

    sidecar = read_sidecar(tmp_path / "closes" / f"{DEN}.json")

    assert close_of(sidecar, DEN_AUG13) == datetime(2026, 8, 14, 6, 59, tzinfo=UTC)
    assert close_of(sidecar, DEN_AUG14) == datetime(2026, 8, 15, 7, 0, tzinfo=UTC)


def test_a_void_is_left_out_of_the_count_and_named_apart(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    printed = pulled(tmp_path, offline, capsys)

    sidecar = read_sidecar(tmp_path / "closes" / f"{DEN}.json")

    assert DEN_VOID not in sidecar.markets
    assert sidecar.voided == (DEN_VOID,)
    assert printed["roots"][DEN]["settled_markets"] == 2
    assert printed["roots"][DEN]["voided"] == 1
    assert printed["settled_markets"] == 3
    assert printed["voided"] == 1


def test_the_printed_closes_bracket_what_the_sidecar_carries(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    printed = pulled(tmp_path, offline, capsys)

    assert printed["roots"][DEN]["first_close"] == "2026-08-14T06:59:00+00:00"
    assert printed["roots"][DEN]["last_close"] == "2026-08-15T07:00:00+00:00"


def test_every_row_of_a_second_page_lands_in_the_count(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    printed = pulled(
        tmp_path,
        offline,
        capsys,
        pages={
            DEN: [
                page([market_row(DEN_AUG13, LEGACY_CLOSE)], cursor="page2"),
                page(
                    [
                        market_row(DEN_AUG13_BRACKET, LEGACY_CLOSE, result="no"),
                        market_row(DEN_AUG14, MOVED_CLOSE),
                    ]
                ),
            ],
            NY: ny_pages(),
        },
    )

    sidecar = read_sidecar(tmp_path / "closes" / f"{DEN}.json")

    assert sorted(sidecar.markets) == [DEN_AUG13_BRACKET, DEN_AUG13, DEN_AUG14]
    assert printed["roots"][DEN]["settled_markets"] == 3


def test_a_root_the_bracket_reaches_nothing_of_carries_no_closes(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    printed = pulled(tmp_path, offline, capsys, pages={DEN: den_pages(), NY: [page([])]})

    assert read_sidecar(tmp_path / "closes" / f"{NY}.json").markets == {}
    assert printed["roots"][NY] == {
        "sha256": printed["roots"][NY]["sha256"],
        "settled_markets": 0,
        "voided": 0,
        "first_close": None,
        "last_close": None,
    }


def test_a_second_pull_refuses_to_overwrite_a_frozen_sidecar(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pulled(tmp_path, offline, capsys)
    offline(both_roots())

    with pytest.raises(FileExistsError):
        main(argv_for(scope_dir(tmp_path, name="again"), tmp_path / "closes"))


def test_a_scope_naming_no_root_of_the_cohort_writes_nothing(
    tmp_path: Path, offline: Offline
) -> None:
    offline(both_roots())
    out = tmp_path / "closes"

    with pytest.raises(ValueError, match="no low series"):
        main(argv_for(scope_dir(tmp_path), out, LOW))

    assert not out.exists()


def test_the_roots_swept_are_the_cohorts_own(
    tmp_path: Path,
    offline: Offline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    offline({LOW_SERIES: [page([market_row("KXLOWTDEN-26AUG13-T44", LEGACY_CLOSE)])]})
    scope = scope_dir(tmp_path, series=(DEN, LOW_SERIES))

    assert main(argv_for(scope, tmp_path / "closes", LOW)) == 0

    printed = json.loads(capsys.readouterr().out)
    assert sorted(printed["roots"]) == [LOW_SERIES]
    assert not (tmp_path / "closes" / f"{DEN}.json").exists()


@pytest.mark.parametrize("flag", ("--run-scope", "--out"))
def test_the_pull_will_not_run_without_the_flags_that_pin_it(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(without(tmp_path / "scope", tmp_path / "closes", flag))


def test_the_pull_names_no_rate_and_no_bar_on_the_command_line() -> None:
    flags = argument_flags(SCRIPT.read_text())

    assert flags == {"run_scope", "out", "cohort"}
    assert [flag for flag in flags if "rate" in flag or "bar" in flag] == []
