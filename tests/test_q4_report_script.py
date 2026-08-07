import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from bot.lag.lock_convergence import (
    STATION_DAY_MIN,
    _gate_payload,
    _replication_payload,
    decide,
)
from bot.lag.run_manifest import MANIFEST_NAME
from bot.lag.taker_flow_run import RESULTS_NAME
from bot.lag.tape_studies import RunScope, load_run_scope
from bot.replay.analysis_stations import HIGH, LOW
from bot.replay.run_scope import DISCOVERY, HOLDOUT
from scripts.q4_report import (
    ARRIVALS_QUERY,
    DEFAULT_RUN_ROOT,
    FLOOR_SOURCES,
    REPO_ROOT,
    SETTLES_NAME,
    build_parser,
    format_report,
    gather_archive,
    gather_settles,
    read_arrivals,
    run,
    slice_observations,
)
from tests.test_lock_convergence import (
    DISCOVERY_DAY,
    HOLDOUT_DAY,
    LADDER_ROWS,
    RUN_ID,
    SCOPE_END,
    SCOPE_START,
    SEED,
    SERIES,
    STATION,
    artifacts_dir,
    both_ladder_scope_dir,
    flat,
    scope_dir,
    spread,
)
from tests.test_q3_near_lock_script import write_state_db
from tests.test_tape_studies import (
    ADEQUATE_SAMPLES,
    SHORT_SAMPLES,
    seeded_repo,
    write_preregistration,
    write_rtt_samples,
)


UTC = timezone.utc
IEM_HOST = "mesonet.agron.iastate.edu"
ACIS_HOST = "data.rcc-acis.org"

BEFORE_WINDOW = datetime(2026, 7, 18, 6, tzinfo=UTC)
IN_DISCOVERY = datetime(2026, 7, 18, 18, tzinfo=UTC)
IN_HOLDOUT = datetime(2026, 7, 19, 18, tzinfo=UTC)
AFTER_WINDOW = datetime(2026, 7, 20, 12, tzinfo=UTC)

IEM_ROWS = (
    (BEFORE_WINDOW, "50.0"),
    (IN_DISCOVERY, "72.0"),
    (IN_HOLDOUT, "73.0"),
    (AFTER_WINDOW, "74.0"),
)
ACIS_HIGHS = {DISCOVERY_DAY: "72", HOLDOUT_DAY: "73"}

BOUNDARY_ARRIVAL = "2026-07-18 07:00:00.000000"
ARRIVAL_ROWS = (
    (STATION, "metar", BOUNDARY_ARRIVAL, "55", "2026-07-18 07:01:00.000000"),
    (STATION, "metar", "2026-07-18 18:00:00.000000", "72", "2026-07-18 18:05:00.000000"),
    (STATION, "metar", "2026-07-21 18:00:00.000000", "80", "2026-07-21 18:05:00.000000"),
)


def iem_body(rows: tuple[tuple[datetime, str], ...] = IEM_ROWS) -> str:
    header = "station,station_name,valid(UTC),tmpf\n"
    return header + "".join(
        f"DEN,DENVER INTL,{stamp.strftime('%Y-%m-%d %H:%M')},{tmpf}\n" for stamp, tmpf in rows
    )


def transport(
    seen: list[httpx.Request],
    *,
    rows: tuple[tuple[datetime, str], ...] = IEM_ROWS,
    highs: dict[date, str] | None = None,
) -> httpx.MockTransport:
    published = ACIS_HIGHS if highs is None else highs

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == IEM_HOST:
            return httpx.Response(200, text=iem_body(rows))
        if request.url.host == ACIS_HOST:
            query = parse_qs(request.url.query.decode())
            day = date.fromisoformat(query["sdate"][0])
            value = published.get(day)
            if value is None:
                return httpx.Response(200, json={"meta": {}, "data": []})
            return httpx.Response(200, json={"meta": {}, "data": [[day.isoformat(), value]]})
        raise AssertionError(f"unexpected host {request.url.host}")

    return httpx.MockTransport(handler)


def refusing(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise AssertionError(f"the cache should have answered {request.url}")

    return httpx.MockTransport(handler)


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    cache = tmp_path / "cache"
    cache.mkdir()
    return {
        "preregistration": write_preregistration(tmp_path / "preregistration.md"),
        "repo": seeded_repo(tmp_path / "tree"),
        "run_scope": scope_dir(tmp_path),
        "artifacts": artifacts_dir(tmp_path, LADDER_ROWS),
        "state_db": write_state_db(tmp_path / "state.db", ARRIVAL_ROWS),
        "cache": cache,
        "rtt_samples": write_rtt_samples(tmp_path / "samples.jsonl", ADEQUATE_SAMPLES),
    }


@pytest.fixture
def scope(paths: dict[str, Path]) -> RunScope:
    return load_run_scope(paths["run_scope"])


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "tape_studies"


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    real = httpx.AsyncClient

    def factory(*_args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real(transport=transport(seen), **kwargs)

    monkeypatch.setattr("scripts.q4_report.httpx.AsyncClient", factory)
    return seen


REQUIRED = (
    "--run-id",
    "--preregistration",
    "--run-scope",
    "--artifacts",
    "--state-db",
    "--cache",
    "--rtt-samples",
    "--floor-source",
    "--seed",
)


def argv_for(paths: dict[str, Path], run_root: Path) -> list[str]:
    return [
        "--run-id",
        RUN_ID,
        "--preregistration",
        str(paths["preregistration"]),
        "--run-scope",
        str(paths["run_scope"]),
        "--artifacts",
        str(paths["artifacts"]),
        "--state-db",
        str(paths["state_db"]),
        "--cache",
        str(paths["cache"]),
        "--rtt-samples",
        str(paths["rtt_samples"]),
        "--floor-source",
        "RTT_read",
        "--seed",
        str(SEED),
        "--run-root",
        str(run_root),
        "--repo",
        str(paths["repo"]),
    ]


def args_for(paths: dict[str, Path], run_root: Path) -> argparse.Namespace:
    return build_parser().parse_args(argv_for(paths, run_root))


def without(argv: list[str], flag: str) -> list[str]:
    index = argv.index(flag)
    return argv[:index] + argv[index + 2 :]


def results_of(run_root: Path) -> dict:
    return json.loads((run_root / RUN_ID / RESULTS_NAME).read_text())


async def test_the_archive_is_pulled_once_for_the_station_not_once_for_its_event_days(
    paths: dict[str, Path], scope: RunScope
) -> None:
    seen: list[httpx.Request] = []

    async with httpx.AsyncClient(transport=transport(seen)) as http:
        readings = await gather_archive(scope, paths["cache"], http)

    assert len({(day.station, day.event_date) for day in scope.event_days.values()}) == 2
    assert [request.url.host for request in seen] == [IEM_HOST]
    assert len(readings[STATION]) == len(IEM_ROWS)


async def test_the_archive_is_sliced_into_the_event_day_whose_window_holds_it(
    paths: dict[str, Path], scope: RunScope
) -> None:
    async with httpx.AsyncClient(transport=transport([])) as http:
        readings = await gather_archive(scope, paths["cache"], http)

    observations = slice_observations(scope, readings)

    assert set(observations) == {(STATION, DISCOVERY_DAY), (STATION, HOLDOUT_DAY)}
    assert [row.valid_time for row in observations[(STATION, DISCOVERY_DAY)]] == [IN_DISCOVERY]
    assert [row.valid_time for row in observations[(STATION, HOLDOUT_DAY)]] == [IN_HOLDOUT]
    assert [row.temp_f for row in observations[(STATION, DISCOVERY_DAY)]] == [Decimal("72.0")]


async def test_a_day_with_no_published_settle_is_absent_rather_than_carrying_a_placeholder(
    paths: dict[str, Path], scope: RunScope
) -> None:
    async with httpx.AsyncClient(transport=transport([], highs={DISCOVERY_DAY: "72"})) as http:
        settles = await gather_settles(scope, paths["cache"], http)

    assert settles == {(STATION, DISCOVERY_DAY): Decimal("72")}
    assert (STATION, HOLDOUT_DAY) not in settles
    assert json.loads((paths["cache"] / SETTLES_NAME).read_text()) == {
        f"{STATION} {DISCOVERY_DAY.isoformat()}": "72"
    }


async def test_a_populated_cache_answers_a_rerun_without_a_single_request(
    paths: dict[str, Path], scope: RunScope
) -> None:
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=transport(seen)) as http:
        first_archive = await gather_archive(scope, paths["cache"], http)
        first_settles = await gather_settles(scope, paths["cache"], http)
    assert len(seen) == 3

    seen.clear()
    async with httpx.AsyncClient(transport=refusing(seen)) as http:
        second_archive = await gather_archive(scope, paths["cache"], http)
        second_settles = await gather_settles(scope, paths["cache"], http)

    assert seen == []
    assert second_settles == first_settles
    assert second_archive == first_archive
    assert all(isinstance(value, Decimal) for value in second_settles.values())
    assert all(isinstance(row.temp_f, Decimal) for row in second_archive[STATION])


async def test_an_archive_reading_survives_the_cache_as_the_decimal_it_arrived_as(
    paths: dict[str, Path], scope: RunScope
) -> None:
    async with httpx.AsyncClient(transport=transport([])) as http:
        await gather_archive(scope, paths["cache"], http)
        cached = await gather_archive(scope, paths["cache"], http)

    assert [row.temp_f for row in cached[STATION]] == [Decimal(tmpf) for _, tmpf in IEM_ROWS]
    assert Decimal("72.0") in [row.temp_f for row in cached[STATION]]
    assert "72.0" in (paths["cache"] / f"{STATION}-2026-07-18-2026-07-20.jsonl").read_text()


def test_the_arrivals_read_bounds_obs_time_the_way_the_column_stores_it(
    paths: dict[str, Path], scope: RunScope
) -> None:
    arrivals = read_arrivals(paths["state_db"], scope)

    conn = sqlite3.connect(paths["state_db"])
    iso = conn.execute(
        ARRIVALS_QUERY,
        (
            STATION,
            SCOPE_START.replace(tzinfo=None).isoformat(),
            SCOPE_END.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
    ).fetchall()
    conn.close()

    assert [row.valid_time for row in arrivals[STATION]] == [
        datetime(2026, 7, 18, 7, tzinfo=UTC),
        datetime(2026, 7, 18, 18, tzinfo=UTC),
    ]
    assert BOUNDARY_ARRIVAL not in {row[2] for row in iso}
    assert iso == []


def test_the_arrivals_row_carries_the_publication_time_the_recorder_stamped(
    paths: dict[str, Path], scope: RunScope
) -> None:
    arrivals = read_arrivals(paths["state_db"], scope)

    first = arrivals[STATION][0]
    assert first.publication_time == datetime(2026, 7, 18, 7, 1, tzinfo=UTC)
    assert first.temp_f == Decimal("55")
    assert first.source == "metar"
    assert first.is_special is False
    assert first.raw == ""


def test_the_arrivals_query_plan_lands_in_the_log_and_seeks_the_index(
    paths: dict[str, Path], scope: RunScope, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="scripts.q4_report")

    read_arrivals(paths["state_db"], scope)

    assert "USING INDEX ix_ws_obs_arrivals_station_obs_time" in caplog.text
    assert "SEARCH ws_obs_arrivals" in caplog.text
    assert "SCAN" not in caplog.text


async def test_a_complete_run_writes_the_manifest_and_the_results_it_prints(
    paths: dict[str, Path],
    run_root: Path,
    mock_http: list[httpx.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert await run(args_for(paths, run_root)) == 0

    results = results_of(run_root)
    manifest = json.loads((run_root / RUN_ID / MANIFEST_NAME).read_text())
    assert results["manifest_sha256"] == manifest["sha256"]
    assert results["bootstrap_seed"] == SEED
    assert results["locks"]["clean_station_days_discovery"] == 1
    assert results["settles"]["unsettled_station_days"] == 0
    assert capsys.readouterr().out == format_report(results) + "\n"


async def test_the_report_reads_the_station_day_counts_before_the_half_life(
    paths: dict[str, Path],
    run_root: Path,
    mock_http: list[httpx.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert await run(args_for(paths, run_root)) == 0

    report = capsys.readouterr().out
    results = results_of(run_root)
    counts = report.index("== STATION-DAYS")
    half_life = report.index("== HALF-LIFE")
    assert counts < half_life
    assert report.index(f"n_min={results['station_day_min']}") < half_life
    assert report.index(f"ceiling={results['locks']['population_ceiling']}") < half_life
    assert counts < report.index("== ARRIVAL ANCHOR") < report.index("== POST-LOCK FILLS")


async def test_a_gate_and_a_replication_that_ran_are_rendered_off_the_verdicts_they_carry(
    paths: dict[str, Path],
    run_root: Path,
    mock_http: list[httpx.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert await run(args_for(paths, run_root)) == 0
    capsys.readouterr()
    decision = decide(
        spread("600", STATION_DAY_MIN, split=DISCOVERY),
        spread("600", (STATION_DAY_MIN + 1) // 2, split=HOLDOUT),
    )
    gate = _gate_payload(decision.gate)
    replication = _replication_payload(decision.replication)

    report = format_report(
        results_of(run_root)
        | {"gate": gate, "replication": replication, "replication_skipped": decision.skipped}
    )

    assert f"estimate={gate['estimate']} threshold={gate['threshold']}" in report
    assert f"p_value={gate['p_value']:.5f}" in report
    assert f"n={gate['n']} n_min={gate['n_min']}" in report
    assert f"passed={gate['passed']}" in report
    assert f"holdout={replication['holdout_estimate']}" in report
    assert (
        f"holdout_n={replication['holdout_n']} holdout_n_min={replication['holdout_n_min']}"
        in report
    )
    assert f"replicated={replication['replicated']}" in report
    assert f"undecidable={gate['undecidable']} passed={gate['passed']}" in report
    assert (
        f"undecidable={replication['undecidable']} replicated={replication['replicated']}" in report
    )
    assert "did not run" not in report


async def test_a_verdict_no_resample_moved_names_the_refusal_in_the_report(
    paths: dict[str, Path],
    run_root: Path,
    mock_http: list[httpx.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert await run(args_for(paths, run_root)) == 0
    capsys.readouterr()
    decision = decide(
        flat("600", STATION_DAY_MIN, split=DISCOVERY),
        flat("600", (STATION_DAY_MIN + 1) // 2, split=HOLDOUT),
    )

    report = format_report(
        results_of(run_root)
        | {
            "gate": _gate_payload(decision.gate),
            "replication": _replication_payload(decision.replication),
            "replication_skipped": decision.skipped,
        }
    )

    assert "undecidable=True passed=False" in report
    assert "undecidable=True replicated=False" in report


async def test_the_run_reuses_the_cache_the_first_attempt_left_behind(
    paths: dict[str, Path],
    run_root: Path,
    mock_http: list[httpx.Request],
) -> None:
    assert await run(args_for(paths, run_root)) == 0
    first = results_of(run_root)
    assert len(mock_http) == 3

    mock_http.clear()
    args = args_for(paths, run_root)
    args.run_id = f"{RUN_ID}-again"

    assert await run(args) == 0

    assert mock_http == []
    assert (
        json.loads((run_root / args.run_id / RESULTS_NAME).read_text())["locks"] == first["locks"]
    )


async def test_a_run_whose_floor_is_unavailable_writes_no_results(
    paths: dict[str, Path],
    run_root: Path,
    mock_http: list[httpx.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_rtt_samples(paths["rtt_samples"], SHORT_SAMPLES)

    assert await run(args_for(paths, run_root)) != 0

    captured = capsys.readouterr()
    assert "latency_floor" in captured.err
    assert captured.out == ""
    assert not run_root.exists()


@pytest.mark.parametrize("flag", REQUIRED)
def test_every_input_the_run_is_read_under_is_required(
    paths: dict[str, Path], run_root: Path, flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(without(argv_for(paths, run_root), flag))

    assert excinfo.value.code != 0
    assert flag in capsys.readouterr().err


def test_the_run_root_and_repo_default_to_the_tree_the_script_ships_in(
    paths: dict[str, Path], run_root: Path
) -> None:
    argv = without(without(argv_for(paths, run_root), "--run-root"), "--repo")

    args = build_parser().parse_args(argv)

    assert args.run_root == DEFAULT_RUN_ROOT
    assert args.repo == REPO_ROOT
    assert FLOOR_SOURCES == ("L", "RTT_read")


def test_the_cohort_is_optional_and_never_names_both(
    paths: dict[str, Path], run_root: Path
) -> None:
    argv = argv_for(paths, run_root)

    assert build_parser().parse_args(argv).cohort is None
    assert build_parser().parse_args([*argv, "--cohort", HIGH]).cohort == HIGH
    assert build_parser().parse_args([*argv, "--cohort", LOW]).cohort == LOW
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([*argv, "--cohort", "both"])
    assert excinfo.value.code != 0


async def test_a_two_ladder_scope_is_read_under_the_cohort_the_run_names(
    paths: dict[str, Path], run_root: Path, tmp_path: Path, mock_http: list[httpx.Request]
) -> None:
    paths["run_scope"] = both_ladder_scope_dir(tmp_path)

    with pytest.raises(ValueError, match="names no cohort"):
        await run(args_for(paths, run_root))

    named = build_parser().parse_args([*argv_for(paths, run_root), "--cohort", HIGH])

    assert await run(named) == 0
    assert results_of(run_root)["locks"]["cities"] == [SERIES]
