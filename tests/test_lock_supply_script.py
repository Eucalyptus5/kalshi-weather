import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from bot.lag.lock_convergence import ROUNDING_MARGIN_F
from bot.lag.lock_supply import DISCOVERY_STATION_DAY_MIN, HOLDOUT_STATION_DAY_MIN, POOLED
from bot.lag.observation_freeze import (
    MAX,
    StationDay,
    read_observation_index,
    write_observation_index,
    write_observation_sidecar,
)
from bot.markets.observation_window import observation_window
from bot.replay.run_scope import DISCOVERY, HOLDOUT
from scripts.lock_supply import build_parser, run
from tests.test_lock_supply import (
    RICH,
    RICH_TEMPS,
    SERIES,
    STATION,
    ZONE,
    closes_dir,
    readings_for,
    scope_path,
)
from tests.test_tape_studies import DISCOVERY_DAY, HOLDOUT_DAY


OBSERVED_AT = datetime(2026, 7, 20, tzinfo=timezone.utc)
REQUIRED = ("--run-scope", "--closes", "--observations")
LADDER_MARKETS = 8


def observations_dir(tmp_path: Path, days: Mapping[date, Sequence[str]]) -> Path:
    directory = tmp_path / "observations"
    directory.mkdir()
    rows = []
    for event_date, temps in days.items():
        start, end = observation_window(ZONE, event_date)
        rows.append(
            StationDay(
                station=STATION,
                event_date=event_date,
                extreme=MAX,
                window_start=start,
                window_end=end,
                readings=tuple(readings_for(event_date, temps)),
                acis_f=None,
            )
        )
    write_observation_sidecar(directory / f"{STATION}.json", STATION, ZONE, rows)
    write_observation_index(directory, OBSERVED_AT)
    return directory


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "run_scope": scope_path(tmp_path),
        "closes": closes_dir(tmp_path, {SERIES: {DISCOVERY_DAY: RICH, HOLDOUT_DAY: RICH}}),
        "observations": observations_dir(
            tmp_path, {DISCOVERY_DAY: RICH_TEMPS, HOLDOUT_DAY: RICH_TEMPS}
        ),
    }


def argv_for(paths: dict[str, Path]) -> list[str]:
    return [
        "--run-scope",
        str(paths["run_scope"]),
        "--closes",
        str(paths["closes"]),
        "--observations",
        str(paths["observations"]),
    ]


def without(argv: list[str], flag: str) -> list[str]:
    index = argv.index(flag)
    return argv[:index] + argv[index + 2 :]


def payload_of(paths: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> dict:
    assert run(build_parser().parse_args(argv_for(paths))) == 0
    return json.loads(capsys.readouterr().out)


def test_the_payload_states_both_floors_beside_the_measured_counts(
    paths: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    payload = payload_of(paths, capsys)

    assert payload["discovery_station_day_min"] == DISCOVERY_STATION_DAY_MIN == 30
    assert payload["holdout_station_day_min"] == HOLDOUT_STATION_DAY_MIN == 15
    assert payload["discovery_mid_day_station_days"] == 1
    assert payload["holdout_mid_day_station_days"] == 1
    assert payload["splits"][DISCOVERY]["mid_day_station_days"] == 1
    assert payload["splits"][HOLDOUT]["mid_day_station_days"] == 1


def test_every_split_reports_both_ratios_as_strings(
    paths: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    payload = payload_of(paths, capsys)

    for name in (DISCOVERY, HOLDOUT, POOLED):
        assert isinstance(payload["splits"][name]["ambiguous_to_clean"], str)
        assert isinstance(payload["splits"][name]["ambiguous_to_mid_day"], str)
    assert payload["splits"][POOLED]["ambiguous_to_clean"] == "0.5"
    assert payload["splits"][POOLED]["ambiguous_to_mid_day"] == "1"


def test_the_payload_carries_one_row_per_in_scope_cell(
    paths: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    payload = payload_of(paths, capsys)

    assert [row["event_date"] for row in payload["station_days"]] == [
        DISCOVERY_DAY.isoformat(),
        HOLDOUT_DAY.isoformat(),
    ]
    assert payload["roots"] == [SERIES]
    assert payload["markets"] == LADDER_MARKETS
    assert payload["by_root"][SERIES]["markets"] == LADDER_MARKETS
    assert payload["rounding_margin_f"] == str(ROUNDING_MARGIN_F)
    assert payload["observations_sha256"] == read_observation_index(paths["observations"]).sha256


@pytest.mark.parametrize("flag", REQUIRED)
def test_every_input_the_survey_is_read_under_is_required(
    paths: dict[str, Path], flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(without(argv_for(paths), flag))

    assert excinfo.value.code != 0
    assert flag in capsys.readouterr().err
