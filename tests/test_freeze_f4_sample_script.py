from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.lag.forecast_sample import read_sample_freeze, sidecar_path
from scripts.freeze_f4_sample import (
    DEFAULT_OUT,
    FREEZE_NAME,
    REPO_ROOT,
    SAMPLE_PLAN,
    build_parser,
    main,
)
from tests.test_forecast_sample import tick_row, write_markets, write_ticks
from tests.test_tape_studies import argument_flags


pytestmark = pytest.mark.skipif(
    not SAMPLE_PLAN.exists(),
    reason="frozen study artifacts under data/ are not committed",
)

SCRIPT = REPO_ROOT / "scripts" / "freeze_f4_sample.py"
UTC = timezone.utc

CLOSE = datetime(2024, 10, 27, 3, 59, tzinfo=UTC)
RUNGS = ("KXHIGHMIA-24OCT26-T80", "KXHIGHMIA-24OCT26-B81.5")


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    markets = write_markets(
        tmp_path / "markets.parquet",
        [
            {
                "ticker": rung,
                "series_ticker": "KXHIGHMIA",
                "result": "no",
                "close_time": CLOSE,
                "floor_strike": None,
            }
            for rung in RUNGS
        ],
    )
    ticks = write_ticks(
        tmp_path / "ticks.parquet",
        [
            tick_row(rung, CLOSE - timedelta(hours=lead, minutes=5), "0.31", 4)
            for rung in RUNGS
            for lead in (24, 36)
        ],
    )
    monkeypatch.setattr("scripts.freeze_f4_sample.MARKETS", markets)
    monkeypatch.setattr("scripts.freeze_f4_sample.TICKS", ticks)
    return tmp_path / "out"


def test_freeze_writes_both_leads_into_one_file(
    artifacts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--out", str(artifacts)]) == 0
    summary = json.loads(capsys.readouterr().out)

    path = artifacts / FREEZE_NAME
    legs = read_sample_freeze(path)
    assert summary["out"] == str(path)
    assert summary["sidecar"] == str(sidecar_path(path))
    assert summary["bytes"] == path.stat().st_size
    assert summary["legs"] == 4
    assert summary["unread_days"] == 343
    assert summary["leads"] == {
        "24": {"legs": 2, "event_days": 1, "discovery_days": 1, "holdout_days": 0},
        "36": {"legs": 2, "event_days": 1, "discovery_days": 1, "holdout_days": 0},
    }
    assert json.loads(sidecar_path(path).read_text())["sha256"] == summary["sha256"]
    assert [leg.lead_hours for leg in legs] == [24, 24, 36, 36]
    assert {leg.ticker for leg in legs} == set(RUNGS)


def test_freeze_refuses_a_second_run(artifacts: Path) -> None:
    assert main(["--out", str(artifacts)]) == 0
    with pytest.raises(FileExistsError):
        main(["--out", str(artifacts)])


def test_out_defaults_to_the_study_inputs() -> None:
    assert build_parser().parse_args([]).out == DEFAULT_OUT
    assert DEFAULT_OUT == REPO_ROOT / "data" / "tape_studies" / "f4_inputs"
    assert argument_flags(SCRIPT.read_text()) == {"out"}
