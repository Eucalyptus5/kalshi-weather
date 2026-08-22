from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.maker_headroom import BAND, MODELLED_NO, MODELLED_YES
from scripts.k1_report import CLOSED, NO_FEE_REGIME, OPEN, PUBLISHED_REGIME, build_parser


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "k1_report.py"
REGIME_HELP = "which fee_type the series api carries for weather series, and so which grid decides"


def invoke(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, cwd=REPO_ROOT
    )


@pytest.fixture(scope="module")
def payload() -> dict:
    completed = invoke("--regime", PUBLISHED_REGIME)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def cell(payload: dict, regime: str, size_kind: str, size: str, price: str) -> dict:
    matches = [
        item
        for item in payload["regimes"][regime]["cells"]
        if item["size_kind"] == size_kind and item["size"] == size and item["price"] == price
    ]
    assert len(matches) == 1
    return matches[0]


def test_the_published_regime_does_not_close_the_question(payload: dict) -> None:
    assert payload["selected_regime"] == PUBLISHED_REGIME
    assert payload["verdict"] == OPEN
    assert payload["regimes"][PUBLISHED_REGIME]["closed_on_arithmetic"] is False
    assert payload["regimes"][PUBLISHED_REGIME]["rate"] == "0.0175"


def test_both_grids_are_emitted_whichever_regime_decides(payload: dict) -> None:
    assert set(payload["regimes"]) == {PUBLISHED_REGIME, NO_FEE_REGIME}
    assert payload["half_tick_capture_cents"] == "0.50"
    for regime in (PUBLISHED_REGIME, NO_FEE_REGIME):
        assert len(payload["regimes"][regime]["cells"]) == 119


def test_the_published_counts_split_a_hundred_and_seventeen_two_and_none(payload: dict) -> None:
    assert payload["regimes"][PUBLISHED_REGIME]["counts"] == {
        "positive": 117,
        "zero": 2,
        "negative": 0,
    }
    assert payload["regimes"][NO_FEE_REGIME]["counts"] == {
        "positive": 119,
        "zero": 0,
        "negative": 0,
    }


@pytest.mark.parametrize(
    "size_kind, size, price, fee, headroom, leaves",
    [
        (MODELLED_YES, "12.36", "0.50", "0.06", "0.0146", True),
        (MODELLED_NO, "26", "0.50", "0.12", "0.0385", True),
        (BAND, "12", "0.50", "0.06", "0.0000", False),
        (BAND, "14", "0.50", "0.07", "0.0000", False),
        (BAND, "16", "0.50", "0.07", "0.0625", True),
        (MODELLED_YES, "12.36", "0.05", "0.02", "0.3382", True),
        (BAND, "13", "0.95", "0.02", "0.3462", True),
    ],
)
def test_anchor_cells_carry_the_preregistered_figures(
    payload: dict,
    size_kind: str,
    size: str,
    price: str,
    fee: str,
    headroom: str,
    leaves: bool,
) -> None:
    item = cell(payload, PUBLISHED_REGIME, size_kind, size, price)

    assert item["aggregate_fee"] == fee
    assert item["headroom_cents_per_contract"] == headroom
    assert item["leaves_headroom"] is leaves


def test_the_free_grid_leaves_the_whole_half_tick(payload: dict) -> None:
    item = cell(payload, NO_FEE_REGIME, BAND, "12", "0.50")

    assert payload["regimes"][NO_FEE_REGIME]["rate"] == "0"
    assert item["aggregate_fee"] == "0.00"
    assert item["headroom_cents_per_contract"] == "0.5000"


def test_the_free_regime_verdict_reads_the_free_grid() -> None:
    completed = invoke("--regime", NO_FEE_REGIME)
    body = json.loads(completed.stdout)

    assert completed.returncode == 0, completed.stderr
    assert body["selected_regime"] == NO_FEE_REGIME
    assert body["verdict"] == OPEN


def test_a_rate_that_swamps_the_half_tick_closes_the_selected_regime() -> None:
    completed = invoke("--regime", PUBLISHED_REGIME, "--rate", "0.25")
    body = json.loads(completed.stdout)

    assert completed.returncode == 0, completed.stderr
    assert body["regimes"][PUBLISHED_REGIME]["rate"] == "0.25"
    assert body["regimes"][PUBLISHED_REGIME]["counts"]["positive"] == 0
    assert body["verdict"] == CLOSED
    assert body["regimes"][NO_FEE_REGIME]["closed_on_arithmetic"] is False


def test_the_script_refuses_to_guess_the_regime() -> None:
    completed = invoke()

    assert completed.returncode == 2
    assert "--regime" in completed.stderr
    assert completed.stdout == ""


def test_the_regime_flag_names_the_series_api_field_and_not_a_schedule() -> None:
    text = " ".join(build_parser().format_help().split())

    assert REGIME_HELP in text
    assert "schedule" not in text


def test_the_rate_is_read_as_a_decimal_not_a_float() -> None:
    completed = invoke("--regime", PUBLISHED_REGIME, "--rate", "0.0175")
    body = json.loads(completed.stdout)
    item = cell(body, PUBLISHED_REGIME, BAND, "16", "0.50")

    assert Decimal(item["fee_cents_per_contract"]) == Decimal("0.4375")
    assert item["fee_cents_per_contract"] == "0.4375"
