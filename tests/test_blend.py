import random
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from bot.forecast import blend
from bot.forecast.blend import (
    DISCOVERY,
    WEIGHT_QUANTUM,
    ClassScore,
    fit_weights,
    freeze_digest,
)
from bot.forecast.blend_score import blend_probability


BASE = date(2026, 3, 1)
ALPHA = "ecmwf_ifs025"
BETA = "icon_global"
GAMMA = "nbm_nbs"


HIGH = (Decimal("0.90"), Decimal("0.88"), Decimal("0.82"))
LOW = (Decimal("0.10"), Decimal("0.14"), Decimal("0.20"))


def leg_outcome(index: int) -> int:
    return 1 if (index * 7) % 11 < 6 else 0


# Each member is wrong on a different quarter of the legs, so no single column fits the outcomes
# and the solver has to spread weight over all of them.
def leg_probability(index: int, offset: int) -> Decimal:
    called = 1 - leg_outcome(index) if index % 4 == offset else leg_outcome(index)
    return HIGH[offset] if called else LOW[offset]


def corpus(
    days: int, members: tuple[str, ...], split: str = DISCOVERY, start: date = BASE
) -> list[ClassScore]:
    rows: list[ClassScore] = []
    for index in range(days):
        event_date = start + timedelta(days=index)
        ticker = f"KXHIGHNY-{event_date.strftime('%y%b%d').upper()}-B60"
        outcome = leg_outcome(index)
        for offset, member in enumerate(members):
            rows.append(
                ClassScore(
                    ticker=ticker,
                    event_date=event_date,
                    split=split,
                    member=member,
                    probability=leg_probability(index, offset),
                    outcome=outcome,
                )
            )
    return rows


@pytest.fixture
def two_member_records() -> list[ClassScore]:
    return corpus(24, (ALPHA, BETA))


@pytest.fixture
def three_member_records() -> list[ClassScore]:
    return corpus(31, (ALPHA, BETA, GAMMA))


@pytest.fixture
def long_records() -> list[ClassScore]:
    return corpus(220, (ALPHA, BETA, GAMMA))


def test_empty_records_raise():
    with pytest.raises(ValueError, match="no records"):
        fit_weights([])


def test_a_holdout_record_refuses_the_fit(two_member_records):
    tainted = list(two_member_records)
    tainted[9] = replace(tainted[9], split="holdout")
    with pytest.raises(ValueError) as excinfo:
        fit_weights(tainted)
    assert "2026-03-05" in str(excinfo.value)


def test_the_first_offending_event_date_is_named(two_member_records):
    tainted = list(two_member_records)
    tainted[9] = replace(tainted[9], split="holdout")
    tainted[3] = replace(tainted[3], split="holdout")
    with pytest.raises(ValueError) as excinfo:
        fit_weights(tainted)
    assert "2026-03-02" in str(excinfo.value)
    assert "2026-03-05" not in str(excinfo.value)


def test_a_ticker_missing_a_member_raises(three_member_records):
    offender = three_member_records[15].ticker
    thinned = [
        record
        for record in three_member_records
        if not (record.ticker == offender and record.member == GAMMA)
    ]
    with pytest.raises(ValueError) as excinfo:
        fit_weights(thinned)
    assert offender in str(excinfo.value)


@pytest.mark.parametrize(
    "days,members",
    [(24, (ALPHA, BETA)), (31, (ALPHA, BETA, GAMMA))],
)
def test_weights_are_non_negative_and_sum_to_one(days, members):
    fitted = fit_weights(corpus(days, members))
    assert fitted.members == tuple(sorted(members))
    assert all(weight >= 0 for weight in fitted.weights.values())
    assert sum(fitted.weights.values(), Decimal(0)) == Decimal("1")
    assert all(weight == weight.quantize(WEIGHT_QUANTUM) for weight in fitted.weights.values())


def test_a_ragged_fit_spends_the_full_quantum(three_member_records):
    fitted = fit_weights(three_member_records)
    assert any(len(weight.normalize().as_tuple().digits) > 6 for weight in fitted.weights.values())


def test_split_and_event_day_count_are_recorded(two_member_records):
    fitted = fit_weights(two_member_records)
    assert fitted.fitted_on_split == "discovery"
    assert fitted.fitted_on_event_days == 24


def test_a_negative_solver_weight_raises(monkeypatch, two_member_records):
    monkeypatch.setattr(blend, "nnls", lambda matrix, target: (np.array([-0.25, 0.75]), 0.0))
    with pytest.raises(ValueError, match="negative weight"):
        fit_weights(two_member_records)


def test_an_all_zero_solver_output_raises(monkeypatch, two_member_records):
    monkeypatch.setattr(blend, "nnls", lambda matrix, target: (np.zeros(2), 0.0))
    with pytest.raises(ValueError, match="no weight"):
        fit_weights(two_member_records)


def test_the_digest_is_stable_and_weight_sensitive(two_member_records):
    first = fit_weights(two_member_records)
    second = fit_weights(two_member_records)
    assert first.sha256 == second.sha256

    nudged = dict(first.weights)
    nudged[first.members[0]] = nudged[first.members[0]] + WEIGHT_QUANTUM
    payload = {
        "members": list(first.members),
        "weights": {member: str(nudged[member]) for member in first.members},
        "fitted_on_event_days": first.fitted_on_event_days,
        "fitted_on_split": first.fitted_on_split,
    }
    assert freeze_digest(payload) != first.sha256


def test_shuffled_records_fit_the_same_weights(three_member_records):
    shuffled = list(three_member_records)
    random.Random(11).shuffle(shuffled)
    assert fit_weights(shuffled).sha256 == fit_weights(three_member_records).sha256


def test_scoring_the_holdout_does_not_move_the_weights(long_records):
    fitted = fit_weights(long_records)
    assert fitted.fitted_on_event_days == 220

    holdout = corpus(30, (ALPHA, BETA, GAMMA), split="holdout", start=date(2026, 11, 1))
    per_ticker: dict[str, dict[str, Decimal]] = {}
    for record in holdout:
        per_ticker.setdefault(record.ticker, {})[record.member] = record.probability
    scored = [blend_probability(fitted, per_class) for per_class in per_ticker.values()]
    assert len(scored) == 30
    assert all(isinstance(value, Decimal) for value in scored)

    refitted = fit_weights(long_records)
    assert refitted.sha256 == fitted.sha256
    assert refitted.weights == fitted.weights
