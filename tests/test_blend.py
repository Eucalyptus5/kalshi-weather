import hashlib
import json
import random
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from decimal import Context, Decimal, localcontext

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

GOLDEN_WEIGHTS = {
    ALPHA: "0.315729131879",
    BETA: "0.320348620714",
    GAMMA: "0.363922247407",
}
GOLDEN_SHA = "eb69924c94455f46e135e036b102bca6f5ef57d74648c9374aea7cc0dc0528a7"


def leg_outcome(index: int) -> int:
    return 1 if (index * 7) % 11 < 6 else 0


# Each member is wrong on a different quarter of the legs, so no single column fits the outcomes
# and the solver has to spread weight over all of them.
def leg_probability(index: int, offset: int) -> Decimal:
    called = 1 - leg_outcome(index) if index % 4 == offset else leg_outcome(index)
    return HIGH[offset] if called else LOW[offset]


def corpus(
    days: int,
    members: tuple[str, ...],
    split: str = DISCOVERY,
    start: date = BASE,
    rungs: int = 1,
) -> list[ClassScore]:
    rows: list[ClassScore] = []
    for index in range(days):
        event_date = start + timedelta(days=index)
        for rung in range(rungs):
            slot = index * rungs + rung
            ticker = f"KXHIGHNY-{event_date.strftime('%y%b%d').upper()}-B{60 + 2 * rung}"
            outcome = leg_outcome(slot)
            for offset, member in enumerate(members):
                rows.append(
                    ClassScore(
                        ticker=ticker,
                        event_date=event_date,
                        split=split,
                        member=member,
                        probability=leg_probability(slot, offset),
                        outcome=outcome,
                    )
                )
    return rows


# The members of one ticker disagree about the outcome, which the upstream seam cannot produce.
# It is the only way to say which member's row the fit target is read off.
def directional(days: int) -> list[ClassScore]:
    rows: list[ClassScore] = []
    for index in range(days):
        event_date = BASE + timedelta(days=index)
        ticker = f"KXHIGHNY-{event_date.strftime('%y%b%d').upper()}-B60"
        outcome = leg_outcome(index)
        for member, probability, stored in (
            (ALPHA, Decimal(outcome), outcome),
            (BETA, Decimal("0.5"), 1 - outcome),
            (GAMMA, Decimal(1 - outcome), 1 - outcome),
        ):
            rows.append(
                ClassScore(
                    ticker=ticker,
                    event_date=event_date,
                    split=DISCOVERY,
                    member=member,
                    probability=probability,
                    outcome=stored,
                )
            )
    return rows


def ragged(ticker: str, event_date: date) -> list[ClassScore]:
    return [
        ClassScore(
            ticker=ticker,
            event_date=event_date,
            split=DISCOVERY,
            member=member,
            probability=Decimal("0.5"),
            outcome=1,
        )
        for member in (ALPHA, BETA)
    ]


@pytest.fixture
def two_member_records() -> list[ClassScore]:
    return corpus(24, (ALPHA, BETA))


@pytest.fixture
def three_member_records() -> list[ClassScore]:
    return corpus(31, (ALPHA, BETA, GAMMA))


@pytest.fixture
def long_records() -> list[ClassScore]:
    return corpus(220, (ALPHA, BETA, GAMMA))


def test_empty_records_raise() -> None:
    with pytest.raises(ValueError, match="no records"):
        fit_weights([])


def test_a_holdout_record_refuses_the_fit(two_member_records: list[ClassScore]) -> None:
    tainted = list(two_member_records)
    tainted[9] = replace(tainted[9], split="holdout")
    with pytest.raises(ValueError) as excinfo:
        fit_weights(tainted)
    assert "2026-03-05" in str(excinfo.value)


def test_the_first_offending_event_date_is_named(two_member_records: list[ClassScore]) -> None:
    tainted = list(two_member_records)
    tainted[9] = replace(tainted[9], split="holdout")
    tainted[3] = replace(tainted[3], split="holdout")
    with pytest.raises(ValueError) as excinfo:
        fit_weights(tainted)
    assert "2026-03-02" in str(excinfo.value)
    assert "2026-03-05" not in str(excinfo.value)


def test_a_ticker_missing_a_member_raises(three_member_records: list[ClassScore]) -> None:
    offender = three_member_records[15].ticker
    thinned = [
        record
        for record in three_member_records
        if not (record.ticker == offender and record.member == GAMMA)
    ]
    with pytest.raises(ValueError) as excinfo:
        fit_weights(thinned)
    assert offender in str(excinfo.value)


def test_a_ticker_carrying_one_member_twice_raises(three_member_records: list[ClassScore]) -> None:
    offender = three_member_records[15].ticker
    doubled = [
        replace(record, member=ALPHA)
        if record.ticker == offender and record.member == GAMMA
        else record
        for record in three_member_records
    ]
    with pytest.raises(ValueError) as excinfo:
        fit_weights(doubled)
    assert offender in str(excinfo.value)


def test_the_member_set_is_read_off_every_ticker(three_member_records: list[ClassScore]) -> None:
    first = three_member_records[0].ticker
    second = three_member_records[3].ticker
    thinned = [
        record
        for record in three_member_records
        if not (record.ticker == first and record.member == GAMMA)
    ]
    with pytest.raises(ValueError) as excinfo:
        fit_weights(thinned)
    assert first in str(excinfo.value)
    assert second not in str(excinfo.value)


def test_the_ragged_ticker_named_is_the_first_in_sorted_order(
    three_member_records: list[ClassScore],
) -> None:
    late = "KXHIGHNY-26MAR31-B99"
    early = "KXHIGHNY-26MAR01-B10"
    records = three_member_records + ragged(late, date(2026, 3, 31)) + ragged(early, BASE)
    with pytest.raises(ValueError) as excinfo:
        fit_weights(records)
    assert early in str(excinfo.value)
    assert late not in str(excinfo.value)


@pytest.mark.parametrize(
    "days,members",
    [(24, (ALPHA, BETA)), (31, (ALPHA, BETA, GAMMA))],
)
def test_weights_are_non_negative_and_sum_to_one(days: int, members: tuple[str, ...]) -> None:
    fitted = fit_weights(corpus(days, members))
    assert fitted.members == tuple(sorted(members))
    assert all(weight >= 0 for weight in fitted.weights.values())
    assert sum(fitted.weights.values(), Decimal(0)) == Decimal("1")
    assert all(weight == weight.quantize(WEIGHT_QUANTUM) for weight in fitted.weights.values())


def test_the_three_member_fit_lands_on_the_golden_weights(
    three_member_records: list[ClassScore],
) -> None:
    fitted = fit_weights(three_member_records)
    assert WEIGHT_QUANTUM == Decimal("1E-12")
    assert {member: str(weight) for member, weight in fitted.weights.items()} == GOLDEN_WEIGHTS
    assert fitted.sha256 == GOLDEN_SHA


def test_the_member_that_called_the_outcome_earns_the_weight() -> None:
    fitted = fit_weights(directional(40))
    assert fitted.weights[ALPHA] > fitted.weights[BETA]
    assert fitted.weights[ALPHA] > fitted.weights[GAMMA]
    assert fitted.weights[ALPHA] == Decimal(1)


def test_the_fit_holds_under_a_low_ambient_precision(
    three_member_records: list[ClassScore],
) -> None:
    with localcontext(Context(prec=6)):
        fitted = fit_weights(three_member_records)
    assert {member: str(weight) for member, weight in fitted.weights.items()} == GOLDEN_WEIGHTS
    assert fitted.sha256 == GOLDEN_SHA


def test_a_ragged_fit_spends_the_full_quantum(three_member_records: list[ClassScore]) -> None:
    fitted = fit_weights(three_member_records)
    assert any(len(weight.normalize().as_tuple().digits) > 6 for weight in fitted.weights.values())


def test_split_and_event_day_count_are_recorded(two_member_records: list[ClassScore]) -> None:
    fitted = fit_weights(two_member_records)
    assert fitted.fitted_on_split == "discovery"
    assert fitted.fitted_on_event_days == 24


def test_a_multi_rung_day_counts_once() -> None:
    records = corpus(10, (ALPHA, BETA, GAMMA), rungs=2)
    assert len({record.ticker for record in records}) == 20
    assert fit_weights(records).fitted_on_event_days == 10


def test_a_negative_solver_weight_raises(
    monkeypatch: pytest.MonkeyPatch, two_member_records: list[ClassScore]
) -> None:
    monkeypatch.setattr(blend, "nnls", lambda matrix, target: (np.array([-0.25, 0.75]), 0.0))
    with pytest.raises(ValueError, match="negative weight"):
        fit_weights(two_member_records)


def test_an_all_zero_solver_output_raises(
    monkeypatch: pytest.MonkeyPatch, two_member_records: list[ClassScore]
) -> None:
    monkeypatch.setattr(blend, "nnls", lambda matrix, target: (np.zeros(2), 0.0))
    with pytest.raises(ValueError, match="no weight"):
        fit_weights(two_member_records)


def test_the_quantised_overshoot_lands_on_the_largest_weight(
    monkeypatch: pytest.MonkeyPatch, three_member_records: list[ClassScore]
) -> None:
    monkeypatch.setattr(blend, "nnls", lambda matrix, target: (np.array([1.0, 1.0, 4.0]), 0.0))
    fitted = fit_weights(three_member_records)
    assert {member: str(weight) for member, weight in fitted.weights.items()} == {
        ALPHA: "0.166666666667",
        BETA: "0.166666666667",
        GAMMA: "0.666666666666",
    }
    assert sum(fitted.weights.values(), Decimal(0)) == Decimal("1")


def test_a_three_way_tie_anchors_on_the_lowest_member_name(
    monkeypatch: pytest.MonkeyPatch, three_member_records: list[ClassScore]
) -> None:
    monkeypatch.setattr(blend, "nnls", lambda matrix, target: (np.ones(3), 0.0))
    fitted = fit_weights(three_member_records)
    assert {member: str(weight) for member, weight in fitted.weights.items()} == {
        ALPHA: "0.333333333334",
        BETA: "0.333333333333",
        GAMMA: "0.333333333333",
    }
    assert sum(fitted.weights.values(), Decimal(0)) == Decimal("1")


def test_the_digest_is_the_specified_construction(
    three_member_records: list[ClassScore],
) -> None:
    fitted = fit_weights(three_member_records)
    expected = hashlib.sha256(
        json.dumps(
            {
                "members": list(fitted.members),
                "weights": {member: str(fitted.weights[member]) for member in fitted.members},
                "fitted_on_event_days": fitted.fitted_on_event_days,
                "fitted_on_split": fitted.fitted_on_split,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert fitted.sha256 == expected
    assert len(fitted.sha256) == 64


def test_the_digest_is_stable_and_weight_sensitive(two_member_records: list[ClassScore]) -> None:
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


def test_the_fitted_artefact_cannot_be_edited(three_member_records: list[ClassScore]) -> None:
    fitted = fit_weights(three_member_records)
    with pytest.raises(FrozenInstanceError):
        fitted.sha256 = "x"
    with pytest.raises(TypeError):
        fitted.weights[ALPHA] = Decimal("0.5")
    assert fitted.sha256 == GOLDEN_SHA


def test_shuffled_records_fit_the_same_weights(three_member_records: list[ClassScore]) -> None:
    shuffled = list(three_member_records)
    random.Random(11).shuffle(shuffled)
    assert fit_weights(shuffled).sha256 == fit_weights(three_member_records).sha256


def test_scoring_the_holdout_does_not_move_the_weights(long_records: list[ClassScore]) -> None:
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
